// Copyright 2025 go-highway Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package nn

import (
	"fmt"
	stdmath "math"
	"testing"

	"github.com/ajroetker/go-highway/hwy/contrib/workerpool"
)

func TestRMSNorm(t *testing.T) {
	tests := []struct {
		name      string
		normSize  int
		useWeight bool
		gemma     bool
	}{
		{"normSize=4/no_weight", 4, false, false},
		{"normSize=4/with_weight", 4, true, false},
		{"normSize=8/no_weight", 8, false, false},
		{"normSize=8/with_weight", 8, true, false},
		{"normSize=16/with_weight", 16, true, false},
		{"normSize=64/with_weight", 64, true, false},
		{"normSize=256/with_weight", 256, true, false},
		{"normSize=4/gemma", 4, true, true},
		{"normSize=16/gemma", 16, true, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			numGroups := 3
			size := numGroups * tt.normSize
			input := make([]float32, size)
			for i := range input {
				input[i] = float32(i)*0.1 - float32(size)*0.05
			}

			var weight []float32
			if tt.useWeight {
				weight = make([]float32, tt.normSize)
				for i := range weight {
					weight[i] = 1.0 + float32(i)*0.01
				}
			}

			output := make([]float32, size)
			RMSNorm(input, output, tt.normSize, weight, 1e-5, tt.gemma)

			// Verify outputs: after RMSNorm without weight, ||x||_rms should be ~1
			if !tt.useWeight {
				for g := range numGroups {
					off := g * tt.normSize
					var sumSq float64
					for i := 0; i < tt.normSize; i++ {
						sumSq += float64(output[off+i]) * float64(output[off+i])
					}
					rms := stdmath.Sqrt(sumSq / float64(tt.normSize))
					if stdmath.Abs(rms-1.0) > 1e-3 {
						t.Errorf("group %d: RMS = %v, want ~1", g, rms)
					}
				}
			}
		})
	}
}

func TestRMSNormScalarMatch(t *testing.T) {
	tests := []struct {
		normSize int
		gemma    bool
	}{
		{4, false},
		{16, false},
		{64, false},
		{256, false},
		{4, true},
		{64, true},
	}

	for _, tt := range tests {
		name := fmt.Sprintf("normSize=%d/gemma=%v", tt.normSize, tt.gemma)
		t.Run(name, func(t *testing.T) {
			numGroups := 4
			size := numGroups * tt.normSize
			input := make([]float32, size)
			for i := range input {
				input[i] = float32(i)*0.1 - float32(size)*0.05
			}

			weight := make([]float32, tt.normSize)
			for i := range weight {
				weight[i] = 1.0 + float32(i)*0.01
			}

			simdOutput := make([]float32, size)
			scalarOutput := make([]float32, size)

			RMSNorm(input, simdOutput, tt.normSize, weight, 1e-5, tt.gemma)
			RMSNormScalar(input, scalarOutput, tt.normSize, weight, 1e-5, tt.gemma)

			for i := range simdOutput {
				if stdmath.Abs(float64(simdOutput[i]-scalarOutput[i])) > 1e-4 {
					t.Errorf("SIMD[%d] = %v, scalar[%d] = %v, mismatch",
						i, simdOutput[i], i, scalarOutput[i])
				}
			}
		})
	}
}

func TestRMSNormForwardSave(t *testing.T) {
	pool := workerpool.New(0)
	defer pool.Close()

	normSize := 16
	numGroups := 4
	size := numGroups * normSize

	input := make([]float32, size)
	for i := range input {
		input[i] = float32(i)*0.1 - float32(size)*0.05
	}

	weight := make([]float32, normSize)
	for i := range weight {
		weight[i] = 1.0 + float32(i)*0.01
	}

	// ForwardSave output
	saveOutput := make([]float32, size)
	savedRRMS := make([]float32, numGroups)
	RMSNormForwardSave(pool, input, saveOutput, normSize, weight, 1e-5, false, savedRRMS)

	// Regular output
	regularOutput := make([]float32, size)
	RMSNorm(input, regularOutput, normSize, weight, 1e-5, false)

	// Outputs must match
	for i := range saveOutput {
		if stdmath.Abs(float64(saveOutput[i]-regularOutput[i])) > 1e-5 {
			t.Errorf("ForwardSave[%d] = %v, Regular[%d] = %v, mismatch",
				i, saveOutput[i], i, regularOutput[i])
		}
	}

	// savedRRMS must be positive
	for i, r := range savedRRMS {
		if r <= 0 {
			t.Errorf("savedRRMS[%d] = %v, expected positive", i, r)
		}
	}
}

func TestRMSNormEmpty(t *testing.T) {
	// Should not panic
	RMSNorm[float32](nil, nil, 4, nil, 1e-5, false)
	RMSNorm([]float32{}, []float32{}, 4, nil, 1e-5, false)
}

func BenchmarkRMSNorm(b *testing.B) {
	sizes := []int{64, 256, 768, 1024}

	for _, normSize := range sizes {
		numGroups := 32
		size := numGroups * normSize

		input := make([]float32, size)
		output := make([]float32, size)
		weight := make([]float32, normSize)
		for i := range input {
			input[i] = float32(i) * 0.01
		}
		for i := range weight {
			weight[i] = 1.0
		}

		b.Run(fmt.Sprintf("SIMD/normSize=%d", normSize), func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				RMSNorm(input, output, normSize, weight, 1e-5, false)
			}
		})

		b.Run(fmt.Sprintf("Scalar/normSize=%d", normSize), func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				RMSNormScalar(input, output, normSize, weight, 1e-5, false)
			}
		})

		b.Run(fmt.Sprintf("LayerNorm/normSize=%d", normSize), func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				LayerNorm(input, output, normSize, weight, weight, 1e-5)
			}
		})
	}
}
