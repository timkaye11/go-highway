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

package grad

import (
	"fmt"
	stdmath "math"
	"testing"

	"github.com/ajroetker/go-highway/hwy/contrib/workerpool"
)

func TestRMSNormBackward_AutoVsScalar(t *testing.T) {
	pool := workerpool.New(0)
	defer pool.Close()

	tests := []struct {
		normSize  int
		numGroups int
		gemma     bool
	}{
		{4, 2, false},
		{8, 3, false},
		{16, 4, false},
		{64, 2, false},
		{4, 2, true},
		{16, 3, true},
	}

	for _, tt := range tests {
		name := fmt.Sprintf("normSize=%d/groups=%d/gemma=%v", tt.normSize, tt.numGroups, tt.gemma)
		t.Run(name, func(t *testing.T) {
			size := tt.numGroups * tt.normSize

			gradOutput := make([]float32, size)
			savedInput := make([]float32, size)
			savedRRMS := make([]float32, tt.numGroups)
			weight := make([]float32, tt.normSize)

			for i := range gradOutput {
				gradOutput[i] = float32(i)*0.01 - float32(size)*0.005
			}
			for i := range savedInput {
				savedInput[i] = float32(i)*0.1 - float32(size)*0.05
			}

			// Compute forward to get savedRRMS
			for g := range tt.numGroups {
				off := g * tt.normSize
				var sumSq float64
				for i := range tt.normSize {
					x := float64(savedInput[off+i])
					sumSq += x * x
				}
				savedRRMS[g] = float32(1.0 / stdmath.Sqrt(sumSq/float64(tt.normSize)+1e-5))
			}

			for i := range weight {
				weight[i] = 1.0 + float32(i)*0.01
			}

			// Auto
			aGI := make([]float32, size)
			aGW := make([]float32, tt.normSize)
			RMSNormBackwardAuto(pool, gradOutput, savedInput, savedRRMS, weight, tt.normSize, tt.gemma, aGI, aGW)

			// Scalar
			sGI := make([]float32, size)
			sGW := make([]float32, tt.normSize)
			RMSNormBackwardScalar(gradOutput, savedInput, savedRRMS, weight, tt.normSize, tt.gemma, sGI, sGW)

			allClose32(t, "gradInput", aGI, sGI, 1e-3)
			allClose32(t, "gradWeight", aGW, sGW, 1e-3)
		})
	}
}

func TestRMSNormBackward_GradCheck(t *testing.T) {
	tests := []struct {
		normSize  int
		numGroups int
		gemma     bool
	}{
		{4, 2, false},
		{8, 2, false},
		{4, 2, true},
	}

	for _, tt := range tests {
		name := fmt.Sprintf("normSize=%d/groups=%d/gemma=%v", tt.normSize, tt.numGroups, tt.gemma)
		t.Run(name, func(t *testing.T) {
			size := tt.numGroups * tt.normSize

			input := make([]float32, size)
			gradOutput := make([]float32, size)
			weight := make([]float32, tt.normSize)

			for i := range input {
				input[i] = float32(i)*0.1 - float32(size)*0.05
			}
			for i := range gradOutput {
				gradOutput[i] = float32(i)*0.01 - float32(size)*0.005
			}
			for i := range weight {
				weight[i] = 1.0 + float32(i)*0.02
			}

			// Compute forward to get savedRRMS
			savedRRMS := make([]float32, tt.numGroups)
			for g := range tt.numGroups {
				off := g * tt.normSize
				var sumSq float64
				for i := range tt.normSize {
					x := float64(input[off+i])
					sumSq += x * x
				}
				savedRRMS[g] = float32(1.0 / stdmath.Sqrt(sumSq/float64(tt.normSize)+1e-5))
			}

			// Analytical gradient
			analyticalGI := make([]float32, size)
			RMSNormBackwardScalar(gradOutput, input, savedRRMS, weight, tt.normSize, tt.gemma,
				analyticalGI, nil)

			// Numerical gradient via finite differences
			numericalGI := make([]float32, size)
			RMSNormBackwardFiniteDiff(input, tt.normSize, weight, 1e-5, tt.gemma, gradOutput, numericalGI)

			allClose32(t, "gradInput_gradcheck", analyticalGI, numericalGI, 5e-3)
		})
	}
}

func TestRMSNormBackward_NoWeight(t *testing.T) {
	pool := workerpool.New(0)
	defer pool.Close()

	normSize := 8
	numGroups := 2
	size := numGroups * normSize

	gradOutput := make([]float32, size)
	savedInput := make([]float32, size)
	savedRRMS := make([]float32, numGroups)

	for i := range gradOutput {
		gradOutput[i] = float32(i) * 0.01
	}
	for i := range savedInput {
		savedInput[i] = float32(i)*0.1 - float32(size)*0.05
	}
	for g := range numGroups {
		off := g * normSize
		var sumSq float64
		for i := range normSize {
			x := float64(savedInput[off+i])
			sumSq += x * x
		}
		savedRRMS[g] = float32(1.0 / stdmath.Sqrt(sumSq/float64(normSize)+1e-5))
	}

	aGI := make([]float32, size)
	RMSNormBackwardAuto[float32](pool, gradOutput, savedInput, savedRRMS, nil, normSize, false, aGI, nil)

	sGI := make([]float32, size)
	RMSNormBackwardScalar[float32](gradOutput, savedInput, savedRRMS, nil, normSize, false, sGI, nil)

	allClose32(t, "gradInput_no_weight", aGI, sGI, 1e-4)
}

func BenchmarkRMSNormBackward(b *testing.B) {
	pool := workerpool.New(0)
	defer pool.Close()

	normSize := 768
	numGroups := 32
	size := numGroups * normSize

	gradOutput := make([]float32, size)
	savedInput := make([]float32, size)
	savedRRMS := make([]float32, numGroups)
	weight := make([]float32, normSize)
	gradInput := make([]float32, size)
	gradWeight := make([]float32, normSize)

	for i := range gradOutput {
		gradOutput[i] = float32(i) * 0.001
	}
	for i := range savedInput {
		savedInput[i] = float32(i) * 0.001
	}
	for i := range savedRRMS {
		savedRRMS[i] = 1.0
	}
	for i := range weight {
		weight[i] = 1.0
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		clear(gradInput)
		clear(gradWeight)
		RMSNormBackwardAuto(pool, gradOutput, savedInput, savedRRMS, weight, normSize, false, gradInput, gradWeight)
	}
}
