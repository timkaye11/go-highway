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

func TestFusedLoRADense_AutoVsScalar(t *testing.T) {
	pool := workerpool.New(0)
	defer pool.Close()

	tests := []struct {
		batch, inF, outF, rank int
	}{
		{1, 8, 4, 2},
		{2, 16, 8, 4},
		{4, 32, 16, 8},
		{3, 7, 5, 2},
	}

	for _, tt := range tests {
		name := fmt.Sprintf("b%d_%dx%d_r%d", tt.batch, tt.inF, tt.outF, tt.rank)
		t.Run(name, func(t *testing.T) {
			x := testInputs(tt.batch * tt.inF)
			W := testInputs(tt.outF * tt.inF)
			bias := testInputs(tt.outF)
			A := testInputs(tt.rank * tt.inF)
			B := make([]float32, tt.outF*tt.rank)
			for i := range B {
				B[i] = float32(i)*0.02 - 0.5
			}
			scale := float32(0.5)

			// Auto
			autoOut := make([]float32, tt.batch*tt.outF)
			autoH := make([]float32, tt.batch*tt.rank)
			FusedLoRADenseAuto(pool, x, W, bias, A, B, scale, autoOut, autoH,
				tt.batch, tt.inF, tt.outF, tt.rank)

			// Scalar
			scalarOut := make([]float32, tt.batch*tt.outF)
			scalarH := make([]float32, tt.batch*tt.rank)
			FusedLoRADenseScalar(x, W, bias, A, B, scale, scalarOut, scalarH,
				tt.batch, tt.inF, tt.outF, tt.rank)

			// Compare outputs
			for i := range autoOut {
				diff := stdmath.Abs(float64(autoOut[i] - scalarOut[i]))
				if diff > 1e-3 {
					t.Errorf("output[%d]: Auto=%v, Scalar=%v, diff=%v",
						i, autoOut[i], scalarOut[i], diff)
				}
			}

			// Compare h
			for i := range autoH {
				diff := stdmath.Abs(float64(autoH[i] - scalarH[i]))
				if diff > 1e-3 {
					t.Errorf("h[%d]: Auto=%v, Scalar=%v, diff=%v",
						i, autoH[i], scalarH[i], diff)
				}
			}
		})
	}
}

func TestFusedLoRADense_NoBias(t *testing.T) {
	pool := workerpool.New(0)
	defer pool.Close()

	batch, inF, outF, rank := 2, 8, 4, 2
	x := testInputs(batch * inF)
	W := testInputs(outF * inF)
	A := testInputs(rank * inF)
	B := testInputs(outF * rank)
	scale := float32(0.5)

	autoOut := make([]float32, batch*outF)
	FusedLoRADenseAuto[float32](pool, x, W, nil, A, B, scale, autoOut, nil, batch, inF, outF, rank)

	scalarOut := make([]float32, batch*outF)
	FusedLoRADenseScalar[float32](x, W, nil, A, B, scale, scalarOut, nil, batch, inF, outF, rank)

	for i := range autoOut {
		diff := stdmath.Abs(float64(autoOut[i] - scalarOut[i]))
		if diff > 1e-3 {
			t.Errorf("[%d]: Auto=%v, Scalar=%v", i, autoOut[i], scalarOut[i])
		}
	}
}

func TestFusedLoRADense_MatchesSeparate(t *testing.T) {
	pool := workerpool.New(0)
	defer pool.Close()

	batch, inF, outF, rank := 4, 16, 8, 4
	x := testInputs(batch * inF)
	W := testInputs(outF * inF)
	bias := testInputs(outF)
	A := testInputs(rank * inF)
	B := make([]float32, outF*rank)
	for i := range B {
		B[i] = float32(i)*0.02 - 0.5
	}
	scale := float32(0.5)

	// Fused
	fusedOut := make([]float32, batch*outF)
	FusedLoRADenseAuto(pool, x, W, bias, A, B, scale, fusedOut, nil,
		batch, inF, outF, rank)

	// Separate: DenseAuto + manual LoRA
	sepOut := make([]float32, batch*outF)
	DenseAuto(pool, x, W, bias, sepOut, batch, inF, outF)

	// Add LoRA: scale * (x @ A^T) @ B^T
	h := make([]float32, batch*rank)
	for i := range batch {
		for r := range rank {
			var sum float64
			for j := range inF {
				sum += float64(x[i*inF+j]) * float64(A[r*inF+j])
			}
			h[i*rank+r] = float32(sum)
		}
	}
	for i := range batch {
		for j := range outF {
			var sum float64
			for r := range rank {
				sum += float64(h[i*rank+r]) * float64(B[j*rank+r])
			}
			sepOut[i*outF+j] += float32(float64(scale) * sum)
		}
	}

	for i := range fusedOut {
		diff := stdmath.Abs(float64(fusedOut[i] - sepOut[i]))
		if diff > 1e-3 {
			t.Errorf("[%d]: fused=%v, separate=%v", i, fusedOut[i], sepOut[i])
		}
	}
}

func BenchmarkFusedLoRADense(b *testing.B) {
	pool := workerpool.New(0)
	defer pool.Close()

	batch, inF, outF, rank := 8, 768, 768, 16
	x := make([]float32, batch*inF)
	W := make([]float32, outF*inF)
	bias := make([]float32, outF)
	A := make([]float32, rank*inF)
	B := make([]float32, outF*rank)
	output := make([]float32, batch*outF)
	h := make([]float32, batch*rank)

	b.Run("Fused", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			FusedLoRADenseAuto(pool, x, W, bias, A, B, 0.5, output, h, batch, inF, outF, rank)
		}
	})

	b.Run("Separate", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			DenseAuto(pool, x, W, bias, output, batch, inF, outF)
		}
	})
}

func testInputs(n int) []float32 {
	s := make([]float32, n)
	for i := range s {
		s[i] = float32(i)*0.1 - float32(n)*0.05
	}
	return s
}
