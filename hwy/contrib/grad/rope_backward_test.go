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

	"github.com/ajroetker/go-highway/hwy/contrib/nn"
	"github.com/ajroetker/go-highway/hwy/contrib/workerpool"
)

func TestRoPEBackward_SIMDvsScalar(t *testing.T) {
	tests := []struct {
		seqLen, headDim int
	}{
		{1, 4},
		{4, 8},
		{8, 16},
		{16, 64},
		{3, 6},
	}

	for _, tt := range tests {
		name := fmt.Sprintf("seq=%d_dim=%d", tt.seqLen, tt.headDim)
		t.Run(name, func(t *testing.T) {
			size := tt.seqLen * tt.headDim
			cos, sin := nn.PrecomputeRoPE[float32](tt.seqLen, tt.headDim, 10000.0)

			// SIMD
			simdData := make([]float32, size)
			for i := range simdData {
				simdData[i] = float32(i)*0.1 - float32(size)*0.05
			}

			scalarData := make([]float32, size)
			copy(scalarData, simdData)

			RoPEBackward(simdData, cos, sin, tt.seqLen, tt.headDim)
			RoPEBackwardScalar(scalarData, cos, sin, tt.seqLen, tt.headDim)

			for i := range simdData {
				diff := stdmath.Abs(float64(simdData[i] - scalarData[i]))
				if diff > 1e-4 {
					t.Errorf("[%d] SIMD=%v, Scalar=%v, diff=%v", i, simdData[i], scalarData[i], diff)
				}
			}
		})
	}
}

func TestRoPEBackward_InverseOfForward(t *testing.T) {
	// RoPE forward then backward should give back the original data
	seqLen, headDim := 8, 16
	size := seqLen * headDim

	cos, sin := nn.PrecomputeRoPE[float32](seqLen, headDim, 10000.0)

	original := make([]float32, size)
	for i := range original {
		original[i] = float32(i)*0.2 - 3.0
	}

	data := make([]float32, size)
	copy(data, original)

	// Forward
	nn.RoPE(data, cos, sin, seqLen, headDim)

	// Backward (should undo the rotation)
	RoPEBackward(data, cos, sin, seqLen, headDim)

	// Should match original
	for i := range data {
		diff := stdmath.Abs(float64(data[i] - original[i]))
		if diff > 1e-4 {
			t.Errorf("[%d] roundtrip=%v, original=%v, diff=%v", i, data[i], original[i], diff)
		}
	}
}

func TestRoPEBackward_GradCheck(t *testing.T) {
	// Verify gradient correctness via finite differences
	seqLen, headDim := 4, 8
	size := seqLen * headDim

	cos, sin := nn.PrecomputeRoPE[float32](seqLen, headDim, 10000.0)

	input := make([]float32, size)
	for i := range input {
		input[i] = float32(i)*0.2 - 2.0
	}

	gradOutput := make([]float32, size)
	for i := range gradOutput {
		gradOutput[i] = float32(i)*0.05 - 1.0
	}

	// Analytical: apply backward to gradOutput
	analyticalGrad := make([]float32, size)
	copy(analyticalGrad, gradOutput)
	RoPEBackward(analyticalGrad, cos, sin, seqLen, headDim)

	// Numerical via finite differences
	eps := float32(1e-4)
	numericalGrad := make([]float32, size)
	output1 := make([]float32, size)
	output2 := make([]float32, size)

	for i := range size {
		orig := input[i]

		// f(x+eps)
		input[i] = orig + eps
		copy(output1, input)
		nn.RoPEScalar(output1, cos, sin, seqLen, headDim)

		// f(x-eps)
		input[i] = orig - eps
		copy(output2, input)
		nn.RoPEScalar(output2, cos, sin, seqLen, headDim)

		input[i] = orig

		// Numerical grad = sum(gradOutput * (f(x+eps) - f(x-eps)) / (2*eps))
		var sum float64
		for j := range size {
			sum += float64(gradOutput[j]) * float64(output1[j]-output2[j]) / float64(2*eps)
		}
		numericalGrad[i] = float32(sum)
	}

	allClose32(t, "rope_gradcheck", analyticalGrad, numericalGrad, 5e-3)
}

func TestRoPEBackwardAuto(t *testing.T) {
	pool := workerpool.New(0)
	defer pool.Close()

	numQHeads, numKVHeads := 4, 2
	seqLen, headDim := 8, 16

	cos, sin := nn.PrecomputeRoPE[float32](seqLen, headDim, 10000.0)

	headStride := seqLen * headDim
	gradQ := make([]float32, numQHeads*headStride)
	gradK := make([]float32, numKVHeads*headStride)
	for i := range gradQ {
		gradQ[i] = float32(i)*0.01 - 1.0
	}
	for i := range gradK {
		gradK[i] = float32(i)*0.02 - 0.5
	}

	// Copy for scalar reference
	sGQ := make([]float32, len(gradQ))
	sGK := make([]float32, len(gradK))
	copy(sGQ, gradQ)
	copy(sGK, gradK)

	// Auto
	RoPEBackwardAuto(pool, gradQ, gradK, cos, sin, numQHeads, numKVHeads, seqLen, headDim)

	// Scalar per-head
	for h := range numQHeads {
		off := h * headStride
		RoPEBackwardScalar(sGQ[off:off+headStride], cos, sin, seqLen, headDim)
	}
	for h := range numKVHeads {
		off := h * headStride
		RoPEBackwardScalar(sGK[off:off+headStride], cos, sin, seqLen, headDim)
	}

	allClose32(t, "gradQ", gradQ, sGQ, 1e-5)
	allClose32(t, "gradK", gradK, sGK, 1e-5)
}

func BenchmarkRoPEBackward(b *testing.B) {
	configs := []struct {
		seqLen, headDim int
	}{
		{128, 64},
		{256, 64},
		{512, 128},
	}

	for _, c := range configs {
		cos, sin := nn.PrecomputeRoPE[float32](c.seqLen, c.headDim, 10000.0)
		data := make([]float32, c.seqLen*c.headDim)
		for i := range data {
			data[i] = float32(i) * 0.01
		}

		b.Run(fmt.Sprintf("Func/seq=%d_dim=%d", c.seqLen, c.headDim), func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				RoPEBackward(data, cos, sin, c.seqLen, c.headDim)
			}
		})

		b.Run(fmt.Sprintf("Scalar/seq=%d_dim=%d", c.seqLen, c.headDim), func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				RoPEBackwardScalar(data, cos, sin, c.seqLen, c.headDim)
			}
		})
	}
}
