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
	stdmath "math"
	"testing"

	"github.com/ajroetker/go-highway/hwy/contrib/matmul"
	"github.com/ajroetker/go-highway/hwy/contrib/workerpool"
)

func TestFusedQLoRADense_NF4(t *testing.T) {
	pool := workerpool.New(0)
	defer pool.Close()

	// Small dimensions for testing
	batch, inF, outF, rank := 2, 16, 8, 4
	groupSize := 4 // outF must be divisible by groupSize

	x := testInputs(batch * inF)
	A := testInputs(rank * inF)
	B := make([]float32, outF*rank)
	for i := range B {
		B[i] = float32(i)*0.02 - 0.5
	}
	bias := testInputs(outF)
	scale := float32(0.5)

	// Create simple quantized weights: all zeros for the quantized part
	// This means the base matmul produces zeros, so output ≈ bias + scale * loRA
	packedSize := inF * outF / 2
	packed := make([]uint8, packedSize)
	// NF4 index 7 = 0.0, so both nibbles = 0x77 gives zeros
	for i := range packed {
		packed[i] = 0x77
	}
	numGroups := outF / groupSize
	scales := make([]float32, inF*numGroups)
	for i := range scales {
		scales[i] = 1.0
	}

	fusedOut := make([]float32, batch*outF)
	FusedQLoRADenseAuto(pool, x, packed, scales, groupSize, bias, A, B, scale,
		fusedOut, nil, batch, inF, outF, rank, QuantNF4)

	// Separate: base (quantized giving ~zeros) + bias + LoRA
	sepOut := make([]float32, batch*outF)
	matmul.BaseFusedNF4MatMul(x, packed, scales, sepOut, batch, inF, outF, groupSize)
	addBias(sepOut, bias, batch, outF)

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
			t.Errorf("[%d]: fused=%v, separate=%v, diff=%v", i, fusedOut[i], sepOut[i], diff)
		}
	}
}
