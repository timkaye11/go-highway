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
	"testing"

	"github.com/ajroetker/go-highway/hwy/contrib/workerpool"
)

func TestLoRABackward_AutoVsScalar(t *testing.T) {
	pool := workerpool.New(0)
	defer pool.Close()

	tests := []struct {
		batch, dIn, dOut, rank int
	}{
		{1, 8, 4, 2},
		{2, 16, 8, 4},
		{4, 32, 16, 8},
		{3, 7, 5, 2},
	}

	for _, tt := range tests {
		name := fmt.Sprintf("b%d_%dx%d_r%d", tt.batch, tt.dIn, tt.dOut, tt.rank)
		t.Run(name, func(t *testing.T) {
			gradOutput := testInputs(tt.batch * tt.dOut)
			x := testInputs(tt.batch * tt.dIn)
			W := testInputs(tt.dOut * tt.dIn)
			A := testInputs(tt.rank * tt.dIn)
			B := make([]float32, tt.dOut*tt.rank)
			for i := range B {
				B[i] = float32(i)*0.02 - 0.5
			}

			// h = x @ A^T — compute from scratch
			h := make([]float32, tt.batch*tt.rank)
			for i := range tt.batch {
				for r := range tt.rank {
					var sum float64
					for j := range tt.dIn {
						sum += float64(x[i*tt.dIn+j]) * float64(A[r*tt.dIn+j])
					}
					h[i*tt.rank+r] = float32(sum)
				}
			}

			scale := float32(0.5)

			// Auto
			aGX := make([]float32, tt.batch*tt.dIn)
			aGA := make([]float32, tt.rank*tt.dIn)
			aGB := make([]float32, tt.dOut*tt.rank)
			LoRABackwardAuto(pool, gradOutput, x, h, W, A, B, scale, aGX, aGA, aGB, tt.batch, tt.dIn, tt.dOut, tt.rank)

			// Scalar
			sGX := make([]float32, tt.batch*tt.dIn)
			sGA := make([]float32, tt.rank*tt.dIn)
			sGB := make([]float32, tt.dOut*tt.rank)
			LoRABackwardScalar(gradOutput, x, h, W, A, B, scale, sGX, sGA, sGB, tt.batch, tt.dIn, tt.dOut, tt.rank)

			allClose32(t, "gradX", aGX, sGX, 1e-2)
			allClose32(t, "gradA", aGA, sGA, 1e-2)
			allClose32(t, "gradB", aGB, sGB, 1e-2)
		})
	}
}

func TestLoRABackward_NilGrads(t *testing.T) {
	pool := workerpool.New(0)
	defer pool.Close()

	batch, dIn, dOut, rank := 2, 8, 4, 2
	gradOutput := testInputs(batch * dOut)
	x := testInputs(batch * dIn)
	W := testInputs(dOut * dIn)
	A := testInputs(rank * dIn)
	B := testInputs(dOut * rank)
	h := make([]float32, batch*rank)

	// Should not panic with nil gradient outputs
	LoRABackwardAuto[float32](pool, gradOutput, x, h, W, A, B, 0.5, nil, nil, nil, batch, dIn, dOut, rank)
}

func BenchmarkLoRABackward(b *testing.B) {
	pool := workerpool.New(0)
	defer pool.Close()

	batch, dIn, dOut, rank := 8, 768, 768, 16
	gradOutput := make([]float32, batch*dOut)
	x := make([]float32, batch*dIn)
	W := make([]float32, dOut*dIn)
	A := make([]float32, rank*dIn)
	B := make([]float32, dOut*rank)
	h := make([]float32, batch*rank)
	gradX := make([]float32, batch*dIn)
	gradA := make([]float32, rank*dIn)
	gradB := make([]float32, dOut*rank)

	for i := range gradOutput {
		gradOutput[i] = float32(i) * 0.001
	}
	for i := range x {
		x[i] = float32(i) * 0.001
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		clear(gradX)
		clear(gradA)
		clear(gradB)
		LoRABackwardAuto(pool, gradOutput, x, h, W, A, B, 0.5, gradX, gradA, gradB, batch, dIn, dOut, rank)
	}
}
