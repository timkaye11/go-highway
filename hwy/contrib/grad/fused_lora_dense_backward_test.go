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

func TestFusedLoRADenseBackward_AutoVsScalar(t *testing.T) {
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

			// Compute h = x @ A^T
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
			aGW := make([]float32, tt.dOut*tt.dIn)
			aGB := make([]float32, tt.dOut)
			aGA := make([]float32, tt.rank*tt.dIn)
			aGBl := make([]float32, tt.dOut*tt.rank)
			FusedLoRADenseBackwardAuto(pool, gradOutput, x, h, W, A, B, scale,
				aGX, aGW, aGB, aGA, aGBl, tt.batch, tt.dIn, tt.dOut, tt.rank)

			// Scalar
			sGX := make([]float32, tt.batch*tt.dIn)
			sGW := make([]float32, tt.dOut*tt.dIn)
			sGB := make([]float32, tt.dOut)
			sGA := make([]float32, tt.rank*tt.dIn)
			sGBl := make([]float32, tt.dOut*tt.rank)
			FusedLoRADenseBackwardScalar(gradOutput, x, h, W, A, B, scale,
				sGX, sGW, sGB, sGA, sGBl, tt.batch, tt.dIn, tt.dOut, tt.rank)

			allClose32(t, "gradX", aGX, sGX, 1e-3)
			allClose32(t, "gradWeight", aGW, sGW, 1e-3)
			allClose32(t, "gradBias", aGB, sGB, 1e-3)
			allClose32(t, "gradA", aGA, sGA, 1e-3)
			allClose32(t, "gradB", aGBl, sGBl, 1e-3)
		})
	}
}

func TestFusedLoRADenseBackward_MatchesSeparate(t *testing.T) {
	pool := workerpool.New(0)
	defer pool.Close()

	batch, dIn, dOut, rank := 4, 16, 8, 4
	gradOutput := testInputs(batch * dOut)
	x := testInputs(batch * dIn)
	W := testInputs(dOut * dIn)
	A := testInputs(rank * dIn)
	B := make([]float32, dOut*rank)
	for i := range B {
		B[i] = float32(i)*0.02 - 0.5
	}
	h := make([]float32, batch*rank)
	for i := range batch {
		for r := range rank {
			var sum float64
			for j := range dIn {
				sum += float64(x[i*dIn+j]) * float64(A[r*dIn+j])
			}
			h[i*rank+r] = float32(sum)
		}
	}
	scale := float32(0.5)

	// Fused
	fGX := make([]float32, batch*dIn)
	fGW := make([]float32, dOut*dIn)
	fGB := make([]float32, dOut)
	fGA := make([]float32, rank*dIn)
	fGBl := make([]float32, dOut*rank)
	FusedLoRADenseBackwardAuto(pool, gradOutput, x, h, W, A, B, scale,
		fGX, fGW, fGB, fGA, fGBl, batch, dIn, dOut, rank)

	// Separate: DenseBackward + LoRABackward
	sGX := make([]float32, batch*dIn)
	sGW := make([]float32, dOut*dIn)
	sGB := make([]float32, dOut)
	DenseBackwardAuto(pool, gradOutput, x, W, sGX, sGW, sGB, batch, dIn, dOut)

	sGA := make([]float32, rank*dIn)
	sGBl := make([]float32, dOut*rank)
	// Add LoRA part to gradX
	loraGX := make([]float32, batch*dIn)
	LoRABackwardAuto(pool, gradOutput, x, h, W, A, B, scale, loraGX, sGA, sGBl,
		batch, dIn, dOut, rank)
	// LoRABackward computes the full gradX (base + LoRA), same as the fused version.
	allClose32(t, "gradX", fGX, loraGX, 1e-3)
	allClose32(t, "gradWeight", fGW, sGW, 1e-3)
	allClose32(t, "gradBias", fGB, sGB, 1e-3)
	allClose32(t, "gradA", fGA, sGA, 1e-3)
	allClose32(t, "gradB", fGBl, sGBl, 1e-3)
}

func TestFusedLoRADenseBackward_NilGrads(t *testing.T) {
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
	FusedLoRADenseBackwardAuto[float32](pool, gradOutput, x, h, W, A, B, 0.5,
		nil, nil, nil, nil, nil, batch, dIn, dOut, rank)
}

func BenchmarkFusedLoRADenseBackward(b *testing.B) {
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
	gradWeight := make([]float32, dOut*dIn)
	gradBias := make([]float32, dOut)
	gradA := make([]float32, rank*dIn)
	gradB := make([]float32, dOut*rank)

	b.Run("Fused", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			clear(gradX)
			clear(gradWeight)
			clear(gradBias)
			clear(gradA)
			clear(gradB)
			FusedLoRADenseBackwardAuto(pool, gradOutput, x, h, W, A, B, 0.5,
				gradX, gradWeight, gradBias, gradA, gradB, batch, dIn, dOut, rank)
		}
	})

	b.Run("Separate", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			clear(gradX)
			clear(gradWeight)
			clear(gradBias)
			clear(gradA)
			clear(gradB)
			DenseBackwardAuto(pool, gradOutput, x, W, gradX, gradWeight, gradBias, batch, dIn, dOut)
			LoRABackwardAuto(pool, gradOutput, x, h, W, A, B, 0.5, gradX, gradA, gradB, batch, dIn, dOut, rank)
		}
	})
}
