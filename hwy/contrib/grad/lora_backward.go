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
	"github.com/ajroetker/go-highway/hwy"
	"github.com/ajroetker/go-highway/hwy/contrib/matmul"
	"github.com/ajroetker/go-highway/hwy/contrib/vec"
	"github.com/ajroetker/go-highway/hwy/contrib/workerpool"
)

// LoRABackwardAuto computes the backward pass for a LoRA-adapted dense layer.
//
// Forward: y = x @ W^T + scale * (x @ A^T) @ B^T
//
// Backward:
//
//	dB += scale * h^T @ gradOutput         [rank, dOut]   where h = x @ A^T
//	dA += scale * x^T @ (gradOutput @ B)   [dIn, rank]    (transposed to match A layout)
//	dX += gradOutput @ W^T + scale * (gradOutput @ B) @ A^T   [batch, dIn]
//
// Parameters:
//   - gradOutput: [batchSize, dOut]
//   - x:          [batchSize, dIn]  — input saved from forward
//   - h:          [batchSize, rank] — intermediate x @ A^T saved from forward
//   - W:          [dOut, dIn]       — frozen weight (only needed if gradX != nil)
//   - A:          [rank, dIn]       — LoRA down-projection
//   - B:          [dOut, rank]      — LoRA up-projection (stored as [dOut, rank])
//   - scale:      LoRA scaling factor (alpha/rank)
//   - gradX:      [batchSize, dIn]  — accumulated (nil to skip)
//   - gradA:      [rank, dIn]       — accumulated (nil to skip)
//   - gradB:      [dOut, rank]      — accumulated (nil to skip)
func LoRABackwardAuto[T hwy.Floats](
	pool *workerpool.Pool,
	gradOutput, x, h []T,
	W, A, B []T,
	scale T,
	gradX, gradA, gradB []T,
	batchSize, dIn, dOut, rank int,
) {
	// dB += scale * h^T @ gradOutput
	// h is [batch, rank], h^T is [rank, batch]
	// h^T @ gradOutput[batch, dOut] => [rank, dOut]
	// But B is stored as [dOut, rank], so we need gradB in [dOut, rank] format.
	// Actually: dB[dOut, rank] += scale * gradOutput^T[dOut, batch] @ h[batch, rank]
	if gradB != nil {
		goT := make([]T, dOut*batchSize)
		matmul.TransposeAuto(pool, gradOutput, batchSize, dOut, goT)
		temp := make([]T, dOut*rank)
		matmul.MatMulAuto(pool, goT, h, temp, dOut, rank, batchSize)
		vec.MulConstAddTo(gradB, scale, temp)
	}

	// Shared computation: gradOutput @ B = gradOutput[batch, dOut] @ B[dOut, rank]
	// MatMulAuto: C[m,n] = A[m,k] @ B[k,n]
	// We need: result[batch, rank] = gradOutput[batch, dOut] @ B^T_col[dOut, rank]
	// B is [dOut, rank], this is already A[m,k] @ B[k,n] with m=batch, k=dOut, n=rank
	var goB []T
	if gradA != nil || gradX != nil {
		goB = make([]T, batchSize*rank)
		matmul.MatMulAuto(pool, gradOutput, B, goB, batchSize, rank, dOut)
	}

	// dA += scale * x^T @ goB
	// x^T is [dIn, batch], goB is [batch, rank]
	// => [dIn, rank], but A is [rank, dIn], so we need result in [rank, dIn].
	// Compute: goB^T[rank, batch] @ x[batch, dIn] => [rank, dIn]
	if gradA != nil {
		goBT := make([]T, rank*batchSize)
		matmul.TransposeAuto(pool, goB, batchSize, rank, goBT)
		temp := make([]T, rank*dIn)
		matmul.MatMulAuto(pool, goBT, x, temp, rank, dIn, batchSize)
		vec.MulConstAddTo(gradA, scale, temp)
	}

	// dX += gradOutput @ W^T + scale * goB @ A^T
	// gradOutput @ W: gradOutput[batch, dOut], W[dOut, dIn]
	// MatMulAuto: C[m,n] = A[m,k] @ B[k,n]
	// => temp1[batch, dIn] = gradOutput[batch, dOut] @ W[dOut, dIn] (m=batch, n=dIn, k=dOut)
	if gradX != nil {
		temp1 := make([]T, batchSize*dIn)
		matmul.MatMulAuto(pool, gradOutput, W, temp1, batchSize, dIn, dOut)
		vec.Add(gradX, temp1)

		// scale * goB @ A^T
		// goB[batch, rank], A[rank, dIn]
		// goB @ A = MatMulAuto(goB, A, temp2, batch, dIn, rank)
		temp2 := make([]T, batchSize*dIn)
		matmul.MatMulAuto(pool, goB, A, temp2, batchSize, dIn, rank)
		vec.MulConstAddTo(gradX, scale, temp2)
	}
}

// LoRABackwardScalar is a scalar reference implementation for testing.
func LoRABackwardScalar[T hwy.Floats](
	gradOutput, x, h []T,
	W, A, B []T,
	scale T,
	gradX, gradA, gradB []T,
	batchSize, dIn, dOut, rank int,
) {
	// dB[dOut, rank] += scale * gradOutput^T[dOut, batch] @ h[batch, rank]
	if gradB != nil {
		for o := range dOut {
			for r := range rank {
				var sum float64
				for i := range batchSize {
					sum += float64(gradOutput[i*dOut+o]) * float64(h[i*rank+r])
				}
				gradB[o*rank+r] += T(float64(scale) * sum)
			}
		}
	}

	// goB = gradOutput @ B (precompute)
	goB := make([]T, batchSize*rank)
	for i := range batchSize {
		for r := range rank {
			var sum float64
			for o := range dOut {
				sum += float64(gradOutput[i*dOut+o]) * float64(B[o*rank+r])
			}
			goB[i*rank+r] = T(sum)
		}
	}

	// dA[rank, dIn] += scale * goB^T[rank, batch] @ x[batch, dIn]
	if gradA != nil {
		for r := range rank {
			for j := range dIn {
				var sum float64
				for i := range batchSize {
					sum += float64(goB[i*rank+r]) * float64(x[i*dIn+j])
				}
				gradA[r*dIn+j] += T(float64(scale) * sum)
			}
		}
	}

	// dX += gradOutput @ W + scale * goB @ A
	if gradX != nil {
		for i := range batchSize {
			for j := range dIn {
				// gradOutput @ W
				var sum1 float64
				for o := range dOut {
					sum1 += float64(gradOutput[i*dOut+o]) * float64(W[o*dIn+j])
				}
				// scale * goB @ A
				var sum2 float64
				for r := range rank {
					sum2 += float64(goB[i*rank+r]) * float64(A[r*dIn+j])
				}
				gradX[i*dIn+j] += T(sum1 + float64(scale)*sum2)
			}
		}
	}
}
