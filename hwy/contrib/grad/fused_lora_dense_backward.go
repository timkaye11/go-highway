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

// FusedLoRADenseBackwardAuto computes the backward pass for a fused LoRA dense
// layer, sharing intermediates between the base and LoRA gradient computations.
//
// Forward: output = x @ W^T + bias + scale * (x @ A^T) @ B^T
//
// The key optimizations over calling DenseBackwardAuto + LoRABackwardAuto separately:
//   - transpose(gradOutput) computed once, used by gradWeight and gradB
//   - goB = gradOutput @ B shared between gradA and the LoRA part of gradX
//
// Parameters:
//   - gradOutput: [batchSize, outFeatures] — upstream gradient
//   - x:          [batchSize, inFeatures] — saved input
//   - h:          [batchSize, rank] — saved intermediate x @ A^T from forward
//   - W:          [outFeatures, inFeatures] — frozen base weight
//   - A:          [rank, inFeatures] — LoRA down-projection
//   - B:          [outFeatures, rank] — LoRA up-projection
//   - scale:      LoRA scaling factor
//   - gradX:      [batchSize, inFeatures] — accumulated (nil to skip)
//   - gradWeight: [outFeatures, inFeatures] — accumulated (nil to skip)
//   - gradBias:   [outFeatures] — accumulated (nil to skip)
//   - gradA:      [rank, inFeatures] — accumulated (nil to skip)
//   - gradB:      [outFeatures, rank] — accumulated (nil to skip)
func FusedLoRADenseBackwardAuto[T hwy.Floats](
	pool *workerpool.Pool,
	gradOutput, x, h, W, A, B []T, scale T,
	gradX, gradWeight, gradBias, gradA, gradB []T,
	batchSize, inFeatures, outFeatures, rank int,
) {
	// Step 1: goT = transpose(gradOutput) — SHARED by gradWeight and gradB
	var goT []T
	if gradWeight != nil || gradB != nil {
		goT = make([]T, outFeatures*batchSize)
		matmul.TransposeAuto(pool, gradOutput, batchSize, outFeatures, goT)
	}

	// Step 2: gradWeight += goT @ x (uses shared goT)
	if gradWeight != nil {
		temp := make([]T, outFeatures*inFeatures)
		matmul.MatMulAuto(pool, goT, x, temp, outFeatures, inFeatures, batchSize)
		vec.Add(gradWeight, temp)
	}

	// Step 3: gradB += scale * goT @ h (uses shared goT)
	if gradB != nil {
		temp := make([]T, outFeatures*rank)
		matmul.MatMulAuto(pool, goT, h, temp, outFeatures, rank, batchSize)
		vec.MulConstAddTo(gradB, scale, temp)
	}

	// Step 4: gradBias += colSum(gradOutput)
	if gradBias != nil {
		biasGradSum(pool, gradOutput, gradBias, batchSize, outFeatures)
	}

	// Step 5: goB = gradOutput @ B — SHARED by gradA and gradX
	var goB []T
	if gradA != nil || gradX != nil {
		goB = make([]T, batchSize*rank)
		matmul.MatMulAuto(pool, gradOutput, B, goB, batchSize, rank, outFeatures)
	}

	// Step 6: gradA += scale * goB^T @ x (uses shared goB)
	if gradA != nil {
		goBT := make([]T, rank*batchSize)
		matmul.TransposeAuto(pool, goB, batchSize, rank, goBT)
		temp := make([]T, rank*inFeatures)
		matmul.MatMulAuto(pool, goBT, x, temp, rank, inFeatures, batchSize)
		vec.MulConstAddTo(gradA, scale, temp)
	}

	// Step 7: gradX += gradOutput @ W + scale * goB @ A
	if gradX != nil {
		// Base part: gradOutput @ W
		temp1 := make([]T, batchSize*inFeatures)
		matmul.MatMulAuto(pool, gradOutput, W, temp1, batchSize, inFeatures, outFeatures)
		vec.Add(gradX, temp1)

		// LoRA part: scale * goB @ A
		temp2 := make([]T, batchSize*inFeatures)
		matmul.MatMulAuto(pool, goB, A, temp2, batchSize, inFeatures, rank)
		vec.MulConstAddTo(gradX, scale, temp2)
	}
}

// FusedLoRADenseBackwardScalar is a scalar reference implementation for testing.
func FusedLoRADenseBackwardScalar[T hwy.Floats](
	gradOutput, x, h, W, A, B []T, scale T,
	gradX, gradWeight, gradBias, gradA, gradB []T,
	batchSize, inFeatures, outFeatures, rank int,
) {
	// gradWeight += gradOutput^T @ x
	if gradWeight != nil {
		for o := range outFeatures {
			for j := range inFeatures {
				var sum float64
				for i := range batchSize {
					sum += float64(gradOutput[i*outFeatures+o]) * float64(x[i*inFeatures+j])
				}
				gradWeight[o*inFeatures+j] += T(sum)
			}
		}
	}

	// gradB += scale * gradOutput^T @ h
	if gradB != nil {
		for o := range outFeatures {
			for r := range rank {
				var sum float64
				for i := range batchSize {
					sum += float64(gradOutput[i*outFeatures+o]) * float64(h[i*rank+r])
				}
				gradB[o*rank+r] += T(float64(scale) * sum)
			}
		}
	}

	// gradBias += colSum(gradOutput)
	if gradBias != nil {
		for j := range outFeatures {
			for i := range batchSize {
				gradBias[j] += gradOutput[i*outFeatures+j]
			}
		}
	}

	// goB = gradOutput @ B
	goB := make([]T, batchSize*rank)
	for i := range batchSize {
		for r := range rank {
			var sum float64
			for o := range outFeatures {
				sum += float64(gradOutput[i*outFeatures+o]) * float64(B[o*rank+r])
			}
			goB[i*rank+r] = T(sum)
		}
	}

	// gradA += scale * goB^T @ x
	if gradA != nil {
		for r := range rank {
			for j := range inFeatures {
				var sum float64
				for i := range batchSize {
					sum += float64(goB[i*rank+r]) * float64(x[i*inFeatures+j])
				}
				gradA[r*inFeatures+j] += T(float64(scale) * sum)
			}
		}
	}

	// gradX += gradOutput @ W + scale * goB @ A
	if gradX != nil {
		for i := range batchSize {
			for j := range inFeatures {
				var sum1 float64
				for o := range outFeatures {
					sum1 += float64(gradOutput[i*outFeatures+o]) * float64(W[o*inFeatures+j])
				}
				var sum2 float64
				for r := range rank {
					sum2 += float64(goB[i*rank+r]) * float64(A[r*inFeatures+j])
				}
				gradX[i*inFeatures+j] += T(sum1 + float64(scale)*sum2)
			}
		}
	}
}
