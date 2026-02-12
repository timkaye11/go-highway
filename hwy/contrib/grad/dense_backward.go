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

// DenseBackwardAuto computes the backward pass for a dense (fully-connected) layer.
//
// Forward: output = x @ weight^T + bias
//
//   - gradOutput: [batchSize, outFeatures] — gradient from upstream
//   - x:          [batchSize, inFeatures]  — input saved from forward pass
//   - weight:     [outFeatures, inFeatures] — weight matrix (PyTorch format)
//   - gradInput:  [batchSize, inFeatures]  — accumulated (nil to skip)
//   - gradWeight: [outFeatures, inFeatures] — accumulated (nil to skip)
//   - gradBias:   [outFeatures]             — accumulated (nil to skip)
//
// Gradient computations:
//
//	gradInput  += gradOutput @ weight         (dL/dx)
//	gradWeight += gradOutput^T @ x            (dL/dW)
//	gradBias   += sum(gradOutput, axis=0)     (dL/db)
func DenseBackwardAuto[T hwy.Floats](
	pool *workerpool.Pool,
	gradOutput, x, weight []T,
	gradInput, gradWeight, gradBias []T,
	batchSize, inFeatures, outFeatures int,
) {
	// gradInput += gradOutput @ weight
	// gradOutput is [batch, outF], weight is [outF, inF]
	// MatMulKLastAuto computes A @ B^T where A[M,K], B[N,K]
	// We want gradOutput[batch, outF] @ weight[outF, inF]
	// = MatMulAuto(gradOutput[batch, outF], weight^T[inF, outF], ..., batch, inF, outF)
	// But weight is [outF, inF], so weight^T is [inF, outF].
	// MatMulAuto(A, B, C, m, n, k): C[m,n] = A[m,k] @ B[k,n]
	// We need: gradInput[batch, inF] += gradOutput[batch, outF] @ weight[outF, inF]
	// This is: C[batch, inF] = A[batch, outF] @ B[outF, inF]
	// => m=batch, n=inF, k=outF, A=gradOutput, B=weight
	if gradInput != nil {
		temp := make([]T, batchSize*inFeatures)
		matmul.MatMulAuto(pool, gradOutput, weight, temp, batchSize, inFeatures, outFeatures)
		vec.Add(gradInput, temp)
	}

	// gradWeight += gradOutput^T @ x
	// gradOutput^T is [outF, batch], x is [batch, inF]
	// C[outF, inF] = gradOutput^T[outF, batch] @ x[batch, inF]
	// => m=outF, n=inF, k=batch
	if gradWeight != nil {
		gradOutputT := make([]T, outFeatures*batchSize)
		matmul.TransposeAuto(pool, gradOutput, batchSize, outFeatures, gradOutputT)
		temp := make([]T, outFeatures*inFeatures)
		matmul.MatMulAuto(pool, gradOutputT, x, temp, outFeatures, inFeatures, batchSize)
		vec.Add(gradWeight, temp)
	}

	// gradBias += sum(gradOutput, axis=0)
	// Sum over the batch dimension: gradBias[j] += Σ_i gradOutput[i*outF + j]
	if gradBias != nil {
		biasGradSum(pool, gradOutput, gradBias, batchSize, outFeatures)
	}
}

// biasGradSum computes column-wise sum: gradBias[j] += Σ_i data[i*cols + j].
func biasGradSum[T hwy.Floats](pool *workerpool.Pool, data, gradBias []T, rows, cols int) {
	lanes := hwy.MaxLanes[T]()
	for i := range rows {
		off := i * cols
		j := 0
		for ; j+lanes <= cols; j += lanes {
			gb := hwy.Load(gradBias[j:])
			d := hwy.Load(data[off+j:])
			hwy.Store(hwy.Add(gb, d), gradBias[j:])
		}
		for ; j < cols; j++ {
			gradBias[j] += data[off+j]
		}
	}
}

// DenseBackwardScalar is a scalar reference implementation for testing.
func DenseBackwardScalar[T hwy.Floats](
	gradOutput, x, weight []T,
	gradInput, gradWeight, gradBias []T,
	batchSize, inFeatures, outFeatures int,
) {
	// gradInput += gradOutput @ weight
	if gradInput != nil {
		for i := range batchSize {
			goOff := i * outFeatures
			giOff := i * inFeatures
			for j := range inFeatures {
				var sum float64
				for k := range outFeatures {
					sum += float64(gradOutput[goOff+k]) * float64(weight[k*inFeatures+j])
				}
				gradInput[giOff+j] += T(sum)
			}
		}
	}

	// gradWeight += gradOutput^T @ x
	if gradWeight != nil {
		for k := range outFeatures {
			for j := range inFeatures {
				var sum float64
				for i := range batchSize {
					sum += float64(gradOutput[i*outFeatures+k]) * float64(x[i*inFeatures+j])
				}
				gradWeight[k*inFeatures+j] += T(sum)
			}
		}
	}

	// gradBias += sum(gradOutput, axis=0)
	if gradBias != nil {
		for j := range outFeatures {
			for i := range batchSize {
				gradBias[j] += gradOutput[i*outFeatures+j]
			}
		}
	}
}
