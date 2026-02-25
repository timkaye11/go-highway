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
	"github.com/ajroetker/go-highway/hwy"
	"github.com/ajroetker/go-highway/hwy/contrib/matmul"
	"github.com/ajroetker/go-highway/hwy/contrib/vec"
	"github.com/ajroetker/go-highway/hwy/contrib/workerpool"
)

// FusedLoRAQKVDenseAuto computes a fused QKV projection with LoRA adapters
// on each of Q, K, V.
//
// The key optimization is that x is read once for all 6 matmuls (3 base + 3 LoRA
// down-projections), keeping it hot in cache. This saves 3 full reads of x
// compared to calling FusedLoRADenseAuto separately for Q, K, V.
//
// Parameters:
//   - x:     [batchSize, inFeatures]
//   - wQKV:  [(qDim + 2*kvDim), inFeatures] — stacked base Q, K, V weights
//   - biasQ: [qDim], biasK: [kvDim], biasV: [kvDim] — optional biases
//   - loraQ, loraK, loraV: LoRA params (nil to skip adapter for that projection)
//   - q: [batchSize, qDim], k: [batchSize, kvDim], v: [batchSize, kvDim] — outputs
//   - hQ, hK, hV: saved intermediates (nil to skip saving)
//   - batchSize, inFeatures, qDim, kvDim: dimensions
func FusedLoRAQKVDenseAuto[T hwy.Floats](
	pool *workerpool.Pool,
	x, wQKV []T,
	biasQ, biasK, biasV []T,
	loraQ, loraK, loraV *LoRAParams[T],
	q, k, v, hQ, hK, hV []T,
	batchSize, inFeatures, qDim, kvDim int,
) {
	// Step 1: All LoRA down-projections first (warms x in cache)
	if loraQ != nil {
		if hQ == nil {
			hQ = make([]T, batchSize*loraQ.Rank)
		}
		matmul.MatMulKLastAuto(pool, x, loraQ.A, hQ, batchSize, loraQ.Rank, inFeatures)
	}

	if loraK != nil {
		if hK == nil {
			hK = make([]T, batchSize*loraK.Rank)
		}
		matmul.MatMulKLastAuto(pool, x, loraK.A, hK, batchSize, loraK.Rank, inFeatures)
	}

	if loraV != nil {
		if hV == nil {
			hV = make([]T, batchSize*loraV.Rank)
		}
		matmul.MatMulKLastAuto(pool, x, loraV.A, hV, batchSize, loraV.Rank, inFeatures)
	}

	// Step 2: Base QKV projection (x still hot from step 1)
	QKVDenseAuto(pool, x, wQKV, biasQ, biasK, biasV, q, k, v,
		batchSize, inFeatures, qDim, kvDim)

	// Step 3: LoRA up-projections
	if loraQ != nil {
		temp := make([]T, batchSize*qDim)
		matmul.MatMulKLastAuto(pool, hQ, loraQ.B, temp, batchSize, qDim, loraQ.Rank)
		vec.MulConstAddTo(q, loraQ.Scale, temp)
	}

	if loraK != nil {
		temp := make([]T, batchSize*kvDim)
		matmul.MatMulKLastAuto(pool, hK, loraK.B, temp, batchSize, kvDim, loraK.Rank)
		vec.MulConstAddTo(k, loraK.Scale, temp)
	}

	if loraV != nil {
		temp := make([]T, batchSize*kvDim)
		matmul.MatMulKLastAuto(pool, hV, loraV.B, temp, batchSize, kvDim, loraV.Rank)
		vec.MulConstAddTo(v, loraV.Scale, temp)
	}
}

// FusedLoRAQKVDenseScalar is a scalar reference implementation for testing.
func FusedLoRAQKVDenseScalar[T hwy.Floats](
	x, wQKV []T,
	biasQ, biasK, biasV []T,
	loraQ, loraK, loraV *LoRAParams[T],
	q, k, v []T,
	batchSize, inFeatures, qDim, kvDim int,
) {
	// Base QKV
	QKVDenseScalar(x, wQKV, biasQ, biasK, biasV, q, k, v,
		batchSize, inFeatures, qDim, kvDim)

	// Add LoRA for Q
	if loraQ != nil {
		addLoRAScalar(x, loraQ.A, loraQ.B, loraQ.Scale, q,
			batchSize, inFeatures, qDim, loraQ.Rank)
	}

	// Add LoRA for K
	if loraK != nil {
		addLoRAScalar(x, loraK.A, loraK.B, loraK.Scale, k,
			batchSize, inFeatures, kvDim, loraK.Rank)
	}

	// Add LoRA for V
	if loraV != nil {
		addLoRAScalar(x, loraV.A, loraV.B, loraV.Scale, v,
			batchSize, inFeatures, kvDim, loraV.Rank)
	}
}

// addLoRAScalar adds scale * (x @ A^T) @ B^T to output.
func addLoRAScalar[T hwy.Floats](x, A, B []T, scale T, output []T,
	batchSize, inFeatures, outDim, rank int,
) {
	for i := range batchSize {
		for j := range outDim {
			var sum float64
			for r := range rank {
				// h = x @ A^T
				var hVal float64
				for k := range inFeatures {
					hVal += float64(x[i*inFeatures+k]) * float64(A[r*inFeatures+k])
				}
				sum += hVal * float64(B[j*rank+r])
			}
			output[i*outDim+j] += T(float64(scale) * sum)
		}
	}
}
