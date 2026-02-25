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

// FusedLoRADenseAuto computes a dense layer with LoRA adaptation in a single
// fused operation that minimizes memory traffic.
//
//	output = x @ W^T + bias + scale * (x @ A^T) @ B^T
//
// The key optimization is that x is read once for both the base matmul (x @ W^T)
// and the LoRA down-projection (x @ A^T), keeping it hot in cache.
//
// Parameters:
//   - x:           [batchSize, inFeatures]
//   - W:           [outFeatures, inFeatures] (PyTorch format)
//   - bias:        [outFeatures] (optional, nil to skip)
//   - A:           [rank, inFeatures] — LoRA down-projection
//   - B:           [outFeatures, rank] — LoRA up-projection
//   - scale:       LoRA scaling factor (alpha / rank)
//   - output:      [batchSize, outFeatures]
//   - h:           [batchSize, rank] — saved intermediate x @ A^T (nil to skip saving)
//   - batchSize, inFeatures, outFeatures, rank: dimensions
func FusedLoRADenseAuto[T hwy.Floats](
	pool *workerpool.Pool,
	x, W, bias, A, B []T, scale T,
	output, h []T,
	batchSize, inFeatures, outFeatures, rank int,
) {
	// Step 1: h = x @ A^T (small matmul, warms x in cache)
	hBuf := h
	if hBuf == nil {
		hBuf = make([]T, batchSize*rank)
	}
	matmul.MatMulKLastAuto(pool, x, A, hBuf, batchSize, rank, inFeatures)

	// Step 2: output = x @ W^T (large matmul, x still hot in L2)
	matmul.MatMulKLastAuto(pool, x, W, output, batchSize, outFeatures, inFeatures)

	// Step 3: output += bias
	if bias != nil {
		addBias(output, bias, batchSize, outFeatures)
	}

	// Step 4: temp = h @ B^T, output += scale * temp
	temp := make([]T, batchSize*outFeatures)
	matmul.MatMulKLastAuto(pool, hBuf, B, temp, batchSize, outFeatures, rank)
	vec.MulConstAddTo(output, scale, temp)
}

// FusedLoRADenseScalar is a scalar reference implementation for testing.
func FusedLoRADenseScalar[T hwy.Floats](
	x, W, bias, A, B []T, scale T,
	output, h []T,
	batchSize, inFeatures, outFeatures, rank int,
) {
	// h = x @ A^T
	hBuf := h
	if hBuf == nil {
		hBuf = make([]T, batchSize*rank)
	}
	for i := range batchSize {
		for r := range rank {
			var sum float64
			for j := range inFeatures {
				sum += float64(x[i*inFeatures+j]) * float64(A[r*inFeatures+j])
			}
			hBuf[i*rank+r] = T(sum)
		}
	}

	// output = x @ W^T + bias
	DenseScalar(x, W, bias, output, batchSize, inFeatures, outFeatures)

	// output += scale * h @ B^T
	for i := range batchSize {
		for j := range outFeatures {
			var sum float64
			for r := range rank {
				sum += float64(hBuf[i*rank+r]) * float64(B[j*rank+r])
			}
			output[i*outFeatures+j] += T(float64(scale) * sum)
		}
	}
}
