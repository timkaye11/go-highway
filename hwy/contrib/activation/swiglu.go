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

package activation

import (
	"github.com/ajroetker/go-highway/hwy"
	"github.com/ajroetker/go-highway/hwy/contrib/workerpool"
)

// ParallelSwiGLU applies SwiGLU(gate, up) → output across a [rows, cols] matrix
// in parallel.
//
// gate, up, output are all [rows, cols]. Each row is processed independently.
func ParallelSwiGLU[T hwy.Floats](pool *workerpool.Pool, gate, up, output []T, rows, cols int) {
	if pool == nil || rows*cols < MinParallelActivationOps {
		for r := range rows {
			off := r * cols
			SwiGLUScalar(gate[off:off+cols], up[off:off+cols], output[off:off+cols])
		}
		return
	}

	pool.ParallelForAtomicBatched(rows, ActivationRowBatch, func(start, end int) {
		for r := start; r < end; r++ {
			off := r * cols
			SwiGLUScalar(gate[off:off+cols], up[off:off+cols], output[off:off+cols])
		}
	})
}

// ParallelGeGLU applies GeGLU(gate, up) → output across a [rows, cols] matrix
// in parallel.
//
// gate, up, output are all [rows, cols]. Each row is processed independently.
func ParallelGeGLU[T hwy.Floats](pool *workerpool.Pool, gate, up, output []T, rows, cols int) {
	if pool == nil || rows*cols < MinParallelActivationOps {
		for r := range rows {
			off := r * cols
			GeGLUScalar(gate[off:off+cols], up[off:off+cols], output[off:off+cols])
		}
		return
	}

	pool.ParallelForAtomicBatched(rows, ActivationRowBatch, func(start, end int) {
		for r := start; r < end; r++ {
			off := r * cols
			GeGLUScalar(gate[off:off+cols], up[off:off+cols], output[off:off+cols])
		}
	})
}
