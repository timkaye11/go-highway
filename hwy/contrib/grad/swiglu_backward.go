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
	"github.com/ajroetker/go-highway/hwy/contrib/activation"
	"github.com/ajroetker/go-highway/hwy/contrib/workerpool"
)

// SwiGLUBackwardAuto computes the backward pass for SwiGLU over a [rows, cols]
// matrix in parallel.
//
// gradOutput, savedGate, savedUp are [rows, cols].
// gradGate, gradUp are [rows, cols] and accumulated into.
func SwiGLUBackwardAuto[T hwy.Floats](
	pool *workerpool.Pool,
	gradOutput, savedGate, savedUp, gradGate, gradUp []T,
	rows, cols int,
) {
	parallelBackward5(pool, gradOutput, savedGate, savedUp, gradGate, gradUp,
		rows, cols, SwiGLUBackward[T])
}

// GeGLUBackwardAuto computes the backward pass for GeGLU over a [rows, cols]
// matrix in parallel.
func GeGLUBackwardAuto[T hwy.Floats](
	pool *workerpool.Pool,
	gradOutput, savedGate, savedUp, gradGate, gradUp []T,
	rows, cols int,
) {
	parallelBackward5(pool, gradOutput, savedGate, savedUp, gradGate, gradUp,
		rows, cols, GeGLUBackward[T])
}

// parallelBackward5 applies a backward function with 5 slice arguments
// (gradOutput, saved1, saved2, grad1, grad2) in parallel across rows.
func parallelBackward5[T hwy.Floats](
	pool *workerpool.Pool,
	gradOutput, saved1, saved2, grad1, grad2 []T,
	rows, cols int,
	fn func(gradOutput, saved1, saved2, grad1, grad2 []T),
) {
	if pool == nil || rows*cols < activation.MinParallelActivationOps {
		for r := range rows {
			off := r * cols
			fn(gradOutput[off:off+cols], saved1[off:off+cols], saved2[off:off+cols],
				grad1[off:off+cols], grad2[off:off+cols])
		}
		return
	}

	pool.ParallelForAtomicBatched(rows, activation.ActivationRowBatch, func(start, end int) {
		for r := start; r < end; r++ {
			off := r * cols
			fn(gradOutput[off:off+cols], saved1[off:off+cols], saved2[off:off+cols],
				grad1[off:off+cols], grad2[off:off+cols])
		}
	})
}
