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

// GELUBackwardAuto computes the backward pass for GELU over a [rows, cols]
// matrix in parallel.
//
// gradOutput and savedInput are [rows, cols]. gradInput is accumulated into.
func GELUBackwardAuto[T hwy.Floats](pool *workerpool.Pool, gradOutput, savedInput, gradInput []T, rows, cols int) {
	parallelBackward3(pool, gradOutput, savedInput, gradInput, rows, cols, GELUBackward[T])
}

// ReLUBackwardAuto computes the backward pass for ReLU over a [rows, cols]
// matrix in parallel.
func ReLUBackwardAuto[T hwy.Floats](pool *workerpool.Pool, gradOutput, savedInput, gradInput []T, rows, cols int) {
	parallelBackward3(pool, gradOutput, savedInput, gradInput, rows, cols, ReLUBackward[T])
}

// SiLUBackwardAuto computes the backward pass for SiLU over a [rows, cols]
// matrix in parallel.
func SiLUBackwardAuto[T hwy.Floats](pool *workerpool.Pool, gradOutput, savedInput, gradInput []T, rows, cols int) {
	parallelBackward3(pool, gradOutput, savedInput, gradInput, rows, cols, SiLUBackward[T])
}

// TanhBackwardAuto computes the backward pass for Tanh over a [rows, cols]
// matrix in parallel.
//
// savedOutput is the tanh output (not the input) saved from the forward pass.
func TanhBackwardAuto[T hwy.Floats](pool *workerpool.Pool, gradOutput, savedOutput, gradInput []T, rows, cols int) {
	parallelBackward3(pool, gradOutput, savedOutput, gradInput, rows, cols, TanhBackward[T])
}

// parallelBackward3 applies a backward function with 3 slice arguments (gradOutput,
// saved, gradInput) in parallel across rows.
func parallelBackward3[T hwy.Floats](
	pool *workerpool.Pool,
	gradOutput, saved, gradInput []T,
	rows, cols int,
	fn func(gradOutput, saved, gradInput []T),
) {
	if pool == nil || rows*cols < activation.MinParallelActivationOps {
		for r := range rows {
			off := r * cols
			fn(gradOutput[off:off+cols], saved[off:off+cols], gradInput[off:off+cols])
		}
		return
	}

	pool.ParallelForAtomicBatched(rows, activation.ActivationRowBatch, func(start, end int) {
		for r := start; r < end; r++ {
			off := r * cols
			fn(gradOutput[off:off+cols], saved[off:off+cols], gradInput[off:off+cols])
		}
	})
}
