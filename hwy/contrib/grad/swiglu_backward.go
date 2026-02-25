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
	stdmath "math"

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
		rows, cols, SwiGLUBackwardScalar[T])
}

// GeGLUBackwardAuto computes the backward pass for GeGLU over a [rows, cols]
// matrix in parallel.
func GeGLUBackwardAuto[T hwy.Floats](
	pool *workerpool.Pool,
	gradOutput, savedGate, savedUp, gradGate, gradUp []T,
	rows, cols int,
) {
	parallelBackward5(pool, gradOutput, savedGate, savedUp, gradGate, gradUp,
		rows, cols, GeGLUBackwardScalar[T])
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

// SwiGLUBackwardScalar computes the backward pass for SwiGLU element-wise (scalar).
func SwiGLUBackwardScalar[T hwy.Floats](gradOutput, savedGate, savedUp, gradGate, gradUp []T) {
	n := min(len(gradOutput), min(len(savedGate), min(len(savedUp), min(len(gradGate), len(gradUp)))))
	for i := range n {
		g := float64(savedGate[i])
		u := float64(savedUp[i])
		go_ := float64(gradOutput[i])

		sig := 1.0 / (1.0 + stdmath.Exp(-g))
		siluG := g * sig
		dSilu := sig * (1.0 + g*(1.0-sig))

		gradGate[i] += T(go_ * u * dSilu)
		gradUp[i] += T(go_ * siluG)
	}
}

// GeGLUBackwardScalar computes the backward pass for GeGLU element-wise (scalar).
func GeGLUBackwardScalar[T hwy.Floats](gradOutput, savedGate, savedUp, gradGate, gradUp []T) {
	n := min(len(gradOutput), min(len(savedGate), min(len(savedUp), min(len(gradGate), len(gradUp)))))
	for i := range n {
		g := float64(savedGate[i])
		u := float64(savedUp[i])
		go_ := float64(gradOutput[i])

		erfVal := stdmath.Erf(g * 0.7071067811865476)
		halfOnePlusErf := 0.5 * (1.0 + erfVal)
		geluG := g * halfOnePlusErf
		gaussianTerm := g * stdmath.Exp(-0.5*g*g) * 0.3989422804014327
		dGelu := halfOnePlusErf + gaussianTerm

		gradGate[i] += T(go_ * u * dGelu)
		gradUp[i] += T(go_ * geluG)
	}
}
