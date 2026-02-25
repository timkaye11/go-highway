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
	stdmath "math"

	"github.com/ajroetker/go-highway/hwy"
	"github.com/ajroetker/go-highway/hwy/contrib/workerpool"
)

// RoPEAuto applies Rotary Position Embeddings in-place to Q and K tensors,
// parallelized across heads.
//
// Parameters:
//   - Q:          [numQHeads, seqLen, headDim] — modified in-place
//   - K:          [numKVHeads, seqLen, headDim] — modified in-place
//   - cos:        [seqLen, headDim/2] — precomputed cosine values
//   - sin:        [seqLen, headDim/2] — precomputed sine values
//   - numQHeads:  number of query heads
//   - numKVHeads: number of key/value heads
//   - seqLen:     sequence length
//   - headDim:    dimension per head (must be even)
func RoPEAuto[T hwy.Floats](
	pool *workerpool.Pool,
	Q, K, cos, sin []T,
	numQHeads, numKVHeads, seqLen, headDim int,
) {
	if seqLen == 0 || headDim == 0 {
		return
	}

	totalHeads := numQHeads + numKVHeads
	headStride := seqLen * headDim

	doHead := func(idx int) {
		var data []T
		if idx < numQHeads {
			off := idx * headStride
			data = Q[off : off+headStride]
		} else {
			kvIdx := idx - numQHeads
			off := kvIdx * headStride
			data = K[off : off+headStride]
		}
		RoPE(data, cos, sin, seqLen, headDim)
	}

	if pool != nil && totalHeads > 1 {
		pool.ParallelForAtomic(totalHeads, doHead)
	} else {
		for i := range totalHeads {
			doHead(i)
		}
	}
}

// RoPEScalar is a scalar reference implementation for comparison and testing.
func RoPEScalar[T hwy.Floats](data, cos, sin []T, seqLen, headDim int) {
	if seqLen == 0 || headDim == 0 {
		return
	}

	halfDim := headDim / 2
	for pos := range seqLen {
		dataOff := pos * headDim
		csOff := pos * halfDim

		for d := range halfDim {
			f := data[dataOff+d]
			s := data[dataOff+halfDim+d]
			c := cos[csOff+d]
			sn := sin[csOff+d]

			data[dataOff+d] = f*c - s*sn
			data[dataOff+halfDim+d] = f*sn + s*c
		}
	}
}

// PrecomputeRoPE precomputes cos and sin tables for RoPE.
//
// Parameters:
//   - seqLen:  maximum sequence length
//   - headDim: dimension per head (must be even)
//   - theta:   RoPE base frequency (e.g. 10000.0)
//
// Returns:
//   - cos: [seqLen, headDim/2]
//   - sin: [seqLen, headDim/2]
func PrecomputeRoPE[T hwy.Floats](seqLen, headDim int, theta float64) (cos, sin []T) {
	halfDim := headDim / 2
	cos = make([]T, seqLen*halfDim)
	sin = make([]T, seqLen*halfDim)

	for pos := range seqLen {
		for d := range halfDim {
			freq := 1.0 / stdmath.Pow(theta, float64(2*d)/float64(headDim))
			angle := float64(pos) * freq

			off := pos*halfDim + d
			cos[off] = T(stdmath.Cos(angle))
			sin[off] = T(stdmath.Sin(angle))
		}
	}
	return
}
