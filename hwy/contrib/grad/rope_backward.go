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
	"github.com/ajroetker/go-highway/hwy/contrib/workerpool"
)

// RoPEBackward applies the transpose of the RoPE rotation in-place.
//
// RoPE forward applies rotation matrix R(θ). The backward pass applies R(-θ) = R^T.
// This is equivalent to RoPE with negated sin values.
//
// Parameters:
//   - data:    [seqLen, headDim] — gradient data, modified in-place
//   - cos:     [seqLen, headDim/2] — same cos values from forward
//   - sin:     [seqLen, headDim/2] — same sin values from forward (negated internally)
//   - seqLen:  sequence length
//   - headDim: dimension per head (must be even)
func RoPEBackward[T hwy.Floats](data, cos, sin []T, seqLen, headDim int) {
	if seqLen == 0 || headDim == 0 {
		return
	}

	halfDim := headDim / 2
	lanes := hwy.MaxLanes[T]()

	for pos := range seqLen {
		dataOff := pos * headDim
		csOff := pos * halfDim

		first := data[dataOff : dataOff+halfDim]
		second := data[dataOff+halfDim : dataOff+headDim]
		c := cos[csOff : csOff+halfDim]
		s := sin[csOff : csOff+halfDim]

		// Backward: R^T means negate sin
		// new_first  = first * cos + second * sin   (note: + because sin is negated)
		// new_second = -first * sin + second * cos
		ii := 0
		for ; ii+lanes <= halfDim; ii += lanes {
			f := hwy.Load(first[ii:])
			se := hwy.Load(second[ii:])
			cv := hwy.Load(c[ii:])
			sv := hwy.Load(s[ii:])

			// new_first = first * cos + second * sin
			newFirst := hwy.MulAdd(se, sv, hwy.Mul(f, cv))
			// new_second = -first * sin + second * cos
			newSecond := hwy.Sub(hwy.Mul(se, cv), hwy.Mul(f, sv))

			hwy.Store(newFirst, first[ii:])
			hwy.Store(newSecond, second[ii:])
		}

		// Scalar tail
		for d := ii; d < halfDim; d++ {
			f := first[d]
			se := second[d]
			first[d] = f*c[d] + se*s[d]
			second[d] = -f*s[d] + se*c[d]
		}
	}
}

// RoPEBackwardAuto applies the RoPE backward pass to gradQ and gradK in
// parallel across heads.
func RoPEBackwardAuto[T hwy.Floats](
	pool *workerpool.Pool,
	gradQ, gradK, cos, sin []T,
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
			data = gradQ[off : off+headStride]
		} else {
			kvIdx := idx - numQHeads
			off := kvIdx * headStride
			data = gradK[off : off+headStride]
		}
		RoPEBackward(data, cos, sin, seqLen, headDim)
	}

	if pool != nil && totalHeads > 1 {
		pool.ParallelForAtomic(totalHeads, doHead)
	} else {
		for i := range totalHeads {
			doHead(i)
		}
	}
}

// RoPEBackwardScalar is a scalar reference implementation for testing.
func RoPEBackwardScalar[T hwy.Floats](data, cos, sin []T, seqLen, headDim int) {
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

			// Backward: transpose of rotation = negate sin
			data[dataOff+d] = f*c + s*sn
			data[dataOff+halfDim+d] = -f*sn + s*c
		}
	}
}
