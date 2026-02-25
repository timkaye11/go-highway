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

import "github.com/ajroetker/go-highway/hwy"

//go:generate go run ../../../cmd/hwygen -input rope_base.go -output . -targets avx2,avx512,neon,fallback

// BaseRoPE applies Rotary Position Embeddings in-place to a single head's data.
//
// Uses the split-half convention (Llama/Gemma/ModernBERT):
//
//	halfDim = headDim / 2
//	For each position pos:
//	  first  = data[pos, 0:halfDim]
//	  second = data[pos, halfDim:headDim]
//	  new_first[d]  = first[d] * cos[pos,d] - second[d] * sin[pos,d]
//	  new_second[d] = first[d] * sin[pos,d] + second[d] * cos[pos,d]
//
// Parameters:
//   - data:    [seqLen, headDim] — modified in-place
//   - cos:     [seqLen, headDim/2] — precomputed cosine values
//   - sin:     [seqLen, headDim/2] — precomputed sine values
//   - seqLen:  number of sequence positions
//   - headDim: dimension per head (must be even)
func BaseRoPE[T hwy.Floats](data, cos, sin []T, seqLen, headDim int) {
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

		// SIMD over halfDim
		ii := 0
		for ; ii+lanes <= halfDim; ii += lanes {
			f := hwy.Load(first[ii:])
			se := hwy.Load(second[ii:])
			cv := hwy.Load(c[ii:])
			sv := hwy.Load(s[ii:])

			// new_first = first * cos - second * sin
			newFirst := hwy.Sub(hwy.Mul(f, cv), hwy.Mul(se, sv))
			// new_second = first * sin + second * cos
			newSecond := hwy.MulAdd(f, sv, hwy.Mul(se, cv))

			hwy.Store(newFirst, first[ii:])
			hwy.Store(newSecond, second[ii:])
		}

		// Scalar tail
		for d := ii; d < halfDim; d++ {
			f := first[d]
			se := second[d]
			first[d] = f*c[d] - se*s[d]
			second[d] = f*s[d] + se*c[d]
		}
	}
}
