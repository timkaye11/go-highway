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
)

//go:generate go run ../../../cmd/hwygen -input rmsnorm_base.go -output . -targets avx2,avx512,neon,fallback

// BaseRMSNorm computes RMS normalization over groups of normSize elements.
//
// For each group of normSize contiguous elements in input:
//
//	rrms = 1 / sqrt(mean(x^2) + epsilon)
//	output[i] = x[i] * rrms * weight[i]
//
// When gemmaMode is true, the weight is applied as (weight[i] + 1.0):
//
//	output[i] = x[i] * rrms * (weight[i] + 1.0)
//
// The input and output slices must have length that is a multiple of normSize.
// weight is optional (pass nil to skip affine transform).
func BaseRMSNorm[T hwy.Floats](input, output []T, normSize int, weight []T, epsilon T, gemmaMode bool) {
	size := min(len(input), len(output))
	if size == 0 || normSize <= 0 {
		return
	}

	numGroups := size / normSize
	invN := T(1.0) / T(normSize)
	lanes := hwy.MaxLanes[T]()

	for g := range numGroups {
		off := g * normSize

		// Pass 1: Compute sum of squares using SIMD accumulation
		sqAcc := hwy.Zero[T]()
		ii := 0
		for ; ii+lanes <= normSize; ii += lanes {
			x := hwy.Load(input[off+ii:])
			xSq := hwy.Mul(x, x)
			sqAcc = hwy.Add(sqAcc, xSq)
		}
		sumSq := hwy.ReduceSum(sqAcc)
		for i := ii; i < normSize; i++ {
			sumSq += input[off+i] * input[off+i]
		}

		// Compute reciprocal RMS
		rrms := T(1.0 / stdmath.Sqrt(float64(sumSq*invN+epsilon)))
		vRRMS := hwy.Set(rrms)

		// Pass 2: Normalize and optionally apply weight
		if weight != nil && gemmaMode {
			vOne := hwy.Const[T](1.0)
			ii = 0
			for ; ii+lanes <= normSize; ii += lanes {
				x := hwy.Load(input[off+ii:])
				normed := hwy.Mul(x, vRRMS)
				w := hwy.Load(weight[ii:])
				wEff := hwy.Add(w, vOne)
				result := hwy.Mul(normed, wEff)
				hwy.Store(result, output[off+ii:])
			}
			for i := ii; i < normSize; i++ {
				output[off+i] = input[off+i] * rrms * (weight[i] + 1.0)
			}
		} else if weight != nil {
			ii = 0
			for ; ii+lanes <= normSize; ii += lanes {
				x := hwy.Load(input[off+ii:])
				normed := hwy.Mul(x, vRRMS)
				w := hwy.Load(weight[ii:])
				result := hwy.Mul(normed, w)
				hwy.Store(result, output[off+ii:])
			}
			for i := ii; i < normSize; i++ {
				output[off+i] = input[off+i] * rrms * weight[i]
			}
		} else {
			ii = 0
			for ; ii+lanes <= normSize; ii += lanes {
				x := hwy.Load(input[off+ii:])
				result := hwy.Mul(x, vRRMS)
				hwy.Store(result, output[off+ii:])
			}
			for i := ii; i < normSize; i++ {
				output[off+i] = input[off+i] * rrms
			}
		}
	}
}
