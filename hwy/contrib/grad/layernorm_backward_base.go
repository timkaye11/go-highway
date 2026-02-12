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

import "github.com/ajroetker/go-highway/hwy"

//go:generate go run ../../../cmd/hwygen -input layernorm_backward_base.go -output . -targets avx2,avx512,neon,fallback -dispatch layernorm_backward

// BaseLayerNormBackward computes the backward pass for layer normalization.
//
// Forward: y = gamma * (x - mean) / std + beta
// Saved from forward: xHat (normalized input), invStd (1/std per group)
//
// Per normalization group (3-pass SIMD):
//
//	dy_gamma[i] = gradOutput[i] * gamma[i]
//	mean_dg = (1/N) * sum(dy_gamma)
//	mean_dg_xh = (1/N) * sum(dy_gamma * xHat)
//	gradInput[i] += invStd * (dy_gamma[i] - mean_dg - xHat[i] * mean_dg_xh)
//	gradGamma[i] += sum_groups(gradOutput[i] * xHat[i])
//	gradBeta[i]  += sum_groups(gradOutput[i])
//
// Parameters:
//   - gradOutput:  flat slice, [numGroups * normSize]
//   - savedXHat:   flat slice, [numGroups * normSize] — normalized input from forward
//   - savedInvStd: [numGroups] — inverse std per group from forward
//   - gamma:       [normSize] — scale parameter (nil if no affine)
//   - normSize:    number of elements per normalization group
//   - gradInput:   flat slice, [numGroups * normSize] — accumulated
//   - gradGamma:   [normSize] — accumulated (nil to skip)
//   - gradBeta:    [normSize] — accumulated (nil to skip)
func BaseLayerNormBackward[T hwy.Floats](
	gradOutput, savedXHat, savedInvStd, gamma []T,
	normSize int,
	gradInput, gradGamma, gradBeta []T,
) {
	size := min(len(gradOutput), min(len(savedXHat), len(gradInput)))
	if size == 0 || normSize <= 0 {
		return
	}

	numGroups := size / normSize
	invN := T(1.0) / T(normSize)
	lanes := hwy.MaxLanes[T]()

	for g := range numGroups {
		off := g * normSize
		invStd := savedInvStd[g]

		sumDG := hwy.Zero[T]()
		sumDGXH := hwy.Zero[T]()

		ii := 0
		if gamma != nil {
			for ; ii+lanes <= normSize; ii += lanes {
				gOut := hwy.Load(gradOutput[off+ii:])
				xHat := hwy.Load(savedXHat[off+ii:])
				gam := hwy.Load(gamma[ii:])

				// dy_gamma = gradOutput * gamma
				dyGamma := hwy.Mul(gOut, gam)
				sumDG = hwy.Add(sumDG, dyGamma)
				sumDGXH = hwy.MulAdd(dyGamma, xHat, sumDGXH)

				// gradGamma += gradOutput * xHat
				if gradGamma != nil {
					gg := hwy.Load(gradGamma[ii:])
					gg = hwy.MulAdd(gOut, xHat, gg)
					hwy.Store(gg, gradGamma[ii:])
				}

				// gradBeta += gradOutput
				if gradBeta != nil {
					gb := hwy.Load(gradBeta[ii:])
					gb = hwy.Add(gb, gOut)
					hwy.Store(gb, gradBeta[ii:])
				}
			}

			// Scalar tail
			meanDGScalar := hwy.ReduceSum(sumDG)
			meanDGXHScalar := hwy.ReduceSum(sumDGXH)
			for i := ii; i < normSize; i++ {
				dyGamma := gradOutput[off+i] * gamma[i]
				meanDGScalar += dyGamma
				meanDGXHScalar += dyGamma * savedXHat[off+i]
				if gradGamma != nil {
					gradGamma[i] += gradOutput[off+i] * savedXHat[off+i]
				}
				if gradBeta != nil {
					gradBeta[i] += gradOutput[off+i]
				}
			}

			meanDG := meanDGScalar * invN
			meanDGXH := meanDGXHScalar * invN

			// Pass 2: Compute gradInput
			vMeanDG := hwy.Set(meanDG)
			vMeanDGXH := hwy.Set(meanDGXH)
			vInvStd := hwy.Set(invStd)
			ii = 0
			for ; ii+lanes <= normSize; ii += lanes {
				gOut := hwy.Load(gradOutput[off+ii:])
				xHat := hwy.Load(savedXHat[off+ii:])
				gIn := hwy.Load(gradInput[off+ii:])
				gam := hwy.Load(gamma[ii:])

				dyGamma := hwy.Mul(gOut, gam)
				// dyGamma - meanDG - xHat * meanDGXH
				term := hwy.Sub(dyGamma, vMeanDG)
				xhMdgxh := hwy.Mul(xHat, vMeanDGXH)
				term = hwy.Sub(term, xhMdgxh)
				// invStd * term
				contrib := hwy.Mul(vInvStd, term)
				result := hwy.Add(gIn, contrib)
				hwy.Store(result, gradInput[off+ii:])
			}
			for i := ii; i < normSize; i++ {
				dyGamma := gradOutput[off+i] * gamma[i]
				term := dyGamma - meanDG - savedXHat[off+i]*meanDGXH
				gradInput[off+i] += invStd * term
			}
		} else {
			// No gamma — dy_gamma = gradOutput
			for ; ii+lanes <= normSize; ii += lanes {
				gOut := hwy.Load(gradOutput[off+ii:])
				xHat := hwy.Load(savedXHat[off+ii:])

				sumDG = hwy.Add(sumDG, gOut)
				sumDGXH = hwy.MulAdd(gOut, xHat, sumDGXH)
			}

			meanDGScalar := hwy.ReduceSum(sumDG)
			meanDGXHScalar := hwy.ReduceSum(sumDGXH)
			for i := ii; i < normSize; i++ {
				meanDGScalar += gradOutput[off+i]
				meanDGXHScalar += gradOutput[off+i] * savedXHat[off+i]
			}

			meanDG := meanDGScalar * invN
			meanDGXH := meanDGXHScalar * invN

			vMeanDG := hwy.Set(meanDG)
			vMeanDGXH := hwy.Set(meanDGXH)
			vInvStd := hwy.Set(invStd)
			ii = 0
			for ; ii+lanes <= normSize; ii += lanes {
				gOut := hwy.Load(gradOutput[off+ii:])
				xHat := hwy.Load(savedXHat[off+ii:])
				gIn := hwy.Load(gradInput[off+ii:])

				term := hwy.Sub(gOut, vMeanDG)
				xhMdgxh := hwy.Mul(xHat, vMeanDGXH)
				term = hwy.Sub(term, xhMdgxh)
				contrib := hwy.Mul(vInvStd, term)
				result := hwy.Add(gIn, contrib)
				hwy.Store(result, gradInput[off+ii:])
			}
			for i := ii; i < normSize; i++ {
				term := gradOutput[off+i] - meanDG - savedXHat[off+i]*meanDGXH
				gradInput[off+i] += invStd * term
			}
		}
	}
}
