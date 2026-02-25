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

//go:generate go run ../../../cmd/hwygen -input rmsnorm_backward_base.go -output . -targets avx2,avx512,neon,fallback -dispatch rmsnorm_backward

// BaseRMSNormBackward computes the backward pass for RMS normalization.
//
// Forward: y = x * rrms * w_eff  where rrms = 1/sqrt(mean(x^2) + eps)
// and w_eff = weight (normal mode) or weight+1 (gemma mode).
//
// Saved from forward: savedInput (original input x), savedRRMS (1/rms per group)
//
// Per normalization group (2-pass SIMD):
//
//	Pass 1: dot = sum(gradOutput[i] * w_eff[i] * x[i])
//	         gradWeight[i] += gradOutput[i] * x[i] * rrms
//	Pass 2: gradInput[i] += rrms * (gradOutput[i] * w_eff[i] - x[i] * rrms^2 * dot / N)
//
// Parameters:
//   - gradOutput:  flat slice, [numGroups * normSize]
//   - savedInput:  flat slice, [numGroups * normSize] — original input from forward
//   - savedRRMS:   [numGroups] — reciprocal RMS per group from forward
//   - weight:      [normSize] — weight parameter (nil if no affine)
//   - normSize:    number of elements per normalization group
//   - gemmaMode:   if true, effective weight is (weight + 1)
//   - gradInput:   flat slice, [numGroups * normSize] — accumulated
//   - gradWeight:  [normSize] — accumulated (nil to skip)
func BaseRMSNormBackward[T hwy.Floats](
	gradOutput, savedInput, savedRRMS, weight []T,
	normSize int, gemmaMode bool,
	gradInput, gradWeight []T,
) {
	size := min(len(gradOutput), min(len(savedInput), len(gradInput)))
	if size == 0 || normSize <= 0 {
		return
	}

	numGroups := size / normSize
	lanes := hwy.MaxLanes[T]()

	for g := range numGroups {
		off := g * normSize
		rrms := savedRRMS[g]

		// Pass 1: Compute dot = sum(gradOutput * w_eff * x), accumulate gradWeight
		dotAcc := hwy.Zero[T]()
		ii := 0

		if weight != nil && gemmaMode {
			vOne := hwy.Const[T](1.0)
			vRRMS := hwy.Set(rrms)
			for ; ii+lanes <= normSize; ii += lanes {
				gOut := hwy.Load(gradOutput[off+ii:])
				x := hwy.Load(savedInput[off+ii:])
				w := hwy.Load(weight[ii:])
				wEff := hwy.Add(w, vOne)

				// dot += gradOutput * wEff * x
				contrib := hwy.Mul(hwy.Mul(gOut, wEff), x)
				dotAcc = hwy.Add(dotAcc, contrib)

				// gradWeight += gradOutput * x * rrms
				if gradWeight != nil {
					gw := hwy.Load(gradWeight[ii:])
					xRrms := hwy.Mul(x, vRRMS)
					newGW := hwy.MulAdd(gOut, xRrms, gw)
					hwy.Store(newGW, gradWeight[ii:])
				}
			}
			dot := hwy.ReduceSum(dotAcc)
			for i := ii; i < normSize; i++ {
				wEff := weight[i] + 1.0
				dot += gradOutput[off+i] * wEff * savedInput[off+i]
				if gradWeight != nil {
					gradWeight[i] += gradOutput[off+i] * savedInput[off+i] * rrms
				}
			}

			// Pass 2: Compute gradInput
			invN := T(1.0) / T(normSize)
			dotCoeff := dot * rrms * rrms * rrms * invN
			vDotCoeff := hwy.Set(dotCoeff)
			vRRMS2 := hwy.Set(rrms) // rrms (not squared — named vRRMS2 for hwygen uniqueness)
			ii = 0
			for ; ii+lanes <= normSize; ii += lanes {
				gOut := hwy.Load(gradOutput[off+ii:])
				x := hwy.Load(savedInput[off+ii:])
				gIn := hwy.Load(gradInput[off+ii:])
				w := hwy.Load(weight[ii:])
				wEff := hwy.Add(w, vOne)

				// rrms * gradOutput * wEff - x * dot * rrms^3 / N
				term1 := hwy.Mul(hwy.Mul(gOut, wEff), vRRMS2)
				term2 := hwy.Mul(x, vDotCoeff)
				contribGI := hwy.Sub(term1, term2)
				result := hwy.Add(gIn, contribGI)
				hwy.Store(result, gradInput[off+ii:])
			}
			for i := ii; i < normSize; i++ {
				wEff := weight[i] + 1.0
				gradInput[off+i] += rrms*gradOutput[off+i]*wEff - savedInput[off+i]*dotCoeff
			}
		} else if weight != nil {
			vRRMS := hwy.Set(rrms)
			for ; ii+lanes <= normSize; ii += lanes {
				gOut := hwy.Load(gradOutput[off+ii:])
				x := hwy.Load(savedInput[off+ii:])
				w := hwy.Load(weight[ii:])

				// dot += gradOutput * weight * x
				contrib := hwy.Mul(hwy.Mul(gOut, w), x)
				dotAcc = hwy.Add(dotAcc, contrib)

				// gradWeight += gradOutput * x * rrms
				if gradWeight != nil {
					gwt := hwy.Load(gradWeight[ii:])
					xRrms := hwy.Mul(x, vRRMS)
					newGWT := hwy.MulAdd(gOut, xRrms, gwt)
					hwy.Store(newGWT, gradWeight[ii:])
				}
			}
			dot := hwy.ReduceSum(dotAcc)
			for i := ii; i < normSize; i++ {
				dot += gradOutput[off+i] * weight[i] * savedInput[off+i]
				if gradWeight != nil {
					gradWeight[i] += gradOutput[off+i] * savedInput[off+i] * rrms
				}
			}

			// Pass 2: Compute gradInput
			invN := T(1.0) / T(normSize)
			dotCoeff := dot * rrms * rrms * rrms * invN
			vDotCoeff := hwy.Set(dotCoeff)
			vRRMS2 := hwy.Set(rrms) // rrms (not squared — named vRRMS2 for hwygen uniqueness)
			ii = 0
			for ; ii+lanes <= normSize; ii += lanes {
				gOut := hwy.Load(gradOutput[off+ii:])
				x := hwy.Load(savedInput[off+ii:])
				gIn := hwy.Load(gradInput[off+ii:])
				w := hwy.Load(weight[ii:])

				term1 := hwy.Mul(hwy.Mul(gOut, w), vRRMS2)
				term2 := hwy.Mul(x, vDotCoeff)
				contribGI := hwy.Sub(term1, term2)
				result := hwy.Add(gIn, contribGI)
				hwy.Store(result, gradInput[off+ii:])
			}
			for i := ii; i < normSize; i++ {
				gradInput[off+i] += rrms*gradOutput[off+i]*weight[i] - savedInput[off+i]*dotCoeff
			}
		} else {
			// No weight
			for ; ii+lanes <= normSize; ii += lanes {
				gOut := hwy.Load(gradOutput[off+ii:])
				x := hwy.Load(savedInput[off+ii:])
				contrib := hwy.Mul(gOut, x)
				dotAcc = hwy.Add(dotAcc, contrib)
			}
			dot := hwy.ReduceSum(dotAcc)
			for i := ii; i < normSize; i++ {
				dot += gradOutput[off+i] * savedInput[off+i]
			}

			// Pass 2: Compute gradInput
			invN := T(1.0) / T(normSize)
			dotCoeff := dot * rrms * rrms * rrms * invN
			vDotCoeff := hwy.Set(dotCoeff)
			vRRMS := hwy.Set(rrms)
			ii = 0
			for ; ii+lanes <= normSize; ii += lanes {
				gOut := hwy.Load(gradOutput[off+ii:])
				x := hwy.Load(savedInput[off+ii:])
				gIn := hwy.Load(gradInput[off+ii:])

				term1 := hwy.Mul(gOut, vRRMS)
				term2 := hwy.Mul(x, vDotCoeff)
				contribGI := hwy.Sub(term1, term2)
				result := hwy.Add(gIn, contribGI)
				hwy.Store(result, gradInput[off+ii:])
			}
			for i := ii; i < normSize; i++ {
				gradInput[off+i] += rrms*gradOutput[off+i] - savedInput[off+i]*dotCoeff
			}
		}
	}
}
