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
	"github.com/ajroetker/go-highway/hwy/contrib/activation"
	"github.com/ajroetker/go-highway/hwy/contrib/workerpool"
)

// LayerNormForwardSave computes layer normalization and saves intermediates
// needed by the backward pass.
//
// This is identical to LayerNorm except it additionally writes:
//   - savedXHat: the normalized input (x - mean) / std, [same shape as input]
//   - savedInvStd: the inverse standard deviation per group, [numGroups]
//
// These saved values are consumed by grad.LayerNormBackwardAuto.
func LayerNormForwardSave[T hwy.Floats](
	pool *workerpool.Pool,
	input, output []T, normSize int,
	gamma, beta []T, epsilon T,
	savedXHat, savedInvStd []T,
) {
	size := min(len(input), len(output))
	if size == 0 || normSize <= 0 {
		return
	}
	numGroups := size / normSize

	if pool == nil || numGroups*normSize < activation.MinParallelActivationOps {
		layerNormSaveSeq(input, output, normSize, gamma, beta, epsilon, savedXHat, savedInvStd)
		return
	}

	pool.ParallelForAtomicBatched(numGroups, activation.ActivationRowBatch, func(start, end int) {
		inSlice := input[start*normSize : end*normSize]
		outSlice := output[start*normSize : end*normSize]
		xHatSlice := savedXHat[start*normSize : end*normSize]
		invStdSlice := savedInvStd[start:end]
		layerNormSaveSeq(inSlice, outSlice, normSize, gamma, beta, epsilon, xHatSlice, invStdSlice)
	})
}

// layerNormSaveSeq is the sequential implementation that saves intermediates.
func layerNormSaveSeq[T hwy.Floats](
	input, output []T, normSize int,
	gamma, beta []T, epsilon T,
	savedXHat, savedInvStd []T,
) {
	size := min(len(input), len(output))
	numGroups := size / normSize
	invN := T(1.0) / T(normSize)
	lanes := hwy.MaxLanes[T]()

	for g := range numGroups {
		off := g * normSize

		// Pass 1: Compute mean
		sumAcc := hwy.Zero[T]()
		ii := 0
		for ; ii+lanes <= normSize; ii += lanes {
			x := hwy.Load(input[off+ii:])
			sumAcc = hwy.Add(sumAcc, x)
		}
		mean := hwy.ReduceSum(sumAcc)
		for i := ii; i < normSize; i++ {
			mean += input[off+i]
		}
		mean *= invN

		// Pass 2: Compute variance
		vMean := hwy.Set(mean)
		varAcc := hwy.Zero[T]()
		ii = 0
		for ; ii+lanes <= normSize; ii += lanes {
			x := hwy.Load(input[off+ii:])
			diff := hwy.Sub(x, vMean)
			varAcc = hwy.MulAdd(diff, diff, varAcc)
		}
		variance := hwy.ReduceSum(varAcc)
		for i := ii; i < normSize; i++ {
			diff := input[off+i] - mean
			variance += diff * diff
		}
		variance *= invN

		// Compute inverse standard deviation
		invStd := T(1.0 / stdmath.Sqrt(float64(variance+epsilon)))
		savedInvStd[g] = invStd
		vInvStd := hwy.Set(invStd)

		// Pass 3: Normalize, save xHat, and apply affine transform
		if gamma != nil && beta != nil {
			ii = 0
			for ; ii+lanes <= normSize; ii += lanes {
				x := hwy.Load(input[off+ii:])
				diff := hwy.Sub(x, vMean)
				normed := hwy.Mul(diff, vInvStd)

				// Save normalized input
				hwy.Store(normed, savedXHat[off+ii:])

				g := hwy.Load(gamma[ii:])
				b := hwy.Load(beta[ii:])
				result := hwy.MulAdd(normed, g, b)
				hwy.Store(result, output[off+ii:])
			}
			for i := ii; i < normSize; i++ {
				normed := (input[off+i] - mean) * invStd
				savedXHat[off+i] = normed
				output[off+i] = normed*gamma[i] + beta[i]
			}
		} else if gamma != nil {
			ii = 0
			for ; ii+lanes <= normSize; ii += lanes {
				x := hwy.Load(input[off+ii:])
				diff := hwy.Sub(x, vMean)
				normed := hwy.Mul(diff, vInvStd)

				hwy.Store(normed, savedXHat[off+ii:])

				gv := hwy.Load(gamma[ii:])
				result := hwy.Mul(normed, gv)
				hwy.Store(result, output[off+ii:])
			}
			for i := ii; i < normSize; i++ {
				normed := (input[off+i] - mean) * invStd
				savedXHat[off+i] = normed
				output[off+i] = normed * gamma[i]
			}
		} else {
			ii = 0
			for ; ii+lanes <= normSize; ii += lanes {
				x := hwy.Load(input[off+ii:])
				diff := hwy.Sub(x, vMean)
				normed := hwy.Mul(diff, vInvStd)

				hwy.Store(normed, savedXHat[off+ii:])
				hwy.Store(normed, output[off+ii:])
			}
			for i := ii; i < normSize; i++ {
				normed := (input[off+i] - mean) * invStd
				savedXHat[off+i] = normed
				output[off+i] = normed
			}
		}
	}
}
