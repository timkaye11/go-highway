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

// RMSNormScalar is a scalar reference implementation for comparison and testing.
func RMSNormScalar[T hwy.Floats](input, output []T, normSize int, weight []T, epsilon T, gemmaMode bool) {
	size := min(len(input), len(output))
	if size == 0 || normSize <= 0 {
		return
	}

	numGroups := size / normSize

	for g := range numGroups {
		off := g * normSize

		// Compute sum of squares
		var sumSq float64
		for i := range normSize {
			x := float64(input[off+i])
			sumSq += x * x
		}

		// Compute reciprocal RMS
		rrms := 1.0 / stdmath.Sqrt(sumSq/float64(normSize)+float64(epsilon))

		// Normalize
		if weight != nil && gemmaMode {
			for i := range normSize {
				output[off+i] = T(float64(input[off+i]) * rrms * (float64(weight[i]) + 1.0))
			}
		} else if weight != nil {
			for i := range normSize {
				output[off+i] = T(float64(input[off+i]) * rrms * float64(weight[i]))
			}
		} else {
			for i := range normSize {
				output[off+i] = T(float64(input[off+i]) * rrms)
			}
		}
	}
}

// RMSNormForwardSave computes RMS normalization and saves intermediates
// needed by the backward pass.
//
// This is identical to RMSNorm except it additionally writes:
//   - savedRRMS: the reciprocal RMS (1/sqrt(mean(x^2)+eps)) per group, [numGroups]
//
// These saved values, along with the original input, are consumed by
// grad.RMSNormBackwardAuto.
func RMSNormForwardSave[T hwy.Floats](
	pool *workerpool.Pool,
	input, output []T, normSize int,
	weight []T, epsilon T, gemmaMode bool,
	savedRRMS []T,
) {
	size := min(len(input), len(output))
	if size == 0 || normSize <= 0 {
		return
	}
	numGroups := size / normSize

	if pool == nil || numGroups*normSize < activation.MinParallelActivationOps {
		rmsNormSaveSeq(input, output, normSize, weight, epsilon, gemmaMode, savedRRMS)
		return
	}

	pool.ParallelForAtomicBatched(numGroups, activation.ActivationRowBatch, func(start, end int) {
		inSlice := input[start*normSize : end*normSize]
		outSlice := output[start*normSize : end*normSize]
		rrmsSlice := savedRRMS[start:end]
		rmsNormSaveSeq(inSlice, outSlice, normSize, weight, epsilon, gemmaMode, rrmsSlice)
	})
}

// rmsNormSaveSeq is the sequential implementation that saves intermediates.
func rmsNormSaveSeq[T hwy.Floats](
	input, output []T, normSize int,
	weight []T, epsilon T, gemmaMode bool,
	savedRRMS []T,
) {
	size := min(len(input), len(output))
	numGroups := size / normSize

	for g := range numGroups {
		off := g * normSize

		// Compute sum of squares
		var sumSq float64
		for i := range normSize {
			x := float64(input[off+i])
			sumSq += x * x
		}

		// Compute and save reciprocal RMS
		rrms := 1.0 / stdmath.Sqrt(sumSq/float64(normSize)+float64(epsilon))
		savedRRMS[g] = T(rrms)

		// Normalize
		if weight != nil && gemmaMode {
			for i := range normSize {
				output[off+i] = T(float64(input[off+i]) * rrms * (float64(weight[i]) + 1.0))
			}
		} else if weight != nil {
			for i := range normSize {
				output[off+i] = T(float64(input[off+i]) * rrms * float64(weight[i]))
			}
		} else {
			for i := range normSize {
				output[off+i] = T(float64(input[off+i]) * rrms)
			}
		}
	}
}

// ParallelRMSNorm computes RMS normalization in parallel across groups.
func ParallelRMSNorm[T hwy.Floats](pool *workerpool.Pool, input, output []T, normSize int, weight []T, epsilon T, gemmaMode bool) {
	size := min(len(input), len(output))
	if size == 0 || normSize <= 0 {
		return
	}
	numGroups := size / normSize

	if pool == nil || numGroups*normSize < activation.MinParallelActivationOps {
		RMSNormScalar(input, output, normSize, weight, epsilon, gemmaMode)
		return
	}

	pool.ParallelForAtomicBatched(numGroups, activation.ActivationRowBatch, func(start, end int) {
		inSlice := input[start*normSize : end*normSize]
		outSlice := output[start*normSize : end*normSize]
		RMSNormScalar(inSlice, outSlice, normSize, weight, epsilon, gemmaMode)
	})
}
