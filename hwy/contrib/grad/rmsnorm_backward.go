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
	"sync"

	"github.com/ajroetker/go-highway/hwy"
	"github.com/ajroetker/go-highway/hwy/contrib/activation"
	"github.com/ajroetker/go-highway/hwy/contrib/workerpool"
)

// RMSNormBackwardAuto computes the backward pass for RMS normalization
// in parallel across normalization groups.
//
// See BaseRMSNormBackward for parameter documentation.
//
// When gradWeight is non-nil, each worker accumulates into a thread-local
// buffer, then reduces into the shared gradWeight to avoid races.
func RMSNormBackwardAuto[T hwy.Floats](
	pool *workerpool.Pool,
	gradOutput, savedInput, savedRRMS, weight []T,
	normSize int, gemmaMode bool,
	gradInput, gradWeight []T,
) {
	size := min(len(gradOutput), min(len(savedInput), len(gradInput)))
	if size == 0 || normSize <= 0 {
		return
	}
	numGroups := size / normSize

	if pool == nil || numGroups*normSize < activation.MinParallelActivationOps {
		RMSNormBackward(gradOutput, savedInput, savedRRMS, weight, normSize, gemmaMode,
			gradInput, gradWeight)
		return
	}

	// For gradWeight, we need per-worker buffers to avoid races.
	// gradInput is safe because each group writes to disjoint regions.
	if gradWeight == nil {
		pool.ParallelForAtomicBatched(numGroups, activation.ActivationRowBatch, func(start, end int) {
			goSlice := gradOutput[start*normSize : end*normSize]
			xiSlice := savedInput[start*normSize : end*normSize]
			rrmsSlice := savedRRMS[start:end]
			giSlice := gradInput[start*normSize : end*normSize]
			RMSNormBackward(goSlice, xiSlice, rrmsSlice, weight, normSize, gemmaMode,
				giSlice, nil)
		})
		return
	}

	// With gradWeight: use per-worker local buffers, then reduce.
	var mu sync.Mutex
	pool.ParallelForAtomicBatched(numGroups, activation.ActivationRowBatch, func(start, end int) {
		goSlice := gradOutput[start*normSize : end*normSize]
		xiSlice := savedInput[start*normSize : end*normSize]
		rrmsSlice := savedRRMS[start:end]
		giSlice := gradInput[start*normSize : end*normSize]

		localGW := make([]T, normSize)
		RMSNormBackward(goSlice, xiSlice, rrmsSlice, weight, normSize, gemmaMode,
			giSlice, localGW)

		// Merge local buffer under lock
		mu.Lock()
		for i := range normSize {
			gradWeight[i] += localGW[i]
		}
		mu.Unlock()
	})
}

// RMSNormBackwardScalar is a scalar reference implementation for testing.
func RMSNormBackwardScalar[T hwy.Floats](
	gradOutput, savedInput, savedRRMS, weight []T,
	normSize int, gemmaMode bool,
	gradInput, gradWeight []T,
) {
	size := min(len(gradOutput), min(len(savedInput), len(gradInput)))
	if size == 0 || normSize <= 0 {
		return
	}
	numGroups := size / normSize
	invN := 1.0 / float64(normSize)

	for g := range numGroups {
		off := g * normSize
		rrms := float64(savedRRMS[g])

		// Pass 1: dot = sum(gradOutput * wEff * x), accumulate gradWeight
		var dot float64
		for i := range normSize {
			go_ := float64(gradOutput[off+i])
			x := float64(savedInput[off+i])

			var wEff float64
			if weight != nil && gemmaMode {
				wEff = float64(weight[i]) + 1.0
			} else if weight != nil {
				wEff = float64(weight[i])
			} else {
				wEff = 1.0
			}

			dot += go_ * wEff * x

			if gradWeight != nil {
				gradWeight[i] += T(go_ * x * rrms)
			}
		}

		// Pass 2: gradInput
		rrms3OverN := rrms * rrms * rrms * invN
		for i := range normSize {
			go_ := float64(gradOutput[off+i])
			x := float64(savedInput[off+i])

			var wEff float64
			if weight != nil && gemmaMode {
				wEff = float64(weight[i]) + 1.0
			} else if weight != nil {
				wEff = float64(weight[i])
			} else {
				wEff = 1.0
			}

			gradInput[off+i] += T(rrms*go_*wEff - x*dot*rrms3OverN)
		}
	}
}

// RMSNormBackwardFiniteDiff computes gradInput numerically via finite differences.
// Used for gradient checking in tests.
func RMSNormBackwardFiniteDiff[T hwy.Floats](
	input []T, normSize int, weight []T, epsilon T, gemmaMode bool,
	gradOutput []T,
	gradInput []T,
) {
	size := len(input)
	eps := 1e-4
	output1 := make([]T, size)
	output2 := make([]T, size)

	for i := range size {
		orig := input[i]

		input[i] = T(float64(orig) + eps)
		RMSNormScalarForGradCheck(input, output1, normSize, weight, epsilon, gemmaMode)

		input[i] = T(float64(orig) - eps)
		RMSNormScalarForGradCheck(input, output2, normSize, weight, epsilon, gemmaMode)

		input[i] = orig

		var sum float64
		for j := range size {
			sum += float64(gradOutput[j]) * (float64(output1[j]) - float64(output2[j])) / (2.0 * eps)
		}
		gradInput[i] += T(sum)
	}
}

// RMSNormScalarForGradCheck is the scalar forward used by gradient checking.
func RMSNormScalarForGradCheck[T hwy.Floats](input, output []T, normSize int, weight []T, epsilon T, gemmaMode bool) {
	size := min(len(input), len(output))
	if size == 0 || normSize <= 0 {
		return
	}

	numGroups := size / normSize

	for g := range numGroups {
		off := g * normSize

		var sumSq float64
		for i := range normSize {
			x := float64(input[off+i])
			sumSq += x * x
		}
		rrms := 1.0 / stdmath.Sqrt(sumSq/float64(normSize)+float64(epsilon))

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
