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
	"sync"

	"github.com/ajroetker/go-highway/hwy"
	"github.com/ajroetker/go-highway/hwy/contrib/activation"
	"github.com/ajroetker/go-highway/hwy/contrib/workerpool"
)

// LayerNormBackwardAuto computes the backward pass for layer normalization
// in parallel across normalization groups.
//
// See BaseLayerNormBackward for parameter documentation.
//
// When gradGamma/gradBeta are non-nil, each worker accumulates into a thread-local
// buffer, then reduces into the shared gradGamma/gradBeta to avoid races.
func LayerNormBackwardAuto[T hwy.Floats](
	pool *workerpool.Pool,
	gradOutput, savedXHat, savedInvStd, gamma []T,
	normSize int,
	gradInput, gradGamma, gradBeta []T,
) {
	size := min(len(gradOutput), min(len(savedXHat), len(gradInput)))
	if size == 0 || normSize <= 0 {
		return
	}
	numGroups := size / normSize

	if pool == nil || numGroups*normSize < activation.MinParallelActivationOps {
		LayerNormBackward(gradOutput, savedXHat, savedInvStd, gamma, normSize,
			gradInput, gradGamma, gradBeta)
		return
	}

	// For gradGamma/gradBeta, we need per-worker buffers to avoid races.
	// gradInput is safe because each group writes to disjoint regions.
	if gradGamma == nil && gradBeta == nil {
		pool.ParallelForAtomicBatched(numGroups, activation.ActivationRowBatch, func(start, end int) {
			goSlice := gradOutput[start*normSize : end*normSize]
			xhSlice := savedXHat[start*normSize : end*normSize]
			isSlice := savedInvStd[start:end]
			giSlice := gradInput[start*normSize : end*normSize]
			LayerNormBackward(goSlice, xhSlice, isSlice, gamma, normSize,
				giSlice, nil, nil)
		})
		return
	}

	// With gradGamma/gradBeta: use per-worker local buffers, then reduce.
	var mu sync.Mutex
	pool.ParallelForAtomicBatched(numGroups, activation.ActivationRowBatch, func(start, end int) {
		goSlice := gradOutput[start*normSize : end*normSize]
		xhSlice := savedXHat[start*normSize : end*normSize]
		isSlice := savedInvStd[start:end]
		giSlice := gradInput[start*normSize : end*normSize]

		var localGG, localGB []T
		if gradGamma != nil {
			localGG = make([]T, normSize)
		}
		if gradBeta != nil {
			localGB = make([]T, normSize)
		}

		LayerNormBackward(goSlice, xhSlice, isSlice, gamma, normSize,
			giSlice, localGG, localGB)

		// Merge local buffers under lock
		mu.Lock()
		if localGG != nil {
			for i := range normSize {
				gradGamma[i] += localGG[i]
			}
		}
		if localGB != nil {
			for i := range normSize {
				gradBeta[i] += localGB[i]
			}
		}
		mu.Unlock()
	})
}

// LayerNormBackwardScalar is a scalar reference implementation for testing.
func LayerNormBackwardScalar[T hwy.Floats](
	gradOutput, savedXHat, savedInvStd, gamma []T,
	normSize int,
	gradInput, gradGamma, gradBeta []T,
) {
	size := min(len(gradOutput), min(len(savedXHat), len(gradInput)))
	if size == 0 || normSize <= 0 {
		return
	}
	numGroups := size / normSize
	invN := 1.0 / float64(normSize)

	for g := range numGroups {
		off := g * normSize
		invStd := float64(savedInvStd[g])

		// Compute dy_gamma and sums
		var meanDG, meanDGXH float64
		for i := range normSize {
			var dyGamma float64
			if gamma != nil {
				dyGamma = float64(gradOutput[off+i]) * float64(gamma[i])
			} else {
				dyGamma = float64(gradOutput[off+i])
			}
			meanDG += dyGamma
			meanDGXH += dyGamma * float64(savedXHat[off+i])

			if gradGamma != nil {
				gradGamma[i] += gradOutput[off+i] * savedXHat[off+i]
			}
			if gradBeta != nil {
				gradBeta[i] += gradOutput[off+i]
			}
		}
		meanDG *= invN
		meanDGXH *= invN

		// Compute gradInput
		for i := range normSize {
			var dyGamma float64
			if gamma != nil {
				dyGamma = float64(gradOutput[off+i]) * float64(gamma[i])
			} else {
				dyGamma = float64(gradOutput[off+i])
			}
			term := dyGamma - meanDG - float64(savedXHat[off+i])*meanDGXH
			gradInput[off+i] += T(invStd * term)
		}
	}
}
