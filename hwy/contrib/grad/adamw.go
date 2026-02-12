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
	"github.com/ajroetker/go-highway/hwy/contrib/activation"
	"github.com/ajroetker/go-highway/hwy/contrib/workerpool"
)

// AdamWStepAuto performs an AdamW optimizer step, parallelized across elements
// using a worker pool.
//
// See BaseAdamWStep for parameter documentation.
func AdamWStepAuto[T hwy.Floats](
	pool *workerpool.Pool,
	param, grad, m, v []T,
	lr, beta1, beta2, epsilon, weightDecay T,
	step int,
) {
	n := min(len(param), min(len(grad), min(len(m), len(v))))
	if n == 0 {
		return
	}

	if pool == nil || n < activation.MinParallelActivationOps {
		AdamWStep(param, grad, m, v, lr, beta1, beta2, epsilon, weightDecay, step)
		return
	}

	// Chunk into batches for parallel execution.
	// Each chunk processes a contiguous slice of elements independently.
	chunkSize := activation.ActivationRowBatch * 256 // ~1024 elements per chunk
	numChunks := (n + chunkSize - 1) / chunkSize

	pool.ParallelForAtomic(numChunks, func(ci int) {
		start := ci * chunkSize
		end := start + chunkSize
		if end > n {
			end = n
		}
		AdamWStep(
			param[start:end], grad[start:end],
			m[start:end], v[start:end],
			lr, beta1, beta2, epsilon, weightDecay, step,
		)
	})
}

// AdamWStepScalar is a scalar reference implementation for testing.
func AdamWStepScalar[T hwy.Floats](
	param, grad, m, v []T,
	lr, beta1, beta2, epsilon, weightDecay T,
	step int,
) {
	n := min(len(param), min(len(grad), min(len(m), len(v))))
	if n == 0 {
		return
	}

	oneMinusB1 := float64(1) - float64(beta1)
	oneMinusB2 := float64(1) - float64(beta2)

	b1Power := 1.0
	for range step {
		b1Power *= float64(beta1)
	}
	b2Power := 1.0
	for range step {
		b2Power *= float64(beta2)
	}
	biasCorr1 := 1.0 / (1.0 - b1Power)
	biasCorr2 := 1.0 / (1.0 - b2Power)

	for i := range n {
		g := float64(grad[i])
		m[i] = T(float64(beta1)*float64(m[i]) + oneMinusB1*g)
		v[i] = T(float64(beta2)*float64(v[i]) + oneMinusB2*g*g)

		mHat := float64(m[i]) * biasCorr1
		vHat := float64(v[i]) * biasCorr2

		update := mHat/(sqrt64(vHat)+float64(epsilon)) + float64(weightDecay)*float64(param[i])
		param[i] -= T(float64(lr) * update)
	}
}
