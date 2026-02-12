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

	"github.com/ajroetker/go-highway/hwy"
	"github.com/ajroetker/go-highway/hwy/contrib/vec"
	"github.com/ajroetker/go-highway/hwy/contrib/workerpool"
)

// SoftmaxBackward computes the backward pass for softmax on a single row.
//
// Given savedProbs = softmax(z), the gradient is:
//
//	dL/dz_i = p_i * (dL/dp_i - dot(p, dL/dp))
//
// where p = savedProbs and dL/dp = gradOutput.
//
// gradInput is accumulated into (not overwritten).
func SoftmaxBackward[T hwy.Floats](gradOutput, savedProbs, gradInput []T) {
	n := min(len(gradOutput), min(len(savedProbs), len(gradInput)))
	if n == 0 {
		return
	}

	// dot(probs, gradOutput)
	dotPG := vec.Dot(savedProbs[:n], gradOutput[:n])

	// gradInput[i] += probs[i] * (gradOutput[i] - dotPG)
	lanes := hwy.MaxLanes[T]()
	vDot := hwy.Set(dotPG)
	ii := 0
	for ; ii+lanes <= n; ii += lanes {
		p := hwy.Load(savedProbs[ii:])
		gOut := hwy.Load(gradOutput[ii:])
		gIn := hwy.Load(gradInput[ii:])

		diff := hwy.Sub(gOut, vDot)
		contrib := hwy.Mul(p, diff)
		result := hwy.Add(gIn, contrib)
		hwy.Store(result, gradInput[ii:])
	}
	for i := ii; i < n; i++ {
		gradInput[i] += savedProbs[i] * (gradOutput[i] - dotPG)
	}
}

// SoftmaxBackwardAuto computes the backward pass for softmax over a [rows, cols]
// matrix in parallel.
//
// Each row is treated independently (softmax was applied per-row in forward).
func SoftmaxBackwardAuto[T hwy.Floats](
	pool *workerpool.Pool,
	gradOutput, savedProbs, gradInput []T,
	rows, cols int,
) {
	parallelBackward3(pool, gradOutput, savedProbs, gradInput, rows, cols, SoftmaxBackward[T])
}

// SoftmaxBackwardScalar is a scalar reference implementation for testing.
func SoftmaxBackwardScalar[T hwy.Floats](gradOutput, savedProbs, gradInput []T) {
	n := min(len(gradOutput), min(len(savedProbs), len(gradInput)))
	if n == 0 {
		return
	}

	// dot(probs, gradOutput)
	var dotPG float64
	for i := range n {
		dotPG += float64(savedProbs[i]) * float64(gradOutput[i])
	}

	for i := range n {
		gradInput[i] += savedProbs[i] * (gradOutput[i] - T(dotPG))
	}
}

// CrossEntropyLoss computes the cross-entropy loss between logits and targets.
//
//	loss = -(1/N) * Σ_i Σ_j targets[i*C+j] * log(softmax(logits[i*C+j]))
//
// where N = batchSize and C = numClasses. Targets should be one-hot or
// soft label distributions that sum to 1 per sample.
func CrossEntropyLoss[T hwy.Floats](logits, targets []T, batchSize, numClasses int) T {
	var totalLoss float64
	probs := make([]T, numClasses)

	for i := range batchSize {
		off := i * numClasses
		softmaxInPlace(logits[off:off+numClasses], probs)

		for j := range numClasses {
			if targets[off+j] > 0 {
				totalLoss -= float64(targets[off+j]) * stdmath.Log(float64(probs[j]))
			}
		}
	}

	return T(totalLoss / float64(batchSize))
}

// softmaxInPlace computes softmax of src into dst (same length).
func softmaxInPlace[T hwy.Floats](src, dst []T) {
	n := len(src)
	maxVal := src[0]
	for j := 1; j < n; j++ {
		if src[j] > maxVal {
			maxVal = src[j]
		}
	}
	var expSum float64
	for j := range n {
		dst[j] = T(stdmath.Exp(float64(src[j]) - float64(maxVal)))
		expSum += float64(dst[j])
	}
	invSum := 1.0 / expSum
	for j := range n {
		dst[j] = T(float64(dst[j]) * invSum)
	}
}

// CrossEntropyBackward computes the backward pass for cross-entropy loss
// with softmax.
//
// Uses the simplified gradient: dL/dlogit_i = (softmax(logit_i) - target_i) / N
//
// gradLogits is accumulated into.
func CrossEntropyBackward[T hwy.Floats](
	pool *workerpool.Pool,
	logits, targets, gradLogits []T,
	batchSize, numClasses int,
) {
	invN := T(1.0) / T(batchSize)
	probs := make([]T, batchSize*numClasses)

	for i := range batchSize {
		off := i * numClasses
		softmaxInPlace(logits[off:off+numClasses], probs[off:off+numClasses])
	}

	// gradLogits += (softmax - target) / batchSize
	lanes := hwy.MaxLanes[T]()
	vInvN := hwy.Set(invN)
	total := batchSize * numClasses
	ii := 0
	for ; ii+lanes <= total; ii += lanes {
		p := hwy.Load(probs[ii:])
		t := hwy.Load(targets[ii:])
		gL := hwy.Load(gradLogits[ii:])
		diff := hwy.Sub(p, t)
		scaled := hwy.Mul(diff, vInvN)
		result := hwy.Add(gL, scaled)
		hwy.Store(result, gradLogits[ii:])
	}
	for i := ii; i < total; i++ {
		gradLogits[i] += (probs[i] - targets[i]) * invN
	}
}
