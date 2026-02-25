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
	"github.com/ajroetker/go-highway/hwy/contrib/math"
	"github.com/ajroetker/go-highway/hwy/contrib/workerpool"
)

// SparseCrossEntropyForward computes the cross-entropy loss with sparse integer labels.
//
//	loss = -(1/N) * Σ_i (logits[i, labels[i]] - max_j(logits[i,:]) - log(Σ_j exp(logits[i,j] - max)))
//
// This is numerically stable via log-sum-exp.
//
// Parameters:
//   - logits: [batchSize, vocabSize] — raw logits
//   - labels: [batchSize] — integer class indices (0-based)
//   - batchSize: number of samples
//   - vocabSize: number of classes
func SparseCrossEntropyForward[T hwy.Floats](logits []T, labels []int32, batchSize, vocabSize int) T {
	var totalLoss float64

	for i := range batchSize {
		off := i * vocabSize
		row := logits[off : off+vocabSize]

		// Max pass
		maxVal := row[0]
		for j := 1; j < vocabSize; j++ {
			if row[j] > maxVal {
				maxVal = row[j]
			}
		}

		// Exp + sum pass
		var expSum float64
		for j := range vocabSize {
			expSum += stdmath.Exp(float64(row[j]) - float64(maxVal))
		}

		// Loss: -logits[label] + max + log(expSum)
		label := int(labels[i])
		if label < 0 || label >= vocabSize {
			continue
		}
		totalLoss += -float64(row[label]) + float64(maxVal) + stdmath.Log(expSum)
	}

	return T(totalLoss / float64(batchSize))
}

// SparseCrossEntropyBackward computes the backward pass for cross-entropy with
// sparse integer labels.
//
//	gradLogits[i,j] += (softmax(logits[i,j]) - onehot(labels[i],j)) / batchSize
//
// Parameters:
//   - logits:     [batchSize, vocabSize] — raw logits
//   - labels:     [batchSize] — integer class indices
//   - gradLogits: [batchSize, vocabSize] — accumulated
//   - batchSize, vocabSize: dimensions
func SparseCrossEntropyBackward[T hwy.Floats](
	logits []T, labels []int32, gradLogits []T,
	batchSize, vocabSize int,
) {
	invN := T(1.0) / T(batchSize)

	for i := range batchSize {
		off := i * vocabSize
		row := logits[off : off+vocabSize]

		// Max pass
		maxVal := row[0]
		for j := 1; j < vocabSize; j++ {
			if row[j] > maxVal {
				maxVal = row[j]
			}
		}

		// Exp + sum pass
		var expSum T
		for j := range vocabSize {
			expSum += T(stdmath.Exp(float64(row[j]) - float64(maxVal)))
		}
		invExpSum := T(1.0) / expSum

		// Gradient: softmax(logit) / batchSize, subtract 1/batchSize at label index
		label := int(labels[i])
		if label < 0 || label >= vocabSize {
			continue
		}
		for j := range vocabSize {
			prob := T(stdmath.Exp(float64(row[j])-float64(maxVal))) * invExpSum
			gradLogits[off+j] += prob * invN
		}
		gradLogits[off+label] -= invN
	}
}

// SparseCrossEntropyForwardBackwardAuto computes both the loss and the gradients
// in a single fused pass, avoiding redundant softmax recomputation.
//
// This is the most efficient variant: 3 passes per sample (max, exp+sum, gradient)
// with parallelism across the batch dimension.
//
// Returns the mean cross-entropy loss.
func SparseCrossEntropyForwardBackwardAuto[T hwy.Floats](
	pool *workerpool.Pool,
	logits []T, labels []int32, gradLogits []T,
	batchSize, vocabSize int,
) T {
	if batchSize == 0 || vocabSize == 0 {
		return 0
	}

	invN := T(1.0) / T(batchSize)
	lanes := hwy.MaxLanes[T]()

	// Per-sample losses accumulated into a shared slice
	losses := make([]T, batchSize)

	doSample := func(i int) {
		off := i * vocabSize
		row := logits[off : off+vocabSize]

		// Pass 1: Max
		maxVal := row[0]
		for j := 1; j < vocabSize; j++ {
			if row[j] > maxVal {
				maxVal = row[j]
			}
		}

		// Pass 2: exp(x - max) and sum — SIMD
		vMax := hwy.Set(maxVal)
		expSumAcc := hwy.Zero[T]()
		// We need the exp values for the gradient pass, store them in gradLogits temporarily
		gRow := gradLogits[off : off+vocabSize]

		ii := 0
		for ; ii+lanes <= vocabSize; ii += lanes {
			x := hwy.Load(row[ii:])
			shifted := hwy.Sub(x, vMax)
			expVal := math.BaseExpVec(shifted)
			hwy.Store(expVal, gRow[ii:]) // temporarily store exp values
			expSumAcc = hwy.Add(expSumAcc, expVal)
		}
		expSum := hwy.ReduceSum(expSumAcc)
		for j := ii; j < vocabSize; j++ {
			ev := T(stdmath.Exp(float64(row[j]) - float64(maxVal)))
			gRow[j] = ev
			expSum += ev
		}

		// Loss: -logits[label] + max + log(expSum)
		label := int(labels[i])
		if label < 0 || label >= vocabSize {
			return
		}
		losses[i] = -row[label] + maxVal + T(stdmath.Log(float64(expSum)))

		// Pass 3: Gradient — convert stored exp values to softmax, scale, subtract at label
		vInvExpSum := hwy.Set(T(1.0) / expSum)
		vInvN := hwy.Set(invN)
		ii = 0
		for ; ii+lanes <= vocabSize; ii += lanes {
			ev := hwy.Load(gRow[ii:])
			prob := hwy.Mul(ev, vInvExpSum)
			grad := hwy.Mul(prob, vInvN)
			hwy.Store(grad, gRow[ii:])
		}
		for j := ii; j < vocabSize; j++ {
			gRow[j] = gRow[j] / expSum * invN
		}
		gRow[label] -= invN
	}

	if pool != nil && batchSize > 1 {
		pool.ParallelForAtomic(batchSize, doSample)
	} else {
		for i := range batchSize {
			doSample(i)
		}
	}

	// Sum losses
	var totalLoss T
	for i := range batchSize {
		totalLoss += losses[i]
	}
	return totalLoss / T(batchSize)
}

// SparseCrossEntropyForwardScalar is a scalar reference implementation.
func SparseCrossEntropyForwardScalar[T hwy.Floats](logits []T, labels []int32, batchSize, vocabSize int) T {
	return SparseCrossEntropyForward(logits, labels, batchSize, vocabSize)
}

// SparseCrossEntropyBackwardScalar is a scalar reference implementation.
func SparseCrossEntropyBackwardScalar[T hwy.Floats](
	logits []T, labels []int32, gradLogits []T,
	batchSize, vocabSize int,
) {
	SparseCrossEntropyBackward(logits, labels, gradLogits, batchSize, vocabSize)
}
