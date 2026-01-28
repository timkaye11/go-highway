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

// Package loss provides SIMD-accelerated loss function kernels.
package loss

//go:generate go run ../../../cmd/hwygen -input cut_cross_entropy.go -dispatch cutce -output . -targets avx2,avx512,neon,fallback

import (
	stdmath "math"

	"github.com/ajroetker/go-highway/hwy"
	"github.com/ajroetker/go-highway/hwy/contrib/math"
)

// BaseCutCrossEntropy computes cross-entropy loss WITHOUT materializing the full logits matrix.
//
// Standard cross-entropy requires materializing [batchSize * seqLen, vocabSize] logits,
// which for a 2B model is ~24 GB. This implementation computes the loss using only
// O(vocabSize) memory per position by streaming the log-sum-exp computation.
//
// Algorithm for each position i with label y_i:
//  1. Compute logit_y = hiddenStates[i] · embeddings[y_i]  (dot product, O(hiddenDim))
//  2. Compute logsumexp = log(sum_v(exp(hiddenStates[i] · embeddings[v]))) via streaming:
//     - For each chunk of vocabulary entries, compute dot products
//     - Maintain running max and sum for numerically stable logsumexp
//  3. Loss_i = logsumexp - logit_y
//
// Memory: O(chunkSize) per position instead of O(vocabSize * batchSize * seqLen)
//
// Parameters:
//   - hiddenStates: [numPositions, hiddenDim] float32 final hidden states
//   - embeddings: [vocabSize, hiddenDim] float32 output embedding matrix (tied weights or lm_head)
//   - labels: [numPositions] int32 ground truth token IDs (-1 = ignore/padding)
//   - numPositions: number of token positions to compute loss for
//   - hiddenDim: dimension of hidden states and embeddings
//   - vocabSize: size of the vocabulary
//
// Returns: scalar mean loss (float32)
func BaseCutCrossEntropy(
	hiddenStates []float32,
	embeddings []float32,
	labels []int32,
	numPositions, hiddenDim, vocabSize int,
) float32 {
	if numPositions == 0 || hiddenDim == 0 || vocabSize == 0 {
		return 0
	}

	totalLoss := float64(0)
	validCount := 0

	for pos := 0; pos < numPositions; pos++ {
		label := labels[pos]
		if label < 0 || int(label) >= vocabSize {
			continue // Skip padding/ignored positions
		}

		hsOffset := pos * hiddenDim

		// Step 1: Compute logit for the correct label
		labelLogit := simdDotProduct(
			hiddenStates[hsOffset:hsOffset+hiddenDim],
			embeddings[int(label)*hiddenDim:(int(label)+1)*hiddenDim],
			hiddenDim,
		)

		// Step 2: Compute logsumexp over all vocabulary entries (streaming)
		lse := streamingLogsumexp(hiddenStates[hsOffset:hsOffset+hiddenDim], embeddings, hiddenDim, vocabSize)

		// Step 3: Loss = logsumexp - logit_y
		loss := lse - float64(labelLogit)
		totalLoss += loss
		validCount++
	}

	if validCount == 0 {
		return 0
	}

	return float32(totalLoss / float64(validCount))
}

// BaseCutCrossEntropyGrad computes gradients for Cut Cross-Entropy.
//
// The gradient of cross-entropy loss w.r.t. hidden states is:
//   dL/dh_i = (1/N) * (sum_v(softmax(h_i · e_v) * e_v) - e_{y_i})
//
// This is computed without materializing full logits by streaming through
// vocabulary chunks, computing softmax contributions on-the-fly.
//
// Parameters:
//   - hiddenStates: [numPositions, hiddenDim] float32
//   - embeddings: [vocabSize, hiddenDim] float32
//   - labels: [numPositions] int32
//   - gradOutput: [numPositions, hiddenDim] float32 pre-allocated gradient buffer
//   - numPositions, hiddenDim, vocabSize: dimensions
func BaseCutCrossEntropyGrad(
	hiddenStates []float32,
	embeddings []float32,
	labels []int32,
	gradOutput []float32,
	numPositions, hiddenDim, vocabSize int,
) {
	if numPositions == 0 || hiddenDim == 0 || vocabSize == 0 {
		return
	}

	// Count valid positions for mean
	validCount := 0
	for _, label := range labels[:numPositions] {
		if label >= 0 && int(label) < vocabSize {
			validCount++
		}
	}
	if validCount == 0 {
		return
	}

	invN := float32(1.0 / float64(validCount))

	lanes := hwy.Zero[float32]().NumLanes()

	for pos := 0; pos < numPositions; pos++ {
		label := labels[pos]
		gradBase := pos * hiddenDim

		if label < 0 || int(label) >= vocabSize {
			// Zero gradient for ignored positions
			for d := 0; d < hiddenDim; d++ {
				gradOutput[gradBase+d] = 0
			}
			continue
		}

		hsOffset := pos * hiddenDim

		// Compute logsumexp for this position (needed for softmax denominator)
		lse := streamingLogsumexp(hiddenStates[hsOffset:hsOffset+hiddenDim], embeddings, hiddenDim, vocabSize)

		// Compute gradient: sum_v(softmax_v * e_v) - e_{label}
		// Initialize gradient to -e_{label}
		labelEmbOffset := int(label) * hiddenDim
		var d int
		for d = 0; d+lanes <= hiddenDim; d += lanes {
			e := hwy.Load(embeddings[labelEmbOffset+d:])
			neg := hwy.Neg(e)
			scaled := hwy.Mul(neg, hwy.Set(invN))
			hwy.Store(scaled, gradOutput[gradBase+d:])
		}
		for ; d < hiddenDim; d++ {
			gradOutput[gradBase+d] = -embeddings[labelEmbOffset+d] * invN
		}

		// Add softmax-weighted embedding contributions
		for v := 0; v < vocabSize; v++ {
			embOffset := v * hiddenDim
			logit := float64(simdDotProduct(
				hiddenStates[hsOffset:hsOffset+hiddenDim],
				embeddings[embOffset:embOffset+hiddenDim],
				hiddenDim,
			))
			softmaxWeight := float32(stdmath.Exp(logit - lse)) * invN

			// gradOutput[pos] += softmaxWeight * embeddings[v]
			vWeight := hwy.Set(softmaxWeight)
			var dd int
			for dd = 0; dd+lanes <= hiddenDim; dd += lanes {
				g := hwy.Load(gradOutput[gradBase+dd:])
				e := hwy.Load(embeddings[embOffset+dd:])
				g = hwy.MulAdd(vWeight, e, g)
				hwy.Store(g, gradOutput[gradBase+dd:])
			}
			for ; dd < hiddenDim; dd++ {
				gradOutput[gradBase+dd] += softmaxWeight * embeddings[embOffset+dd]
			}
		}
	}
}

// streamingLogsumexp computes log(sum_v(exp(h · e_v))) without materializing all logits.
//
// Uses the numerically stable streaming formula:
//   max = -inf
//   sumExp = 0
//   For each vocab entry v:
//     logit = h · e_v
//     if logit > max:
//       sumExp = sumExp * exp(max - logit) + 1
//       max = logit
//     else:
//       sumExp += exp(logit - max)
//   logsumexp = max + log(sumExp)
//
// This processes vocabulary in chunks for better cache locality.
func streamingLogsumexp(
	hiddenState []float32,
	embeddings []float32,
	hiddenDim, vocabSize int,
) float64 {
	// Chunk size chosen to keep dot-product buffer in L1 cache
	const chunkSize = 256

	currentMax := -stdmath.MaxFloat64
	sumExp := float64(0)

	for vStart := 0; vStart < vocabSize; vStart += chunkSize {
		vEnd := vStart + chunkSize
		if vEnd > vocabSize {
			vEnd = vocabSize
		}

		for v := vStart; v < vEnd; v++ {
			embOffset := v * hiddenDim
			logit := float64(simdDotProduct(
				hiddenState,
				embeddings[embOffset:embOffset+hiddenDim],
				hiddenDim,
			))

			if logit > currentMax {
				// Rescale existing sum to new max
				sumExp = sumExp*stdmath.Exp(currentMax-logit) + 1.0
				currentMax = logit
			} else {
				sumExp += stdmath.Exp(logit - currentMax)
			}
		}
	}

	if sumExp == 0 {
		return currentMax
	}
	return currentMax + stdmath.Log(sumExp)
}

// simdDotProduct computes the dot product of two float32 vectors using SIMD.
func simdDotProduct(a, b []float32, n int) float32 {
	lanes := hwy.Zero[float32]().NumLanes()

	acc := hwy.Zero[float32]()

	var i int
	for i = 0; i+lanes <= n; i += lanes {
		va := hwy.Load(a[i:])
		vb := hwy.Load(b[i:])
		acc = hwy.MulAdd(va, vb, acc)
	}

	sum := hwy.ReduceSum(acc)

	// Scalar tail
	for ; i < n; i++ {
		sum += a[i] * b[i]
	}

	return sum
}

// BaseCutCrossEntropyWithLogits is a variant that also returns per-position losses
// and the logit for the correct token (useful for metrics/debugging).
//
// Parameters:
//   - hiddenStates: [numPositions, hiddenDim] float32
//   - embeddings: [vocabSize, hiddenDim] float32
//   - labels: [numPositions] int32
//   - perPositionLoss: [numPositions] float32 pre-allocated (receives per-position losses)
//   - correctLogits: [numPositions] float32 pre-allocated (receives logit for correct token)
//   - numPositions, hiddenDim, vocabSize: dimensions
//
// Returns: scalar mean loss
func BaseCutCrossEntropyWithLogits(
	hiddenStates []float32,
	embeddings []float32,
	labels []int32,
	perPositionLoss []float32,
	correctLogits []float32,
	numPositions, hiddenDim, vocabSize int,
) float32 {
	if numPositions == 0 || hiddenDim == 0 || vocabSize == 0 {
		return 0
	}

	totalLoss := float64(0)
	validCount := 0

	for pos := 0; pos < numPositions; pos++ {
		label := labels[pos]
		if label < 0 || int(label) >= vocabSize {
			perPositionLoss[pos] = 0
			correctLogits[pos] = 0
			continue
		}

		hsOffset := pos * hiddenDim

		labelLogit := simdDotProduct(
			hiddenStates[hsOffset:hsOffset+hiddenDim],
			embeddings[int(label)*hiddenDim:(int(label)+1)*hiddenDim],
			hiddenDim,
		)
		correctLogits[pos] = labelLogit

		lse := streamingLogsumexp(hiddenStates[hsOffset:hsOffset+hiddenDim], embeddings, hiddenDim, vocabSize)

		loss := float32(lse - float64(labelLogit))
		perPositionLoss[pos] = loss
		totalLoss += float64(loss)
		validCount++
	}

	if validCount == 0 {
		return 0
	}

	return float32(totalLoss / float64(validCount))
}

// BaseSwiGLUVec computes SwiGLU in-place on paired gate/up vectors.
// This is a helper used by the fused matmul+activation kernels.
//
// SwiGLU(gate, up) = SiLU(gate) * up = (gate * sigmoid(gate)) * up
func BaseSwiGLUVec(gate, up hwy.Vec[float32]) hwy.Vec[float32] {
	sig := math.BaseSigmoidVec[float32](gate)
	silu := hwy.Mul(gate, sig)
	return hwy.Mul(silu, up)
}
