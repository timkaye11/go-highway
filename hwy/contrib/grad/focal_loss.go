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
)

// FocalLoss computes the focal loss between predicted probabilities and targets.
//
//	FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
//
// where p_t = p for target=1 and p_t = 1-p for target=0.
//
// Parameters:
//   - probs:      [batchSize, numClasses] — predicted probabilities (after softmax)
//   - targets:    [batchSize, numClasses] — one-hot or soft labels
//   - alpha:      balancing factor (typically 0.25)
//   - gamma:      focusing parameter (typically 2.0)
//   - batchSize:  number of samples
//   - numClasses: number of classes
//
// Returns the mean focal loss across the batch.
func FocalLoss[T hwy.Floats](probs, targets []T, alpha, gamma T, batchSize, numClasses int) T {
	var totalLoss float64

	for i := range batchSize {
		for j := range numClasses {
			idx := i*numClasses + j
			p := clampProb(float64(probs[idx]))
			t := float64(targets[idx])

			if t > 0 {
				totalLoss -= t * float64(alpha) * pow64(1.0-p, float64(gamma)) * stdmath.Log(p)
			}
			if t < 1 {
				totalLoss -= (1.0 - t) * (1.0 - float64(alpha)) * pow64(p, float64(gamma)) * stdmath.Log(1.0-p)
			}
		}
	}

	return T(totalLoss / float64(batchSize))
}

// clampProb clamps a probability to [eps, 1-eps] for numerical stability.
func clampProb(p float64) float64 {
	const eps = 1e-7
	if p < eps {
		return eps
	}
	if p > 1.0-eps {
		return 1.0 - eps
	}
	return p
}

// FocalLossBackward computes the backward pass for focal loss.
//
// For the positive class term: -alpha * (1-p)^gamma * log(p)
//
//	d/dp = -alpha * [-gamma * (1-p)^(gamma-1) * log(p) + (1-p)^gamma / p]
//	     = alpha * (1-p)^(gamma-1) * [gamma * log(p) * (1-p) - (1-p)^gamma] / p ... (simplified below)
//
// For simplicity we use the direct derivative of each term:
//
//	d/dp [-alpha * t * (1-p)^gamma * log(p)]
//	  = -alpha * t * [ -gamma*(1-p)^(gamma-1)*log(p) + (1-p)^gamma * (1/p) ]
//	  = alpha * t * (1-p)^(gamma-1) * [ gamma*log(p) - (1-p)/p ]
//
//	d/dp [-(1-alpha) * (1-t) * p^gamma * log(1-p)]
//	  = -(1-alpha) * (1-t) * [ gamma*p^(gamma-1)*log(1-p) + p^gamma * (-1/(1-p)) ]
//	  = (1-alpha) * (1-t) * p^(gamma-1) * [ -gamma*log(1-p) + p/(1-p) ]
//
// gradProbs is accumulated into.
func FocalLossBackward[T hwy.Floats](
	probs, targets, gradProbs []T,
	alpha, gamma T,
	batchSize, numClasses int,
) {
	invN := 1.0 / float64(batchSize)
	a := float64(alpha)
	g := float64(gamma)

	for i := range batchSize {
		for j := range numClasses {
			idx := i*numClasses + j
			p := clampProb(float64(probs[idx]))
			t := float64(targets[idx])

			var grad float64

			// Positive class: d/dp [-alpha * t * (1-p)^gamma * log(p)]
			if t > 0 {
				omp := 1.0 - p // (1 - p)
				logp := stdmath.Log(p)
				// -alpha * t * [-gamma*(1-p)^(gamma-1)*log(p) + (1-p)^gamma/p]
				term1 := -g * pow64(omp, g-1) * logp
				term2 := pow64(omp, g) / p
				grad += -a * t * (term1 + term2)
			}

			// Negative class: d/dp [-(1-alpha) * (1-t) * p^gamma * log(1-p)]
			if t < 1 {
				omp := 1.0 - p
				logOmp := stdmath.Log(omp)
				// -(1-alpha) * (1-t) * [gamma*p^(gamma-1)*log(1-p) + p^gamma*(-1/(1-p))]
				term1 := g * pow64(p, g-1) * logOmp
				term2 := -pow64(p, g) / omp
				grad += -(1.0 - a) * (1.0 - t) * (term1 + term2)
			}

			gradProbs[idx] += T(grad * invN)
		}
	}
}
