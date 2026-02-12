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
	"testing"
)

func TestFocalLoss_Basic(t *testing.T) {
	batchSize, numClasses := 2, 3
	alpha, gamma := float32(0.25), float32(2.0)

	// Correct predictions with high confidence
	probs := []float32{0.9, 0.05, 0.05, 0.1, 0.8, 0.1}
	targets := []float32{1, 0, 0, 0, 1, 0}

	loss := FocalLoss(probs, targets, alpha, gamma, batchSize, numClasses)

	if loss <= 0 || stdmath.IsNaN(float64(loss)) || stdmath.IsInf(float64(loss), 0) {
		t.Errorf("unexpected loss: %v", loss)
	}

	// Loss should be small for correct, high-confidence predictions
	if loss > 0.1 {
		t.Errorf("focal loss should be small for confident correct predictions, got %v", loss)
	}

	// Compare with wrong predictions
	wrongProbs := []float32{0.1, 0.45, 0.45, 0.45, 0.1, 0.45}
	wrongLoss := FocalLoss(wrongProbs, targets, alpha, gamma, batchSize, numClasses)

	if wrongLoss <= loss {
		t.Errorf("loss for wrong predictions (%v) should be > loss for correct (%v)", wrongLoss, loss)
	}
}

func TestFocalLoss_ReducesToCE(t *testing.T) {
	// With gamma=0 and alpha=1, focal loss reduces to standard cross-entropy
	// for the positive class term.
	batchSize, numClasses := 1, 2
	alpha := float32(1.0)
	gamma := float32(0.0)

	probs := []float32{0.7, 0.3}
	targets := []float32{1, 0}

	focalLoss := FocalLoss(probs, targets, alpha, gamma, batchSize, numClasses)

	// CE = -log(0.7) ≈ 0.3567
	expectedCE := float32(-stdmath.Log(0.7))
	if stdmath.Abs(float64(focalLoss-expectedCE)) > 0.01 {
		t.Errorf("focal(gamma=0) = %v, expected CE = %v", focalLoss, expectedCE)
	}
}

func TestFocalLossBackward_NumericalGradient(t *testing.T) {
	batchSize, numClasses := 2, 3
	alpha, gamma := float32(0.25), float32(2.0)

	probs := []float32{0.7, 0.2, 0.1, 0.1, 0.6, 0.3}
	targets := []float32{1, 0, 0, 0, 1, 0}

	gradProbs := make([]float32, batchSize*numClasses)
	FocalLossBackward(probs, targets, gradProbs, alpha, gamma, batchSize, numClasses)

	// Numerical gradient
	eps := float32(1e-4)
	for i := range len(probs) {
		orig := probs[i]
		probs[i] = orig + eps
		lPlus := FocalLoss(probs, targets, alpha, gamma, batchSize, numClasses)
		probs[i] = orig - eps
		lMinus := FocalLoss(probs, targets, alpha, gamma, batchSize, numClasses)
		probs[i] = orig

		numerical := (lPlus - lMinus) / (2 * eps)
		diff := stdmath.Abs(float64(gradProbs[i] - numerical))
		denom := stdmath.Max(1e-6, stdmath.Abs(float64(numerical)))
		if diff/denom > 5e-2 {
			t.Errorf("[%d]: analytical=%v, numerical=%v, diff=%v", i, gradProbs[i], numerical, diff)
		}
	}
}
