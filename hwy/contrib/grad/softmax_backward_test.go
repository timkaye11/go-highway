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

	"github.com/ajroetker/go-highway/hwy/contrib/workerpool"
)

func softmaxRow(logits []float32) []float32 {
	n := len(logits)
	probs := make([]float32, n)
	maxVal := logits[0]
	for _, v := range logits[1:] {
		if v > maxVal {
			maxVal = v
		}
	}
	var sum float64
	for i, v := range logits {
		probs[i] = float32(stdmath.Exp(float64(v - maxVal)))
		sum += float64(probs[i])
	}
	inv := float32(1.0 / sum)
	for i := range probs {
		probs[i] *= inv
	}
	return probs
}

func TestSoftmaxBackward_AutoVsScalar(t *testing.T) {
	pool := workerpool.New(0)
	defer pool.Close()

	sizes := []struct{ rows, cols int }{
		{1, 8}, {2, 16}, {4, 7}, {8, 32},
	}

	for _, s := range sizes {
		n := s.rows * s.cols
		// Compute softmax probs first
		savedProbs := make([]float32, n)
		for r := range s.rows {
			logits := testInputs(s.cols)
			probs := softmaxRow(logits)
			copy(savedProbs[r*s.cols:], probs)
		}

		gradOutput := testInputs(n)

		gradAuto := make([]float32, n)
		gradScalar := make([]float32, n)

		SoftmaxBackwardAuto(pool, gradOutput, savedProbs, gradAuto, s.rows, s.cols)

		for r := range s.rows {
			off := r * s.cols
			SoftmaxBackwardScalar(gradOutput[off:off+s.cols], savedProbs[off:off+s.cols], gradScalar[off:off+s.cols])
		}

		allClose32(t, "SoftmaxBackward", gradAuto, gradScalar, 1e-5)
	}
}

func TestSoftmaxBackward_GradientSumProperty(t *testing.T) {
	// For softmax, the Jacobian has the property: sum_j (dL/dz_j) * p_j = sum_j J_ij * dL/dp_i
	// More practically: sum of gradients of z should relate to sum of gradients of p.
	// The specific property: sum_i dL/dz_i = sum_i p_i * (dL/dp_i - dot(p, dL/dp)) = 0
	// This is because sum_i p_i * dL/dp_i - dot(p,dL/dp) * sum_i p_i = dot - dot = 0.
	cols := 16
	logits := testInputs(cols)
	probs := softmaxRow(logits)
	gradOutput := testInputs(cols)

	gradInput := make([]float32, cols)
	SoftmaxBackward(gradOutput, probs, gradInput)

	var sum float64
	for i := range cols {
		sum += float64(gradInput[i])
	}
	// Sum should be approximately 0
	if stdmath.Abs(sum) > 1e-4 {
		t.Errorf("softmax backward grad sum should be ~0, got %v", sum)
	}
}

func TestCrossEntropyLoss(t *testing.T) {
	// Simple 2-class cross-entropy
	batchSize, numClasses := 2, 3

	// logits = [[1, 0, 0], [0, 1, 0]]
	logits := []float32{1, 0, 0, 0, 1, 0}
	// targets = one-hot matching logits
	targets := []float32{1, 0, 0, 0, 1, 0}

	loss := CrossEntropyLoss(logits, targets, batchSize, numClasses)

	// Loss should be positive and finite
	if loss <= 0 || stdmath.IsNaN(float64(loss)) || stdmath.IsInf(float64(loss), 0) {
		t.Errorf("unexpected loss: %v", loss)
	}

	// When predictions match targets exactly (as logits), loss should be relatively small
	// softmax([1,0,0]) = [0.576, 0.212, 0.212], -log(0.576) ≈ 0.551
	expected := float32(0.551)
	if stdmath.Abs(float64(loss-expected)) > 0.01 {
		t.Errorf("loss=%v, expected ~%v", loss, expected)
	}
}

func TestCrossEntropyBackward_AutoVsNumerical(t *testing.T) {
	batchSize, numClasses := 2, 4
	logits := []float32{1.0, 0.5, -0.5, 0.0, 0.0, 1.0, 0.5, -1.0}
	targets := []float32{1, 0, 0, 0, 0, 1, 0, 0}

	pool := workerpool.New(0)
	defer pool.Close()

	gradLogits := make([]float32, batchSize*numClasses)
	CrossEntropyBackward(pool, logits, targets, gradLogits, batchSize, numClasses)

	// Numerical gradient check
	eps := float32(1e-4)
	for i := range len(logits) {
		orig := logits[i]
		logits[i] = orig + eps
		lPlus := CrossEntropyLoss(logits, targets, batchSize, numClasses)
		logits[i] = orig - eps
		lMinus := CrossEntropyLoss(logits, targets, batchSize, numClasses)
		logits[i] = orig

		numerical := (lPlus - lMinus) / (2 * eps)
		diff := stdmath.Abs(float64(gradLogits[i] - numerical))
		denom := stdmath.Max(1.0, stdmath.Abs(float64(numerical)))
		if diff/denom > 1e-2 {
			t.Errorf("[%d]: analytical=%v, numerical=%v, diff=%v", i, gradLogits[i], numerical, diff)
		}
	}
}
