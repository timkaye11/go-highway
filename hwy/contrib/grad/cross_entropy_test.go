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
	"fmt"
	stdmath "math"
	"testing"

	"github.com/ajroetker/go-highway/hwy/contrib/workerpool"
)

func TestSparseCrossEntropy_Forward(t *testing.T) {
	// Simple 2-class case: logits = [[2, 1]], label = [0]
	// softmax([2,1]) = [e^2/(e^2+e), e/(e^2+e)] ≈ [0.731, 0.269]
	// loss = -log(0.731) ≈ 0.3133
	logits := []float32{2, 1}
	labels := []int32{0}
	loss := SparseCrossEntropyForward(logits, labels, 1, 2)

	expected := float32(0.3133)
	if stdmath.Abs(float64(loss-expected)) > 0.01 {
		t.Errorf("loss = %v, want ~%v", loss, expected)
	}

	// Label=1: loss = -log(0.269) ≈ 1.3133
	labels2 := []int32{1}
	loss2 := SparseCrossEntropyForward(logits, labels2, 1, 2)
	expected2 := float32(1.3133)
	if stdmath.Abs(float64(loss2-expected2)) > 0.01 {
		t.Errorf("loss = %v, want ~%v", loss2, expected2)
	}
}

func TestSparseCrossEntropy_BackwardVsForward(t *testing.T) {
	// Verify backward is consistent with forward via finite differences
	tests := []struct {
		batch, vocab int
	}{
		{1, 4},
		{2, 8},
		{4, 16},
		{2, 3},
	}

	for _, tt := range tests {
		name := fmt.Sprintf("b%d_v%d", tt.batch, tt.vocab)
		t.Run(name, func(t *testing.T) {
			logits := make([]float32, tt.batch*tt.vocab)
			labels := make([]int32, tt.batch)
			for i := range logits {
				logits[i] = float32(i)*0.3 - float32(tt.batch*tt.vocab)*0.15
			}
			for i := range labels {
				labels[i] = int32(i % tt.vocab)
			}

			// Analytical gradient
			analyticalGrad := make([]float32, tt.batch*tt.vocab)
			SparseCrossEntropyBackward(logits, labels, analyticalGrad, tt.batch, tt.vocab)

			// Numerical gradient
			eps := float32(1e-4)
			numericalGrad := make([]float32, tt.batch*tt.vocab)
			for i := range logits {
				orig := logits[i]

				logits[i] = orig + eps
				lossPlus := SparseCrossEntropyForward(logits, labels, tt.batch, tt.vocab)

				logits[i] = orig - eps
				lossMinus := SparseCrossEntropyForward(logits, labels, tt.batch, tt.vocab)

				logits[i] = orig
				numericalGrad[i] = (lossPlus - lossMinus) / (2 * eps)
			}

			allClose32(t, "grad_logits", analyticalGrad, numericalGrad, 5e-3)
		})
	}
}

func TestSparseCrossEntropy_FusedVsSeparate(t *testing.T) {
	pool := workerpool.New(0)
	defer pool.Close()

	tests := []struct {
		batch, vocab int
	}{
		{1, 4},
		{2, 8},
		{4, 16},
		{8, 32},
	}

	for _, tt := range tests {
		name := fmt.Sprintf("b%d_v%d", tt.batch, tt.vocab)
		t.Run(name, func(t *testing.T) {
			logits := make([]float32, tt.batch*tt.vocab)
			labels := make([]int32, tt.batch)
			for i := range logits {
				logits[i] = float32(i)*0.2 - float32(tt.batch*tt.vocab)*0.1
			}
			for i := range labels {
				labels[i] = int32(i % tt.vocab)
			}

			// Separate forward + backward
			sepLoss := SparseCrossEntropyForward(logits, labels, tt.batch, tt.vocab)
			sepGrad := make([]float32, tt.batch*tt.vocab)
			SparseCrossEntropyBackward(logits, labels, sepGrad, tt.batch, tt.vocab)

			// Fused forward + backward
			fusedGrad := make([]float32, tt.batch*tt.vocab)
			fusedLoss := SparseCrossEntropyForwardBackwardAuto(pool, logits, labels, fusedGrad, tt.batch, tt.vocab)

			// Losses must match
			if stdmath.Abs(float64(fusedLoss-sepLoss)) > 1e-4 {
				t.Errorf("fused loss = %v, separate loss = %v", fusedLoss, sepLoss)
			}

			// Gradients must match
			allClose32(t, "grad_fused_vs_sep", fusedGrad, sepGrad, 1e-4)
		})
	}
}

func TestSparseCrossEntropy_PoolVsNil(t *testing.T) {
	pool := workerpool.New(0)
	defer pool.Close()

	batch, vocab := 8, 32
	logits := make([]float32, batch*vocab)
	labels := make([]int32, batch)
	for i := range logits {
		logits[i] = float32(i) * 0.1
	}
	for i := range labels {
		labels[i] = int32(i % vocab)
	}

	seqGrad := make([]float32, batch*vocab)
	seqLoss := SparseCrossEntropyForwardBackwardAuto[float32](nil, logits, labels, seqGrad, batch, vocab)

	parGrad := make([]float32, batch*vocab)
	parLoss := SparseCrossEntropyForwardBackwardAuto(pool, logits, labels, parGrad, batch, vocab)

	if stdmath.Abs(float64(parLoss-seqLoss)) > 1e-5 {
		t.Errorf("par loss = %v, seq loss = %v", parLoss, seqLoss)
	}
	allClose32(t, "grad_par_vs_seq", parGrad, seqGrad, 1e-5)
}

func BenchmarkSparseCrossEntropy(b *testing.B) {
	pool := workerpool.New(0)
	defer pool.Close()

	configs := []struct{ batch, vocab int }{
		{8, 256},
		{8, 32000},
		{32, 32000},
	}

	for _, c := range configs {
		logits := make([]float32, c.batch*c.vocab)
		labels := make([]int32, c.batch)
		gradLogits := make([]float32, c.batch*c.vocab)
		for i := range logits {
			logits[i] = float32(i) * 0.001
		}
		for i := range labels {
			labels[i] = int32(i % c.vocab)
		}

		b.Run(fmt.Sprintf("Fused/%dx%d", c.batch, c.vocab), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				clear(gradLogits)
				SparseCrossEntropyForwardBackwardAuto(pool, logits, labels, gradLogits, c.batch, c.vocab)
			}
		})

		b.Run(fmt.Sprintf("Separate/%dx%d", c.batch, c.vocab), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				clear(gradLogits)
				SparseCrossEntropyForward(logits, labels, c.batch, c.vocab)
				SparseCrossEntropyBackward(logits, labels, gradLogits, c.batch, c.vocab)
			}
		})
	}
}
