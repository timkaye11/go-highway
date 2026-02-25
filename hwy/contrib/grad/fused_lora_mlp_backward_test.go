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
	"testing"

	"github.com/ajroetker/go-highway/hwy/contrib/nn"
	"github.com/ajroetker/go-highway/hwy/contrib/workerpool"
)

func TestFusedLoRAMLPBackward_NoLoRA(t *testing.T) {
	pool := workerpool.New(0)
	defer pool.Close()

	batch, inF, interSize := 2, 8, 16
	x := testInputs(batch * inF)
	wGate := testInputs(interSize * inF)
	wUp := testInputs(interSize * inF)
	for i := range wUp {
		wUp[i] += 0.1
	}
	wDown := testInputs(inF * interSize)

	// Forward pass to get saved intermediates
	output := make([]float32, batch*inF)
	saved := &nn.MLPSaved[float32]{}
	nn.FusedLoRAMLPForwardAuto[float32](pool, x, wGate, wUp, wDown,
		nil, nil, nil, output, saved,
		batch, inF, interSize)

	// Backward pass
	gradOutput := testInputs(batch * inF)
	gradX := make([]float32, batch*inF)
	gradWGate := make([]float32, interSize*inF)
	gradWUp := make([]float32, interSize*inF)
	gradWDown := make([]float32, inF*interSize)

	FusedLoRAMLPBackwardAuto[float32](pool, gradOutput, saved,
		wGate, wUp, wDown,
		nil, nil, nil,
		gradX, gradWGate, gradWUp, gradWDown,
		nil, nil, nil,
		batch, inF, interSize)

	// Verify gradX is non-zero
	var sumAbs float64
	for _, v := range gradX {
		if v != 0 {
			sumAbs += float64(v)
		}
	}
	if sumAbs == 0 {
		t.Errorf("gradX is all zeros")
	}

	// Verify gradWeight is non-zero
	for _, v := range gradWGate {
		if v != 0 {
			sumAbs += float64(v)
		}
	}
	if sumAbs == 0 {
		t.Errorf("gradWGate is all zeros")
	}
}

func TestFusedLoRAMLPBackward_WithLoRA(t *testing.T) {
	pool := workerpool.New(0)
	defer pool.Close()

	batch, inF, interSize, rank := 2, 8, 16, 2
	x := testInputs(batch * inF)
	wGate := testInputs(interSize * inF)
	wUp := testInputs(interSize * inF)
	for i := range wUp {
		wUp[i] += 0.1
	}
	wDown := testInputs(inF * interSize)

	loraGate := &nn.LoRAParams[float32]{
		A: testInputs(rank * inF), B: testInputs(interSize * rank),
		Scale: 0.5, Rank: rank,
	}
	loraUp := &nn.LoRAParams[float32]{
		A: testInputs(rank * inF), B: testInputs(interSize * rank),
		Scale: 0.3, Rank: rank,
	}
	for i := range loraUp.A {
		loraUp.A[i] += 0.2
	}
	loraDown := &nn.LoRAParams[float32]{
		A: testInputs(rank * interSize), B: testInputs(inF * rank),
		Scale: 0.4, Rank: rank,
	}

	// Forward pass
	output := make([]float32, batch*inF)
	saved := &nn.MLPSaved[float32]{}
	nn.FusedLoRAMLPForwardAuto(pool, x, wGate, wUp, wDown,
		loraGate, loraUp, loraDown, output, saved,
		batch, inF, interSize)

	// Backward pass
	gradOutput := testInputs(batch * inF)
	gradX := make([]float32, batch*inF)
	gradWGate := make([]float32, interSize*inF)
	gradWUp := make([]float32, interSize*inF)
	gradWDown := make([]float32, inF*interSize)

	gradsGate := &nn.LoRAGrads[float32]{
		GradA: make([]float32, rank*inF),
		GradB: make([]float32, interSize*rank),
	}
	gradsUp := &nn.LoRAGrads[float32]{
		GradA: make([]float32, rank*inF),
		GradB: make([]float32, interSize*rank),
	}
	gradsDown := &nn.LoRAGrads[float32]{
		GradA: make([]float32, rank*interSize),
		GradB: make([]float32, inF*rank),
	}

	FusedLoRAMLPBackwardAuto(pool, gradOutput, saved,
		wGate, wUp, wDown,
		loraGate, loraUp, loraDown,
		gradX, gradWGate, gradWUp, gradWDown,
		gradsGate, gradsUp, gradsDown,
		batch, inF, interSize)

	// Verify all gradients are non-zero
	checkNonZero := func(name string, s []float32) {
		t.Helper()
		var any bool
		for _, v := range s {
			if v != 0 {
				any = true
				break
			}
		}
		if !any {
			t.Errorf("%s is all zeros", name)
		}
	}
	checkNonZero("gradX", gradX)
	checkNonZero("gradWGate", gradWGate)
	checkNonZero("gradWUp", gradWUp)
	checkNonZero("gradWDown", gradWDown)
	checkNonZero("gradsGate.GradA", gradsGate.GradA)
	checkNonZero("gradsGate.GradB", gradsGate.GradB)
	checkNonZero("gradsUp.GradA", gradsUp.GradA)
	checkNonZero("gradsUp.GradB", gradsUp.GradB)
	checkNonZero("gradsDown.GradA", gradsDown.GradA)
	checkNonZero("gradsDown.GradB", gradsDown.GradB)
}

func BenchmarkFusedLoRAMLPBackward(b *testing.B) {
	pool := workerpool.New(0)
	defer pool.Close()

	configs := []struct {
		batch, inF, interSize, rank int
	}{
		{4, 768, 3072, 16},
	}

	for _, c := range configs {
		x := testInputs(c.batch * c.inF)
		wGate := testInputs(c.interSize * c.inF)
		wUp := testInputs(c.interSize * c.inF)
		wDown := testInputs(c.inF * c.interSize)

		loraGate := &nn.LoRAParams[float32]{
			A: make([]float32, c.rank*c.inF), B: make([]float32, c.interSize*c.rank),
			Scale: 0.5, Rank: c.rank,
		}
		loraUp := &nn.LoRAParams[float32]{
			A: make([]float32, c.rank*c.inF), B: make([]float32, c.interSize*c.rank),
			Scale: 0.5, Rank: c.rank,
		}
		loraDown := &nn.LoRAParams[float32]{
			A: make([]float32, c.rank*c.interSize), B: make([]float32, c.inF*c.rank),
			Scale: 0.5, Rank: c.rank,
		}

		output := make([]float32, c.batch*c.inF)
		saved := &nn.MLPSaved[float32]{}
		nn.FusedLoRAMLPForwardAuto(pool, x, wGate, wUp, wDown,
			loraGate, loraUp, loraDown, output, saved,
			c.batch, c.inF, c.interSize)

		gradOutput := testInputs(c.batch * c.inF)
		gradX := make([]float32, c.batch*c.inF)
		gradWGate := make([]float32, c.interSize*c.inF)
		gradWUp := make([]float32, c.interSize*c.inF)
		gradWDown := make([]float32, c.inF*c.interSize)

		label := fmt.Sprintf("b%d_%dx%d_r%d", c.batch, c.inF, c.interSize, c.rank)

		b.Run("Fused/"+label, func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				clear(gradX)
				clear(gradWGate)
				clear(gradWUp)
				clear(gradWDown)
				FusedLoRAMLPBackwardAuto[float32](pool, gradOutput, saved,
					wGate, wUp, wDown,
					loraGate, loraUp, loraDown,
					gradX, gradWGate, gradWUp, gradWDown,
					nil, nil, nil,
					c.batch, c.inF, c.interSize)
			}
		})

		b.Run("Separate/"+label, func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				clear(gradX)
				clear(gradWGate)
				clear(gradWUp)
				clear(gradWDown)
				// Down backward
				gradHidden := make([]float32, c.batch*c.interSize)
				DenseBackwardAuto(pool, gradOutput, saved.Hidden, wDown,
					gradHidden, gradWDown, nil, c.batch, c.interSize, c.inF)
				LoRABackwardAuto(pool, gradOutput, saved.Hidden, saved.HDown,
					wDown, loraDown.A, loraDown.B, loraDown.Scale,
					gradHidden, nil, nil, c.batch, c.interSize, c.inF, loraDown.Rank)
				// SwiGLU backward
				gradGate := make([]float32, c.batch*c.interSize)
				gradUp := make([]float32, c.batch*c.interSize)
				SwiGLUBackwardAuto(pool, gradHidden, saved.Gate, saved.Up,
					gradGate, gradUp, c.batch, c.interSize)
				// Gate backward
				DenseBackwardAuto(pool, gradGate, saved.X, wGate,
					gradX, gradWGate, nil, c.batch, c.inF, c.interSize)
				LoRABackwardAuto(pool, gradGate, saved.X, saved.HGate,
					wGate, loraGate.A, loraGate.B, loraGate.Scale,
					gradX, nil, nil, c.batch, c.inF, c.interSize, loraGate.Rank)
				// Up backward
				DenseBackwardAuto(pool, gradUp, saved.X, wUp,
					gradX, gradWUp, nil, c.batch, c.inF, c.interSize)
				LoRABackwardAuto(pool, gradUp, saved.X, saved.HUp,
					wUp, loraUp.A, loraUp.B, loraUp.Scale,
					gradX, nil, nil, c.batch, c.inF, c.interSize, loraUp.Rank)
			}
		})
	}
}
