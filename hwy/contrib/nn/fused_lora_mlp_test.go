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

package nn

import (
	"fmt"
	stdmath "math"
	"testing"

	"github.com/ajroetker/go-highway/hwy/contrib/workerpool"
)

func TestFusedLoRAMLP_NoLoRA(t *testing.T) {
	pool := workerpool.New(0)
	defer pool.Close()

	batch, inF, interSize := 2, 8, 16
	x := testInputs(batch * inF)
	wGate := testInputs(interSize * inF)
	wUp := testInputs(interSize * inF)
	for i := range wUp {
		wUp[i] += 0.1 // Make different from wGate
	}
	wDown := testInputs(inF * interSize)

	// Fused MLP
	fusedOut := make([]float32, batch*inF)
	saved := &MLPSaved[float32]{}
	FusedLoRAMLPForwardAuto[float32](pool, x, wGate, wUp, wDown,
		nil, nil, nil, fusedOut, saved,
		batch, inF, interSize)

	// Manual: gate = x @ wGate^T, up = x @ wUp^T, hidden = SwiGLU(gate, up), out = hidden @ wDown^T
	gate := make([]float32, batch*interSize)
	DenseScalar(x, wGate, nil, gate, batch, inF, interSize)
	up := make([]float32, batch*interSize)
	DenseScalar(x, wUp, nil, up, batch, inF, interSize)

	hidden := make([]float32, batch*interSize)
	for i := range hidden {
		g := float64(gate[i])
		sig := 1.0 / (1.0 + stdmath.Exp(-g))
		hidden[i] = float32(g * sig * float64(up[i]))
	}

	manualOut := make([]float32, batch*inF)
	DenseScalar(hidden, wDown, nil, manualOut, batch, interSize, inF)

	for i := range fusedOut {
		diff := stdmath.Abs(float64(fusedOut[i] - manualOut[i]))
		if diff > 1e-2 {
			t.Errorf("[%d]: fused=%v, manual=%v, diff=%v", i, fusedOut[i], manualOut[i], diff)
		}
	}
}

func TestFusedLoRAMLP_WithLoRA(t *testing.T) {
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

	loraGate := &LoRAParams[float32]{
		A: testInputs(rank * inF), B: testInputs(interSize * rank),
		Scale: 0.5, Rank: rank,
	}
	loraUp := &LoRAParams[float32]{
		A: testInputs(rank * inF), B: testInputs(interSize * rank),
		Scale: 0.3, Rank: rank,
	}
	for i := range loraUp.A {
		loraUp.A[i] += 0.2
	}
	loraDown := &LoRAParams[float32]{
		A: testInputs(rank * interSize), B: testInputs(inF * rank),
		Scale: 0.4, Rank: rank,
	}

	fusedOut := make([]float32, batch*inF)
	saved := &MLPSaved[float32]{}
	FusedLoRAMLPForwardAuto(pool, x, wGate, wUp, wDown,
		loraGate, loraUp, loraDown, fusedOut, saved,
		batch, inF, interSize)

	// Verify saved intermediates are populated
	if saved.X == nil || saved.Gate == nil || saved.Up == nil || saved.Hidden == nil {
		t.Fatal("saved intermediates not populated")
	}
	if saved.HGate == nil || saved.HUp == nil || saved.HDown == nil {
		t.Fatal("LoRA intermediates not saved")
	}

	// Verify output is non-zero (basic smoke test)
	var sumAbs float64
	for _, v := range fusedOut {
		sumAbs += stdmath.Abs(float64(v))
	}
	if sumAbs < 1e-6 {
		t.Errorf("output is all zeros")
	}
}

func BenchmarkFusedLoRAMLP(b *testing.B) {
	pool := workerpool.New(0)
	defer pool.Close()

	configs := []struct {
		batch, inF, interSize, rank int
	}{
		{4, 768, 3072, 16},
		{8, 768, 3072, 16},
	}

	for _, c := range configs {
		x := make([]float32, c.batch*c.inF)
		wGate := make([]float32, c.interSize*c.inF)
		wUp := make([]float32, c.interSize*c.inF)
		wDown := make([]float32, c.inF*c.interSize)
		output := make([]float32, c.batch*c.inF)

		loraGate := &LoRAParams[float32]{
			A: make([]float32, c.rank*c.inF), B: make([]float32, c.interSize*c.rank),
			Scale: 0.5, Rank: c.rank,
		}
		loraUp := &LoRAParams[float32]{
			A: make([]float32, c.rank*c.inF), B: make([]float32, c.interSize*c.rank),
			Scale: 0.5, Rank: c.rank,
		}
		loraDown := &LoRAParams[float32]{
			A: make([]float32, c.rank*c.interSize), B: make([]float32, c.inF*c.rank),
			Scale: 0.5, Rank: c.rank,
		}

		b.Run(fmt.Sprintf("b%d_%dx%d_r%d", c.batch, c.inF, c.interSize, c.rank), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				FusedLoRAMLPForwardAuto(pool, x, wGate, wUp, wDown,
					loraGate, loraUp, loraDown, output, nil,
					c.batch, c.inF, c.interSize)
			}
		})
	}
}
