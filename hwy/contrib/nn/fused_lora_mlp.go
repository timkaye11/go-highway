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
	"github.com/ajroetker/go-highway/hwy"
	"github.com/ajroetker/go-highway/hwy/contrib/activation"
	"github.com/ajroetker/go-highway/hwy/contrib/workerpool"
)

// FusedLoRAMLPForwardAuto computes the entire transformer MLP block with
// optional LoRA adapters on gate, up, and down projections.
//
// Architecture: gate-up SwiGLU MLP (LLaMA/Gemma style):
//
//	gate = FusedLoRADense(x, wGate, loraGate)
//	up   = FusedLoRADense(x, wUp, loraUp)
//	hidden = SwiGLU(gate, up)
//	output = FusedLoRADense(hidden, wDown, loraDown)
//
// The key optimization is that x is read once for all gate+up matmuls,
// and hidden is read once for the down matmul.
//
// Parameters:
//   - x:         [batchSize, inFeatures]
//   - wGate:     [intermediateSize, inFeatures]
//   - wUp:       [intermediateSize, inFeatures]
//   - wDown:     [inFeatures, intermediateSize]
//   - loraGate, loraUp, loraDown: LoRA params (nil to skip adapter)
//   - output:    [batchSize, inFeatures]
//   - saved:     MLPSaved struct for backward pass intermediates
//   - batchSize, inFeatures, intermediateSize: dimensions
func FusedLoRAMLPForwardAuto[T hwy.Floats](
	pool *workerpool.Pool,
	x, wGate, wUp, wDown []T,
	loraGate, loraUp, loraDown *LoRAParams[T],
	output []T, saved *MLPSaved[T],
	batchSize, inFeatures, intermediateSize int,
) {
	// Save input for backward
	if saved != nil {
		saved.X = make([]T, len(x))
		copy(saved.X, x)
	}

	// Step 1: Gate projection with optional LoRA (warms x in cache)
	gate := make([]T, batchSize*intermediateSize)
	var hGate []T
	if loraGate != nil && saved != nil {
		hGate = make([]T, batchSize*loraGate.Rank)
	}
	if loraGate != nil {
		FusedLoRADenseAuto(pool, x, wGate, nil, loraGate.A, loraGate.B, loraGate.Scale,
			gate, hGate, batchSize, inFeatures, intermediateSize, loraGate.Rank)
	} else {
		DenseAuto(pool, x, wGate, nil, gate, batchSize, inFeatures, intermediateSize)
	}

	// Step 2: Up projection with optional LoRA (x still hot from step 1)
	up := make([]T, batchSize*intermediateSize)
	var hUp []T
	if loraUp != nil && saved != nil {
		hUp = make([]T, batchSize*loraUp.Rank)
	}
	if loraUp != nil {
		FusedLoRADenseAuto(pool, x, wUp, nil, loraUp.A, loraUp.B, loraUp.Scale,
			up, hUp, batchSize, inFeatures, intermediateSize, loraUp.Rank)
	} else {
		DenseAuto(pool, x, wUp, nil, up, batchSize, inFeatures, intermediateSize)
	}

	// Step 3: SwiGLU activation
	hidden := make([]T, batchSize*intermediateSize)
	activation.ParallelSwiGLU(pool, gate, up, hidden, batchSize, intermediateSize)

	// Step 4: Down projection with optional LoRA
	var hDown []T
	if loraDown != nil && saved != nil {
		hDown = make([]T, batchSize*loraDown.Rank)
	}
	if loraDown != nil {
		FusedLoRADenseAuto(pool, hidden, wDown, nil, loraDown.A, loraDown.B, loraDown.Scale,
			output, hDown, batchSize, intermediateSize, inFeatures, loraDown.Rank)
	} else {
		DenseAuto(pool, hidden, wDown, nil, output, batchSize, intermediateSize, inFeatures)
	}

	// Save intermediates for backward
	if saved != nil {
		saved.Gate = gate
		saved.Up = up
		saved.Hidden = hidden
		saved.HGate = hGate
		saved.HUp = hUp
		saved.HDown = hDown
	}
}
