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
	"github.com/ajroetker/go-highway/hwy"
	"github.com/ajroetker/go-highway/hwy/contrib/nn"
	"github.com/ajroetker/go-highway/hwy/contrib/workerpool"
)

// FusedLoRAMLPBackwardAuto computes the backward pass for the entire fused
// LoRA MLP block.
//
// Forward: output = down(SwiGLU(gate(x), up(x)))
//
// The backward reverses the pipeline:
//  1. Down-proj backward → gradHidden
//  2. SwiGLU backward → gradGate, gradUp
//  3. Gate-proj backward + Up-proj backward → gradX (accumulated from both)
//
// Parameters:
//   - gradOutput:  [batchSize, inFeatures] — upstream gradient
//   - saved:       MLPSaved from forward pass
//   - wGate:       [intermediateSize, inFeatures]
//   - wUp:         [intermediateSize, inFeatures]
//   - wDown:       [inFeatures, intermediateSize]
//   - loraGate, loraUp, loraDown: LoRA params (nil if not used)
//   - gradX:       [batchSize, inFeatures] — accumulated
//   - gradWGate:   [intermediateSize, inFeatures] — accumulated (nil to skip)
//   - gradWUp:     [intermediateSize, inFeatures] — accumulated (nil to skip)
//   - gradWDown:   [inFeatures, intermediateSize] — accumulated (nil to skip)
//   - gradsGate, gradsUp, gradsDown: LoRA gradient structs (nil to skip)
func FusedLoRAMLPBackwardAuto[T hwy.Floats](
	pool *workerpool.Pool,
	gradOutput []T, saved *nn.MLPSaved[T],
	wGate, wUp, wDown []T,
	loraGate, loraUp, loraDown *nn.LoRAParams[T],
	gradX, gradWGate, gradWUp, gradWDown []T,
	gradsGate, gradsUp, gradsDown *nn.LoRAGrads[T],
	batchSize, inFeatures, intermediateSize int,
) {
	// Step 1: Down-projection backward → gradHidden
	gradHidden := make([]T, batchSize*intermediateSize)

	if loraDown != nil {
		var gdA, gdB []T
		if gradsDown != nil {
			gdA = gradsDown.GradA
			gdB = gradsDown.GradB
		}
		FusedLoRADenseBackwardAuto(pool, gradOutput, saved.Hidden, saved.HDown,
			wDown, loraDown.A, loraDown.B, loraDown.Scale,
			gradHidden, gradWDown, nil, gdA, gdB,
			batchSize, intermediateSize, inFeatures, loraDown.Rank)
	} else {
		DenseBackwardAuto(pool, gradOutput, saved.Hidden, wDown,
			gradHidden, gradWDown, nil,
			batchSize, intermediateSize, inFeatures)
	}

	// Step 2: SwiGLU backward → gradGate, gradUp
	gradGate := make([]T, batchSize*intermediateSize)
	gradUp := make([]T, batchSize*intermediateSize)
	SwiGLUBackwardAuto(pool, gradHidden, saved.Gate, saved.Up,
		gradGate, gradUp, batchSize, intermediateSize)

	// Step 3: Gate-proj backward → gradX (accumulated)
	if loraGate != nil {
		var ggA, ggB []T
		if gradsGate != nil {
			ggA = gradsGate.GradA
			ggB = gradsGate.GradB
		}
		FusedLoRADenseBackwardAuto(pool, gradGate, saved.X, saved.HGate,
			wGate, loraGate.A, loraGate.B, loraGate.Scale,
			gradX, gradWGate, nil, ggA, ggB,
			batchSize, inFeatures, intermediateSize, loraGate.Rank)
	} else {
		DenseBackwardAuto(pool, gradGate, saved.X, wGate,
			gradX, gradWGate, nil,
			batchSize, inFeatures, intermediateSize)
	}

	// Step 4: Up-proj backward → gradX (accumulated from both gate + up)
	if loraUp != nil {
		var guA, guB []T
		if gradsUp != nil {
			guA = gradsUp.GradA
			guB = gradsUp.GradB
		}
		FusedLoRADenseBackwardAuto(pool, gradUp, saved.X, saved.HUp,
			wUp, loraUp.A, loraUp.B, loraUp.Scale,
			gradX, gradWUp, nil, guA, guB,
			batchSize, inFeatures, intermediateSize, loraUp.Rank)
	} else {
		DenseBackwardAuto(pool, gradUp, saved.X, wUp,
			gradX, gradWUp, nil,
			batchSize, inFeatures, intermediateSize)
	}
}
