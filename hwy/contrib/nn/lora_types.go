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

import "github.com/ajroetker/go-highway/hwy"

// LoRAParams holds the parameters for a single LoRA adapter.
type LoRAParams[T hwy.Floats] struct {
	A     []T // [rank, inFeatures] — down-projection
	B     []T // [outDim, rank] — up-projection
	Scale T   // scaling factor (alpha / rank)
	Rank  int
}

// LoRAGrads holds the gradients for a single LoRA adapter.
type LoRAGrads[T hwy.Floats] struct {
	GradA []T // [rank, inFeatures]
	GradB []T // [outDim, rank]
}

// MLPSaved holds saved intermediates from a fused LoRA MLP forward pass,
// needed by the backward pass.
type MLPSaved[T hwy.Floats] struct {
	X      []T // [batchSize, inFeatures] — original input
	Gate   []T // [batchSize, intermediateSize] — gate projection output (pre-activation)
	Up     []T // [batchSize, intermediateSize] — up projection output
	Hidden []T // [batchSize, intermediateSize] — SwiGLU output (= SiLU(gate) * up)

	// LoRA intermediates (nil if no LoRA on that projection)
	HGate []T // [batchSize, rank_gate] — x @ A_gate^T
	HUp   []T // [batchSize, rank_up] — x @ A_up^T
	HDown []T // [batchSize, rank_down] — hidden @ A_down^T
}
