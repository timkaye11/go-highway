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

// Package grad provides backward-pass (gradient) kernels for neural network
// operations defined in the nn, activation, and matmul packages.
//
// # Accumulate Semantics
//
// All backward functions use accumulate semantics for gradient outputs:
//
//	gradInput[i] += <computed gradient>
//
// The caller is responsible for zeroing gradient buffers before the first
// backward call in a training step. Accumulate semantics naturally support
// residual connections and parameter sharing (e.g., tied embeddings),
// where a single parameter receives gradients from multiple paths.
//
// # Naming Convention
//
// Functions follow the PyTorch convention: FooBackward computes the backward
// pass for the corresponding Foo forward operation.
//
// # Nil-able Gradient Outputs
//
// For operations with multiple gradient outputs (e.g., DenseBackwardAuto
// produces gradInput, gradWeight, gradBias), any gradient output can be nil
// to skip its computation. This allows callers to compute only the gradients
// they need (e.g., skip gradWeight for frozen layers).
//
// # SIMD Acceleration
//
// Element-wise backward ops (GELU, ReLU, SiLU, Tanh, LayerNorm, AdamW) use
// hwygen-generated SIMD specializations. Composition-based backward ops
// (Dense, Softmax, SDPA, LoRA) delegate to existing SIMD-accelerated
// matmul and vec operations.
package grad
