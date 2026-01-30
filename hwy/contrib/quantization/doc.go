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

// Package quantization provides SIMD-accelerated dequantization functions
// for common neural network weight quantization formats.
//
// Supported formats:
//   - NF4 (4-bit NormalFloat): Used in QLoRA for efficient LLM fine-tuning
//   - Int4 (4-bit signed integer): Symmetric quantization with range [-8, 7]
//   - Int8 (8-bit signed integer): Standard quantization with range [-128, 127]
//
// All formats use per-group scaling for improved accuracy. The groupSize
// parameter controls how many weights share a single scale factor.
//
// Example usage:
//
//	// Dequantize NF4-packed weights
//	output := make([]float32, numElements)
//	quantization.DequantizeNF4(packedWeights, scales, output, groupSize)
//
// For fused dequantization + matmul operations, see the matmul package
// which provides memory-efficient fused kernels.
package quantization
