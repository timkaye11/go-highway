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
	"github.com/ajroetker/go-highway/hwy/contrib/matmul"
	"github.com/ajroetker/go-highway/hwy/contrib/vec"
	"github.com/ajroetker/go-highway/hwy/contrib/workerpool"
)

// QuantType selects the quantization format for fused QLoRA operations.
type QuantType int

const (
	QuantNF4  QuantType = iota // NormalFloat 4-bit (QLoRA)
	QuantInt4                  // Symmetric INT4
)

// FusedQLoRADenseAuto computes a quantized dense layer with LoRA adaptation.
//
//	output = x @ dequant(W_q)^T + bias + scale * (x @ A^T) @ B^T
//
// The base weight matrix W_q is stored in quantized format (NF4 or INT4),
// while the LoRA adapters A, B remain in float32.
//
// Parameters:
//   - x:          [batchSize, inFeatures] — float32 input
//   - packed:     [inFeatures, outFeatures/2] — uint8 packed quantized weights
//   - scales:     per-group dequantization scales
//   - groupSize:  number of columns per scale group
//   - bias:       [outFeatures] — optional bias (nil to skip)
//   - A:          [rank, inFeatures] — LoRA down-projection (float32)
//   - B:          [outFeatures, rank] — LoRA up-projection (float32)
//   - loraScale:  LoRA scaling factor (alpha / rank)
//   - output:     [batchSize, outFeatures] — float32 output
//   - h:          [batchSize, rank] — saved intermediate (nil to skip)
//   - batchSize, inFeatures, outFeatures, rank: dimensions
//   - quantType:  QuantNF4 or QuantInt4
func FusedQLoRADenseAuto(
	pool *workerpool.Pool,
	x []float32, packed []uint8, scales []float32, groupSize int,
	bias, A, B []float32, loraScale float32,
	output, h []float32,
	batchSize, inFeatures, outFeatures, rank int,
	quantType QuantType,
) {
	// Step 1: h = x @ A^T (small matmul, warms x in cache)
	hBuf := h
	if hBuf == nil {
		hBuf = make([]float32, batchSize*rank)
	}
	matmul.MatMulKLastAuto(pool, x, A, hBuf, batchSize, rank, inFeatures)

	// Step 2: output = x @ dequant(W_q)^T (quantized matmul, x still hot)
	switch quantType {
	case QuantNF4:
		matmul.BaseFusedNF4MatMul(x, packed, scales, output, batchSize, inFeatures, outFeatures, groupSize)
	case QuantInt4:
		matmul.BaseFusedInt4MatMul(x, packed, scales, output, batchSize, inFeatures, outFeatures, groupSize)
	default:
		panic("fused_qlora_dense: unsupported quantType")
	}

	// Step 3: output += bias
	if bias != nil {
		addBias(output, bias, batchSize, outFeatures)
	}

	// Step 4: temp = h @ B^T, output += loraScale * temp
	temp := make([]float32, batchSize*outFeatures)
	matmul.MatMulKLastAuto(pool, hBuf, B, temp, batchSize, outFeatures, rank)
	vec.MulConstAddTo(output, loraScale, temp)
}
