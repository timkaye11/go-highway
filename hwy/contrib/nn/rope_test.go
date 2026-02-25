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

func TestRoPE_SIMDvsScalar(t *testing.T) {
	tests := []struct {
		seqLen, headDim int
	}{
		{1, 4},
		{4, 8},
		{8, 16},
		{16, 64},
		{32, 128},
		{3, 6}, // odd seq, small dim
	}

	for _, tt := range tests {
		name := fmt.Sprintf("seq=%d_dim=%d", tt.seqLen, tt.headDim)
		t.Run(name, func(t *testing.T) {
			size := tt.seqLen * tt.headDim
			cos, sin := PrecomputeRoPE[float32](tt.seqLen, tt.headDim, 10000.0)

			// SIMD version
			simdData := make([]float32, size)
			for i := range simdData {
				simdData[i] = float32(i)*0.1 - float32(size)*0.05
			}

			// Scalar version (copy same input)
			scalarData := make([]float32, size)
			copy(scalarData, simdData)

			RoPE(simdData, cos, sin, tt.seqLen, tt.headDim)
			RoPEScalar(scalarData, cos, sin, tt.seqLen, tt.headDim)

			for i := range simdData {
				diff := stdmath.Abs(float64(simdData[i] - scalarData[i]))
				if diff > 1e-4 {
					t.Errorf("[%d] SIMD=%v, Scalar=%v, diff=%v", i, simdData[i], scalarData[i], diff)
				}
			}
		})
	}
}

func TestRoPE_Orthogonality(t *testing.T) {
	// RoPE should preserve vector norms (it's a rotation)
	seqLen, headDim := 4, 8

	cos, sin := PrecomputeRoPE[float32](seqLen, headDim, 10000.0)

	data := make([]float32, seqLen*headDim)
	for i := range data {
		data[i] = float32(i)*0.3 - 2.0
	}

	// Compute norms before
	normsBefore := make([]float64, seqLen)
	for pos := range seqLen {
		off := pos * headDim
		var norm float64
		for d := range headDim {
			norm += float64(data[off+d]) * float64(data[off+d])
		}
		normsBefore[pos] = stdmath.Sqrt(norm)
	}

	RoPE(data, cos, sin, seqLen, headDim)

	// Compute norms after
	for pos := range seqLen {
		off := pos * headDim
		var norm float64
		for d := range headDim {
			norm += float64(data[off+d]) * float64(data[off+d])
		}
		normAfter := stdmath.Sqrt(norm)

		relDiff := stdmath.Abs(normAfter-normsBefore[pos]) / normsBefore[pos]
		if relDiff > 1e-5 {
			t.Errorf("pos %d: norm before=%v, after=%v, rel_diff=%v",
				pos, normsBefore[pos], normAfter, relDiff)
		}
	}
}

func TestRoPEAuto(t *testing.T) {
	pool := workerpool.New(0)
	defer pool.Close()

	numQHeads, numKVHeads := 4, 2
	seqLen, headDim := 8, 16

	cos, sin := PrecomputeRoPE[float32](seqLen, headDim, 10000.0)

	headStride := seqLen * headDim
	Q := make([]float32, numQHeads*headStride)
	K := make([]float32, numKVHeads*headStride)
	for i := range Q {
		Q[i] = float32(i)*0.01 - 1.0
	}
	for i := range K {
		K[i] = float32(i)*0.02 - 0.5
	}

	// Copy for scalar reference
	Qs := make([]float32, len(Q))
	Ks := make([]float32, len(K))
	copy(Qs, Q)
	copy(Ks, K)

	// Apply via RoPEAuto
	RoPEAuto(pool, Q, K, cos, sin, numQHeads, numKVHeads, seqLen, headDim)

	// Apply per-head scalar reference
	for h := range numQHeads {
		off := h * headStride
		RoPEScalar(Qs[off:off+headStride], cos, sin, seqLen, headDim)
	}
	for h := range numKVHeads {
		off := h * headStride
		RoPEScalar(Ks[off:off+headStride], cos, sin, seqLen, headDim)
	}

	// Compare
	for i := range Q {
		if stdmath.Abs(float64(Q[i]-Qs[i])) > 1e-5 {
			t.Errorf("Q[%d]: Auto=%v, Scalar=%v", i, Q[i], Qs[i])
		}
	}
	for i := range K {
		if stdmath.Abs(float64(K[i]-Ks[i])) > 1e-5 {
			t.Errorf("K[%d]: Auto=%v, Scalar=%v", i, K[i], Ks[i])
		}
	}
}

func TestRoPE_Empty(t *testing.T) {
	RoPE[float32](nil, nil, nil, 0, 0)
	RoPEScalar[float32](nil, nil, nil, 0, 0)
}

func BenchmarkRoPE(b *testing.B) {
	configs := []struct {
		seqLen, headDim int
	}{
		{128, 64},
		{256, 64},
		{512, 128},
	}

	for _, c := range configs {
		cos, sin := PrecomputeRoPE[float32](c.seqLen, c.headDim, 10000.0)
		data := make([]float32, c.seqLen*c.headDim)
		for i := range data {
			data[i] = float32(i) * 0.01
		}

		b.Run(fmt.Sprintf("SIMD/seq=%d_dim=%d", c.seqLen, c.headDim), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				RoPE(data, cos, sin, c.seqLen, c.headDim)
			}
		})

		b.Run(fmt.Sprintf("Scalar/seq=%d_dim=%d", c.seqLen, c.headDim), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				RoPEScalar(data, cos, sin, c.seqLen, c.headDim)
			}
		})
	}
}
