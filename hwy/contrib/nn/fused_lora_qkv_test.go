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

	"github.com/ajroetker/go-highway/hwy/contrib/matmul"
	"github.com/ajroetker/go-highway/hwy/contrib/vec"
	"github.com/ajroetker/go-highway/hwy/contrib/workerpool"
)

func TestFusedLoRAQKV_AutoVsScalar(t *testing.T) {
	pool := workerpool.New(0)
	defer pool.Close()

	tests := []struct {
		batch, inF, qDim, kvDim, rank int
	}{
		{1, 8, 4, 4, 2},
		{2, 16, 8, 4, 2},
		{4, 32, 16, 8, 4},
	}

	for _, tt := range tests {
		name := fmt.Sprintf("b%d_%d_q%d_kv%d_r%d", tt.batch, tt.inF, tt.qDim, tt.kvDim, tt.rank)
		t.Run(name, func(t *testing.T) {
			totalOut := tt.qDim + 2*tt.kvDim
			x := testInputs(tt.batch * tt.inF)
			wQKV := testInputs(totalOut * tt.inF)
			biasQ := testInputs(tt.qDim)
			biasK := testInputs(tt.kvDim)
			biasV := testInputs(tt.kvDim)

			loraQ := &LoRAParams[float32]{
				A: testInputs(tt.rank * tt.inF), B: testInputs(tt.qDim * tt.rank),
				Scale: 0.5, Rank: tt.rank,
			}
			loraK := &LoRAParams[float32]{
				A: testInputs(tt.rank * tt.inF), B: testInputs(tt.kvDim * tt.rank),
				Scale: 0.3, Rank: tt.rank,
			}
			// Differentiate K's A from Q's A
			for i := range loraK.A {
				loraK.A[i] += 0.1
			}
			loraV := &LoRAParams[float32]{
				A: testInputs(tt.rank * tt.inF), B: testInputs(tt.kvDim * tt.rank),
				Scale: 0.4, Rank: tt.rank,
			}
			for i := range loraV.A {
				loraV.A[i] -= 0.1
			}

			// Auto
			aQ := make([]float32, tt.batch*tt.qDim)
			aK := make([]float32, tt.batch*tt.kvDim)
			aV := make([]float32, tt.batch*tt.kvDim)
			FusedLoRAQKVDenseAuto(pool, x, wQKV, biasQ, biasK, biasV,
				loraQ, loraK, loraV, aQ, aK, aV, nil, nil, nil,
				tt.batch, tt.inF, tt.qDim, tt.kvDim)

			// Scalar
			sQ := make([]float32, tt.batch*tt.qDim)
			sK := make([]float32, tt.batch*tt.kvDim)
			sV := make([]float32, tt.batch*tt.kvDim)
			FusedLoRAQKVDenseScalar(x, wQKV, biasQ, biasK, biasV,
				loraQ, loraK, loraV, sQ, sK, sV,
				tt.batch, tt.inF, tt.qDim, tt.kvDim)

			for i := range aQ {
				diff := stdmath.Abs(float64(aQ[i] - sQ[i]))
				if diff > 1e-2 {
					t.Errorf("Q[%d]: Auto=%v, Scalar=%v, diff=%v", i, aQ[i], sQ[i], diff)
				}
			}
			for i := range aK {
				diff := stdmath.Abs(float64(aK[i] - sK[i]))
				if diff > 1e-2 {
					t.Errorf("K[%d]: Auto=%v, Scalar=%v, diff=%v", i, aK[i], sK[i], diff)
				}
			}
			for i := range aV {
				diff := stdmath.Abs(float64(aV[i] - sV[i]))
				if diff > 1e-2 {
					t.Errorf("V[%d]: Auto=%v, Scalar=%v, diff=%v", i, aV[i], sV[i], diff)
				}
			}
		})
	}
}

func TestFusedLoRAQKV_NilLoRA(t *testing.T) {
	pool := workerpool.New(0)
	defer pool.Close()

	batch, inF, qDim, kvDim := 2, 8, 4, 4
	totalOut := qDim + 2*kvDim
	x := testInputs(batch * inF)
	wQKV := testInputs(totalOut * inF)

	// Without LoRA, should match plain QKVDenseAuto
	aQ := make([]float32, batch*qDim)
	aK := make([]float32, batch*kvDim)
	aV := make([]float32, batch*kvDim)
	FusedLoRAQKVDenseAuto[float32](pool, x, wQKV, nil, nil, nil,
		nil, nil, nil, aQ, aK, aV, nil, nil, nil,
		batch, inF, qDim, kvDim)

	sQ := make([]float32, batch*qDim)
	sK := make([]float32, batch*kvDim)
	sV := make([]float32, batch*kvDim)
	QKVDenseAuto(pool, x, wQKV, nil, nil, nil, sQ, sK, sV,
		batch, inF, qDim, kvDim)

	for i := range aQ {
		diff := stdmath.Abs(float64(aQ[i] - sQ[i]))
		if diff > 1e-5 {
			t.Errorf("Q[%d]: Auto=%v, Plain=%v", i, aQ[i], sQ[i])
		}
	}
}

func BenchmarkFusedLoRAQKV(b *testing.B) {
	pool := workerpool.New(0)
	defer pool.Close()

	batch, inF, qDim, kvDim, rank := 8, 768, 768, 256, 16
	totalOut := qDim + 2*kvDim
	x := make([]float32, batch*inF)
	wQKV := make([]float32, totalOut*inF)

	loraQ := &LoRAParams[float32]{
		A: make([]float32, rank*inF), B: make([]float32, qDim*rank),
		Scale: 0.5, Rank: rank,
	}
	loraK := &LoRAParams[float32]{
		A: make([]float32, rank*inF), B: make([]float32, kvDim*rank),
		Scale: 0.5, Rank: rank,
	}
	loraV := &LoRAParams[float32]{
		A: make([]float32, rank*inF), B: make([]float32, kvDim*rank),
		Scale: 0.5, Rank: rank,
	}

	q := make([]float32, batch*qDim)
	k := make([]float32, batch*kvDim)
	v := make([]float32, batch*kvDim)

	b.Run("Fused", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			FusedLoRAQKVDenseAuto(pool, x, wQKV, nil, nil, nil,
				loraQ, loraK, loraV, q, k, v, nil, nil, nil,
				batch, inF, qDim, kvDim)
		}
	})

	b.Run("Separate", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			// Base QKV
			QKVDenseAuto(pool, x, wQKV, nil, nil, nil, q, k, v,
				batch, inF, qDim, kvDim)
			// LoRA Q: h = x@A^T, temp = h@B^T, q += scale*temp
			hQ := make([]float32, batch*rank)
			matmul.MatMulKLastAuto(pool, x, loraQ.A, hQ, batch, rank, inF)
			tempQ := make([]float32, batch*qDim)
			matmul.MatMulKLastAuto(pool, hQ, loraQ.B, tempQ, batch, qDim, rank)
			vec.MulConstAddTo(q, loraQ.Scale, tempQ)
			// LoRA K
			hK := make([]float32, batch*rank)
			matmul.MatMulKLastAuto(pool, x, loraK.A, hK, batch, rank, inF)
			tempK := make([]float32, batch*kvDim)
			matmul.MatMulKLastAuto(pool, hK, loraK.B, tempK, batch, kvDim, rank)
			vec.MulConstAddTo(k, loraK.Scale, tempK)
			// LoRA V
			hV := make([]float32, batch*rank)
			matmul.MatMulKLastAuto(pool, x, loraV.A, hV, batch, rank, inF)
			tempV := make([]float32, batch*kvDim)
			matmul.MatMulKLastAuto(pool, hV, loraV.B, tempV, batch, kvDim, rank)
			vec.MulConstAddTo(v, loraV.Scale, tempV)
		}
	})
}
