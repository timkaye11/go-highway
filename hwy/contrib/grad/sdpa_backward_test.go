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
	stdmath "math"
	"testing"

	"github.com/ajroetker/go-highway/hwy/contrib/workerpool"
)

// sdpaForwardScalar computes a simple scalar SDPA forward and returns saved state.
func sdpaForwardScalar(Q, K, V []float32, seqLen, kvLen, headDim int, scale float32) (output []float32, saved *SDPASaved[float32]) {
	// scores = Q @ K^T * scale
	scores := make([]float32, seqLen*kvLen)
	for i := range seqLen {
		for j := range kvLen {
			var sum float64
			for d := range headDim {
				sum += float64(Q[i*headDim+d]) * float64(K[j*headDim+d])
			}
			scores[i*kvLen+j] = float32(sum * float64(scale))
		}
	}

	// softmax per row
	probs := make([]float32, seqLen*kvLen)
	for i := range seqLen {
		off := i * kvLen
		row := scores[off : off+kvLen]

		maxVal := row[0]
		for _, v := range row[1:] {
			if v > maxVal {
				maxVal = v
			}
		}
		var expSum float64
		for j := range kvLen {
			probs[off+j] = float32(stdmath.Exp(float64(row[j] - maxVal)))
			expSum += float64(probs[off+j])
		}
		inv := float32(1.0 / expSum)
		for j := range kvLen {
			probs[off+j] *= inv
		}
	}

	// output = probs @ V
	output = make([]float32, seqLen*headDim)
	for i := range seqLen {
		for d := range headDim {
			var sum float64
			for j := range kvLen {
				sum += float64(probs[i*kvLen+j]) * float64(V[j*headDim+d])
			}
			output[i*headDim+d] = float32(sum)
		}
	}

	saved = &SDPASaved[float32]{
		Q:       Q,
		K:       K,
		V:       V,
		Probs:   probs,
		Scale:   scale,
		SeqLen:  seqLen,
		KVLen:   kvLen,
		HeadDim: headDim,
	}
	return
}

func TestSDPABackward_AutoVsScalar(t *testing.T) {
	pool := workerpool.New(0)
	defer pool.Close()

	tests := []struct {
		seqLen, kvLen, headDim int
	}{
		{4, 4, 8},
		{8, 8, 16},
		{4, 6, 8},
		{3, 5, 7},
		{16, 16, 32},
	}

	for _, tt := range tests {
		name := fmt.Sprintf("s%d_k%d_d%d", tt.seqLen, tt.kvLen, tt.headDim)
		t.Run(name, func(t *testing.T) {
			Q := testInputs(tt.seqLen * tt.headDim)
			K := testInputs(tt.kvLen * tt.headDim)
			V := make([]float32, tt.kvLen*tt.headDim)
			for i := range V {
				V[i] = float32(i)*0.02 - 0.5
			}
			scale := float32(1.0 / stdmath.Sqrt(float64(tt.headDim)))

			_, saved := sdpaForwardScalar(Q, K, V, tt.seqLen, tt.kvLen, tt.headDim, scale)

			gradOutput := testInputs(tt.seqLen * tt.headDim)

			// Auto
			aGQ := make([]float32, tt.seqLen*tt.headDim)
			aGK := make([]float32, tt.kvLen*tt.headDim)
			aGV := make([]float32, tt.kvLen*tt.headDim)
			SDPABackwardAuto(pool, gradOutput, saved, aGQ, aGK, aGV)

			// Scalar
			sGQ := make([]float32, tt.seqLen*tt.headDim)
			sGK := make([]float32, tt.kvLen*tt.headDim)
			sGV := make([]float32, tt.kvLen*tt.headDim)
			SDPABackwardScalar(gradOutput, saved, sGQ, sGK, sGV)

			allClose32(t, "gradQ", aGQ, sGQ, 1e-2)
			allClose32(t, "gradK", aGK, sGK, 1e-2)
			allClose32(t, "gradV", aGV, sGV, 1e-2)
		})
	}
}

func TestMultiHeadSDPABackward(t *testing.T) {
	pool := workerpool.New(0)
	defer pool.Close()

	batchSize := 2
	numHeads := 4
	numKVHeads := 2 // GQA: 2 query heads per KV head
	seqLen := 4
	kvLen := 4
	headDim := 8
	scale := float32(1.0 / stdmath.Sqrt(float64(headDim)))

	totalQHeads := batchSize * numHeads
	qSize := totalQHeads * seqLen * headDim
	kvSize := batchSize * numKVHeads * kvLen * headDim

	// Build per-head saved state
	savedPerHead := make([]SDPASaved[float32], totalQHeads)
	for idx := range totalQHeads {
		Q := testInputs(seqLen * headDim)
		K := testInputs(kvLen * headDim)
		V := make([]float32, kvLen*headDim)
		for i := range V {
			V[i] = float32(i)*0.02 - 0.3
		}

		// Compute probs
		probs := make([]float32, seqLen*kvLen)
		for i := range seqLen {
			for j := range kvLen {
				var sum float64
				for d := range headDim {
					sum += float64(Q[i*headDim+d]) * float64(K[j*headDim+d])
				}
				probs[i*kvLen+j] = float32(sum * float64(scale))
			}
		}
		// softmax
		for i := range seqLen {
			off := i * kvLen
			maxV := probs[off]
			for j := 1; j < kvLen; j++ {
				if probs[off+j] > maxV {
					maxV = probs[off+j]
				}
			}
			var expSum float64
			for j := range kvLen {
				probs[off+j] = float32(stdmath.Exp(float64(probs[off+j] - maxV)))
				expSum += float64(probs[off+j])
			}
			inv := float32(1.0 / expSum)
			for j := range kvLen {
				probs[off+j] *= inv
			}
		}

		savedPerHead[idx] = SDPASaved[float32]{
			Q: Q, K: K, V: V, Probs: probs,
			Scale: scale, SeqLen: seqLen, KVLen: kvLen, HeadDim: headDim,
		}
	}

	gradOutput := testInputs(qSize)

	gradQ := make([]float32, qSize)
	gradK := make([]float32, kvSize)
	gradV := make([]float32, kvSize)

	MultiHeadSDPABackwardAuto(pool, gradOutput, savedPerHead, gradQ, gradK, gradV,
		batchSize, numHeads, numKVHeads, seqLen, kvLen, headDim)

	// Verify non-zero gradients
	nonZeroQ, nonZeroK, nonZeroV := false, false, false
	for _, v := range gradQ {
		if v != 0 {
			nonZeroQ = true
			break
		}
	}
	for _, v := range gradK {
		if v != 0 {
			nonZeroK = true
			break
		}
	}
	for _, v := range gradV {
		if v != 0 {
			nonZeroV = true
			break
		}
	}
	if !nonZeroQ {
		t.Error("gradQ should be non-zero")
	}
	if !nonZeroK {
		t.Error("gradK should be non-zero")
	}
	if !nonZeroV {
		t.Error("gradV should be non-zero")
	}

	// Verify no NaN/Inf
	for i, v := range gradQ {
		if stdmath.IsNaN(float64(v)) || stdmath.IsInf(float64(v), 0) {
			t.Errorf("gradQ[%d] = %v", i, v)
		}
	}
}
