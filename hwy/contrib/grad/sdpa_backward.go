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
	"github.com/ajroetker/go-highway/hwy/contrib/matmul"
	"github.com/ajroetker/go-highway/hwy/contrib/vec"
	"github.com/ajroetker/go-highway/hwy/contrib/workerpool"
)

// SDPASaved holds intermediates saved during the forward pass of scaled
// dot-product attention, needed by the backward pass.
type SDPASaved[T hwy.Floats] struct {
	Q, K, V, Probs         []T
	Scale                  T
	SeqLen, KVLen, HeadDim int
}

// SDPABackwardAuto computes the backward pass for single-head scaled
// dot-product attention.
//
// Forward: output = softmax(Q @ K^T * scale + mask) @ V
//
// Backward:
//
//	dV = P^T @ dO                     [kvLen, headDim]
//	dP = dO @ V^T                     [seqLen, kvLen]
//	dS = softmax_backward(dP, P)      [seqLen, kvLen]
//	dQ = scale * dS @ K               [seqLen, headDim]
//	dK = scale * dS^T @ Q             [kvLen, headDim]
//
// All gradient outputs are accumulated.
func SDPABackwardAuto[T hwy.Floats](
	pool *workerpool.Pool,
	gradOutput []T,
	saved *SDPASaved[T],
	gradQ, gradK, gradV []T,
) {
	seqLen := saved.SeqLen
	kvLen := saved.KVLen
	headDim := saved.HeadDim
	scale := saved.Scale

	if seqLen == 0 || kvLen == 0 || headDim == 0 {
		return
	}

	// dV += P^T @ dO
	// P is [seqLen, kvLen], dO is [seqLen, headDim]
	// P^T is [kvLen, seqLen], P^T @ dO => [kvLen, headDim]
	// => m=kvLen, n=headDim, k=seqLen
	if gradV != nil {
		pT := make([]T, kvLen*seqLen)
		matmul.TransposeAuto(pool, saved.Probs, seqLen, kvLen, pT)
		temp := make([]T, kvLen*headDim)
		matmul.MatMulAuto(pool, pT, gradOutput, temp, kvLen, headDim, seqLen)
		vec.Add(gradV, temp)
	}

	// dP = dO @ V^T
	// dO is [seqLen, headDim], V is [kvLen, headDim]
	// dO @ V^T => [seqLen, kvLen]
	// MatMulKLastAuto: A[M,K] @ B[N,K]^T => C[M,N]
	// A=dO[seqLen, headDim], B=V[kvLen, headDim] => C[seqLen, kvLen]
	dP := make([]T, seqLen*kvLen)
	matmul.MatMulKLastAuto(pool, gradOutput, saved.V, dP, seqLen, kvLen, headDim)

	// dS = softmax_backward(dP, P) for each row
	dS := make([]T, seqLen*kvLen)
	for i := range seqLen {
		off := i * kvLen
		SoftmaxBackward(dP[off:off+kvLen], saved.Probs[off:off+kvLen], dS[off:off+kvLen])
	}

	// dQ += scale * dS @ K
	// dS is [seqLen, kvLen], K is [kvLen, headDim]
	// => m=seqLen, n=headDim, k=kvLen
	if gradQ != nil {
		temp := make([]T, seqLen*headDim)
		matmul.MatMulAuto(pool, dS, saved.K, temp, seqLen, headDim, kvLen)
		vec.MulConstAddTo(gradQ, scale, temp)
	}

	// dK += scale * dS^T @ Q
	// dS^T is [kvLen, seqLen], Q is [seqLen, headDim]
	// => m=kvLen, n=headDim, k=seqLen
	if gradK != nil {
		dST := make([]T, kvLen*seqLen)
		matmul.TransposeAuto(pool, dS, seqLen, kvLen, dST)
		temp := make([]T, kvLen*headDim)
		matmul.MatMulAuto(pool, dST, saved.Q, temp, kvLen, headDim, seqLen)
		vec.MulConstAddTo(gradK, scale, temp)
	}
}

// MultiHeadSDPABackwardAuto computes the backward pass for multi-head scaled
// dot-product attention with optional grouped-query attention (GQA).
//
//   - gradOutput: [batchSize, numHeads, seqLen, headDim]
//   - savedPerHead: per-head saved state from forward pass
//   - gradQ: [batchSize, numHeads, seqLen, headDim] — accumulated
//   - gradK: [batchSize, numKVHeads, kvLen, headDim] — accumulated
//   - gradV: [batchSize, numKVHeads, kvLen, headDim] — accumulated
//
// When numKVHeads < numHeads (GQA), multiple query heads share the same
// KV head, so their gradients are accumulated into the shared gradK/gradV.
func MultiHeadSDPABackwardAuto[T hwy.Floats](
	pool *workerpool.Pool,
	gradOutput []T,
	savedPerHead []SDPASaved[T],
	gradQ, gradK, gradV []T,
	batchSize, numHeads, numKVHeads, seqLen, kvLen, headDim int,
) {
	if batchSize == 0 || numHeads == 0 || seqLen == 0 || kvLen == 0 || headDim == 0 {
		return
	}

	headsPerKVHead := numHeads / numKVHeads
	qHeadStride := seqLen * headDim
	kvHeadStride := kvLen * headDim
	totalHeads := batchSize * numHeads

	doHead := func(idx int) {
		b := idx / numHeads
		h := idx % numHeads
		kvHead := h / headsPerKVHead

		qOff := (b*numHeads + h) * qHeadStride
		kOff := (b*numKVHeads + kvHead) * kvHeadStride
		vOff := kOff
		oOff := qOff

		goSlice := gradOutput[oOff : oOff+qHeadStride]
		gqSlice := gradQ[qOff : qOff+qHeadStride]
		gkSlice := gradK[kOff : kOff+kvHeadStride]
		gvSlice := gradV[vOff : vOff+kvHeadStride]

		saved := &savedPerHead[idx]

		// For GQA: multiple heads accumulate into same gradK/gradV.
		// Since we use atomic accumulation semantics, this is safe
		// when running sequentially. For parallel execution with GQA,
		// we run per-head sequentially to avoid races on shared K/V grads.
		SDPABackwardAuto(nil, goSlice, saved, gqSlice, gkSlice, gvSlice)
	}

	// When GQA is active (multiple heads share K/V), we must run sequentially
	// to avoid race conditions on the shared gradK/gradV.
	if headsPerKVHead > 1 || pool == nil {
		for i := range totalHeads {
			doHead(i)
		}
	} else {
		pool.ParallelForAtomic(totalHeads, doHead)
	}
}

// SDPABackwardScalar is a scalar reference implementation for testing.
func SDPABackwardScalar[T hwy.Floats](
	gradOutput []T,
	saved *SDPASaved[T],
	gradQ, gradK, gradV []T,
) {
	seqLen := saved.SeqLen
	kvLen := saved.KVLen
	headDim := saved.HeadDim
	scale := saved.Scale

	// dV += P^T @ dO
	if gradV != nil {
		for j := range kvLen {
			for d := range headDim {
				var sum float64
				for i := range seqLen {
					sum += float64(saved.Probs[i*kvLen+j]) * float64(gradOutput[i*headDim+d])
				}
				gradV[j*headDim+d] += T(sum)
			}
		}
	}

	// dP = dO @ V^T
	dP := make([]T, seqLen*kvLen)
	for i := range seqLen {
		for j := range kvLen {
			var sum float64
			for d := range headDim {
				sum += float64(gradOutput[i*headDim+d]) * float64(saved.V[j*headDim+d])
			}
			dP[i*kvLen+j] = T(sum)
		}
	}

	// dS = softmax_backward(dP, P) per row
	dS := make([]T, seqLen*kvLen)
	for i := range seqLen {
		off := i * kvLen
		SoftmaxBackwardScalar(dP[off:off+kvLen], saved.Probs[off:off+kvLen], dS[off:off+kvLen])
	}

	// dQ += scale * dS @ K
	if gradQ != nil {
		for i := range seqLen {
			for d := range headDim {
				var sum float64
				for j := range kvLen {
					sum += float64(dS[i*kvLen+j]) * float64(saved.K[j*headDim+d])
				}
				gradQ[i*headDim+d] += T(float64(scale) * sum)
			}
		}
	}

	// dK += scale * dS^T @ Q
	if gradK != nil {
		for j := range kvLen {
			for d := range headDim {
				var sum float64
				for i := range seqLen {
					sum += float64(dS[i*kvLen+j]) * float64(saved.Q[i*headDim+d])
				}
				gradK[j*headDim+d] += T(float64(scale) * sum)
			}
		}
	}
}
