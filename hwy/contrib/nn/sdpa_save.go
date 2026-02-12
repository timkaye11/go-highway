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
	"github.com/ajroetker/go-highway/hwy/contrib/workerpool"
)

// SDPAAutoSave computes single-head scaled dot-product attention and saves
// the attention probabilities needed by the backward pass.
//
//   - q:          [seqLen, headDim] (queries)
//   - k:          [kvLen, headDim] (keys)
//   - v:          [kvLen, headDim] (values)
//   - mask:       [seqLen, kvLen] (additive mask, nil for no mask)
//   - output:     [seqLen, headDim] (result)
//   - savedProbs: [seqLen, kvLen] (attention probabilities, saved for backward)
//   - scale:      typically 1/sqrt(headDim)
//
// savedProbs receives the softmax(Q@K^T * scale + mask) attention weights.
// These are consumed by grad.SDPABackwardAuto.
func SDPAAutoSave[T hwy.Floats](
	q, k, v, mask, output, savedProbs []T,
	seqLen, kvLen, headDim int, scale T,
) {
	// Use savedProbs directly as the scores buffer — after softmax they
	// become the attention probabilities that the backward pass needs.
	SDPA(q, k, v, mask, savedProbs, output, seqLen, kvLen, headDim, scale)
}

// MultiHeadSDPAAutoSave computes multi-head scaled dot-product attention
// with GQA support and saves per-head attention probabilities.
//
//   - pool:           worker pool for parallelizing across batch×head
//   - q:              [batchSize, numHeads, seqLen, headDim]
//   - k:              [batchSize, numKVHeads, kvLen, headDim]
//   - v:              [batchSize, numKVHeads, kvLen, headDim]
//   - mask:           additive mask (nil for no mask)
//   - output:         [batchSize, numHeads, seqLen, headDim]
//   - savedProbs:     [batchSize, numHeads, seqLen, kvLen]
//   - maskBatchStride, maskHeadStride: stride control for mask broadcasting
//   - scale:          typically 1/sqrt(headDim)
func MultiHeadSDPAAutoSave[T hwy.Floats](
	pool *workerpool.Pool,
	q, k, v, mask, output, savedProbs []T,
	batchSize, numHeads, numKVHeads, seqLen, kvLen, headDim int,
	maskBatchStride, maskHeadStride int,
	scale T,
) {
	if batchSize == 0 || numHeads == 0 || seqLen == 0 || kvLen == 0 || headDim == 0 {
		return
	}

	headsPerKVHead := numHeads / numKVHeads
	qHeadStride := seqLen * headDim
	kvHeadStride := kvLen * headDim
	probsHeadStride := seqLen * kvLen
	maskSliceLen := seqLen * kvLen
	totalHeads := batchSize * numHeads

	doHead := func(idx int) {
		b := idx / numHeads
		h := idx % numHeads
		kvHead := h / headsPerKVHead

		qOff := (b*numHeads + h) * qHeadStride
		kOff := (b*numKVHeads + kvHead) * kvHeadStride
		vOff := kOff
		oOff := qOff
		pOff := (b*numHeads + h) * probsHeadStride

		qSlice := q[qOff : qOff+qHeadStride]
		kSlice := k[kOff : kOff+kvHeadStride]
		vSlice := v[vOff : vOff+kvHeadStride]
		oSlice := output[oOff : oOff+qHeadStride]
		pSlice := savedProbs[pOff : pOff+probsHeadStride]

		var maskSlice []T
		if mask != nil {
			maskOff := b*maskBatchStride + h*maskHeadStride
			maskSlice = mask[maskOff : maskOff+maskSliceLen]
		}

		SDPAAutoSave(qSlice, kSlice, vSlice, maskSlice, oSlice, pSlice,
			seqLen, kvLen, headDim, scale)
	}

	if pool != nil {
		pool.ParallelForAtomic(totalHeads, doHead)
	} else {
		for i := range totalHeads {
			doHead(i)
		}
	}
}
