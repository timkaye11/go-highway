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
	stdmath "math"
	"testing"

	"github.com/ajroetker/go-highway/hwy/contrib/activation"
	"github.com/ajroetker/go-highway/hwy/contrib/nn"
	"github.com/ajroetker/go-highway/hwy/contrib/vec"
	"github.com/ajroetker/go-highway/hwy/contrib/workerpool"
)

// TestIntegration_TransformerBlock tests a full transformer encoder block:
//
//	residual1 = x
//	x = LayerNorm(x)
//	q, k, v = Dense_qkv(x)
//	attn = SDPA(q, k, v)
//	x = Dense_out(attn)
//	x = x + residual1
//	residual2 = x
//	x = LayerNorm(x)
//	x = Dense_ffn(x)
//	x = GELU(x)
//	x = Dense_proj(x)
//	x = x + residual2
//	loss = sum(x)
//
// Then backward through the full chain, verifying no NaN/Inf and that
// gradient magnitudes are reasonable.
func TestIntegration_TransformerBlock(t *testing.T) {
	pool := workerpool.New(0)
	defer pool.Close()

	// Small dimensions for testing
	batchSize := 2
	seqLen := 4
	dModel := 16
	numHeads := 2
	headDim := dModel / numHeads
	dFF := 32
	eps := float32(1e-5)

	tokens := batchSize * seqLen
	inputSize := tokens * dModel

	// Initialize random-ish weights and input
	x := make([]float32, inputSize)
	for i := range x {
		x[i] = float32(i)*0.01 - float32(inputSize)*0.005
	}

	// LayerNorm 1 params
	gamma1 := make([]float32, dModel)
	beta1 := make([]float32, dModel)
	for i := range dModel {
		gamma1[i] = 1.0
		beta1[i] = 0.0
	}

	// QKV projection: [dModel, 3*dModel]
	wQKV := make([]float32, 3*dModel*dModel)
	for i := range wQKV {
		wQKV[i] = float32(i)*0.002 - float32(len(wQKV))*0.001
	}

	// Output projection: [dModel, dModel]
	wOut := make([]float32, dModel*dModel)
	for i := range wOut {
		wOut[i] = float32(i)*0.003 - float32(len(wOut))*0.0015
	}

	// LayerNorm 2 params
	gamma2 := make([]float32, dModel)
	beta2 := make([]float32, dModel)
	for i := range dModel {
		gamma2[i] = 1.0
		beta2[i] = 0.0
	}

	// FFN: dModel -> dFF
	wFFN := make([]float32, dFF*dModel)
	for i := range wFFN {
		wFFN[i] = float32(i)*0.001 - float32(len(wFFN))*0.0005
	}

	// Proj: dFF -> dModel
	wProj := make([]float32, dModel*dFF)
	for i := range wProj {
		wProj[i] = float32(i)*0.001 - float32(len(wProj))*0.0005
	}

	// ===== FORWARD PASS =====

	// Save residual 1
	residual1 := make([]float32, inputSize)
	copy(residual1, x)

	// LayerNorm 1
	ln1Out := make([]float32, inputSize)
	ln1XHat := make([]float32, inputSize)
	ln1InvStd := make([]float32, tokens)
	nn.LayerNormForwardSave(pool, x, ln1Out, dModel, gamma1, beta1, eps, ln1XHat, ln1InvStd)

	// QKV projection: [tokens, dModel] @ [3*dModel, dModel]^T -> [tokens, 3*dModel]
	qkvOut := make([]float32, tokens*3*dModel)
	nn.DenseAuto(pool, ln1Out, wQKV, nil, qkvOut, tokens, dModel, 3*dModel)

	// Split Q, K, V
	q := make([]float32, tokens*dModel)
	k := make([]float32, tokens*dModel)
	v := make([]float32, tokens*dModel)
	for i := range tokens {
		copy(q[i*dModel:], qkvOut[i*3*dModel:i*3*dModel+dModel])
		copy(k[i*dModel:], qkvOut[i*3*dModel+dModel:i*3*dModel+2*dModel])
		copy(v[i*dModel:], qkvOut[i*3*dModel+2*dModel:i*3*dModel+3*dModel])
	}

	// SDPA per head
	scale := float32(1.0 / stdmath.Sqrt(float64(headDim)))
	attnOut := make([]float32, tokens*dModel)
	savedPerHead := make([]SDPASaved[float32], batchSize*numHeads)

	for b := range batchSize {
		for h := range numHeads {
			idx := b*numHeads + h
			qOff := (b*seqLen)*dModel + h*headDim
			kOff := qOff
			vOff := qOff
			oOff := qOff

			// Extract per-head slices (interleaved layout)
			qh := make([]float32, seqLen*headDim)
			kh := make([]float32, seqLen*headDim)
			vh := make([]float32, seqLen*headDim)
			for s := range seqLen {
				srcOff := (b*seqLen+s)*dModel + h*headDim
				copy(qh[s*headDim:], q[srcOff:srcOff+headDim])
				copy(kh[s*headDim:], k[srcOff:srcOff+headDim])
				copy(vh[s*headDim:], v[srcOff:srcOff+headDim])
			}

			oh := make([]float32, seqLen*headDim)
			probs := make([]float32, seqLen*seqLen)
			nn.SDPAAutoSave(qh, kh, vh, nil, oh, probs, seqLen, seqLen, headDim, scale)

			savedPerHead[idx] = SDPASaved[float32]{
				Q: qh, K: kh, V: vh, Probs: probs,
				Scale: scale, SeqLen: seqLen, KVLen: seqLen, HeadDim: headDim,
			}

			// Scatter back to interleaved output
			for s := range seqLen {
				dstOff := (b*seqLen+s)*dModel + h*headDim
				copy(attnOut[dstOff:dstOff+headDim], oh[s*headDim:(s+1)*headDim])
			}
			_ = qOff
			_ = kOff
			_ = vOff
			_ = oOff
		}
	}

	// Output projection
	outProjOut := make([]float32, inputSize)
	nn.DenseAuto(pool, attnOut, wOut, nil, outProjOut, tokens, dModel, dModel)

	// Residual add 1
	afterRes1 := make([]float32, inputSize)
	copy(afterRes1, residual1)
	vec.Add(afterRes1, outProjOut)

	// Save residual 2
	residual2 := make([]float32, inputSize)
	copy(residual2, afterRes1)

	// LayerNorm 2
	ln2Out := make([]float32, inputSize)
	ln2XHat := make([]float32, inputSize)
	ln2InvStd := make([]float32, tokens)
	nn.LayerNormForwardSave(pool, afterRes1, ln2Out, dModel, gamma2, beta2, eps, ln2XHat, ln2InvStd)

	// FFN up: [tokens, dModel] -> [tokens, dFF]
	ffnOut := make([]float32, tokens*dFF)
	nn.DenseAuto(pool, ln2Out, wFFN, nil, ffnOut, tokens, dModel, dFF)

	// GELU
	geluOut := make([]float32, tokens*dFF)
	activation.ParallelGELU(pool, ffnOut, geluOut, tokens, dFF)
	savedFFNInput := make([]float32, tokens*dFF)
	copy(savedFFNInput, ffnOut) // save pre-GELU for backward

	// FFN down: [tokens, dFF] -> [tokens, dModel]
	projOut := make([]float32, inputSize)
	nn.DenseAuto(pool, geluOut, wProj, nil, projOut, tokens, dFF, dModel)

	// Residual add 2
	finalOut := make([]float32, inputSize)
	copy(finalOut, residual2)
	vec.Add(finalOut, projOut)

	// Loss = sum(output)
	var loss float64
	for _, v := range finalOut {
		loss += float64(v)
	}
	t.Logf("Forward loss: %v", loss)

	// ===== BACKWARD PASS =====
	// gradOutput = d(sum)/d(output) = all ones
	gradOut := make([]float32, inputSize)
	for i := range gradOut {
		gradOut[i] = 1.0
	}

	// Backward through residual 2: grad flows to both branches
	gradResidual2 := make([]float32, inputSize)
	copy(gradResidual2, gradOut)
	gradProjOut := make([]float32, inputSize)
	copy(gradProjOut, gradOut)

	// Backward through FFN down projection
	gradGeluOut := make([]float32, tokens*dFF)
	gradWProj := make([]float32, dModel*dFF)
	DenseBackwardAuto(pool, gradProjOut, geluOut, wProj, gradGeluOut, gradWProj, nil, tokens, dFF, dModel)

	// Backward through GELU
	gradFFNOut := make([]float32, tokens*dFF)
	GELUBackwardAuto(pool, gradGeluOut, savedFFNInput, gradFFNOut, tokens, dFF)

	// Backward through FFN up
	gradLN2Out := make([]float32, inputSize)
	gradWFFN := make([]float32, dFF*dModel)
	DenseBackwardAuto(pool, gradFFNOut, ln2Out, wFFN, gradLN2Out, gradWFFN, nil, tokens, dModel, dFF)

	// Backward through LayerNorm 2
	gradAfterRes1 := make([]float32, inputSize)
	gradGamma2 := make([]float32, dModel)
	gradBeta2 := make([]float32, dModel)
	LayerNormBackwardAuto(pool, gradLN2Out, ln2XHat, ln2InvStd, gamma2, dModel,
		gradAfterRes1, gradGamma2, gradBeta2)

	// Add gradient from residual 2
	vec.Add(gradAfterRes1, gradResidual2)

	// Backward through residual 1: grad flows to both branches
	gradResidual1 := make([]float32, inputSize)
	copy(gradResidual1, gradAfterRes1)
	gradOutProjOut := make([]float32, inputSize)
	copy(gradOutProjOut, gradAfterRes1)

	// Backward through output projection
	gradAttnOut := make([]float32, inputSize)
	gradWOut := make([]float32, dModel*dModel)
	DenseBackwardAuto(pool, gradOutProjOut, attnOut, wOut, gradAttnOut, gradWOut, nil, tokens, dModel, dModel)

	// Backward through SDPA (per head)
	gradQ := make([]float32, tokens*dModel)
	gradK := make([]float32, tokens*dModel)
	gradV := make([]float32, tokens*dModel)

	for b := range batchSize {
		for h := range numHeads {
			idx := b*numHeads + h

			// Gather per-head gradients from interleaved layout
			goh := make([]float32, seqLen*headDim)
			for s := range seqLen {
				srcOff := (b*seqLen+s)*dModel + h*headDim
				copy(goh[s*headDim:], gradAttnOut[srcOff:srcOff+headDim])
			}

			gqh := make([]float32, seqLen*headDim)
			gkh := make([]float32, seqLen*headDim)
			gvh := make([]float32, seqLen*headDim)
			SDPABackwardAuto(pool, goh, &savedPerHead[idx], gqh, gkh, gvh)

			// Scatter back
			for s := range seqLen {
				dstOff := (b*seqLen+s)*dModel + h*headDim
				for d := range headDim {
					gradQ[dstOff+d] += gqh[s*headDim+d]
					gradK[dstOff+d] += gkh[s*headDim+d]
					gradV[dstOff+d] += gvh[s*headDim+d]
				}
			}
		}
	}

	// Backward through QKV split: merge gradQ, gradK, gradV into gradQKV
	gradQKV := make([]float32, tokens*3*dModel)
	for i := range tokens {
		for j := range dModel {
			gradQKV[i*3*dModel+j] = gradQ[i*dModel+j]
			gradQKV[i*3*dModel+dModel+j] = gradK[i*dModel+j]
			gradQKV[i*3*dModel+2*dModel+j] = gradV[i*dModel+j]
		}
	}

	// Backward through QKV projection
	gradLN1Out := make([]float32, inputSize)
	gradWQKV := make([]float32, 3*dModel*dModel)
	DenseBackwardAuto(pool, gradQKV, ln1Out, wQKV, gradLN1Out, gradWQKV, nil, tokens, dModel, 3*dModel)

	// Backward through LayerNorm 1
	gradX := make([]float32, inputSize)
	gradGamma1 := make([]float32, dModel)
	gradBeta1 := make([]float32, dModel)
	LayerNormBackwardAuto(pool, gradLN1Out, ln1XHat, ln1InvStd, gamma1, dModel,
		gradX, gradGamma1, gradBeta1)

	// Add gradient from residual 1
	vec.Add(gradX, gradResidual1)

	// ===== VERIFICATION =====

	// Check no NaN/Inf in any gradient
	checkNoNaNInf(t, "gradX", gradX)
	checkNoNaNInf(t, "gradWQKV", gradWQKV)
	checkNoNaNInf(t, "gradWOut", gradWOut)
	checkNoNaNInf(t, "gradWFFN", gradWFFN)
	checkNoNaNInf(t, "gradWProj", gradWProj)
	checkNoNaNInf(t, "gradGamma1", gradGamma1)
	checkNoNaNInf(t, "gradBeta1", gradBeta1)
	checkNoNaNInf(t, "gradGamma2", gradGamma2)
	checkNoNaNInf(t, "gradBeta2", gradBeta2)

	// Check gradients are non-zero
	checkNonZero(t, "gradX", gradX)
	checkNonZero(t, "gradWQKV", gradWQKV)
	checkNonZero(t, "gradWOut", gradWOut)
	checkNonZero(t, "gradWFFN", gradWFFN)
	checkNonZero(t, "gradWProj", gradWProj)

	// Log gradient norms for debugging
	t.Logf("||gradX||     = %v", norm(gradX))
	t.Logf("||gradWQKV||  = %v", norm(gradWQKV))
	t.Logf("||gradWOut||  = %v", norm(gradWOut))
	t.Logf("||gradWFFN||  = %v", norm(gradWFFN))
	t.Logf("||gradWProj|| = %v", norm(gradWProj))
}

func checkNoNaNInf(t *testing.T, name string, data []float32) {
	t.Helper()
	for i, v := range data {
		if stdmath.IsNaN(float64(v)) || stdmath.IsInf(float64(v), 0) {
			t.Errorf("%s[%d] = %v (NaN/Inf)", name, i, v)
			return
		}
	}
}

func checkNonZero(t *testing.T, name string, data []float32) {
	t.Helper()
	for _, v := range data {
		if v != 0 {
			return
		}
	}
	t.Errorf("%s is all zeros", name)
}

func norm(data []float32) float64 {
	var sum float64
	for _, v := range data {
		sum += float64(v) * float64(v)
	}
	return stdmath.Sqrt(sum)
}
