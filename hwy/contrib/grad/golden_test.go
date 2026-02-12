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
	"testing"

	"github.com/ajroetker/go-highway/hwy/contrib/grad/gradcheck"
	"github.com/ajroetker/go-highway/hwy/contrib/workerpool"
)

func TestGolden_DenseBackward(t *testing.T) {
	gd, err := gradcheck.LoadGolden("dense_backward_f32")
	if err != nil {
		t.Fatal(err)
	}
	if gd == nil {
		t.Skip("golden data not found (run scripts/gen_golden_data.py)")
	}

	pool := workerpool.New(0)
	defer pool.Close()

	batch, _ := gd.GetDim("batchSize")
	inF, _ := gd.GetDim("inFeatures")
	outF, _ := gd.GetDim("outFeatures")
	gradOutput, _ := gd.GetArray("gradOutput")
	x, _ := gd.GetArray("x")
	weight, _ := gd.GetArray("weight")
	wantGI, _ := gd.GetArray("gradInput")
	wantGW, _ := gd.GetArray("gradWeight")
	wantGB, _ := gd.GetArray("gradBias")

	gotGI := make([]float32, batch*inF)
	gotGW := make([]float32, outF*inF)
	gotGB := make([]float32, outF)

	DenseBackwardAuto(pool, gradOutput, x, weight, gotGI, gotGW, gotGB, batch, inF, outF)

	tol := gradcheck.DenseTolerance.Float32RelTol
	checkGoldenArray(t, "gradInput", gotGI, wantGI, tol)
	checkGoldenArray(t, "gradWeight", gotGW, wantGW, tol)
	checkGoldenArray(t, "gradBias", gotGB, wantGB, tol)
}

func TestGolden_GELUBackward(t *testing.T) {
	gd, err := gradcheck.LoadGolden("gelu_backward_f32")
	if err != nil {
		t.Fatal(err)
	}
	if gd == nil {
		t.Skip("golden data not found (run scripts/gen_golden_data.py)")
	}

	pool := workerpool.New(0)
	defer pool.Close()

	n, _ := gd.GetDim("n")
	gradOutput, _ := gd.GetArray("gradOutput")
	savedInput, _ := gd.GetArray("savedInput")
	wantGI, _ := gd.GetArray("gradInput")

	gotGI := make([]float32, n)
	GELUBackwardAuto(pool, gradOutput, savedInput, gotGI, 1, n)

	checkGoldenArray(t, "gradInput", gotGI, wantGI, gradcheck.GELUTolerance.Float32RelTol)
}

func TestGolden_SoftmaxBackward(t *testing.T) {
	gd, err := gradcheck.LoadGolden("softmax_backward_f32")
	if err != nil {
		t.Fatal(err)
	}
	if gd == nil {
		t.Skip("golden data not found (run scripts/gen_golden_data.py)")
	}

	pool := workerpool.New(0)
	defer pool.Close()

	rows, _ := gd.GetDim("rows")
	cols, _ := gd.GetDim("cols")
	gradOutput, _ := gd.GetArray("gradOutput")
	savedProbs, _ := gd.GetArray("savedProbs")
	wantGI, _ := gd.GetArray("gradInput")

	gotGI := make([]float32, rows*cols)
	SoftmaxBackwardAuto(pool, gradOutput, savedProbs, gotGI, rows, cols)

	checkGoldenArray(t, "gradInput", gotGI, wantGI, gradcheck.SoftmaxTolerance.Float32RelTol)
}

func TestGolden_LayerNormBackward(t *testing.T) {
	gd, err := gradcheck.LoadGolden("layernorm_backward_f32")
	if err != nil {
		t.Fatal(err)
	}
	if gd == nil {
		t.Skip("golden data not found (run scripts/gen_golden_data.py)")
	}

	pool := workerpool.New(0)
	defer pool.Close()

	normSize, _ := gd.GetDim("normSize")
	gradOutput, _ := gd.GetArray("gradOutput")
	savedXHat, _ := gd.GetArray("savedXHat")
	savedInvStd, _ := gd.GetArray("savedInvStd")
	gamma, _ := gd.GetArray("gamma")
	wantGI, _ := gd.GetArray("gradInput")
	wantGG, _ := gd.GetArray("gradGamma")
	wantGBeta, _ := gd.GetArray("gradBeta")

	n := len(gradOutput)
	gotGI := make([]float32, n)
	gotGG := make([]float32, normSize)
	gotGBeta := make([]float32, normSize)

	LayerNormBackwardAuto(pool, gradOutput, savedXHat, savedInvStd, gamma, normSize,
		gotGI, gotGG, gotGBeta)

	tol := gradcheck.LayerNormTolerance.Float32RelTol
	checkGoldenArray(t, "gradInput", gotGI, wantGI, tol)
	checkGoldenArray(t, "gradGamma", gotGG, wantGG, tol)
	checkGoldenArray(t, "gradBeta", gotGBeta, wantGBeta, tol)
}

func TestGolden_SDPABackward(t *testing.T) {
	gd, err := gradcheck.LoadGolden("sdpa_backward_f32")
	if err != nil {
		t.Fatal(err)
	}
	if gd == nil {
		t.Skip("golden data not found (run scripts/gen_golden_data.py)")
	}

	pool := workerpool.New(0)
	defer pool.Close()

	seqLen, _ := gd.GetDim("seqLen")
	kvLen, _ := gd.GetDim("kvLen")
	headDim, _ := gd.GetDim("headDim")

	Q, _ := gd.GetArray("Q")
	K, _ := gd.GetArray("K")
	V, _ := gd.GetArray("V")
	probs, _ := gd.GetArray("probs")
	scaleArr, _ := gd.GetArray("scale")
	gradOutput, _ := gd.GetArray("gradOutput")
	wantGQ, _ := gd.GetArray("gradQ")
	wantGK, _ := gd.GetArray("gradK")
	wantGV, _ := gd.GetArray("gradV")

	saved := &SDPASaved[float32]{
		Q: Q, K: K, V: V, Probs: probs,
		Scale:   scaleArr[0],
		SeqLen:  seqLen,
		KVLen:   kvLen,
		HeadDim: headDim,
	}

	gotGQ := make([]float32, seqLen*headDim)
	gotGK := make([]float32, kvLen*headDim)
	gotGV := make([]float32, kvLen*headDim)

	SDPABackwardAuto(pool, gradOutput, saved, gotGQ, gotGK, gotGV)

	tol := gradcheck.SDPATolerance.Float32RelTol
	checkGoldenArray(t, "gradQ", gotGQ, wantGQ, tol)
	checkGoldenArray(t, "gradK", gotGK, wantGK, tol)
	checkGoldenArray(t, "gradV", gotGV, wantGV, tol)
}

func TestGolden_LoRABackward(t *testing.T) {
	gd, err := gradcheck.LoadGolden("lora_backward_f32")
	if err != nil {
		t.Fatal(err)
	}
	if gd == nil {
		t.Skip("golden data not found (run scripts/gen_golden_data.py)")
	}

	pool := workerpool.New(0)
	defer pool.Close()

	batch, _ := gd.GetDim("batchSize")
	dIn, _ := gd.GetDim("dIn")
	dOut, _ := gd.GetDim("dOut")
	rank, _ := gd.GetDim("rank")

	gradOutput, _ := gd.GetArray("gradOutput")
	x, _ := gd.GetArray("x")
	h, _ := gd.GetArray("h")
	W, _ := gd.GetArray("W")
	A, _ := gd.GetArray("A")
	B, _ := gd.GetArray("B")
	scaleArr, _ := gd.GetArray("scale")
	wantGX, _ := gd.GetArray("gradX")
	wantGA, _ := gd.GetArray("gradA")
	wantGB, _ := gd.GetArray("gradB")

	scale := scaleArr[0]

	gotGX := make([]float32, batch*dIn)
	gotGA := make([]float32, rank*dIn)
	gotGB := make([]float32, dOut*rank)

	LoRABackwardAuto(pool, gradOutput, x, h, W, A, B, scale, gotGX, gotGA, gotGB,
		batch, dIn, dOut, rank)

	tol := gradcheck.LoRATolerance.Float32RelTol
	checkGoldenArray(t, "gradX", gotGX, wantGX, tol)
	checkGoldenArray(t, "gradA", gotGA, wantGA, tol)
	checkGoldenArray(t, "gradB", gotGB, wantGB, tol)
}

func TestGolden_AdamWStep(t *testing.T) {
	gd, err := gradcheck.LoadGolden("adamw_step_f32")
	if err != nil {
		t.Fatal(err)
	}
	if gd == nil {
		t.Skip("golden data not found (run scripts/gen_golden_data.py)")
	}

	n, _ := gd.GetDim("n")
	numSteps, _ := gd.GetDim("step")

	paramInit, _ := gd.GetArray("paramInit")
	grad, _ := gd.GetArray("grad")
	lrArr, _ := gd.GetArray("lr")
	beta1Arr, _ := gd.GetArray("beta1")
	beta2Arr, _ := gd.GetArray("beta2")
	epsArr, _ := gd.GetArray("epsilon")
	wdArr, _ := gd.GetArray("weightDecay")
	wantParam, _ := gd.GetArray("paramFinal")
	wantM, _ := gd.GetArray("mFinal")
	wantV, _ := gd.GetArray("vFinal")

	param := make([]float32, n)
	copy(param, paramInit)
	m := make([]float32, n)
	v := make([]float32, n)

	lr, beta1, beta2, eps, wd := lrArr[0], beta1Arr[0], beta2Arr[0], epsArr[0], wdArr[0]

	for s := 1; s <= numSteps; s++ {
		gradCopy := make([]float32, n)
		copy(gradCopy, grad)
		AdamWStep(param, gradCopy, m, v, lr, beta1, beta2, eps, wd, s)
	}

	tol := gradcheck.AdamWTolerance.Float32RelTol
	checkGoldenArray(t, "param", param, wantParam, tol)
	checkGoldenArray(t, "m", m, wantM, tol)
	checkGoldenArray(t, "v", v, wantV, tol)
}

// checkGoldenArray compares got vs want using gradcheck.CompareArrays and reports errors.
func checkGoldenArray(t *testing.T, name string, got, want []float32, relTol float64) {
	t.Helper()
	mismatches := gradcheck.CompareArrays(name, got, want, relTol)
	for _, m := range mismatches {
		if m.Index < 0 {
			t.Errorf("%s: length mismatch: got %d, want %d", name, len(got), len(want))
		} else {
			t.Errorf("%s[%d]: got=%v, want=%v, rel_err=%.2e",
				name, m.Index, m.Analytical, m.Numerical, m.RelError)
		}
	}
}
