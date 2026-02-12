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

// mustDim retrieves a dimension from golden data, failing the test if absent.
func mustDim(t *testing.T, gd *gradcheck.GoldenData, name string) int {
	t.Helper()
	v, err := gd.GetDim(name)
	if err != nil {
		t.Fatal(err)
	}
	return v
}

// mustArray retrieves an array from golden data, failing the test if absent.
func mustArray(t *testing.T, gd *gradcheck.GoldenData, name string) []float32 {
	t.Helper()
	arr, err := gd.GetArray(name)
	if err != nil {
		t.Fatal(err)
	}
	return arr
}

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

	batch := mustDim(t, gd, "batchSize")
	inF := mustDim(t, gd, "inFeatures")
	outF := mustDim(t, gd, "outFeatures")
	gradOutput := mustArray(t, gd, "gradOutput")
	x := mustArray(t, gd, "x")
	weight := mustArray(t, gd, "weight")
	wantGI := mustArray(t, gd, "gradInput")
	wantGW := mustArray(t, gd, "gradWeight")
	wantGB := mustArray(t, gd, "gradBias")

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

	n := mustDim(t, gd, "n")
	gradOutput := mustArray(t, gd, "gradOutput")
	savedInput := mustArray(t, gd, "savedInput")
	wantGI := mustArray(t, gd, "gradInput")

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

	rows := mustDim(t, gd, "rows")
	cols := mustDim(t, gd, "cols")
	gradOutput := mustArray(t, gd, "gradOutput")
	savedProbs := mustArray(t, gd, "savedProbs")
	wantGI := mustArray(t, gd, "gradInput")

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

	normSize := mustDim(t, gd, "normSize")
	gradOutput := mustArray(t, gd, "gradOutput")
	savedXHat := mustArray(t, gd, "savedXHat")
	savedInvStd := mustArray(t, gd, "savedInvStd")
	gamma := mustArray(t, gd, "gamma")
	wantGI := mustArray(t, gd, "gradInput")
	wantGG := mustArray(t, gd, "gradGamma")
	wantGBeta := mustArray(t, gd, "gradBeta")

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

	seqLen := mustDim(t, gd, "seqLen")
	kvLen := mustDim(t, gd, "kvLen")
	headDim := mustDim(t, gd, "headDim")

	Q := mustArray(t, gd, "Q")
	K := mustArray(t, gd, "K")
	V := mustArray(t, gd, "V")
	probs := mustArray(t, gd, "probs")
	scaleArr := mustArray(t, gd, "scale")
	gradOutput := mustArray(t, gd, "gradOutput")
	wantGQ := mustArray(t, gd, "gradQ")
	wantGK := mustArray(t, gd, "gradK")
	wantGV := mustArray(t, gd, "gradV")

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

	batch := mustDim(t, gd, "batchSize")
	dIn := mustDim(t, gd, "dIn")
	dOut := mustDim(t, gd, "dOut")
	rank := mustDim(t, gd, "rank")

	gradOutput := mustArray(t, gd, "gradOutput")
	x := mustArray(t, gd, "x")
	h := mustArray(t, gd, "h")
	W := mustArray(t, gd, "W")
	A := mustArray(t, gd, "A")
	B := mustArray(t, gd, "B")
	scaleArr := mustArray(t, gd, "scale")
	wantGX := mustArray(t, gd, "gradX")
	wantGA := mustArray(t, gd, "gradA")
	wantGB := mustArray(t, gd, "gradB")

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

	n := mustDim(t, gd, "n")
	numSteps := mustDim(t, gd, "step")

	paramInit := mustArray(t, gd, "paramInit")
	grad := mustArray(t, gd, "grad")
	lrArr := mustArray(t, gd, "lr")
	beta1Arr := mustArray(t, gd, "beta1")
	beta2Arr := mustArray(t, gd, "beta2")
	epsArr := mustArray(t, gd, "epsilon")
	wdArr := mustArray(t, gd, "weightDecay")
	wantParam := mustArray(t, gd, "paramFinal")
	wantM := mustArray(t, gd, "mFinal")
	wantV := mustArray(t, gd, "vFinal")

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
