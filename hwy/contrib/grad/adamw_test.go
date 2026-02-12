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

func TestAdamWStep_AutoVsScalar(t *testing.T) {
	tests := []struct {
		n    int
		step int
	}{
		{8, 1},
		{16, 1},
		{32, 5},
		{7, 3},
		{64, 10},
	}

	lr := float32(0.001)
	beta1 := float32(0.9)
	beta2 := float32(0.999)
	eps := float32(1e-8)
	wd := float32(0.01)

	for _, tt := range tests {
		name := fmt.Sprintf("n%d_step%d", tt.n, tt.step)
		t.Run(name, func(t *testing.T) {
			// Initialize identically for both
			aParam := make([]float32, tt.n)
			aGrad := make([]float32, tt.n)
			aM := make([]float32, tt.n)
			aV := make([]float32, tt.n)

			sParam := make([]float32, tt.n)
			sGrad := make([]float32, tt.n)
			sM := make([]float32, tt.n)
			sV := make([]float32, tt.n)

			for i := range tt.n {
				val := float32(i)*0.01 - 0.5
				aParam[i] = val
				sParam[i] = val
				g := float32(i) * 0.1
				aGrad[i] = g
				sGrad[i] = g
			}

			AdamWStep(aParam, aGrad, aM, aV, lr, beta1, beta2, eps, wd, tt.step)
			AdamWStepScalar(sParam, sGrad, sM, sV, lr, beta1, beta2, eps, wd, tt.step)

			allClose32(t, "param", aParam, sParam, 1e-3)
			allClose32(t, "m", aM, sM, 1e-3)
			allClose32(t, "v", aV, sV, 1e-3)
		})
	}
}

func TestAdamWStep_MultiStep(t *testing.T) {
	pool := workerpool.New(0)
	defer pool.Close()

	n := 32
	lr := float32(0.001)
	beta1 := float32(0.9)
	beta2 := float32(0.999)
	eps := float32(1e-8)
	wd := float32(0.01)

	aParam := make([]float32, n)
	aM := make([]float32, n)
	aV := make([]float32, n)
	sParam := make([]float32, n)
	sM := make([]float32, n)
	sV := make([]float32, n)

	for i := range n {
		val := float32(i) * 0.1
		aParam[i] = val
		sParam[i] = val
	}

	for step := 1; step <= 10; step++ {
		grad := make([]float32, n)
		for i := range n {
			grad[i] = float32(i) * 0.01 * float32(step)
		}
		gradCopy := make([]float32, n)
		copy(gradCopy, grad)

		AdamWStepAuto(pool, aParam, grad, aM, aV, lr, beta1, beta2, eps, wd, step)
		AdamWStepScalar(sParam, gradCopy, sM, sV, lr, beta1, beta2, eps, wd, step)
	}

	allClose32(t, "param_10steps", aParam, sParam, 1e-2)
	allClose32(t, "m_10steps", aM, sM, 1e-2)
	allClose32(t, "v_10steps", aV, sV, 1e-2)
}

func TestAdamWStep_ParamsDecrease(t *testing.T) {
	// With weight decay and positive gradients, params should generally decrease
	n := 16
	param := make([]float32, n)
	grad := make([]float32, n)
	m := make([]float32, n)
	v := make([]float32, n)

	for i := range n {
		param[i] = 1.0
		grad[i] = 0.1
	}

	origSum := float64(0)
	for _, p := range param {
		origSum += float64(p)
	}

	AdamWStep(param, grad, m, v, 0.01, 0.9, 0.999, 1e-8, 0.01, 1)

	newSum := float64(0)
	for _, p := range param {
		newSum += float64(p)
	}

	if newSum >= origSum {
		t.Errorf("expected params to decrease: before=%v, after=%v", origSum, newSum)
	}
}

func TestAdamWStep_Float64(t *testing.T) {
	n := 16
	param := make([]float64, n)
	grad := make([]float64, n)
	m := make([]float64, n)
	v := make([]float64, n)
	paramS := make([]float64, n)
	gradS := make([]float64, n)
	mS := make([]float64, n)
	vS := make([]float64, n)

	for i := range n {
		param[i] = float64(i) * 0.1
		paramS[i] = param[i]
		grad[i] = float64(i) * 0.01
		gradS[i] = grad[i]
	}

	AdamWStep(param, grad, m, v, 0.001, 0.9, 0.999, 1e-8, 0.01, 1)
	AdamWStepScalar(paramS, gradS, mS, vS, 0.001, 0.9, 0.999, 1e-8, 0.01, 1)

	for i := range n {
		if stdmath.Abs(param[i]-paramS[i]) > 1e-10 {
			t.Errorf("param[%d]: auto=%v, scalar=%v", i, param[i], paramS[i])
		}
	}
}

func BenchmarkAdamWStep(b *testing.B) {
	pool := workerpool.New(0)
	defer pool.Close()

	n := 768 * 768
	param := make([]float32, n)
	grad := make([]float32, n)
	m := make([]float32, n)
	v := make([]float32, n)

	for i := range n {
		param[i] = float32(i) * 0.0001
		grad[i] = float32(i) * 0.00001
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		AdamWStepAuto(pool, param, grad, m, v, 0.001, 0.9, 0.999, 1e-8, 0.01, i+1)
	}
}
