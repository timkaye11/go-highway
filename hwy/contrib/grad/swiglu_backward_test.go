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

	"github.com/ajroetker/go-highway/hwy/contrib/activation"
	"github.com/ajroetker/go-highway/hwy/contrib/workerpool"
)

func TestSwiGLUBackward_SIMDvsScalar(t *testing.T) {
	sizes := []int{4, 7, 16, 33, 64, 256}

	for _, size := range sizes {
		t.Run(fmt.Sprintf("size=%d", size), func(t *testing.T) {
			gradOutput := make([]float32, size)
			savedGate := make([]float32, size)
			savedUp := make([]float32, size)
			for i := range size {
				gradOutput[i] = float32(i)*0.01 - float32(size)*0.005
				savedGate[i] = float32(i)*0.1 - float32(size)*0.05
				savedUp[i] = float32(i)*0.05 + 0.5
			}

			simdGGate := make([]float32, size)
			simdGUp := make([]float32, size)
			SwiGLUBackward(gradOutput, savedGate, savedUp, simdGGate, simdGUp)

			scalarGGate := make([]float32, size)
			scalarGUp := make([]float32, size)
			swiGLUBackwardScalar(gradOutput, savedGate, savedUp, scalarGGate, scalarGUp)

			allClose32(t, "gradGate", simdGGate, scalarGGate, 1e-4)
			allClose32(t, "gradUp", simdGUp, scalarGUp, 1e-4)
		})
	}
}

func TestGeGLUBackward_SIMDvsScalar(t *testing.T) {
	sizes := []int{4, 7, 16, 33, 64, 256}

	for _, size := range sizes {
		t.Run(fmt.Sprintf("size=%d", size), func(t *testing.T) {
			gradOutput := make([]float32, size)
			savedGate := make([]float32, size)
			savedUp := make([]float32, size)
			for i := range size {
				gradOutput[i] = float32(i)*0.01 - float32(size)*0.005
				savedGate[i] = float32(i)*0.1 - float32(size)*0.05
				savedUp[i] = float32(i)*0.05 + 0.5
			}

			simdGGate := make([]float32, size)
			simdGUp := make([]float32, size)
			GeGLUBackward(gradOutput, savedGate, savedUp, simdGGate, simdGUp)

			scalarGGate := make([]float32, size)
			scalarGUp := make([]float32, size)
			geGLUBackwardScalar(gradOutput, savedGate, savedUp, scalarGGate, scalarGUp)

			allClose32(t, "gradGate", simdGGate, scalarGGate, 1e-4)
			allClose32(t, "gradUp", simdGUp, scalarGUp, 1e-4)
		})
	}
}

func TestSwiGLUBackward_GradCheck(t *testing.T) {
	size := 8
	savedGate := make([]float32, size)
	savedUp := make([]float32, size)
	gradOutput := make([]float32, size)
	for i := range size {
		savedGate[i] = float32(i)*0.3 - 1.0
		savedUp[i] = float32(i)*0.2 + 0.5
		gradOutput[i] = float32(i)*0.1 - 0.3
	}

	// Analytical gradients
	analyticalGGate := make([]float32, size)
	analyticalGUp := make([]float32, size)
	swiGLUBackwardScalar(gradOutput, savedGate, savedUp, analyticalGGate, analyticalGUp)

	// Numerical gradients via finite differences
	eps := 1e-4
	numericalGGate := make([]float32, size)
	numericalGUp := make([]float32, size)
	output1 := make([]float32, size)
	output2 := make([]float32, size)

	// Gradient w.r.t. gate
	for i := range size {
		orig := savedGate[i]
		savedGate[i] = float32(float64(orig) + eps)
		activation.SwiGLUScalar(savedGate, savedUp, output1)
		savedGate[i] = float32(float64(orig) - eps)
		activation.SwiGLUScalar(savedGate, savedUp, output2)
		savedGate[i] = orig

		var sum float64
		for j := range size {
			sum += float64(gradOutput[j]) * (float64(output1[j]) - float64(output2[j])) / (2.0 * eps)
		}
		numericalGGate[i] = float32(sum)
	}

	// Gradient w.r.t. up
	for i := range size {
		orig := savedUp[i]
		savedUp[i] = float32(float64(orig) + eps)
		activation.SwiGLUScalar(savedGate, savedUp, output1)
		savedUp[i] = float32(float64(orig) - eps)
		activation.SwiGLUScalar(savedGate, savedUp, output2)
		savedUp[i] = orig

		var sum float64
		for j := range size {
			sum += float64(gradOutput[j]) * (float64(output1[j]) - float64(output2[j])) / (2.0 * eps)
		}
		numericalGUp[i] = float32(sum)
	}

	allClose32(t, "gradGate_gradcheck", analyticalGGate, numericalGGate, 5e-3)
	allClose32(t, "gradUp_gradcheck", analyticalGUp, numericalGUp, 5e-3)
}

func TestSwiGLUBackwardAuto_ParVsSeq(t *testing.T) {
	pool := workerpool.New(0)
	defer pool.Close()

	rows, cols := 16, 32
	size := rows * cols
	gradOutput := make([]float32, size)
	savedGate := make([]float32, size)
	savedUp := make([]float32, size)
	for i := range size {
		gradOutput[i] = float32(i)*0.01 - float32(size)*0.005
		savedGate[i] = float32(i)*0.1 - float32(size)*0.05
		savedUp[i] = float32(i)*0.05 + 0.5
	}

	seqGGate := make([]float32, size)
	seqGUp := make([]float32, size)
	SwiGLUBackwardAuto[float32](nil, gradOutput, savedGate, savedUp, seqGGate, seqGUp, rows, cols)

	parGGate := make([]float32, size)
	parGUp := make([]float32, size)
	SwiGLUBackwardAuto(pool, gradOutput, savedGate, savedUp, parGGate, parGUp, rows, cols)

	allClose32(t, "gradGate_par", parGGate, seqGGate, 1e-6)
	allClose32(t, "gradUp_par", parGUp, seqGUp, 1e-6)
}

// swiGLUBackwardScalar is a scalar reference for testing.
func swiGLUBackwardScalar(gradOutput, savedGate, savedUp, gradGate, gradUp []float32) {
	n := min(len(gradOutput), min(len(savedGate), min(len(savedUp), min(len(gradGate), len(gradUp)))))
	for i := range n {
		g := float64(savedGate[i])
		u := float64(savedUp[i])
		go_ := float64(gradOutput[i])

		sig := 1.0 / (1.0 + stdmath.Exp(-g))
		siluG := g * sig
		dSilu := sig * (1.0 + g*(1.0-sig))

		gradGate[i] += float32(go_ * u * dSilu)
		gradUp[i] += float32(go_ * siluG)
	}
}

// geGLUBackwardScalar is a scalar reference for testing.
func geGLUBackwardScalar(gradOutput, savedGate, savedUp, gradGate, gradUp []float32) {
	n := min(len(gradOutput), min(len(savedGate), min(len(savedUp), min(len(gradGate), len(gradUp)))))
	for i := range n {
		g := float64(savedGate[i])
		u := float64(savedUp[i])
		go_ := float64(gradOutput[i])

		erfVal := stdmath.Erf(g * 0.7071067811865476)
		halfOnePlusErf := 0.5 * (1.0 + erfVal)
		geluG := g * halfOnePlusErf
		gaussianTerm := g * stdmath.Exp(-0.5*g*g) * 0.3989422804014327
		dGelu := halfOnePlusErf + gaussianTerm

		gradGate[i] += float32(go_ * u * dGelu)
		gradUp[i] += float32(go_ * geluG)
	}
}

func BenchmarkSwiGLUBackward(b *testing.B) {
	pool := workerpool.New(0)
	defer pool.Close()

	rows, cols := 32, 768
	size := rows * cols
	gradOutput := make([]float32, size)
	savedGate := make([]float32, size)
	savedUp := make([]float32, size)
	gradGate := make([]float32, size)
	gradUp := make([]float32, size)

	for i := range size {
		gradOutput[i] = float32(i) * 0.001
		savedGate[i] = float32(i) * 0.001
		savedUp[i] = float32(i) * 0.001
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		clear(gradGate)
		clear(gradUp)
		SwiGLUBackwardAuto(pool, gradOutput, savedGate, savedUp, gradGate, gradUp, rows, cols)
	}
}
