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

func TestDenseBackward_AutoVsScalar(t *testing.T) {
	pool := workerpool.New(0)
	defer pool.Close()

	tests := []struct {
		batch, inF, outF int
	}{
		{1, 4, 4},
		{2, 8, 4},
		{4, 16, 8},
		{3, 7, 5},
		{8, 64, 32},
	}

	for _, tt := range tests {
		name := fmt.Sprintf("%dx%dx%d", tt.batch, tt.inF, tt.outF)
		t.Run(name, func(t *testing.T) {
			gradOutput := make([]float32, tt.batch*tt.outF)
			x := make([]float32, tt.batch*tt.inF)
			weight := make([]float32, tt.outF*tt.inF)

			for i := range gradOutput {
				gradOutput[i] = float32(i)*0.01 - 0.5
			}
			for i := range x {
				x[i] = float32(i)*0.005 - 0.25
			}
			for i := range weight {
				weight[i] = float32(i)*0.003 - 0.15
			}

			// Auto
			aGradInput := make([]float32, tt.batch*tt.inF)
			aGradWeight := make([]float32, tt.outF*tt.inF)
			aGradBias := make([]float32, tt.outF)
			DenseBackwardAuto(pool, gradOutput, x, weight, aGradInput, aGradWeight, aGradBias, tt.batch, tt.inF, tt.outF)

			// Scalar
			sGradInput := make([]float32, tt.batch*tt.inF)
			sGradWeight := make([]float32, tt.outF*tt.inF)
			sGradBias := make([]float32, tt.outF)
			DenseBackwardScalar(gradOutput, x, weight, sGradInput, sGradWeight, sGradBias, tt.batch, tt.inF, tt.outF)

			allClose32(t, "gradInput", aGradInput, sGradInput, 1e-3)
			allClose32(t, "gradWeight", aGradWeight, sGradWeight, 1e-3)
			allClose32(t, "gradBias", aGradBias, sGradBias, 1e-5)
		})
	}
}

func TestDenseBackward_Float64(t *testing.T) {
	pool := workerpool.New(0)
	defer pool.Close()

	batch, inF, outF := 2, 16, 8

	gradOutput := make([]float64, batch*outF)
	x := make([]float64, batch*inF)
	weight := make([]float64, outF*inF)

	for i := range gradOutput {
		gradOutput[i] = float64(i)*0.01 - 0.5
	}
	for i := range x {
		x[i] = float64(i)*0.005 - 0.25
	}
	for i := range weight {
		weight[i] = float64(i)*0.003 - 0.15
	}

	aGI := make([]float64, batch*inF)
	aGW := make([]float64, outF*inF)
	aGB := make([]float64, outF)
	DenseBackwardAuto(pool, gradOutput, x, weight, aGI, aGW, aGB, batch, inF, outF)

	sGI := make([]float64, batch*inF)
	sGW := make([]float64, outF*inF)
	sGB := make([]float64, outF)
	DenseBackwardScalar(gradOutput, x, weight, sGI, sGW, sGB, batch, inF, outF)

	for i := range aGI {
		if stdmath.Abs(aGI[i]-sGI[i]) > 1e-8 {
			t.Errorf("gradInput[%d]: auto=%v, scalar=%v", i, aGI[i], sGI[i])
		}
	}
	for i := range aGW {
		if stdmath.Abs(aGW[i]-sGW[i]) > 1e-8 {
			t.Errorf("gradWeight[%d]: auto=%v, scalar=%v", i, aGW[i], sGW[i])
		}
	}
	for i := range aGB {
		if stdmath.Abs(aGB[i]-sGB[i]) > 1e-10 {
			t.Errorf("gradBias[%d]: auto=%v, scalar=%v", i, aGB[i], sGB[i])
		}
	}
}

func TestDenseBackward_NilGrads(t *testing.T) {
	pool := workerpool.New(0)
	defer pool.Close()

	batch, inF, outF := 2, 8, 4
	gradOutput := testInputs(batch * outF)
	x := testInputs(batch * inF)
	weight := testInputs(outF * inF)

	// Should not panic with nil gradient outputs
	DenseBackwardAuto[float32](pool, gradOutput, x, weight, nil, nil, nil, batch, inF, outF)

	gradInput := make([]float32, batch*inF)
	DenseBackwardAuto(pool, gradOutput, x, weight, gradInput, nil, nil, batch, inF, outF)

	// Verify non-zero output
	anyNonZero := false
	for _, v := range gradInput {
		if v != 0 {
			anyNonZero = true
			break
		}
	}
	if !anyNonZero {
		t.Error("gradInput should be non-zero")
	}
}

func BenchmarkDenseBackward(b *testing.B) {
	pool := workerpool.New(0)
	defer pool.Close()

	batch, inF, outF := 8, 768, 768
	gradOutput := make([]float32, batch*outF)
	x := make([]float32, batch*inF)
	weight := make([]float32, outF*inF)
	gradInput := make([]float32, batch*inF)
	gradWeight := make([]float32, outF*inF)
	gradBias := make([]float32, outF)

	for i := range gradOutput {
		gradOutput[i] = float32(i) * 0.001
	}
	for i := range x {
		x[i] = float32(i) * 0.001
	}
	for i := range weight {
		weight[i] = float32(i) * 0.0005
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		clear(gradInput)
		clear(gradWeight)
		clear(gradBias)
		DenseBackwardAuto(pool, gradOutput, x, weight, gradInput, gradWeight, gradBias, batch, inF, outF)
	}
}
