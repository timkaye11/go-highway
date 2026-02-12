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

// layerNormForwardScalar computes layer norm forward and saves intermediates.
func layerNormForwardScalar(input []float32, normSize int, gamma, beta []float32, eps float32) (output, xHat, invStd []float32) {
	n := len(input)
	numGroups := n / normSize
	output = make([]float32, n)
	xHat = make([]float32, n)
	invStd = make([]float32, numGroups)

	for g := range numGroups {
		off := g * normSize

		// Compute mean
		var mean float64
		for i := range normSize {
			mean += float64(input[off+i])
		}
		mean /= float64(normSize)

		// Compute variance
		var variance float64
		for i := range normSize {
			d := float64(input[off+i]) - mean
			variance += d * d
		}
		variance /= float64(normSize)

		std := stdmath.Sqrt(variance + float64(eps))
		invStd[g] = float32(1.0 / std)

		// Normalize
		for i := range normSize {
			xHat[off+i] = float32((float64(input[off+i]) - mean) / std)
			if gamma != nil {
				output[off+i] = xHat[off+i]*gamma[i] + beta[i]
			} else {
				output[off+i] = xHat[off+i]
			}
		}
	}
	return
}

func TestLayerNormBackward_AutoVsScalar(t *testing.T) {
	pool := workerpool.New(0)
	defer pool.Close()

	tests := []struct {
		numGroups, normSize int
		useGamma            bool
	}{
		{1, 8, true},
		{2, 16, true},
		{4, 32, true},
		{2, 7, true},
		{8, 17, true},
		{2, 16, false},
	}

	for _, tt := range tests {
		name := fmt.Sprintf("g%d_n%d_gamma%v", tt.numGroups, tt.normSize, tt.useGamma)
		t.Run(name, func(t *testing.T) {
			n := tt.numGroups * tt.normSize

			// Create input and compute forward pass
			input := testInputs(n)
			var gamma, beta []float32
			if tt.useGamma {
				gamma = make([]float32, tt.normSize)
				beta = make([]float32, tt.normSize)
				for i := range tt.normSize {
					gamma[i] = 1.0 + float32(i)*0.01
					beta[i] = float32(i) * 0.01
				}
			}

			_, savedXHat, savedInvStd := layerNormForwardScalar(input, tt.normSize, gamma, beta, 1e-5)

			gradOutput := testInputs(n)

			// Auto
			aGradInput := make([]float32, n)
			var aGradGamma, aGradBeta []float32
			if tt.useGamma {
				aGradGamma = make([]float32, tt.normSize)
				aGradBeta = make([]float32, tt.normSize)
			}
			LayerNormBackwardAuto(pool, gradOutput, savedXHat, savedInvStd, gamma, tt.normSize, aGradInput, aGradGamma, aGradBeta)

			// Scalar
			sGradInput := make([]float32, n)
			var sGradGamma, sGradBeta []float32
			if tt.useGamma {
				sGradGamma = make([]float32, tt.normSize)
				sGradBeta = make([]float32, tt.normSize)
			}
			LayerNormBackwardScalar(gradOutput, savedXHat, savedInvStd, gamma, tt.normSize, sGradInput, sGradGamma, sGradBeta)

			allClose32(t, "gradInput", aGradInput, sGradInput, 5e-3)
			if tt.useGamma {
				allClose32(t, "gradGamma", aGradGamma, sGradGamma, 5e-3)
				allClose32(t, "gradBeta", aGradBeta, sGradBeta, 5e-3)
			}
		})
	}
}

func TestLayerNormBackward_NumericalGradient(t *testing.T) {
	normSize := 8
	numGroups := 2
	n := numGroups * normSize
	eps := float32(1e-5)

	input := make([]float64, n)
	for i := range n {
		input[i] = float64(i)*0.1 - float64(n)*0.05
	}

	gamma := make([]float32, normSize)
	beta := make([]float32, normSize)
	for i := range normSize {
		gamma[i] = 1.0 + float32(i)*0.1
		beta[i] = float32(i) * 0.05
	}

	// Forward function: returns sum of output (as a scalar loss)
	forwardLoss := func(inp []float64) float64 {
		f32Input := make([]float32, n)
		for i := range n {
			f32Input[i] = float32(inp[i])
		}
		output, _, _ := layerNormForwardScalar(f32Input, normSize, gamma, beta, eps)
		var sum float64
		for _, v := range output {
			sum += float64(v)
		}
		return sum
	}

	// Numerical gradient w.r.t. input
	h := 1e-4
	numericalGrad := make([]float64, n)
	for i := range n {
		orig := input[i]
		input[i] = orig + h
		lPlus := forwardLoss(input)
		input[i] = orig - h
		lMinus := forwardLoss(input)
		input[i] = orig
		numericalGrad[i] = (lPlus - lMinus) / (2 * h)
	}

	// Analytical gradient (gradOutput = all ones since loss = sum(output))
	f32Input := make([]float32, n)
	for i := range n {
		f32Input[i] = float32(input[i])
	}
	_, savedXHat, savedInvStd := layerNormForwardScalar(f32Input, normSize, gamma, beta, eps)
	gradOutput := make([]float32, n)
	for i := range n {
		gradOutput[i] = 1.0
	}
	analyticalGrad := make([]float32, n)
	LayerNormBackwardScalar(gradOutput, savedXHat, savedInvStd, gamma, normSize, analyticalGrad, nil, nil)

	for i := range n {
		diff := stdmath.Abs(float64(analyticalGrad[i]) - numericalGrad[i])
		denom := stdmath.Max(1.0, stdmath.Abs(numericalGrad[i]))
		if diff/denom > 5e-3 {
			t.Errorf("[%d]: analytical=%v, numerical=%v, diff=%v", i, analyticalGrad[i], numericalGrad[i], diff)
		}
	}
}

func BenchmarkLayerNormBackward(b *testing.B) {
	pool := workerpool.New(0)
	defer pool.Close()

	normSize := 768
	numGroups := 32
	n := numGroups * normSize

	gradOutput := make([]float32, n)
	savedXHat := make([]float32, n)
	savedInvStd := make([]float32, numGroups)
	gamma := make([]float32, normSize)
	gradInput := make([]float32, n)
	gradGamma := make([]float32, normSize)
	gradBeta := make([]float32, normSize)

	for i := range n {
		gradOutput[i] = float32(i) * 0.001
		savedXHat[i] = float32(i) * 0.001
	}
	for i := range numGroups {
		savedInvStd[i] = 1.0
	}
	for i := range normSize {
		gamma[i] = 1.0
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		clear(gradInput)
		clear(gradGamma)
		clear(gradBeta)
		LayerNormBackwardAuto(pool, gradOutput, savedXHat, savedInvStd, gamma, normSize, gradInput, gradGamma, gradBeta)
	}
}
