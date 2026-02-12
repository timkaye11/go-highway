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

	"github.com/ajroetker/go-highway/hwy"
	"github.com/ajroetker/go-highway/hwy/contrib/math"
)

//go:generate go run ../../../cmd/hwygen -input gelu_backward_base.go -output . -targets avx2,avx512,neon,fallback -dispatch gelu_backward

// BaseGELUBackward computes the backward pass for exact GELU activation.
//
//	GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
//	dGELU/dx = 0.5 * (1 + erf(x/√2)) + x * (1/√(2π)) * exp(-x²/2)
//	gradInput[i] += gradOutput[i] * dGELU/dx(savedInput[i])
//
// savedInput is the pre-activation input saved from the forward pass.
func BaseGELUBackward[T hwy.Floats](gradOutput, savedInput, gradInput []T) {
	n := min(len(gradOutput), min(len(savedInput), len(gradInput)))
	if n == 0 {
		return
	}

	// Constants
	vHalf := hwy.Const[T](0.5)
	vOne := hwy.Const[T](1.0)
	vInvSqrt2 := hwy.Const[T](0.7071067811865476)    // 1/sqrt(2)
	vInvSqrt2Pi := hwy.Const[T](0.3989422804014327)   // 1/sqrt(2*pi)
	vNegHalf := hwy.Const[T](-0.5)

	lanes := vOne.NumLanes()
	ii := 0

	for ; ii+lanes <= n; ii += lanes {
		gOut := hwy.Load(gradOutput[ii:])
		x := hwy.Load(savedInput[ii:])
		gIn := hwy.Load(gradInput[ii:])

		// erf(x / sqrt(2))
		xScaled := hwy.Mul(x, vInvSqrt2)
		erfX := math.BaseErfVec(xScaled)

		// term1 = 0.5 * (1 + erf(x/√2))
		term1 := hwy.Mul(vHalf, hwy.Add(vOne, erfX))

		// term2 = x * (1/√(2π)) * exp(-x²/2)
		xSq := hwy.Mul(x, x)
		negHalfXSq := hwy.Mul(vNegHalf, xSq)
		expTerm := math.BaseExpVec(negHalfXSq)
		term2 := hwy.Mul(x, hwy.Mul(vInvSqrt2Pi, expTerm))

		// dGELU/dx = term1 + term2
		dGelu := hwy.Add(term1, term2)

		// Accumulate: gradInput += gradOutput * dGELU/dx
		result := hwy.MulAdd(gOut, dGelu, gIn)
		hwy.Store(result, gradInput[ii:])
	}

	// Scalar tail
	for i := ii; i < n; i++ {
		x := float64(savedInput[i])
		erfVal := stdmath.Erf(x * 0.7071067811865476)
		term1 := 0.5 * (1.0 + erfVal)
		term2 := x * 0.3989422804014327 * stdmath.Exp(-0.5*x*x)
		gradInput[i] += gradOutput[i] * T(term1+term2)
	}
}
