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

//go:generate go run ../../../cmd/hwygen -input geglu_backward_base.go -output . -targets avx2,avx512,neon,fallback -dispatch geglu_backward

// BaseGeGLUBackward computes the backward pass for GeGLU gated activation.
//
// Forward: output = GELU(gate) * up
//
// GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
// dGELU/dx = 0.5 * (1 + erf(x/sqrt(2))) + x * exp(-x^2/2) / sqrt(2*pi)
//
// Backward:
//
//	gradGate[i] += gradOutput[i] * savedUp[i] * dGELU(savedGate[i])
//	gradUp[i]   += gradOutput[i] * GELU(savedGate[i])
func BaseGeGLUBackward[T hwy.Floats](gradOutput, savedGate, savedUp, gradGate, gradUp []T) {
	n := min(len(gradOutput), min(len(savedGate), min(len(savedUp), min(len(gradGate), len(gradUp)))))
	if n == 0 {
		return
	}

	vHalf := hwy.Const[T](0.5)
	vOne := hwy.Const[T](1.0)
	vInvSqrt2 := hwy.Const[T](0.7071067811865476)
	vInvSqrt2Pi := hwy.Const[T](0.3989422804014327) // 1/sqrt(2*pi)
	vNegHalf := hwy.Const[T](-0.5)

	lanes := vOne.NumLanes()
	ii := 0

	for ; ii+lanes <= n; ii += lanes {
		gOut := hwy.Load(gradOutput[ii:])
		g := hwy.Load(savedGate[ii:])
		u := hwy.Load(savedUp[ii:])
		gGate := hwy.Load(gradGate[ii:])
		gUp := hwy.Load(gradUp[ii:])

		// GELU(gate) = gate * 0.5 * (1 + erf(gate / sqrt(2)))
		gScaled := hwy.Mul(g, vInvSqrt2)
		erfG := math.BaseErfVec(gScaled)
		onePlusErf := hwy.Add(vOne, erfG)
		halfOnePlusErf := hwy.Mul(vHalf, onePlusErf)
		geluG := hwy.Mul(g, halfOnePlusErf)

		// dGELU/dx = 0.5*(1+erf(x/sqrt(2))) + x*exp(-x^2/2)/sqrt(2*pi)
		gSq := hwy.Mul(g, g)
		negHalfGSq := hwy.Mul(vNegHalf, gSq)
		expTerm := math.BaseExpVec(negHalfGSq)
		gaussianTerm := hwy.Mul(hwy.Mul(g, expTerm), vInvSqrt2Pi)
		dGelu := hwy.Add(halfOnePlusErf, gaussianTerm)

		// gradGate += gradOutput * up * dGELU
		gOutUp := hwy.Mul(gOut, u)
		newGGate := hwy.MulAdd(gOutUp, dGelu, gGate)
		hwy.Store(newGGate, gradGate[ii:])

		// gradUp += gradOutput * GELU(gate)
		newGUp := hwy.MulAdd(gOut, geluG, gUp)
		hwy.Store(newGUp, gradUp[ii:])
	}

	// Scalar tail
	for i := ii; i < n; i++ {
		g := float64(savedGate[i])
		u := float64(savedUp[i])
		go_ := float64(gradOutput[i])

		// GELU(gate)
		erfVal := stdmath.Erf(g * 0.7071067811865476)
		halfOnePlusErf := 0.5 * (1.0 + erfVal)
		geluG := g * halfOnePlusErf

		// dGELU/dgate
		gaussianTerm := g * stdmath.Exp(-0.5*g*g) * 0.3989422804014327
		dGelu := halfOnePlusErf + gaussianTerm

		gradGate[i] += T(go_ * u * dGelu)
		gradUp[i] += T(go_ * geluG)
	}
}
