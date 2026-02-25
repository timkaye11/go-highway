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

//go:generate go run ../../../cmd/hwygen -input swiglu_backward_base.go -output . -targets avx2,avx512,neon,fallback -dispatch swiglu_backward

// BaseSwiGLUBackward computes the backward pass for SwiGLU gated activation.
//
// Forward: output = SiLU(gate) * up = gate * sigmoid(gate) * up
//
// Backward (recomputes sigmoid from savedGate to avoid storing intermediate):
//
//	sig = sigmoid(savedGate)
//	siluGate = savedGate * sig
//	dSiLU = sig * (1 + savedGate * (1 - sig))
//	gradGate[i] += gradOutput[i] * savedUp[i] * dSiLU[i]
//	gradUp[i]   += gradOutput[i] * siluGate[i]
//
// All slices must have the same length. gradGate and gradUp are accumulated into.
func BaseSwiGLUBackward[T hwy.Floats](gradOutput, savedGate, savedUp, gradGate, gradUp []T) {
	n := min(len(gradOutput), min(len(savedGate), min(len(savedUp), min(len(gradGate), len(gradUp)))))
	if n == 0 {
		return
	}

	vOne := hwy.Const[T](1.0)
	lanes := vOne.NumLanes()
	ii := 0

	for ; ii+lanes <= n; ii += lanes {
		gOut := hwy.Load(gradOutput[ii:])
		g := hwy.Load(savedGate[ii:])
		u := hwy.Load(savedUp[ii:])
		gGate := hwy.Load(gradGate[ii:])
		gUp := hwy.Load(gradUp[ii:])

		// Recompute sigmoid(gate)
		sigG := math.BaseSigmoidVec(g)

		// SiLU(gate) = gate * sigmoid(gate)
		siluG := hwy.Mul(g, sigG)

		// dSiLU/dgate = sig * (1 + gate * (1 - sig))
		oneMinusSig := hwy.Sub(vOne, sigG)
		gTimesOMS := hwy.Mul(g, oneMinusSig)
		innerTerm := hwy.Add(vOne, gTimesOMS)
		dSilu := hwy.Mul(sigG, innerTerm)

		// gradGate += gradOutput * up * dSiLU
		gOutUp := hwy.Mul(gOut, u)
		newGGate := hwy.MulAdd(gOutUp, dSilu, gGate)
		hwy.Store(newGGate, gradGate[ii:])

		// gradUp += gradOutput * SiLU(gate)
		newGUp := hwy.MulAdd(gOut, siluG, gUp)
		hwy.Store(newGUp, gradUp[ii:])
	}

	// Scalar tail
	for i := ii; i < n; i++ {
		g := float64(savedGate[i])
		u := float64(savedUp[i])
		go_ := float64(gradOutput[i])

		sig := 1.0 / (1.0 + stdmath.Exp(-g))
		siluG := g * sig
		dSilu := sig * (1.0 + g*(1.0-sig))

		gradGate[i] += T(go_ * u * dSilu)
		gradUp[i] += T(go_ * siluG)
	}
}
