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

//go:generate go run ../../../cmd/hwygen -input silu_backward_base.go -output . -targets avx2,avx512,neon,fallback -dispatch silu_backward

// BaseSiLUBackward computes the backward pass for SiLU (Swish) activation.
//
//	SiLU(x) = x * σ(x)
//	dSiLU/dx = σ(x) * (1 + x * (1 - σ(x)))
//	gradInput[i] += gradOutput[i] * dSiLU/dx(savedInput[i])
//
// savedInput is the pre-activation input saved from the forward pass.
func BaseSiLUBackward[T hwy.Floats](gradOutput, savedInput, gradInput []T) {
	n := min(len(gradOutput), min(len(savedInput), len(gradInput)))
	if n == 0 {
		return
	}

	vOne := hwy.Const[T](1.0)
	lanes := vOne.NumLanes()
	ii := 0

	for ; ii+lanes <= n; ii += lanes {
		gOut := hwy.Load(gradOutput[ii:])
		x := hwy.Load(savedInput[ii:])
		gIn := hwy.Load(gradInput[ii:])

		// σ(x) = sigmoid(x)
		sigX := math.BaseSigmoidVec(x)

		// 1 - σ(x)
		oneMinusSig := hwy.Sub(vOne, sigX)

		// x * (1 - σ(x))
		xTimesTerm := hwy.Mul(x, oneMinusSig)

		// 1 + x * (1 - σ(x))
		innerTerm := hwy.Add(vOne, xTimesTerm)

		// dSiLU/dx = σ(x) * (1 + x * (1 - σ(x)))
		dSilu := hwy.Mul(sigX, innerTerm)

		// Accumulate: gradInput += gradOutput * dSiLU/dx
		result := hwy.MulAdd(gOut, dSilu, gIn)
		hwy.Store(result, gradInput[ii:])
	}

	// Scalar tail
	for i := ii; i < n; i++ {
		x := float64(savedInput[i])
		sig := 1.0 / (1.0 + stdmath.Exp(-x))
		dSilu := sig * (1.0 + x*(1.0-sig))
		gradInput[i] += gradOutput[i] * T(dSilu)
	}
}
