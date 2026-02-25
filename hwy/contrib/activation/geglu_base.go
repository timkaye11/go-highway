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

package activation

import (
	stdmath "math"

	"github.com/ajroetker/go-highway/hwy"
	"github.com/ajroetker/go-highway/hwy/contrib/math"
)

//go:generate go run ../../../cmd/hwygen -input geglu_base.go -output . -targets avx2,avx512,neon,fallback

// BaseGeGLU computes the GeGLU gated activation: output = GELU(gate) * up.
//
// GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2))), so
// GeGLU(gate, up) = GELU(gate) * up.
//
// gate, up, and output must have the same length.
func BaseGeGLU[T hwy.Floats](gate, up, output []T) {
	size := min(len(gate), min(len(up), len(output)))
	if size == 0 {
		return
	}

	vHalf := hwy.Const[T](0.5)
	vOne := hwy.Const[T](1.0)
	vInvSqrt2 := hwy.Const[T](0.7071067811865476)

	lanes := vOne.NumLanes()
	ii := 0

	for ; ii+lanes <= size; ii += lanes {
		g := hwy.Load(gate[ii:])
		u := hwy.Load(up[ii:])

		// GELU(gate) = gate * 0.5 * (1 + erf(gate / sqrt(2)))
		gScaled := hwy.Mul(g, vInvSqrt2)
		erfG := math.BaseErfVec(gScaled)
		onePlusErf := hwy.Add(vOne, erfG)
		halfOnePlusErf := hwy.Mul(vHalf, onePlusErf)
		geluG := hwy.Mul(g, halfOnePlusErf)

		// output = GELU(gate) * up
		result := hwy.Mul(geluG, u)
		hwy.Store(result, output[ii:])
	}

	// Scalar tail
	for i := ii; i < size; i++ {
		g := float64(gate[i])
		gelu := g * 0.5 * (1.0 + stdmath.Erf(g*0.7071067811865476))
		output[i] = T(gelu * float64(up[i]))
	}
}

// GeGLUScalar is a scalar reference implementation for testing.
func GeGLUScalar[T hwy.Floats](gate, up, output []T) {
	size := min(len(gate), min(len(up), len(output)))
	for i := range size {
		g := float64(gate[i])
		gelu := g * 0.5 * (1.0 + stdmath.Erf(g*0.7071067811865476))
		output[i] = T(gelu * float64(up[i]))
	}
}
