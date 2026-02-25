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

//go:generate go run ../../../cmd/hwygen -input swiglu_base.go -output . -targets avx2,avx512,neon,fallback

// BaseSwiGLU computes the SwiGLU gated activation: output = SiLU(gate) * up.
//
// SiLU(x) = x * sigmoid(x), so SwiGLU(gate, up) = gate * sigmoid(gate) * up.
//
// gate, up, and output must have the same length. This is the element-wise
// activation used in LLaMA, Gemma, and other modern transformer MLPs.
func BaseSwiGLU[T hwy.Floats](gate, up, output []T) {
	size := min(len(gate), min(len(up), len(output)))
	if size == 0 {
		return
	}

	lanes := hwy.MaxLanes[T]()
	ii := 0

	for ; ii+lanes <= size; ii += lanes {
		g := hwy.Load(gate[ii:])
		u := hwy.Load(up[ii:])

		// SiLU(gate) = gate * sigmoid(gate)
		sigG := math.BaseSigmoidVec(g)
		siluG := hwy.Mul(g, sigG)

		// output = SiLU(gate) * up
		result := hwy.Mul(siluG, u)
		hwy.Store(result, output[ii:])
	}

	// Scalar tail
	for i := ii; i < size; i++ {
		g := float64(gate[i])
		sig := 1.0 / (1.0 + stdmath.Exp(-g))
		output[i] = T(g * sig * float64(up[i]))
	}
}

// SwiGLUScalar is a scalar reference implementation for testing.
func SwiGLUScalar[T hwy.Floats](gate, up, output []T) {
	size := min(len(gate), min(len(up), len(output)))
	for i := range size {
		g := float64(gate[i])
		sig := 1.0 / (1.0 + stdmath.Exp(-g))
		output[i] = T(g * sig * float64(up[i]))
	}
}
