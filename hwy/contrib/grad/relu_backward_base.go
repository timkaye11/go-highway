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

import "github.com/ajroetker/go-highway/hwy"

//go:generate go run ../../../cmd/hwygen -input relu_backward_base.go -output . -targets avx2,avx512,neon,fallback -dispatch relu_backward

// BaseReLUBackward computes the backward pass for ReLU activation.
//
//	dReLU/dx = 1 if savedInput > 0, else 0
//	gradInput[i] += gradOutput[i] * (savedInput[i] > 0 ? 1 : 0)
//
// savedInput is the pre-activation input saved from the forward pass.
func BaseReLUBackward[T hwy.Floats](gradOutput, savedInput, gradInput []T) {
	n := min(len(gradOutput), min(len(savedInput), len(gradInput)))
	if n == 0 {
		return
	}

	vZero := hwy.Const[T](0.0)
	lanes := vZero.NumLanes()
	ii := 0

	for ; ii+lanes <= n; ii += lanes {
		gOut := hwy.Load(gradOutput[ii:])
		inp := hwy.Load(savedInput[ii:])
		gIn := hwy.Load(gradInput[ii:])

		// Mask: 1 where input > 0, 0 otherwise
		mask := hwy.Greater(inp, vZero)
		// Select gradOutput where mask is true, zero otherwise
		masked := hwy.Merge(gOut, vZero, mask)
		// Accumulate
		result := hwy.Add(gIn, masked)
		hwy.Store(result, gradInput[ii:])
	}

	// Scalar tail
	for i := ii; i < n; i++ {
		if savedInput[i] > 0 {
			gradInput[i] += gradOutput[i]
		}
	}
}
