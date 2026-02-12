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

//go:generate go run ../../../cmd/hwygen -input tanh_backward_base.go -output . -targets avx2,avx512,neon,fallback -dispatch tanh_backward

// BaseTanhBackward computes the backward pass for Tanh activation.
//
//	dTanh/dx = 1 - tanh(x)²
//	gradInput[i] += gradOutput[i] * (1 - savedOutput[i]²)
//
// savedOutput is the tanh output saved from the forward pass (avoids
// recomputing tanh).
func BaseTanhBackward[T hwy.Floats](gradOutput, savedOutput, gradInput []T) {
	n := min(len(gradOutput), min(len(savedOutput), len(gradInput)))
	if n == 0 {
		return
	}

	vNegOne := hwy.Const[T](-1.0)
	lanes := vNegOne.NumLanes()
	ii := 0

	for ; ii+lanes <= n; ii += lanes {
		gOut := hwy.Load(gradOutput[ii:])
		tanhX := hwy.Load(savedOutput[ii:])
		gIn := hwy.Load(gradInput[ii:])

		// Compute: gradInput += gradOutput * (1 - tanh²)
		//        = gradInput + gradOutput - gradOutput * tanh * tanh
		// Using FMA: result = (gradInput + gradOutput) + (-gradOutput * tanh) * tanh
		gOutTanh := hwy.Mul(gOut, tanhX)
		negGOutTanh := hwy.Mul(vNegOne, gOutTanh)
		gSum := hwy.Add(gIn, gOut)
		result := hwy.MulAdd(negGOutTanh, tanhX, gSum)
		hwy.Store(result, gradInput[ii:])
	}

	// Scalar tail
	for i := ii; i < n; i++ {
		tanhX := savedOutput[i]
		dTanh := T(1) - tanhX*tanhX
		gradInput[i] += gradOutput[i] * dTanh
	}
}
