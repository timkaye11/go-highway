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
	"github.com/ajroetker/go-highway/hwy"
	"github.com/ajroetker/go-highway/hwy/contrib/vec"
)

// ResidualAddBackward computes the backward pass for a residual (skip)
// connection: output = x + f(x).
//
// For the skip connection path, the gradient simply passes through:
//
//	gradInput[i] += gradOutput[i]
//
// The function-path gradient is handled by that function's own backward op.
func ResidualAddBackward[T hwy.Floats](gradOutput, gradInput []T) {
	vec.Add(gradInput, gradOutput)
}
