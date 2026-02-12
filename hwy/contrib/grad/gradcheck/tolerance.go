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

package gradcheck

// OpTolerance defines per-operation tolerance thresholds for gradient checking.
type OpTolerance struct {
	Float32RelTol float64
	Float64RelTol float64
}

// Per-operation tolerance tables, calibrated against numerical gradient checking
// and PyTorch autograd. The tolerances account for SIMD approximation error
// (e.g., polynomial erf/exp) compounding through the backward formula.
var (
	DenseTolerance     = OpTolerance{Float32RelTol: 1e-3, Float64RelTol: 1e-5}
	GELUTolerance      = OpTolerance{Float32RelTol: 2e-3, Float64RelTol: 1e-5}
	ReLUTolerance      = OpTolerance{Float32RelTol: 1e-5, Float64RelTol: 1e-10}
	SiLUTolerance      = OpTolerance{Float32RelTol: 2e-3, Float64RelTol: 1e-5}
	TanhTolerance      = OpTolerance{Float32RelTol: 1e-3, Float64RelTol: 1e-5}
	SoftmaxTolerance   = OpTolerance{Float32RelTol: 5e-3, Float64RelTol: 1e-4}
	LayerNormTolerance = OpTolerance{Float32RelTol: 5e-3, Float64RelTol: 1e-4}
	SDPATolerance      = OpTolerance{Float32RelTol: 1e-2, Float64RelTol: 1e-3}
	LoRATolerance      = OpTolerance{Float32RelTol: 1e-3, Float64RelTol: 1e-5}
	CrossEntropyTolerance = OpTolerance{Float32RelTol: 5e-3, Float64RelTol: 1e-4}
	FocalLossTolerance   = OpTolerance{Float32RelTol: 5e-3, Float64RelTol: 1e-4}
	AdamWTolerance     = OpTolerance{Float32RelTol: 1e-3, Float64RelTol: 1e-5}
)

// ConfigFor64 returns a GradCheckConfig for float64 precision.
func (t OpTolerance) ConfigFor64() GradCheckConfig {
	return GradCheckConfig{
		Epsilon:   1e-5,
		RelTol:    t.Float64RelTol,
		AbsTol:    1e-8,
		MaxErrors: 10,
	}
}

// ConfigFor32 returns a GradCheckConfig for float32 precision.
func (t OpTolerance) ConfigFor32() GradCheckConfig {
	return GradCheckConfig{
		Epsilon:   1e-3,
		RelTol:    t.Float32RelTol,
		AbsTol:    1e-5,
		MaxErrors: 10,
	}
}
