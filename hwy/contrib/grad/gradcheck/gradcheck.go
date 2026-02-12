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

// Package gradcheck provides numerical gradient checking utilities for
// verifying hand-coded backward pass implementations against finite differences.
package gradcheck

import (
	"fmt"
	"math"
)

// GradCheckConfig controls the behavior of gradient checking.
type GradCheckConfig struct {
	// Epsilon is the step size for finite differences. Default: 1e-5 for float64.
	Epsilon float64

	// RelTol is the relative tolerance for comparing analytical vs numerical gradients.
	RelTol float64

	// AbsTol is the absolute tolerance for comparing analytical vs numerical gradients.
	AbsTol float64

	// MaxErrors is the maximum number of mismatches to report. 0 means report all.
	MaxErrors int
}

// DefaultConfig returns a default configuration suitable for float64 gradient checking.
func DefaultConfig() GradCheckConfig {
	return GradCheckConfig{
		Epsilon:   1e-5,
		RelTol:    1e-4,
		AbsTol:    1e-6,
		MaxErrors: 10,
	}
}

// Float32Config returns a configuration tuned for float32 precision.
func Float32Config() GradCheckConfig {
	return GradCheckConfig{
		Epsilon:   1e-3,
		RelTol:    2e-3,
		AbsTol:    1e-5,
		MaxErrors: 10,
	}
}

// GradMismatch describes a single gradient mismatch.
type GradMismatch struct {
	Index      int
	Analytical float64
	Numerical  float64
	RelError   float64
}

// GradCheckResult holds the results of a gradient check.
type GradCheckResult struct {
	ParamName  string
	Passed     bool
	Mismatches []GradMismatch
	MaxRelErr  float64
	AvgRelErr  float64
}

func (r GradCheckResult) String() string {
	if r.Passed {
		return fmt.Sprintf("PASS %s: max_rel_err=%.2e avg_rel_err=%.2e",
			r.ParamName, r.MaxRelErr, r.AvgRelErr)
	}
	s := fmt.Sprintf("FAIL %s: %d mismatches, max_rel_err=%.2e avg_rel_err=%.2e\n",
		r.ParamName, len(r.Mismatches), r.MaxRelErr, r.AvgRelErr)
	for _, m := range r.Mismatches {
		s += fmt.Sprintf("  [%d] analytical=%.8e numerical=%.8e rel_err=%.2e\n",
			m.Index, m.Analytical, m.Numerical, m.RelError)
	}
	return s
}

// CheckGradient verifies analytical gradients against central finite differences.
//
// param is the parameter slice whose gradient is being checked.
// analyticalGrad is the gradient computed by the backward pass being tested.
// forwardFn computes a scalar loss value from the current state of param.
// cfg controls tolerances and reporting.
//
// The function uses central finite differences:
//
//	numerical[i] = (f(x+h) - f(x-h)) / (2*h)
//
// This is O(h²) accurate, providing much tighter agreement than forward differences.
func CheckGradient(
	paramName string,
	param, analyticalGrad []float64,
	forwardFn func() float64,
	cfg GradCheckConfig,
) GradCheckResult {
	if cfg.Epsilon == 0 {
		cfg.Epsilon = 1e-5
	}
	if cfg.RelTol == 0 {
		cfg.RelTol = 1e-4
	}

	result := GradCheckResult{
		ParamName: paramName,
		Passed:    true,
	}

	n := min(len(param), len(analyticalGrad))
	var totalRelErr float64
	numChecked := 0

	for i := range n {
		orig := param[i]

		// f(x + h)
		param[i] = orig + cfg.Epsilon
		fPlus := forwardFn()

		// f(x - h)
		param[i] = orig - cfg.Epsilon
		fMinus := forwardFn()

		// Restore
		param[i] = orig

		numerical := (fPlus - fMinus) / (2 * cfg.Epsilon)
		analytical := analyticalGrad[i]

		// Compute relative error
		denom := math.Max(math.Abs(numerical), math.Abs(analytical))
		denom = math.Max(denom, 1e-8) // avoid division by zero

		relErr := math.Abs(numerical-analytical) / denom
		absErr := math.Abs(numerical - analytical)

		totalRelErr += relErr
		numChecked++

		if relErr > result.MaxRelErr {
			result.MaxRelErr = relErr
		}

		if relErr > cfg.RelTol && absErr > cfg.AbsTol {
			result.Passed = false
			if cfg.MaxErrors == 0 || len(result.Mismatches) < cfg.MaxErrors {
				result.Mismatches = append(result.Mismatches, GradMismatch{
					Index:      i,
					Analytical: analytical,
					Numerical:  numerical,
					RelError:   relErr,
				})
			}
		}
	}

	if numChecked > 0 {
		result.AvgRelErr = totalRelErr / float64(numChecked)
	}

	return result
}

// CheckGradientFloat32 is a convenience wrapper that converts float32 slices
// to float64 for more precise numerical gradient checking.
func CheckGradientFloat32(
	paramName string,
	param, analyticalGrad []float32,
	forwardFn func() float32,
	cfg GradCheckConfig,
) GradCheckResult {
	// Convert to float64 for numerical precision
	param64 := make([]float64, len(param))
	grad64 := make([]float64, len(analyticalGrad))
	for i := range param {
		param64[i] = float64(param[i])
	}
	for i := range analyticalGrad {
		grad64[i] = float64(analyticalGrad[i])
	}

	return CheckGradient(paramName, param64, grad64, func() float64 {
		// Copy float64 params back to float32 for forward pass
		for i := range param {
			param[i] = float32(param64[i])
		}
		return float64(forwardFn())
	}, cfg)
}
