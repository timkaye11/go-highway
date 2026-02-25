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
	"testing"

	"github.com/ajroetker/go-highway/hwy/contrib/workerpool"
)

// testInputs returns a slice of floats centered around zero for gradient testing.
func testInputs(n int) []float32 {
	out := make([]float32, n)
	for i := range n {
		out[i] = float32(i)*0.1 - float32(n)*0.05
	}
	return out
}

func allClose32(t *testing.T, name string, got, want []float32, relTol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: length mismatch: got %d, want %d", name, len(got), len(want))
	}
	for i := range got {
		diff := stdmath.Abs(float64(got[i] - want[i]))
		denom := stdmath.Max(1.0, stdmath.Abs(float64(want[i])))
		if diff/denom > relTol {
			t.Errorf("%s[%d]: got %v, want %v, diff=%v", name, i, got[i], want[i], diff)
		}
	}
}

func geluScalar(x float64) float64 {
	return 0.5 * x * (1.0 + stdmath.Erf(x/stdmath.Sqrt2))
}

func geluDerivScalar(x float64) float64 {
	cdf := 0.5 * (1.0 + stdmath.Erf(x/stdmath.Sqrt2))
	pdf := (1.0 / stdmath.Sqrt(2*stdmath.Pi)) * stdmath.Exp(-0.5*x*x)
	return cdf + x*pdf
}

func TestGELUBackward_ScalarMatch(t *testing.T) {
	pool := workerpool.New(0)
	defer pool.Close()

	sizes := []struct {
		rows, cols int
	}{
		{1, 8}, {2, 16}, {4, 32}, {3, 7}, {8, 17},
	}

	for _, s := range sizes {
		n := s.rows * s.cols
		gradOutput := testInputs(n)
		savedInput := testInputs(n)

		gradAuto := make([]float32, n)
		gradScalar := make([]float32, n)

		for i := range n {
			gradScalar[i] = gradOutput[i] * float32(geluDerivScalar(float64(savedInput[i])))
		}

		GELUBackwardAuto(pool, gradOutput, savedInput, gradAuto, s.rows, s.cols)
		allClose32(t, "GELUBackward", gradAuto, gradScalar, 1e-4)
	}
}

func TestReLUBackward_ScalarMatch(t *testing.T) {
	pool := workerpool.New(0)
	defer pool.Close()

	sizes := []struct {
		rows, cols int
	}{
		{4, 16}, {2, 32}, {3, 7}, {8, 17}, {1, 8},
	}

	for _, s := range sizes {
		n := s.rows * s.cols
		gradOutput := testInputs(n)
		savedInput := testInputs(n)

		gradAuto := make([]float32, n)
		gradScalar := make([]float32, n)

		for i := range n {
			if savedInput[i] > 0 {
				gradScalar[i] = gradOutput[i]
			}
		}

		ReLUBackwardAuto(pool, gradOutput, savedInput, gradAuto, s.rows, s.cols)
		allClose32(t, "ReLUBackward", gradAuto, gradScalar, 1e-6)
	}
}

func TestSiLUBackward_ScalarMatch(t *testing.T) {
	pool := workerpool.New(0)
	defer pool.Close()

	sizes := []struct {
		rows, cols int
	}{
		{4, 16}, {2, 32}, {3, 7}, {8, 17}, {1, 8},
	}

	for _, s := range sizes {
		n := s.rows * s.cols
		gradOutput := testInputs(n)
		savedInput := testInputs(n)

		gradAuto := make([]float32, n)
		gradScalar := make([]float32, n)

		for i := range n {
			x := float64(savedInput[i])
			sig := 1.0 / (1.0 + stdmath.Exp(-x))
			deriv := sig * (1.0 + x*(1.0-sig))
			gradScalar[i] = gradOutput[i] * float32(deriv)
		}

		SiLUBackwardAuto(pool, gradOutput, savedInput, gradAuto, s.rows, s.cols)
		allClose32(t, "SiLUBackward", gradAuto, gradScalar, 1e-4)
	}
}

func TestTanhBackward_ScalarMatch(t *testing.T) {
	pool := workerpool.New(0)
	defer pool.Close()

	sizes := []struct {
		rows, cols int
	}{
		{4, 16}, {2, 32}, {3, 7}, {8, 17}, {1, 8},
	}

	for _, s := range sizes {
		n := s.rows * s.cols
		gradOutput := testInputs(n)
		inputs := testInputs(n)
		savedOutput := make([]float32, n)
		for i := range n {
			savedOutput[i] = float32(stdmath.Tanh(float64(inputs[i])))
		}

		gradAuto := make([]float32, n)
		gradScalar := make([]float32, n)

		for i := range n {
			t2 := float64(savedOutput[i])
			gradScalar[i] = gradOutput[i] * float32(1.0-t2*t2)
		}

		TanhBackwardAuto(pool, gradOutput, savedOutput, gradAuto, s.rows, s.cols)
		allClose32(t, "TanhBackward", gradAuto, gradScalar, 1e-5)
	}
}

func TestGELUBackward_NumericalGradient(t *testing.T) {
	n := 32
	x := testInputs(n)
	eps := 1e-4

	for i := range n {
		xi := float64(x[i])
		fPlus := geluScalar(xi + eps)
		fMinus := geluScalar(xi - eps)
		numerical := (fPlus - fMinus) / (2 * eps)
		analytical := geluDerivScalar(xi)

		diff := stdmath.Abs(numerical - analytical)
		denom := stdmath.Max(1.0, stdmath.Abs(analytical))
		if diff/denom > 1e-3 {
			t.Errorf("x=%v: numerical=%v, analytical=%v, diff=%v", xi, numerical, analytical, diff)
		}
	}
}

func TestResidualAddBackward(t *testing.T) {
	n := 32
	gradOutput := testInputs(n)
	gradInput := make([]float32, n)
	for i := range n {
		gradInput[i] = float32(i) * 0.01
	}

	want := make([]float32, n)
	for i := range n {
		want[i] = gradInput[i] + gradOutput[i]
	}

	ResidualAddBackward(gradOutput, gradInput)
	allClose32(t, "ResidualAddBackward", gradInput, want, 1e-6)
}

func BenchmarkGELUBackward(b *testing.B) {
	pool := workerpool.New(0)
	defer pool.Close()

	n := 8 * 768
	gradOutput := make([]float32, n)
	savedInput := make([]float32, n)
	gradInput := make([]float32, n)
	for i := range n {
		gradOutput[i] = float32(i) * 0.001
		savedInput[i] = float32(i)*0.01 - 3.0
	}

	b.Run("Auto", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			clear(gradInput)
			GELUBackwardAuto(pool, gradOutput, savedInput, gradInput, 8, 768)
		}
	})
	b.Run("SIMD", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			clear(gradInput)
			GELUBackward(gradOutput, savedInput, gradInput)
		}
	})
	b.Run("Scalar", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			clear(gradInput)
			GELUBackwardScalar(gradOutput, savedInput, gradInput)
		}
	})
}

func BenchmarkReLUBackward(b *testing.B) {
	n := 8 * 768
	gradOutput := make([]float32, n)
	savedInput := make([]float32, n)
	gradInput := make([]float32, n)
	for i := range n {
		gradOutput[i] = float32(i) * 0.001
		savedInput[i] = float32(i)*0.01 - 3.0
	}

	b.Run("SIMD", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			clear(gradInput)
			ReLUBackward(gradOutput, savedInput, gradInput)
		}
	})
	b.Run("Scalar", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			clear(gradInput)
			ReLUBackwardScalar(gradOutput, savedInput, gradInput)
		}
	})
}

func BenchmarkSiLUBackward(b *testing.B) {
	n := 8 * 768
	gradOutput := make([]float32, n)
	savedInput := make([]float32, n)
	gradInput := make([]float32, n)
	for i := range n {
		gradOutput[i] = float32(i) * 0.001
		savedInput[i] = float32(i)*0.01 - 3.0
	}

	b.Run("SIMD", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			clear(gradInput)
			SiLUBackward(gradOutput, savedInput, gradInput)
		}
	})
	b.Run("Scalar", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			clear(gradInput)
			SiLUBackwardScalar(gradOutput, savedInput, gradInput)
		}
	})
}

func BenchmarkTanhBackward(b *testing.B) {
	n := 8 * 768
	gradOutput := make([]float32, n)
	savedOutput := make([]float32, n)
	gradInput := make([]float32, n)
	for i := range n {
		gradOutput[i] = float32(i) * 0.001
		savedOutput[i] = float32(stdmath.Tanh(float64(i)*0.01 - 3.0))
	}

	b.Run("SIMD", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			clear(gradInput)
			TanhBackward(gradOutput, savedOutput, gradInput)
		}
	})
	b.Run("Scalar", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			clear(gradInput)
			TanhBackwardScalar(gradOutput, savedOutput, gradInput)
		}
	})
}
