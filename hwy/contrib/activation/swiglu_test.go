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
	"fmt"
	stdmath "math"
	"testing"

	"github.com/ajroetker/go-highway/hwy/contrib/workerpool"
)

func TestSwiGLU_SIMDvsScalar(t *testing.T) {
	sizes := []int{1, 4, 7, 16, 33, 64, 127, 256}

	for _, size := range sizes {
		t.Run(fmt.Sprintf("size=%d", size), func(t *testing.T) {
			gate := make([]float32, size)
			up := make([]float32, size)
			for i := range size {
				gate[i] = float32(i)*0.1 - float32(size)*0.05
				up[i] = float32(i)*0.05 + 0.5
			}

			simdOut := make([]float32, size)
			scalarOut := make([]float32, size)

			SwiGLU(gate, up, simdOut)
			SwiGLUScalar(gate, up, scalarOut)

			for i := range size {
				diff := stdmath.Abs(float64(simdOut[i] - scalarOut[i]))
				if diff > 1e-4 {
					t.Errorf("[%d] SIMD=%v, Scalar=%v, diff=%v", i, simdOut[i], scalarOut[i], diff)
				}
			}
		})
	}
}

func TestGeGLU_SIMDvsScalar(t *testing.T) {
	sizes := []int{1, 4, 7, 16, 33, 64, 127, 256}

	for _, size := range sizes {
		t.Run(fmt.Sprintf("size=%d", size), func(t *testing.T) {
			gate := make([]float32, size)
			up := make([]float32, size)
			for i := range size {
				gate[i] = float32(i)*0.1 - float32(size)*0.05
				up[i] = float32(i)*0.05 + 0.5
			}

			simdOut := make([]float32, size)
			scalarOut := make([]float32, size)

			GeGLU(gate, up, simdOut)
			GeGLUScalar(gate, up, scalarOut)

			for i := range size {
				diff := stdmath.Abs(float64(simdOut[i] - scalarOut[i]))
				if diff > 1e-4 {
					t.Errorf("[%d] SIMD=%v, Scalar=%v, diff=%v", i, simdOut[i], scalarOut[i], diff)
				}
			}
		})
	}
}

func TestSwiGLU_KnownValues(t *testing.T) {
	// SwiGLU(0, x) = SiLU(0) * x = 0 * sigmoid(0) * x = 0
	gate := []float32{0, 0, 0}
	up := []float32{1, 2, 3}
	output := make([]float32, 3)
	SwiGLU(gate, up, output)
	for i, v := range output {
		if stdmath.Abs(float64(v)) > 1e-6 {
			t.Errorf("SwiGLU(0, %v) = %v, want 0", up[i], v)
		}
	}

	// SwiGLU(large_pos, 1) ≈ large_pos (since SiLU(x) → x for large x)
	gate2 := []float32{10}
	up2 := []float32{1}
	output2 := make([]float32, 1)
	SwiGLU(gate2, up2, output2)
	if stdmath.Abs(float64(output2[0]-10)) > 0.01 {
		t.Errorf("SwiGLU(10, 1) = %v, want ~10", output2[0])
	}
}

func TestParallelSwiGLU(t *testing.T) {
	pool := workerpool.New(0)
	defer pool.Close()

	rows, cols := 32, 64
	size := rows * cols
	gate := make([]float32, size)
	up := make([]float32, size)
	for i := range size {
		gate[i] = float32(i)*0.01 - float32(size)*0.005
		up[i] = float32(i)*0.005 + 0.5
	}

	seqOut := make([]float32, size)
	parOut := make([]float32, size)

	ParallelSwiGLU[float32](nil, gate, up, seqOut, rows, cols)
	ParallelSwiGLU(pool, gate, up, parOut, rows, cols)

	for i := range size {
		diff := stdmath.Abs(float64(seqOut[i] - parOut[i]))
		if diff > 1e-6 {
			t.Errorf("[%d] seq=%v, par=%v, diff=%v", i, seqOut[i], parOut[i], diff)
		}
	}
}

func TestSwiGLU_Empty(t *testing.T) {
	// Should not panic
	SwiGLU[float32](nil, nil, nil)
	SwiGLU([]float32{}, []float32{}, []float32{})
}

func BenchmarkSwiGLU(b *testing.B) {
	pool := workerpool.New(0)
	defer pool.Close()

	sizes := []struct{ rows, cols int }{
		{32, 256},
		{32, 768},
		{32, 3072},
	}

	for _, s := range sizes {
		size := s.rows * s.cols
		gate := make([]float32, size)
		up := make([]float32, size)
		output := make([]float32, size)
		for i := range size {
			gate[i] = float32(i) * 0.001
			up[i] = float32(i) * 0.001
		}

		b.Run(fmt.Sprintf("SIMD/%dx%d", s.rows, s.cols), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				ParallelSwiGLU(pool, gate, up, output, s.rows, s.cols)
			}
		})

		b.Run(fmt.Sprintf("Scalar/%dx%d", s.rows, s.cols), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				for r := range s.rows {
					off := r * s.cols
					SwiGLUScalar(gate[off:off+s.cols], up[off:off+s.cols], output[off:off+s.cols])
				}
			}
		})
	}
}
