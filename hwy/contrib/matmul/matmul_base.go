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

package matmul

// matmulScalar is the pure Go scalar implementation.
// C[i,j] = sum(A[i,p] * B[p,j]) for p in 0..K-1
// This is kept for reference and benchmarking; the generated BaseMatMul_fallback
// is used as the actual fallback implementation.
func matmulScalar(a, b, c []float32, m, n, k int) {
	// Clear output
	for i := range c[:m*n] {
		c[i] = 0
	}

	// Standard triple-loop matrix multiply
	for i := range m {
		for p := range k {
			aip := a[i*k+p]
			for j := range n {
				c[i*n+j] += aip * b[p*n+j]
			}
		}
	}
}

// matmulScalar64 is the pure Go scalar implementation for float64.
func matmulScalar64(a, b, c []float64, m, n, k int) {
	// Clear output
	for i := range c[:m*n] {
		c[i] = 0
	}

	// Standard triple-loop matrix multiply
	for i := range m {
		for p := range k {
			aip := a[i*k+p]
			for j := range n {
				c[i*n+j] += aip * b[p*n+j]
			}
		}
	}
}
