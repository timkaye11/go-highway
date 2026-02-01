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

//go:build !noasm && darwin && arm64

// NOTE: This file is named "sme_blockedmatmul_arm64.go" (starting with 's')
// to ensure its init() runs AFTER "dispatch_blockedmatmul_arm64.gen.go" (starting with 'd').
// Go executes init() functions in lexicographic filename order within a package.
// The generated dispatch sets BlockedMatMulFloat32/64 to NEON; this file's init()
// must run afterward to override with the SME implementation when available.

package matmul

import (
	"unsafe"

	"github.com/ajroetker/go-highway/hwy"
)

// Blocked FMOPA implementation for SME.
//
// Combines the cache-tiled blocking strategy (48×48 blocks for L1 cache) with
// SME FMOPA outer product tiles (16×16 for f32, 8×8 for f64).
//
// Key optimizations:
//   - Enters streaming mode ONCE for the entire operation (vs per-tile)
//   - Pre-transposes A for contiguous column loads during FMOPA
//   - Cache-tiled to keep working set in L1 for large matrices
//
// Apple M4 SVL = 512 bits:
//   - f32: 16 lanes, 16×16 tiles, 512 FLOPs per FMOPA
//   - f64: 8 lanes, 8×8 tiles, 128 FLOPs per FMOPA

// Minimum dimensions to use SME blocked FMOPA.
// Below this, the streaming mode overhead outweighs the benefits.
const minDimForBlockedSME = 64

//go:noescape
func blockedmatmul_fmopa_at_f32(at, b, c unsafe.Pointer, m, n, k int64)

//go:noescape
func blockedmatmul_fmopa_at_f64(at, b, c unsafe.Pointer, m, n, k int64)

// blockedMatMulFMOPA uses ARM SME FMOPA for blocked matrix multiplication (f32).
// Uses outer product accumulate with ZA tiles and cache-tiled blocking.
// Pre-transposes A for contiguous column access, enabling fast vector loads.
func blockedMatMulFMOPA(a, b, c []float32, m, n, k int) {
	// For non-aligned sizes (16×16 tiles for f32), fall back to NEON
	if m%16 != 0 || n%16 != 0 || k%16 != 0 {
		BaseBlockedMatMul_neon(a, b, c, m, n, k)
		return
	}

	// For small matrices, NEON is faster (streaming mode has overhead)
	if m < minDimForBlockedSME || n < minDimForBlockedSME || k < minDimForBlockedSME {
		BaseBlockedMatMul_neon(a, b, c, m, n, k)
		return
	}

	// Get transpose buffer from pool
	atSize := m * k
	atBuf := transposePool32.Get().([]float32)
	if cap(atBuf) < atSize {
		atBuf = make([]float32, atSize)
	} else {
		atBuf = atBuf[:atSize]
	}

	// Transpose A (M×K) to AT (K×M) for contiguous column access
	transposeMatrix(a, m, k, atBuf)

	// Call blocked FMOPA with transposed A
	blockedmatmul_fmopa_at_f32(
		unsafe.Pointer(unsafe.SliceData(atBuf)),
		unsafe.Pointer(unsafe.SliceData(b)),
		unsafe.Pointer(unsafe.SliceData(c)),
		int64(m),
		int64(n),
		int64(k),
	)

	// Return buffer to pool
	transposePool32.Put(atBuf)
}

// blockedMatMulFMOPA64 uses ARM SME FMOPA for blocked matrix multiplication (f64).
// Uses outer product accumulate with ZA tiles (8×8 for f64) and cache-tiled blocking.
// Pre-transposes A for contiguous column access, enabling fast vector loads.
func blockedMatMulFMOPA64(a, b, c []float64, m, n, k int) {
	// For non-aligned sizes (8×8 tiles for f64), fall back to NEON
	if m%8 != 0 || n%8 != 0 || k%8 != 0 {
		BaseBlockedMatMul_neon_Float64(a, b, c, m, n, k)
		return
	}

	// For small matrices, NEON is faster (streaming mode has overhead)
	if m < minDimForBlockedSME || n < minDimForBlockedSME || k < minDimForBlockedSME {
		BaseBlockedMatMul_neon_Float64(a, b, c, m, n, k)
		return
	}

	// Get transpose buffer from pool
	atSize := m * k
	atBuf := transposePool64.Get().([]float64)
	if cap(atBuf) < atSize {
		atBuf = make([]float64, atSize)
	} else {
		atBuf = atBuf[:atSize]
	}

	// Transpose A (M×K) to AT (K×M) for contiguous column access
	transposeMatrix(a, m, k, atBuf)

	// Call blocked FMOPA with transposed A
	blockedmatmul_fmopa_at_f64(
		unsafe.Pointer(unsafe.SliceData(atBuf)),
		unsafe.Pointer(unsafe.SliceData(b)),
		unsafe.Pointer(unsafe.SliceData(c)),
		int64(m),
		int64(n),
		int64(k),
	)

	// Return buffer to pool
	transposePool64.Put(atBuf)
}

func init() {
	if hwy.HasSME() {
		// Use blocked FMOPA implementation which works on Apple M4
		// This overrides the generated dispatch for large aligned matrices
		BlockedMatMulFloat32 = blockedMatMulFMOPA
		BlockedMatMulFloat64 = blockedMatMulFMOPA64
	}
}
