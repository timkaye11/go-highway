//go:build !noasm && darwin && arm64

#include "textflag.h"

// func matmul_fmopa_at_f64(at, b, c unsafe.Pointer, m, n, k int64)
//
// FMOPA-based matrix multiplication for float64 with A TRANSPOSED: C = AT^T * B = A * B
//   AT is K x M (row-major) = A transposed
//   B is K x N (row-major)
//   C is M x N (row-major)
//
// With A transposed, loading A columns becomes contiguous!
//   A[i:i+8, k] = AT[k, i:i+8] = contiguous in memory
//
// Apple M4 SVL = 512 bits = 8 × float64
// ZA tiles for float64 = 8×8 = 64 float64 elements
// Single FMOPA does 8×8×2 = 128 FP64 ops
//
// Key encoding differences from float32:
//   - FMOPA za0.d: 0x80C0xxxx (bit 22 set for double)
//   - MOVA za0h.d to z: 0xc0c2xxxx (size bits 23:22 = 11 for 64-bit)
//   - LD1D: 0xa5eaxxxx (double-word load with scalar+scalar)
//   - ST1D: 0xe5e0xxxx (double-word store)
//   - PTRUE p0.d: 0x25d8e3e0 (64-bit predicate)
//
// Register usage:
//   R19: AT base pointer
//   R20: B base pointer
//   R21: C base pointer
//   R22: M dimension
//   R23: N dimension
//   R24: K dimension
//   R25: N * 8 (B and C row stride in bytes)
//   R26: M * 8 (AT row stride in bytes)
//   R0: i (tile row index)
//   R2: j (tile column index)
//   R3: kk (k loop index)
//   R10: zero register for LD1D offset
//
TEXT ·matmul_fmopa_at_f64(SB), NOSPLIT, $64-48
	// Save callee-saved registers
	MOVD	R19, 0(RSP)
	MOVD	R20, 8(RSP)
	MOVD	R21, 16(RSP)
	MOVD	R22, 24(RSP)
	MOVD	R23, 32(RSP)
	MOVD	R24, 40(RSP)
	MOVD	R25, 48(RSP)
	MOVD	R26, 56(RSP)

	// Load arguments
	MOVD	at+0(FP), R19     // AT matrix base (K x M, row-major)
	MOVD	b+8(FP), R20      // B matrix base
	MOVD	c+16(FP), R21     // C matrix base
	MOVD	m+24(FP), R22     // M dimension
	MOVD	n+32(FP), R23     // N dimension
	MOVD	k+40(FP), R24     // K dimension

	// Calculate strides (in bytes) - float64 is 8 bytes
	LSL	$3, R23, R25       // R25 = N * 8 (B and C row stride)
	LSL	$3, R22, R26       // R26 = M * 8 (AT row stride)

	// Enter streaming mode with ZA enabled
	WORD	$0xd503477f        // smstart

	// Set up predicates for 64-bit elements
	WORD	$0x25d8e3e0        // ptrue p0.d
	WORD	$0x25d8e3e1        // ptrue p1.d

	// Zero R10 for use as offset register in LD1D
	MOVD	$0, R10

	// Tile loop: process 8x8 tiles of C (float64 tiles)
	// Outer loop: for i = 0; i < M; i += 8
	MOVD	$0, R0             // R0 = i (row tile index)

tile_row_loop_f64:
	CMP	R22, R0            // i - M
	BGE	done_tiles_f64

	// Inner loop: for j = 0; j < N; j += 8
	MOVD	$0, R2             // R2 = j (column tile index)

tile_col_loop_f64:
	CMP	R23, R2            // j - N
	BGE	next_tile_row_f64

	// Refresh predicates
	WORD	$0x25d8e3e0        // ptrue p0.d
	WORD	$0x25d8e3e1        // ptrue p1.d

	// Zero all ZA tiles for accumulation
	WORD	$0xc00800ff        // zero {za}

	// K loop: accumulate outer products
	MOVD	$0, R3             // R3 = kk

k_loop_f64:
	CMP	R24, R3            // kk - K
	BGE	store_tile_f64

	// === Load A column vector: A[i:i+8, kk] ===
	// With AT (K x M row-major), A[i:i+8, kk] = AT[kk, i:i+8]
	// This is CONTIGUOUS at: AT_base + kk * M * 8 + i * 8
	MUL	R26, R3, R4        // R4 = kk * M * 8
	ADD	R19, R4, R4        // R4 = &AT[kk, 0]
	LSL	$3, R0, R5         // R5 = i * 8
	ADD	R4, R5, R4         // R4 = &AT[kk, i] = &A[i, kk] (contiguous!)

	// LD1D z2.d, p0/z, [x4, x10, lsl #3] - contiguous load of A column!
	// Scalar+scalar with Rm=x10 (zero): 0xa5e04000 + (10 << 16) + (4 << 5) + 2
	WORD	$0xa5ea4082        // ld1d {z2.d}, p0/z, [x4, x10, lsl #3]

	// === Load B row vector: B[kk, j:j+8] (contiguous) ===
	MUL	R25, R3, R6        // R6 = kk * N * 8
	ADD	R20, R6, R6        // R6 = &B[kk, 0]
	LSL	$3, R2, R7         // R7 = j * 8
	ADD	R6, R7, R6         // R6 = &B[kk, j]

	// LD1D z0.d, p0/z, [x6, x10, lsl #3]
	WORD	$0xa5ea40c0        // ld1d {z0.d}, p0/z, [x6, x10, lsl #3]

	// === FMOPA: za0 += z2 ⊗ z0 (outer product, float64) ===
	// fmopa za0.d, p0/m, p1/m, z2.d, z0.d
	// FP32 encoding: 0x80802040, FP64 sets bit 22: 0x80C02040
	WORD	$0x80c02040

	ADD	$1, R3, R3
	B	k_loop_f64

store_tile_f64:
	// Extract za0 rows and store to C
	// Calculate C pointer: &C[i, j]
	MUL	R25, R0, R4        // R4 = i * N * 8
	ADD	R21, R4, R4        // R4 = &C[i, 0]
	LSL	$3, R2, R5         // R5 = j * 8
	ADD	R4, R5, R4         // R4 = &C[i, j]

	// Extract and store all 8 rows of za0
	MOVD	$0, R12            // W12 = slice index
	MOVD	R4, R6             // R6 = current C row pointer

store_rows_f64:
	CMP	$8, R12
	BGE	next_tile_col_f64

	// MOVA z2.d, p0/m, za0h.d[w12, 0]
	// Float32: 0xc0820002, Float64 sets bit 22: 0xc0C20002
	WORD	$0xc0c20002

	// ST1D {z2.d}, p0, [x6]
	// Float32 ST1W: 0xe540e0c2, Float64 ST1D: 0xe5e0e0c2
	WORD	$0xe5e0e0c2

	// Move to next row
	ADD	R25, R6, R6        // Next C row (stride = N*8)
	ADD	$1, R12, R12       // Next ZA slice
	B	store_rows_f64

next_tile_col_f64:
	ADD	$8, R2, R2
	B	tile_col_loop_f64

next_tile_row_f64:
	ADD	$8, R0, R0
	B	tile_row_loop_f64

done_tiles_f64:
	// Exit streaming mode
	WORD	$0xd503467f        // smstop

	// Restore callee-saved registers
	MOVD	0(RSP), R19
	MOVD	8(RSP), R20
	MOVD	16(RSP), R21
	MOVD	24(RSP), R22
	MOVD	32(RSP), R23
	MOVD	40(RSP), R24
	MOVD	48(RSP), R25
	MOVD	56(RSP), R26

	RET
