//go:build !noasm && darwin && arm64

#include "textflag.h"

// func matmul_fmopa_at_f32(at, b, c unsafe.Pointer, m, n, k int64)
//
// FMOPA-based matrix multiplication with A TRANSPOSED: C = AT^T * B = A * B
//   AT is K x M (row-major) = A transposed
//   B is K x N (row-major)
//   C is M x N (row-major)
//
// With A transposed, loading A columns becomes contiguous!
//   A[i:i+16, k] = AT[k, i:i+16] = contiguous in memory
//
// This eliminates the slow scalar loop for building A column vectors.
//
// Apple M4 SVL = 512 bits = 16 × float32
// ZA tiles = 16×16 = 256 float32 elements
// Single FMOPA does 16×16×2 = 512 FLOPs
//
// Register usage:
//   R19: AT base pointer
//   R20: B base pointer
//   R21: C base pointer
//   R22: M dimension
//   R23: N dimension
//   R24: K dimension
//   R25: N * 4 (B and C row stride)
//   R26: M * 4 (AT row stride)
//   R0: i (tile row index)
//   R2: j (tile column index)
//   R3: kk (k loop index)
//   R10: zero register for LD1W offset
//
TEXT ·matmul_fmopa_at_f32(SB), NOSPLIT, $64-48
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

	// Calculate strides (in bytes)
	LSL	$2, R23, R25       // R25 = N * 4 (B and C row stride)
	LSL	$2, R22, R26       // R26 = M * 4 (AT row stride)

	// Enter streaming mode with ZA enabled
	WORD	$0xd503477f        // smstart

	// Set up predicates for 32-bit elements
	WORD	$0x2598e3e0        // ptrue p0.s
	WORD	$0x2598e3e1        // ptrue p1.s

	// Zero R10 for use as offset register in LD1W
	MOVD	$0, R10

	// Tile loop: process 16x16 tiles of C
	// Outer loop: for i = 0; i < M; i += 16
	MOVD	$0, R0             // R0 = i (row tile index)

tile_row_loop_at:
	CMP	R22, R0            // i - M
	BGE	done_tiles_at

	// Inner loop: for j = 0; j < N; j += 16
	MOVD	$0, R2             // R2 = j (column tile index)

tile_col_loop_at:
	CMP	R23, R2            // j - N
	BGE	next_tile_row_at

	// Refresh predicates
	WORD	$0x2598e3e0        // ptrue p0.s
	WORD	$0x2598e3e1        // ptrue p1.s

	// Zero all ZA tiles for accumulation
	WORD	$0xc00800ff        // zero {za}

	// K loop: accumulate outer products
	MOVD	$0, R3             // R3 = kk

k_loop_at:
	CMP	R24, R3            // kk - K
	BGE	store_tile_at

	// === Load A column vector: A[i:i+16, kk] ===
	// With AT (K x M row-major), A[i:i+16, kk] = AT[kk, i:i+16]
	// This is CONTIGUOUS at: AT_base + kk * M * 4 + i * 4
	MUL	R26, R3, R4        // R4 = kk * M * 4
	ADD	R19, R4, R4        // R4 = &AT[kk, 0]
	LSL	$2, R0, R5         // R5 = i * 4
	ADD	R4, R5, R4         // R4 = &AT[kk, i] = &A[i, kk] (contiguous!)

	// LD1W z2.s, p0/z, [x4, x10, lsl #2] - contiguous load of A column!
	// Scalar+scalar with Rm=x10 (zero): 0xa5404000 + (10 << 16) + (4 << 5) + 2
	WORD	$0xa54a4082        // ld1w {z2.s}, p0/z, [x4, x10, lsl #2]

	// === Load B row vector: B[kk, j:j+16] (contiguous) ===
	MUL	R25, R3, R6        // R6 = kk * N * 4
	ADD	R20, R6, R6        // R6 = &B[kk, 0]
	LSL	$2, R2, R7         // R7 = j * 4
	ADD	R6, R7, R6         // R6 = &B[kk, j]

	// LD1W z0.s, p0/z, [x6, x10, lsl #2]
	WORD	$0xa54a40c0        // ld1w {z0.s}, p0/z, [x6, x10, lsl #2]

	// === FMOPA: za0 += z2 ⊗ z0 (outer product) ===
	// fmopa za0.s, p0/m, p1/m, z2.s, z0.s
	WORD	$0x80802040

	ADD	$1, R3, R3
	B	k_loop_at

store_tile_at:
	// Extract za0 rows and store to C
	// Calculate C pointer: &C[i, j]
	MUL	R25, R0, R4        // R4 = i * N * 4
	ADD	R21, R4, R4        // R4 = &C[i, 0]
	LSL	$2, R2, R5         // R5 = j * 4
	ADD	R4, R5, R4         // R4 = &C[i, j]

	// Extract and store all 16 rows of za0
	MOVD	$0, R12            // W12 = slice index
	MOVD	R4, R6             // R6 = current C row pointer

store_rows_at:
	CMP	$16, R12
	BGE	next_tile_col_at

	// MOVA z2.s, p0/m, za0h.s[w12, 0]
	WORD	$0xc0820002

	// ST1W {z2.s}, p0, [x6]
	WORD	$0xe540e0c2

	// Move to next row
	ADD	R25, R6, R6        // Next C row (stride = N*4)
	ADD	$1, R12, R12       // Next ZA slice
	B	store_rows_at

next_tile_col_at:
	ADD	$16, R2, R2
	B	tile_col_loop_at

next_tile_row_at:
	ADD	$16, R0, R0
	B	tile_row_loop_at

done_tiles_at:
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
