//go:build !noasm && darwin && arm64

#include "textflag.h"

// func matmul_fmopa_f32(a, b, c unsafe.Pointer, m, n, k int64)
//
// FMOPA-based matrix multiplication: C = A * B
//   A is M x K (row-major)
//   B is K x N (row-major)
//   C is M x N (row-major)
//
// Uses ARM SME FMOPA instruction for outer product accumulation.
// Processes 16x16 tiles.
//
// Apple M4 SVL = 512 bits = 16 × float32
// ZA tiles = 16×16 = 256 float32 elements
// Single FMOPA does 16×16×2 = 512 FLOPs
//
// CRITICAL: On Apple M4, tile-to-vector MOVA only reliably reads za0!
//   Tile-to-vector encodings 0xc0820001, 0xc0820005, etc all read za0.
//   So we MUST use za0 for FMOPA accumulation (need to read it at the end).
//
// MOVA encodings (32-bit):
//   Vector-to-tile: tile in bits 1:0
//     za0h: 0xc0800000, za1h: 0xc0800001
//   Tile-to-vector: always reads za0!
//     za0h to z0: 0xc0820000, za0h to z2: 0xc0820002
//     za0v to z0: 0xc0828000 (bit 15 for vertical)
//
// Strategy:
//   - Use stack to build A column vector (16 strided scalars → stack → ld1w)
//   - Use ld1w to load B row (contiguous)
//   - FMOPA to za0, read za0 at end
//
// Register usage:
//   za0: FMOPA accumulation
//   z0: B vector
//   z2: A vector (from stack)
//   R9: pointer to stack temp area (scratch register)
//   RSP+0..63: callee-saved registers (R19-R26)
//   RSP+64..127: A column temp storage (16 x float32)
//
TEXT ·matmul_fmopa_f32(SB), NOSPLIT, $128-48
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
	MOVD	a+0(FP), R19      // A matrix base
	MOVD	b+8(FP), R20      // B matrix base
	MOVD	c+16(FP), R21     // C matrix base
	MOVD	m+24(FP), R22     // M dimension
	MOVD	n+32(FP), R23     // N dimension
	MOVD	k+40(FP), R24     // K dimension

	// Calculate strides (in bytes)
	LSL	$2, R23, R25       // R25 = N * 4 (B and C row stride)
	LSL	$2, R24, R26       // R26 = K * 4 (A row stride)

	// Stack pointer for A column temp storage (RSP+64)
	ADD	$64, RSP, R9       // R9 = stack temp area (scratch register)

	// Enter streaming mode with ZA enabled
	WORD	$0xd503477f        // smstart

	// Set up predicates for 32-bit elements
	WORD	$0x2598e3e0        // ptrue p0.s
	WORD	$0x2598e3e1        // ptrue p1.s

	// Tile loop: process 16x16 tiles of C
	// Outer loop: for i = 0; i < M; i += 16
	MOVD	$0, R0             // R0 = i (row tile index)

tile_row_loop:
	CMP	R22, R0            // i - M
	BGE	done_tiles

	// Inner loop: for j = 0; j < N; j += 16
	MOVD	$0, R2             // R2 = j (column tile index)

tile_col_loop:
	CMP	R23, R2            // j - N
	BGE	next_tile_row

	// Refresh predicates (in case they got corrupted)
	WORD	$0x2598e3e0        // ptrue p0.s
	WORD	$0x2598e3e1        // ptrue p1.s

	// Zero all ZA tiles for accumulation
	WORD	$0xc00800ff        // zero {za} - all tiles

	// K loop: accumulate outer products
	MOVD	$0, R3             // R3 = kk

k_loop:
	CMP	R24, R3            // kk - K
	BGE	store_tile

	// Zero R10 for use as offset register in LD1W scalar+scalar form
	// (R0 contains tile row index which would corrupt addresses)
	MOVD	$0, R10

	// === Build A column vector: A[i:i+16, kk] ===
	// Load 16 strided scalars to stack, then vector load

	// Calculate A column base: &A[i, kk]
	MUL	R26, R0, R4        // R4 = i * K * 4
	ADD	R19, R4, R4        // R4 = &A[i, 0]
	LSL	$2, R3, R5         // R5 = kk * 4
	ADD	R4, R5, R4         // R4 = &A[i, kk]

	// Load 16 scalars from A column to stack
	MOVD	$0, R12            // row index
	MOVD	R4, R6             // current A pointer
	MOVD	R9, R8             // stack pointer

build_a_stack:
	CMP	$16, R12
	BGE	load_a_vector

	MOVW	(R6), R5           // Load A[i+row, kk]
	MOVW	R5, (R8)           // Store to stack

	ADD	R26, R6, R6        // Next A row (stride = K*4)
	ADD	$4, R8, R8         // Next stack slot
	ADD	$1, R12, R12
	B	build_a_stack

load_a_vector:
	// LD1W z2.s, p0/z, [x9, x10, lsl #2] - load A vector from stack
	// Use x10 (zeroed above) as offset register instead of x0 (which contains tile row index)
	// Scalar+scalar with Rm=x10: 0xa5404000 + (10 << 16) + (9 << 5) + 2 = 0xa54a4122
	WORD	$0xa54a4122        // ld1w {z2.s}, p0/z, [x9, x10, lsl #2]

	// === Load B row vector: B[kk, j:j+16] (contiguous) ===
	// Calculate B row base: &B[kk, j]
	MUL	R25, R3, R6        // R6 = kk * N * 4
	ADD	R20, R6, R6        // R6 = &B[kk, 0]
	LSL	$2, R2, R7         // R7 = j * 4
	ADD	R6, R7, R6         // R6 = &B[kk, j]

	// LD1W z0.s, p0/z, [x6, x10, lsl #2] - load B vector directly
	// Use x10 (zeroed above) as offset register instead of x0 (which contains tile row index)
	// Scalar+scalar with Rm=x10: 0xa5404000 + (10 << 16) + (6 << 5) + 0 = 0xa54a40c0
	WORD	$0xa54a40c0        // ld1w {z0.s}, p0/z, [x6, x10, lsl #2]

	// === FMOPA: za0 += z2 ⊗ z0 (outer product) ===
	// fmopa za0.s, p0/m, p1/m, z2.s, z0.s
	// ZAda=0, Pn=0, Pm=1, Zn=2, Zm=0
	// 0x80800000 | (0 << 16) | (1 << 13) | (2 << 5) | 0 = 0x80802040
	WORD	$0x80802040

	ADD	$1, R3, R3
	B	k_loop

store_tile:
	// Extract za0 rows and store to C
	// C[i+row, j:j+16] = za0[row, :]

	// Calculate C pointer: &C[i, j]
	MUL	R25, R0, R4        // R4 = i * N * 4
	ADD	R21, R4, R4        // R4 = &C[i, 0]
	LSL	$2, R2, R5         // R5 = j * 4
	ADD	R4, R5, R4         // R4 = &C[i, j]

	// Extract and store all 16 rows of za0
	MOVD	$0, R12            // W12 = slice index
	MOVD	R4, R6             // R6 = current C row pointer

store_rows:
	CMP	$16, R12
	BGE	next_tile_col

	// MOVA z2.s, p0/m, za0h.s[w12, 0]
	// za0h to z2: 0xc0820002 (using z2 like the working test)
	WORD	$0xc0820002

	// ST1W {z2.s}, p0, [x6]
	WORD	$0xe540e0c2

	// Move to next row
	ADD	R25, R6, R6        // Next C row (stride = N*4)
	ADD	$1, R12, R12       // Next ZA slice
	B	store_rows

next_tile_col:
	ADD	$16, R2, R2
	B	tile_col_loop

next_tile_row:
	ADD	$16, R0, R0
	B	tile_row_loop

done_tiles:
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
