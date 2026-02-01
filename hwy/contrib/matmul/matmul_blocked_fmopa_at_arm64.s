//go:build !noasm && darwin && arm64

#include "textflag.h"

// func blockedmatmul_fmopa_at_f32(at, b, c unsafe.Pointer, m, n, k int64)
//
// Blocked FMOPA-based matrix multiplication with A TRANSPOSED: C = AT^T * B = A * B
// Combines cache-tiled blocking (48×48 blocks for M/N) with FMOPA outer product (16×16 tiles).
//
// IMPORTANT: Only blocks over M and N, NOT K. This is critical for SME performance:
// - Blocking over K would require loading C into ZA for every k-block (slow!)
// - Instead, we process ALL of K for each tile, accumulating in ZA
// - M/N blocking ensures C tiles stay in cache
//
//   AT is K x M (row-major) = A transposed
//   B is K x N (row-major)
//   C is M x N (row-major)
//
// Block size = 48 (tuned for L1 cache, multiple of 16 for FMOPA tiles)
// FMOPA tile size = 16 (Apple M4 SVL = 512 bits = 16 × float32)
//
// Key optimization:
// - Enters streaming mode ONCE and processes all blocks
// - Uses incremental pointer updates in k-loop (avoids MUL per iteration)
//
// Register usage:
//   R19: AT base pointer
//   R20: B base pointer
//   R21: C base pointer
//   R22: M dimension
//   R23: N dimension
//   R24: K dimension
//   R25: N * 4 (B and C row stride in bytes)
//   R26: M * 4 (AT row stride in bytes)
//   R0: i0 (block row start)
//   R1: j0 (block column start)
//   R3: i (tile row, actual row = i0 + ti*16)
//   R4: j (tile col, actual col = j0 + tj*16)
//   R5: AT pointer for current k (incremented by R26 each iteration)
//   R6: B pointer for current k (incremented by R25 each iteration)
//   R7: k loop counter
//   R10: zero register for LD1W offset
//   R11-R17: temporaries
//   Note: R27 (g) and R28 (g0) are reserved by Go - do not use
//
TEXT ·blockedmatmul_fmopa_at_f32(SB), NOSPLIT, $64-48
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

	// Enter streaming mode with ZA enabled (done ONCE for all blocks)
	WORD	$0xd503477f        // smstart

	// Set up predicates for 32-bit elements
	WORD	$0x2598e3e0        // ptrue p0.s
	WORD	$0x2598e3e1        // ptrue p1.s

	// Zero R10 for use as offset register
	MOVD	$0, R10

	// Block loop: i0 over M (blocks of 48)
	MOVD	$0, R0             // R0 = i0

block_i_loop:
	CMP	R22, R0            // i0 - M
	BGE	done_all_blocks

	// Calculate i block end (clamped to M)
	ADD	$48, R0, R11       // i0 + 48
	CMP	R22, R11
	CSEL	LT, R11, R22, R11  // iEnd = min(i0+48, M)

	// Block loop: j0 over N (blocks of 48)
	MOVD	$0, R1             // R1 = j0

block_j_loop:
	CMP	R23, R1            // j0 - N
	BGE	next_block_i

	// Calculate j block end (clamped to N)
	ADD	$48, R1, R12       // j0 + 48
	CMP	R23, R12
	CSEL	LT, R12, R23, R12  // jEnd = min(j0+48, N)

	// Tile loop: i over tiles in this block (step by 16)
	MOVD	R0, R3             // R3 = i, starts at i0

tile_i_loop:
	CMP	R11, R3            // i - iEnd
	BGE	next_block_j

	// Tile loop: j over tiles in this block (step by 16)
	MOVD	R1, R4             // R4 = j, starts at j0

tile_j_loop:
	CMP	R12, R4            // j - jEnd
	BGE	next_tile_i

	// Refresh predicates (may have been corrupted)
	WORD	$0x2598e3e0        // ptrue p0.s
	WORD	$0x2598e3e1        // ptrue p1.s

	// Zero ZA for accumulation (we process ALL of K for each tile)
	WORD	$0xc00800ff        // zero {za}

	// Initialize pointers for k-loop (computed ONCE per tile)
	// AT pointer: &AT[0, i] = AT_base + i * 4
	LSL	$2, R3, R5         // R5 = i * 4
	ADD	R19, R5, R5        // R5 = &AT[0, i]

	// B pointer: &B[0, j] = B_base + j * 4
	LSL	$2, R4, R6         // R6 = j * 4
	ADD	R20, R6, R6        // R6 = &B[0, j]

	// K loop: process ALL K values for this tile
	MOVD	$0, R7             // R7 = k (loop counter)

k_loop:
	CMP	R24, R7            // k - K
	BGE	store_tile

	// Load A column: AT[k, i:i+16] using current AT pointer R5
	// LD1W z2.s, p0/z, [x5, x10, lsl #2]
	WORD	$0xa54a40a2        // ld1w {z2.s}, p0/z, [x5, x10, lsl #2]

	// Load B row: B[k, j:j+16] using current B pointer R6
	// LD1W z0.s, p0/z, [x6, x10, lsl #2]
	WORD	$0xa54a40c0        // ld1w {z0.s}, p0/z, [x6, x10, lsl #2]

	// FMOPA: za0 += z2 ⊗ z0
	WORD	$0x80802040

	// Advance pointers to next k row
	ADD	R26, R5, R5        // AT_ptr += M * 4 (next row of AT)
	ADD	R25, R6, R6        // B_ptr += N * 4 (next row of B)

	ADD	$1, R7, R7
	B	k_loop

store_tile:
	// Store ZA tile to C
	// IMPORTANT: R12 holds jEnd, so we must NOT overwrite it
	// Use R9 for the slice counter instead
	MUL	R25, R3, R14       // R14 = i * N * 4
	ADD	R21, R14, R14      // R14 = &C[i, 0]
	LSL	$2, R4, R15        // R15 = j * 4
	ADD	R14, R15, R14      // R14 = &C[i, j]

	MOVD	$0, R9             // R9 = slice index

store_rows:
	CMP	$16, R9
	BGE	next_tile_j

	// MOVA z2.s, p0/m, za0h.s[w12, 0]
	// Need to use W12 (bits 12:10 of encoding)
	// But we can't corrupt R12, so we save/restore it
	MOVD	R12, R8            // Save jEnd
	MOVD	R9, R12            // W12 = slice index
	WORD	$0xc0820002
	MOVD	R8, R12            // Restore jEnd

	// ST1W {z2.s}, p0, [x14]
	WORD	$0xe540e1c2

	ADD	R25, R14, R14      // Next C row
	ADD	$1, R9, R9
	B	store_rows

next_tile_j:
	ADD	$16, R4, R4
	B	tile_j_loop

next_tile_i:
	ADD	$16, R3, R3
	B	tile_i_loop

next_block_j:
	ADD	$48, R1, R1
	B	block_j_loop

next_block_i:
	ADD	$48, R0, R0
	B	block_i_loop

done_all_blocks:
	// Exit streaming mode (done ONCE after all blocks)
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
