//go:build !noasm && darwin && arm64

#include "textflag.h"

// func blockedmatmul_fmopa_at_f64(at, b, c unsafe.Pointer, m, n, k int64)
//
// Blocked FMOPA-based matrix multiplication for float64 with A TRANSPOSED: C = AT^T * B = A * B
// Combines cache-tiled blocking (48×48 blocks for M/N) with FMOPA outer product (8×8 tiles).
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
// Block size = 48 (tuned for L1 cache, multiple of 8 for FMOPA tiles)
// FMOPA tile size = 8 (Apple M4 SVL = 512 bits = 8 × float64)
//
// Key optimization: enters streaming mode ONCE and processes all blocks,
// minimizing the streaming mode entry/exit overhead.
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
//   R0: i0 (block row start)
//   R1: j0 (block column start)
//   R3: i (tile row, actual row = i0 + ti*8)
//   R4: j (tile col, actual col = j0 + tj*8)
//   R5: p (k loop index, 0 to K)
//   R10: zero register for LD1D offset
//   R11-R17: temporaries
//   Note: R27 (g) and R28 (g0) are reserved by Go - do not use
//
TEXT ·blockedmatmul_fmopa_at_f64(SB), NOSPLIT, $64-48
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

	// Enter streaming mode with ZA enabled (done ONCE for all blocks)
	WORD	$0xd503477f        // smstart

	// Set up predicates for 64-bit elements
	WORD	$0x25d8e3e0        // ptrue p0.d
	WORD	$0x25d8e3e1        // ptrue p1.d

	// Zero R10 for use as offset register
	MOVD	$0, R10

	// Block loop: i0 over M (blocks of 48)
	MOVD	$0, R0             // R0 = i0

block_i_loop_f64:
	CMP	R22, R0            // i0 - M
	BGE	done_all_blocks_f64

	// Calculate i block end (clamped to M)
	ADD	$48, R0, R11       // i0 + 48
	CMP	R22, R11
	CSEL	LT, R11, R22, R11  // iEnd = min(i0+48, M)

	// Block loop: j0 over N (blocks of 48)
	MOVD	$0, R1             // R1 = j0

block_j_loop_f64:
	CMP	R23, R1            // j0 - N
	BGE	next_block_i_f64

	// Calculate j block end (clamped to N)
	ADD	$48, R1, R12       // j0 + 48
	CMP	R23, R12
	CSEL	LT, R12, R23, R12  // jEnd = min(j0+48, N)

	// Tile loop: i over tiles in this block (step by 8 for f64)
	MOVD	R0, R3             // R3 = i, starts at i0

tile_i_loop_f64:
	CMP	R11, R3            // i - iEnd
	BGE	next_block_j_f64

	// Tile loop: j over tiles in this block (step by 8 for f64)
	MOVD	R1, R4             // R4 = j, starts at j0

tile_j_loop_f64:
	CMP	R12, R4            // j - jEnd
	BGE	next_tile_i_f64

	// Refresh predicates (may have been corrupted)
	WORD	$0x25d8e3e0        // ptrue p0.d
	WORD	$0x25d8e3e1        // ptrue p1.d

	// Zero ZA for accumulation (we process ALL of K for each tile)
	WORD	$0xc00800ff        // zero {za}

	// K loop: process ALL K values for this tile
	MOVD	$0, R5             // R5 = p (k index)

k_loop_f64:
	CMP	R24, R5            // p - K
	BGE	store_tile_f64

	// === Load A column: AT[p, i:i+8] (contiguous due to transpose) ===
	MUL	R26, R5, R14       // R14 = p * M * 8
	ADD	R19, R14, R14      // R14 = &AT[p, 0]
	LSL	$3, R3, R15        // R15 = i * 8
	ADD	R14, R15, R14      // R14 = &AT[p, i]

	// LD1D z2.d, p0/z, [x14, x10, lsl #3]
	WORD	$0xa5ea41c2        // ld1d {z2.d}, p0/z, [x14, x10, lsl #3]

	// === Load B row: B[p, j:j+8] (contiguous) ===
	MUL	R25, R5, R15       // R15 = p * N * 8
	ADD	R20, R15, R15      // R15 = &B[p, 0]
	LSL	$3, R4, R16        // R16 = j * 8
	ADD	R15, R16, R15      // R15 = &B[p, j]

	// LD1D z0.d, p0/z, [x15, x10, lsl #3]
	WORD	$0xa5ea41e0        // ld1d {z0.d}, p0/z, [x15, x10, lsl #3]

	// === FMOPA: za0 += z2 ⊗ z0 (float64) ===
	// fmopa za0.d, p0/m, p1/m, z2.d, z0.d
	WORD	$0x80c02040

	ADD	$1, R5, R5
	B	k_loop_f64

store_tile_f64:
	// === Store ZA tile to C ===
	// IMPORTANT: R12 holds jEnd, so we must NOT overwrite it
	// Use R9 for the slice counter instead
	MUL	R25, R3, R14       // R14 = i * N * 8
	ADD	R21, R14, R14      // R14 = &C[i, 0]
	LSL	$3, R4, R15        // R15 = j * 8
	ADD	R14, R15, R14      // R14 = &C[i, j]

	MOVD	$0, R9             // R9 = slice index

store_rows_f64:
	CMP	$8, R9
	BGE	next_tile_j_f64

	// MOVA z2.d, p0/m, za0h.d[w12, 0]
	// Need to use W12 (bits 12:10 of encoding)
	// But we can't corrupt R12, so we save/restore it
	MOVD	R12, R8            // Save jEnd
	MOVD	R9, R12            // W12 = slice index
	WORD	$0xc0c20002
	MOVD	R8, R12            // Restore jEnd

	// ST1D {z2.d}, p0, [x14]
	WORD	$0xe5e0e1c2

	ADD	R25, R14, R14      // Next C row
	ADD	$1, R9, R9
	B	store_rows_f64

next_tile_j_f64:
	ADD	$8, R4, R4
	B	tile_j_loop_f64

next_tile_i_f64:
	ADD	$8, R3, R3
	B	tile_i_loop_f64

next_block_j_f64:
	ADD	$48, R1, R1
	B	block_j_loop_f64

next_block_i_f64:
	ADD	$48, R0, R0
	B	block_i_loop_f64

done_all_blocks_f64:
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
