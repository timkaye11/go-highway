//go:build !noasm && darwin && arm64

#include "textflag.h"

// func matmul_sme_f32(a, b, c unsafe.Pointer, m, n, k int64)
//
// Computes C = A * B where:
//   A is M x K (row-major)
//   B is K x N (row-major)
//   C is M x N (row-major)
//
// Uses ARM SME streaming mode with SVE instructions.
// SVL on M4 is 512 bits = 16 x float32.
//
// Algorithm:
//   For each row i of C:
//     For each k:
//       C[i, :] += A[i, k] * B[k, :]
//
// Note: On macOS, NEON FMOVS instructions fail in streaming mode.
// We use integer loads and SVE dup to broadcast scalars.
TEXT ·matmul_sme_f32(SB), NOSPLIT, $64-48
	// Use stack for callee-saved registers
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
	LSL	$2, R23, R25       // R25 = N * 4 (row stride for B and C in bytes)
	LSL	$2, R24, R26       // R26 = K * 4 (row stride for A in bytes)

	// Calculate vector elements (N rounded down to multiple of 16)
	AND	$~15, R23, R27     // R27 = N & ~15 (vector portion)

	// Enter streaming mode
	WORD	$0xd503477f        // smstart

	// Set up predicate (all lanes active for 32-bit elements)
	WORD	$0x2598e3e0        // ptrue p0.s

	// Outer loop: for each row i of C (i in R0)
	MOVD	$0, R0

row_loop:
	CMP	R0, R22
	BGE	done_vector

	// Calculate A row pointer: &A[i, 0] = A + i * K * 4
	MUL	R26, R0, R1
	ADD	R19, R1, R1        // R1 = &A[i, 0]

	// Calculate C row pointer: &C[i, 0] = C + i * N * 4
	MUL	R25, R0, R2
	ADD	R21, R2, R2        // R2 = &C[i, 0]

	// Zero the vector portion of C row
	MOVD	R2, R3             // R3 = current C pointer
	MOVD	R27, R4            // R4 = vector elements count
	WORD	$0x25b8c000        // mov z0.s, #0

zero_loop:
	CBZ	R4, k_loop_init
	WORD	$0xe540e060        // st1w {z0.s}, p0, [x3]
	ADD	$64, R3, R3
	SUB	$16, R4, R4
	B	zero_loop

k_loop_init:
	// Middle loop: for each kk in 0..K-1
	MOVD	$0, R3             // R3 = kk

k_loop:
	CMP	R3, R24
	BGE	next_row

	// Load A[i, kk] as integer bits
	LSL	$2, R3, R4         // R4 = kk * 4
	ADD	R1, R4, R4         // R4 = &A[i, kk]
	MOVW	(R4), R5           // R5 = A[i, kk] as int32 bits

	// Broadcast R5 to z0.s using dup
	// dup z0.s, w5 = 0x05a03800 | (Rm << 5) = 0x05a038a0
	WORD	$0x05a038a0        // dup z0.s, w5

	// Calculate B row pointer: &B[kk, 0] = B + kk * N * 4
	MUL	R25, R3, R4
	ADD	R20, R4, R4        // R4 = &B[kk, 0]

	// Inner loop: C[i, j:j+16] += A[i,kk] * B[kk, j:j+16]
	MOVD	R2, R6             // R6 = &C[i, 0]
	MOVD	R4, R7             // R7 = &B[kk, 0]
	MOVD	R27, R8            // R8 = vector elements remaining

inner_loop:
	CBZ	R8, k_next
	// ld1w {z1.s}, p0/z, [x7] - load B row chunk
	WORD	$0xa540a0e1        // ld1w {z1.s}, p0/z, [x7]
	// ld1w {z2.s}, p0/z, [x6] - load C row chunk
	WORD	$0xa540a0c2        // ld1w {z2.s}, p0/z, [x6]

	// Compute z2 = z2 + z0 * z1 using FMUL+FADD (FMLA has encoding collision with FADD when using p0)
	// Step 1: z3 = z0 * z1 (FMUL unpredicated)
	// fmul z3.s, z0.s, z1.s: 01100101 10 0 00001 000010 00000 00011 = 0x65810803
	WORD	$0x65810803        // fmul z3.s, z0.s, z1.s
	// Step 2: z2 = z2 + z3 (FADD unpredicated)
	// fadd z2.s, z2.s, z3.s: 01100101 10 0 00011 000000 00010 00010 = 0x65830042
	WORD	$0x65830042        // fadd z2.s, z2.s, z3.s

	// st1w {z2.s}, p0, [x6] - store C row chunk
	WORD	$0xe540e0c2        // st1w {z2.s}, p0, [x6]

	ADD	$64, R6, R6
	ADD	$64, R7, R7
	SUB	$16, R8, R8
	B	inner_loop

k_next:
	ADD	$1, R3, R3
	B	k_loop

next_row:
	ADD	$1, R0, R0
	B	row_loop

done_vector:
	// Exit streaming mode before scalar tail processing
	WORD	$0xd503467f        // smstop

	// Now handle scalar tail (N % 16 remaining elements per row)
	AND	$15, R23, R8       // R8 = N % 16 (tail elements)
	CBZ	R8, done           // No tail, we're done

	// Process scalar tail for each row
	MOVD	$0, R0             // row counter

scalar_row_loop:
	CMP	R0, R22
	BGE	done

	// Calculate A row pointer
	MUL	R26, R0, R1
	ADD	R19, R1, R1        // R1 = &A[i, 0]

	// Calculate C row pointer + vector offset
	MUL	R25, R0, R2
	ADD	R21, R2, R2
	LSL	$2, R27, R9        // R9 = vector_elements * 4
	ADD	R9, R2, R2         // R2 = &C[i, vector_elements]

	// Zero the tail of C row
	MOVD	R2, R3
	MOVD	R8, R4
zero_tail:
	CBZ	R4, scalar_k_init
	MOVW	$0, R10
	MOVW	R10, (R3)
	ADD	$4, R3, R3
	SUB	$1, R4, R4
	B	zero_tail

scalar_k_init:
	MOVD	$0, R3             // kk counter

scalar_k_loop:
	CMP	R3, R24
	BGE	scalar_next_row

	// Load A[i, kk]
	LSL	$2, R3, R4
	ADD	R1, R4, R4
	FMOVS	(R4), F0           // A[i, kk]

	// Calculate B row pointer + vector offset
	MUL	R25, R3, R4
	ADD	R20, R4, R4
	ADD	R9, R4, R4         // R4 = &B[kk, vector_elements]

	// Inner scalar loop
	MOVD	R2, R5             // C tail ptr
	MOVD	R4, R6             // B tail ptr
	MOVD	R8, R7             // tail count

scalar_inner:
	CBZ	R7, scalar_k_next
	FMOVS	(R6), F1           // B[kk, j]
	FMOVS	(R5), F2           // C[i, j]
	FMADDS	F0, F1, F2, F2     // C += A * B
	FMOVS	F2, (R5)
	ADD	$4, R5, R5
	ADD	$4, R6, R6
	SUB	$1, R7, R7
	B	scalar_inner

scalar_k_next:
	ADD	$1, R3, R3
	B	scalar_k_loop

scalar_next_row:
	ADD	$1, R0, R0
	B	scalar_row_loop

done:
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
