//go:build !noasm && arm64

#include "textflag.h"

// func matmul_neon_f32(a, b, c unsafe.Pointer, m, n, k int64)
//
// Computes C = A * B where:
//   A is M x K (row-major)
//   B is K x N (row-major)
//   C is M x N (row-major)
//
// Uses ARM NEON FMLA for vectorized multiply-add.
// Processes 4 floats at a time using 128-bit NEON vectors.
TEXT ·matmul_neon_f32(SB), NOSPLIT, $64-48
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

	// Calculate vector elements (N rounded down to multiple of 4)
	AND	$~3, R23, R27      // R27 = N & ~3 (vector portion)

	// Outer loop: for each row i of C (i in R0)
	MOVD	$0, R0

row_loop:
	CMP	R22, R0
	BGE	done

	// Calculate A row pointer: &A[i, 0] = A + i * K * 4
	MUL	R26, R0, R1
	ADD	R19, R1, R1        // R1 = &A[i, 0]

	// Calculate C row pointer: &C[i, 0] = C + i * N * 4
	MUL	R25, R0, R2
	ADD	R21, R2, R2        // R2 = &C[i, 0]

	// Zero the vector portion of C row using NEON
	MOVD	R2, R3             // R3 = current C pointer
	MOVD	R27, R4            // R4 = vector elements count
	VEOR	V0.B16, V0.B16, V0.B16  // V0 = 0

zero_loop:
	CBZ	R4, k_loop_init
	VST1	[V0.S4], (R3)      // Store 4 zeros
	ADD	$16, R3, R3
	SUB	$4, R4, R4
	B	zero_loop

k_loop_init:
	// Middle loop: for each kk in 0..K-1
	MOVD	$0, R3             // R3 = kk

k_loop:
	CMP	R24, R3
	BGE	next_row

	// Load A[i, kk] and broadcast to V0
	LSL	$2, R3, R4         // R4 = kk * 4
	ADD	R1, R4, R4         // R4 = &A[i, kk]
	FMOVS	(R4), F0           // Load A[i, kk] into s0
	VDUP	V0.S[0], V0.S4     // Broadcast s0 to all lanes of V0

	// Calculate B row pointer: &B[kk, 0] = B + kk * N * 4
	MUL	R25, R3, R4
	ADD	R20, R4, R4        // R4 = &B[kk, 0]

	// Inner loop: C[i, j:j+4] += A[i,kk] * B[kk, j:j+4]
	MOVD	R2, R6             // R6 = &C[i, 0]
	MOVD	R4, R7             // R7 = &B[kk, 0]
	MOVD	R27, R8            // R8 = vector elements remaining

inner_loop:
	CBZ	R8, k_next
	// Load B[kk, j:j+4] into V1
	VLD1	(R7), [V1.S4]
	// Load C[i, j:j+4] into V2
	VLD1	(R6), [V2.S4]
	// V2 = V0 * V1 + V2 (fmla)
	VFMLA	V0.S4, V1.S4, V2.S4
	// Store V2 to C[i, j:j+4]
	VST1	[V2.S4], (R6)

	ADD	$16, R6, R6
	ADD	$16, R7, R7
	SUB	$4, R8, R8
	B	inner_loop

k_next:
	ADD	$1, R3, R3
	B	k_loop

next_row:
	ADD	$1, R0, R0
	B	row_loop

done:
	// Handle scalar tail (N % 4 remaining elements per row)
	AND	$3, R23, R8        // R8 = N % 4 (tail elements)
	CBZ	R8, exit           // No tail, we're done

	// Process scalar tail for each row
	MOVD	$0, R0             // row counter

scalar_row_loop:
	CMP	R22, R0
	BGE	exit

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
	CMP	R24, R3
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

exit:
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
