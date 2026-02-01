//go:build smedebug && !noasm && darwin && arm64

#include "textflag.h"

// SME debug assembly functions. See sme_debug_test.go for usage.
// These are isolated under smedebug tag because LD1W crashes in streaming mode.

// Simple test: enter streaming mode, write a value, exit
TEXT ·sme_write_test(SB), NOSPLIT, $0-12
	MOVD	dst+0(FP), R0
	FMOVS	val+8(FP), F0

	// Convert float to int bits for testing
	FMOVS	F0, R1

	// Enter streaming mode
	WORD	$0xd503477f        // smstart

	// Write the value using integer store (should work in streaming mode)
	MOVW	R1, (R0)

	// Exit streaming mode
	WORD	$0xd503467f        // smstop

	RET

// Test SVE zero store in streaming mode
// func sme_sve_zero_test(dst unsafe.Pointer, count int64)
TEXT ·sme_sve_zero_test(SB), NOSPLIT, $0-16
	MOVD	dst+0(FP), R0
	MOVD	count+8(FP), R1

	// Enter streaming mode
	WORD	$0xd503477f        // smstart

	// Set up predicate
	WORD	$0x2598e3e0        // ptrue p0.s

	// Zero z0
	WORD	$0x25b8c000        // mov z0.s, #0

	// Store 16 zeros (64 bytes)
	WORD	$0xe540e000        // st1w {z0.s}, p0, [x0]

	// Exit streaming mode
	WORD	$0xd503467f        // smstop

	RET

// Test simple matmul: writes K (as float) to every element of C
// func sme_simple_matmul_test(c unsafe.Pointer, m, n, k int64)
TEXT ·sme_simple_matmul_test(SB), NOSPLIT, $0-32
	MOVD	c+0(FP), R0       // C matrix base
	MOVD	m+8(FP), R1       // M
	MOVD	n+16(FP), R2      // N
	MOVD	k+24(FP), R3      // K

	// Calculate total elements
	MUL	R1, R2, R4         // R4 = M * N (total elements)

	// Convert K to float (outside streaming mode)
	SCVTFWS	R3, F0             // Convert K (int64) to float32 in s0
	FMOVS	F0, R5             // Get float bits as integer for broadcast

	// Enter streaming mode
	WORD	$0xd503477f        // smstart

	// Set up predicate
	WORD	$0x2598e3e0        // ptrue p0.s

	// Broadcast R5 (K as float bits) to z0.s using DUP
	// dup z0.s, w5 = 0x05a03800 | (Rm << 5) = 0x05a038a0
	WORD	$0x05a038a0        // dup z0.s, w5

	// Loop: store K to all elements of C
	MOVD	R0, R6             // R6 = current ptr
	MOVD	R4, R7             // R7 = elements remaining

store_loop:
	CMP	$16, R7
	BLT	store_tail

	// Store 16 elements: st1w {z0.s}, p0, [x6]
	// x6 = reg 6, z0 = reg 0: lower byte = (6 << 5) | 0 = 0xc0
	WORD	$0xe540e0c0        // st1w {z0.s}, p0, [x6]
	ADD	$64, R6, R6
	SUB	$16, R7, R7
	B	store_loop

store_tail:
	CBZ	R7, store_done

	// Exit streaming mode for scalar stores
	WORD	$0xd503467f        // smstop

store_scalar:
	// Store single element using integer
	MOVW	R5, (R6)
	ADD	$4, R6, R6
	SUB	$1, R7, R7
	CBNZ	R7, store_scalar
	RET

store_done:
	// Exit streaming mode
	WORD	$0xd503467f        // smstop
	RET

// Test ld1w: load 16 floats from src, store to dst
// func sme_ld1w_test(dst, src unsafe.Pointer)
TEXT ·sme_ld1w_test(SB), NOSPLIT, $0-16
	MOVD	dst+0(FP), R0
	MOVD	src+8(FP), R1

	// Enter streaming mode
	WORD	$0xd503477f        // smstart

	// Set up predicate
	WORD	$0x2598e3e0        // ptrue p0.s

	// Try LD1RW (load and replicate word) - loads one float and broadcasts to all lanes
	// ld1rw {z0.s}, p0/z, [x1]
	// Encoding: 1000 0101 01 0 imm6=0 1 Pg=000 Rn=00001 Zt=00000
	// = 1000 0101 0100 0001 0000 0000 0010 0000 = 0x85410020
	WORD	$0x85410020        // ld1rw {z0.s}, p0/z, [x1]

	// Store z0 to dst: st1w {z0.s}, p0, [x0]
	WORD	$0xe540e000        // st1w {z0.s}, p0, [x0]

	// Exit streaming mode
	WORD	$0xd503467f        // smstop

	RET

// Test broadcast: dst[0..15] = 42.0
// func sme_broadcast_test(dst unsafe.Pointer)
TEXT ·sme_broadcast_test(SB), NOSPLIT, $0-8
	MOVD	dst+0(FP), R0

	// 42.0 = 0x42280000
	MOVW	$0x42280000, R1

	// Enter streaming mode
	WORD	$0xd503477f        // smstart

	// Set up predicate
	WORD	$0x2598e3e0        // ptrue p0.s

	// Broadcast 42.0 to z0: dup z0.s, w1 = 0x05a03800 + (1 << 5) = 0x05a03820
	WORD	$0x05a03820        // dup z0.s, w1

	// Store z0 to dst
	WORD	$0xe540e000        // st1w {z0.s}, p0, [x0]

	// Exit streaming mode
	WORD	$0xd503467f        // smstop

	RET

// Test FMUL: dst[0:16] = z0 * z1 = 2.0 * 3.0 = 6.0
// func sme_fmul_test(dst unsafe.Pointer)
TEXT ·sme_fmul_test(SB), NOSPLIT, $0-8
	MOVD	dst+0(FP), R0

	// 2.0 = 0x40000000, 3.0 = 0x40400000
	MOVW	$0x40000000, R1
	MOVW	$0x40400000, R2

	// Enter streaming mode
	WORD	$0xd503477f        // smstart

	// Set up predicate
	WORD	$0x2598e3e0        // ptrue p0.s

	// Broadcast 2.0 to z0
	WORD	$0x05a03820        // dup z0.s, w1

	// Broadcast 3.0 to z1
	WORD	$0x05a03841        // dup z1.s, w2

	// FMUL (unpredicated vectors): fmul z2.s, z0.s, z1.s
	// Encoding: 01100101 size=10 0 Zm=00001 opc=000010 Zn=00000 Zd=00010
	// = 0110 0101 1000 0001 0000 1000 0000 0010
	// = 0x65810802
	WORD	$0x65810802        // fmul z2.s, z0.s, z1.s

	// Store z2 to dst
	WORD	$0xe540e002        // st1w {z2.s}, p0, [x0]

	// Exit streaming mode
	WORD	$0xd503467f        // smstop

	RET

// Test FMLA via FMUL+FADD: z4 = z2 + (z0 * z1) = 1.0 + (2.0 * 3.0) = 7.0
// dst[0:16] = result (7.0), dst[16:32] = z0 (2.0), dst[32:48] = z1 (3.0)
// func sme_fmla_test(dst unsafe.Pointer)
TEXT ·sme_fmla_test(SB), NOSPLIT, $0-8
	MOVD	dst+0(FP), R0

	// Prepare float bits outside streaming mode
	// 2.0 = 0x40000000
	MOVW	$0x40000000, R1
	// 3.0 = 0x40400000
	MOVW	$0x40400000, R2
	// 1.0 = 0x3f800000
	MOVW	$0x3f800000, R3

	// Enter streaming mode
	WORD	$0xd503477f        // smstart

	// Set up predicate
	WORD	$0x2598e3e0        // ptrue p0.s

	// Broadcast 2.0 to z0
	WORD	$0x05a03820        // dup z0.s, w1

	// Broadcast 3.0 to z1
	WORD	$0x05a03841        // dup z1.s, w2

	// Broadcast 1.0 to z2
	WORD	$0x05a03862        // dup z2.s, w3

	// Step 1: z3 = z0 * z1 = 2.0 * 3.0 = 6.0
	// fmul z3.s, z0.s, z1.s (unpredicated)
	// Encoding: 01100101 size=10 0 Zm=00001 opc=000010 Zn=00000 Zd=00011
	// = 0110 0101 1000 0001 0000 1000 0000 0011
	// = 0x65810803
	WORD	$0x65810803        // fmul z3.s, z0.s, z1.s

	// Step 2: z4 = z2 + z3 = 1.0 + 6.0 = 7.0
	// fadd z4.s, z2.s, z3.s (unpredicated)
	// Encoding: 01100101 size=10 0 Zm=00011 opc=000000 Zn=00010 Zd=00100
	// = 0110 0101 1000 0011 0000 0000 0100 0100
	// = 0x65830044
	WORD	$0x65830044        // fadd z4.s, z2.s, z3.s

	// Store z4 to dst[0:16]: st1w {z4.s}, p0, [x0]
	WORD	$0xe540e004        // st1w {z4.s}, p0, [x0]

	// Store z0 to dst[16:32]: st1w {z0.s}, p0, [x0, #1, mul vl]
	WORD	$0xe541e000        // st1w {z0.s}, p0, [x0, #1, mul vl]

	// Store z1 to dst[32:48]: st1w {z1.s}, p0, [x0, #2, mul vl]
	WORD	$0xe542e001        // st1w {z1.s}, p0, [x0, #2, mul vl]

	// Exit streaming mode
	WORD	$0xd503467f        // smstop

	RET
