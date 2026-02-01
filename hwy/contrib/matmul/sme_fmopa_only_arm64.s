//go:build !noasm && darwin && arm64

#include "textflag.h"

// sme_fmopa_only_test: Test FMOPA without storing result
// Just to see if the instruction executes without SIGILL
// func sme_fmopa_only_test(dst unsafe.Pointer)
TEXT ·sme_fmopa_only_test(SB), NOSPLIT, $0-8
	MOVD	dst+0(FP), R0

	// Enter streaming SVE mode with ZA enabled
	// SMSTART (SM + ZA)
	WORD	$0xd503477f

	// PTRUE p0.s - all lanes active
	WORD	$0x2598e3e0

	// ZERO {ZA} - zero all of ZA array
	WORD	$0xc00800ff

	// Set z0 = all 2.0
	// 2.0 in float32 = 0x40000000
	MOVD	$0x40000000, R2
	// DUP z0.s, w2
	WORD	$0x05a03840

	// Set z1 = all 3.0
	// 3.0 in float32 = 0x40400000
	MOVD	$0x40400000, R2
	// DUP z1.s, w2
	WORD	$0x05a03841

	// FMOPA ZA0.S, p0/m, p0/m, z0.s, z1.s
	// Outer product: ZA0 += z0 * z1^T
	// Encoding: 1000 0000 100 Zm(5) 00 Pm(3) Pn(3) Zn(5) S(1) imm(2)
	// Zm=1, Pm=0, Pn=0, Zn=0, S=0, imm=00
	// = 1000 0000 1000 0001 0000 0000 0000 0000 = 0x80810000
	WORD	$0x80810000

	// Don't try to store ZA - just exit and see if we get here

	// Exit streaming mode
	// SMSTOP
	WORD	$0xd503467f

	RET
