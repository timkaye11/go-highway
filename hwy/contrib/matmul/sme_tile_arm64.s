//go:build !noasm && darwin && arm64

#include "textflag.h"

// SME tile operations test functions
// These test the real SME instructions: ZERO ZA, LD1W to ZA, ST1W from ZA, FMOPA

// sme_za_store_test: Zero ZA and store row 0 to dst
// Uses MOVA to move from ZA to Z register, then ST1W from Z register
// func sme_za_store_test(dst unsafe.Pointer)
TEXT ·sme_za_store_test(SB), NOSPLIT, $0-8
	MOVD	dst+0(FP), R0

	// Enter streaming SVE mode with ZA enabled
	// SMSTART (SM + ZA)
	WORD	$0xd503477f

	// PTRUE p0.s - all lanes active
	WORD	$0x2598e3e0

	// ZERO {ZA} - zero all of ZA array
	// Encoding: 1100 0000 0000 1000 0000 0000 1111 1111
	WORD	$0xc00800ff

	// Set up W12 = 0 for slice index
	MOVD	$0, R12

	// MOVA z0.s, p0/m, ZA0H.S[W12, #0]
	// Move horizontal slice from ZA to Z register
	// Correct encoding from LLVM: [0x00,0x00,0x82,0xc0] → 0xc0820000
	WORD	$0xc0820000

	// ST1W {z0.s}, p0, [x0]
	// Store Z register to memory (this works from earlier tests)
	WORD	$0xe540e000

	// Exit streaming mode
	// SMSTOP
	WORD	$0xd503467f

	RET

// sme_za_load_store_test: Load from src into ZA, store to dst
// func sme_za_load_store_test(dst, src unsafe.Pointer)
TEXT ·sme_za_load_store_test(SB), NOSPLIT, $0-16
	MOVD	dst+0(FP), R0
	MOVD	src+8(FP), R1

	// Enter streaming SVE mode with ZA enabled
	WORD	$0xd503477f		// SMSTART

	// PTRUE p0.s
	WORD	$0x2598e3e0

	// ZERO {ZA}
	WORD	$0xc00800ff

	// Set up W12 = 0 for slice index
	MOVD	$0, R12

	// LD1W {ZA0H.S[W12, #0]}, p0/z, [x1, xzr, LSL #2]
	// Load horizontal slice from src into ZA
	// Encoding: 1110 0000 00 0 Rm(5) 0 V(2) 0 Pg(2) Rn(5) 0 i2(2) Rv(2)
	// msz=00 (.S), bit21=0 (load), Rm=11111, V=00, Pg=00, Rn=00001, i2=00, Rv=00
	// = 1110 0000 0001 1111 0000 0000 0010 0000 = 0xe01f0020
	WORD	$0xe01f0020

	// ST1W {ZA0H.S[W12, #0]}, p0, [x0, xzr, LSL #2]
	// Store horizontal slice to dst
	// = 1110 0000 0011 1111 0000 0000 0000 0000 = 0xe03f0000
	WORD	$0xe03f0000

	// Exit streaming mode
	WORD	$0xd503467f		// SMSTOP

	RET

// sme_fmopa_test: Test outer product accumulate
// Uses MOVA to extract result from ZA
// func sme_fmopa_test(dst unsafe.Pointer)
TEXT ·sme_fmopa_test(SB), NOSPLIT, $0-8
	MOVD	dst+0(FP), R0

	// Enter streaming SVE mode with ZA enabled
	WORD	$0xd503477f		// SMSTART

	// PTRUE p0.s, p1.s
	WORD	$0x2598e3e0
	WORD	$0x2598e3e1

	// ZERO {ZA}
	WORD	$0xc00800ff

	// Set z0 = all 2.0
	MOVD	$0x40000000, R2
	WORD	$0x05a03840		// dup z0.s, w2

	// Set z1 = all 3.0
	MOVD	$0x40400000, R2
	WORD	$0x05a03841		// dup z1.s, w2

	// FMOPA ZA0.S, p0/m, p1/m, z0.s, z1.s
	// From LLVM: fmopa za0.s, p0/m, p0/m, z0.s, z0.s → 0x80800000
	// With Zm=1, Pm=1: 0x80812000
	WORD	$0x80812000

	// Set up W12 = 0 for slice index
	MOVD	$0, R12

	// MOVA z2.s, p0/m, ZA0H.S[W12, #0]
	// From LLVM: 0xc0820000 for z0, 0xc0820002 for z2
	WORD	$0xc0820002

	// ST1W {z2.s}, p0, [x0]
	// st1w z0: 0xe540e000, for z2: 0xe540e002
	WORD	$0xe540e002

	// Exit streaming mode
	WORD	$0xd503467f		// SMSTOP

	RET

// sme_fmopa_debug_test: Debug test - stores z0, z1, and ZA result
// func sme_fmopa_debug_test(z0out, z1out, zaout unsafe.Pointer)
TEXT ·sme_fmopa_debug_test(SB), NOSPLIT, $0-24
	MOVD	z0out+0(FP), R0
	MOVD	z1out+8(FP), R1
	MOVD	zaout+16(FP), R3

	// Enter streaming SVE mode with ZA enabled
	WORD	$0xd503477f		// SMSTART

	// PTRUE p0.b - set all predicate bits (byte granularity)
	// Encoding: 0010 0101 0001 1000 1110 0011 1110 0000 = 0x2518e3e0
	WORD	$0x2518e3e0

	// PTRUE p1.b - set second predicate
	// Same encoding but Pd=1: 0x2518e3e1
	WORD	$0x2518e3e1

	// ZERO {ZA}
	WORD	$0xc00800ff

	// Set z0 = all 2.0
	MOVD	$0x40000000, R2
	WORD	$0x05a03840		// dup z0.s, w2

	// Store z0 to z0out (use p0 for 32-bit store)
	// Need ptrue p0.s for the store: 0x2598e3e0
	WORD	$0x2598e3e0		// ptrue p0.s
	WORD	$0xe540e000		// st1w {z0.s}, p0, [x0]

	// Set z1 = all 3.0
	MOVD	$0x40400000, R2
	WORD	$0x05a03841		// dup z1.s, w2

	// Store z1 to z1out
	WORD	$0xe540e021		// st1w {z1.s}, p0, [x1]

	// Re-setup predicates for FMOPA (32-bit granularity for FP32)
	WORD	$0x2598e3e0		// ptrue p0.s
	WORD	$0x2598e3e1		// ptrue p1.s

	// FMOPA ZA0.S, p0/m, p1/m, z0.s, z1.s
	// Encoding: bits 31:23=100000001, 22:21=00, 20:16=Zm, 15:13=Pm, 12:10=Pn, 9:5=Zn, 4=S, 3:2=ZAda, 1:0=00
	// Zm=1, Pm=1, Pn=0, Zn=0, S=0, ZAda=0
	// = 1000 0000 1000 0001 001 000 00000 0 00 00
	// Wait, Pm is 3 bits at 15:13, Pn is 3 bits at 12:10
	// Pm=001 (p1), Pn=000 (p0)
	// = 1000 0000 1000 0001 0010 0000 0000 0000 = 0x80812000
	WORD	$0x80812000

	// Set up W12 = 0 for slice index
	MOVD	$0, R12

	// Re-setup p0.s for MOVA and store
	WORD	$0x2598e3e0		// ptrue p0.s

	// MOVA z2.s, p0/m, ZA0H.S[W12, #0]
	// Encoding from LLVM: [0x00,0x00,0x82,0xc0] for z0 → 0xc0820000
	// For z2: 0xc0820002
	WORD	$0xc0820002

	// Store z2 (ZA result) to zaout
	WORD	$0xe540e062		// st1w {z2.s}, p0, [x3]

	// Exit streaming mode
	WORD	$0xd503467f		// SMSTOP

	RET

// sme_build_vector_test: Test building vector via horizontal write + vertical read
// Writes different values to each horizontal slice, reads back vertically
// func sme_build_vector_test(dst unsafe.Pointer)
TEXT ·sme_build_vector_test(SB), NOSPLIT, $0-8
	MOVD	dst+0(FP), R0

	// Enter streaming SVE mode with ZA enabled
	WORD	$0xd503477f		// SMSTART

	// PTRUE p0.s
	WORD	$0x2598e3e0

	// ZERO {ZA} - zero all tiles
	WORD	$0xc00800ff

	// Write values 0.0, 1.0, 2.0, ..., 15.0 to horizontal slices
	// Each horizontal slice will have all lanes set to the same value
	// After vertical read, we should get [0, 1, 2, ..., 15]

	// Use ZA0 as scratch (we just zeroed it)

	// Row 0: set to 0.0
	MOVD	$0x00000000, R2		// 0.0 in float32
	WORD	$0x05a03840		// dup z0.s, w2
	MOVD	$0, R12			// W12 = 0
	WORD	$0xc0800000		// mova za0h.s[w12, 0], p0/m, z0.s

	// Row 1: set to 1.0
	MOVD	$0x3f800000, R2		// 1.0 in float32
	WORD	$0x05a03840		// dup z0.s, w2
	ADD	$1, R12, R12		// W12 = 1
	WORD	$0xc0800000		// mova za0h.s[w12, 0], p0/m, z0.s

	// Row 2: set to 2.0
	MOVD	$0x40000000, R2		// 2.0 in float32
	WORD	$0x05a03840		// dup z0.s, w2
	ADD	$1, R12, R12		// W12 = 2
	WORD	$0xc0800000		// mova za0h.s[w12, 0], p0/m, z0.s

	// Row 3: set to 3.0
	MOVD	$0x40400000, R2		// 3.0 in float32
	WORD	$0x05a03840		// dup z0.s, w2
	ADD	$1, R12, R12		// W12 = 3
	WORD	$0xc0800000		// mova za0h.s[w12, 0], p0/m, z0.s

	// Rows 4-15: set to 4.0, 5.0, ..., 15.0
	// For brevity, just set them all to 4.0 (we'll verify the pattern works)
	MOVD	$0x40800000, R2		// 4.0 in float32
	WORD	$0x05a03840		// dup z0.s, w2
	ADD	$1, R12, R12		// W12 = 4
	WORD	$0xc0800000		// mova za0h.s[w12, 0], p0/m, z0.s

	MOVD	$0x40a00000, R2		// 5.0
	WORD	$0x05a03840
	ADD	$1, R12, R12		// W12 = 5
	WORD	$0xc0800000

	MOVD	$0x40c00000, R2		// 6.0
	WORD	$0x05a03840
	ADD	$1, R12, R12		// W12 = 6
	WORD	$0xc0800000

	MOVD	$0x40e00000, R2		// 7.0
	WORD	$0x05a03840
	ADD	$1, R12, R12		// W12 = 7
	WORD	$0xc0800000

	MOVD	$0x41000000, R2		// 8.0
	WORD	$0x05a03840
	ADD	$1, R12, R12		// W12 = 8
	WORD	$0xc0800000

	MOVD	$0x41100000, R2		// 9.0
	WORD	$0x05a03840
	ADD	$1, R12, R12		// W12 = 9
	WORD	$0xc0800000

	MOVD	$0x41200000, R2		// 10.0
	WORD	$0x05a03840
	ADD	$1, R12, R12		// W12 = 10
	WORD	$0xc0800000

	MOVD	$0x41300000, R2		// 11.0
	WORD	$0x05a03840
	ADD	$1, R12, R12		// W12 = 11
	WORD	$0xc0800000

	MOVD	$0x41400000, R2		// 12.0
	WORD	$0x05a03840
	ADD	$1, R12, R12		// W12 = 12
	WORD	$0xc0800000

	MOVD	$0x41500000, R2		// 13.0
	WORD	$0x05a03840
	ADD	$1, R12, R12		// W12 = 13
	WORD	$0xc0800000

	MOVD	$0x41600000, R2		// 14.0
	WORD	$0x05a03840
	ADD	$1, R12, R12		// W12 = 14
	WORD	$0xc0800000

	MOVD	$0x41700000, R2		// 15.0
	WORD	$0x05a03840
	ADD	$1, R12, R12		// W12 = 15
	WORD	$0xc0800000

	// Now read VERTICAL slice 0 - should give us [0, 1, 2, ..., 15]
	// MOVA z1.s, p0/m, za0v.s[w12, 0]
	// Horizontal read: 0xc0820001 for z1
	// Vertical read: bit 15 set, so 0xc082 + 0x8000 = 0xc08a for z1
	// Actually, looking at ARM encoding, for MOVA (tile to vector, 32-bit):
	// bit 15 is V (0=horizontal, 1=vertical)
	// So vertical would be: 0xc0820000 | 0x00008000 = 0xc082 8000
	MOVD	$0, R12
	WORD	$0xc0828001		// mova z1.s, p0/m, za0v.s[w12, 0]

	// Store z1 to dst
	WORD	$0xe540e001		// st1w {z1.s}, p0, [x0]

	// Exit streaming mode
	WORD	$0xd503467f		// SMSTOP

	RET

// sme_za1_build_vector_test: Test building vector using ZA1 (tile 1) as scratch
// func sme_za1_build_vector_test(dst unsafe.Pointer)
TEXT ·sme_za1_build_vector_test(SB), NOSPLIT, $0-8
	MOVD	dst+0(FP), R0

	// Enter streaming SVE mode with ZA enabled
	WORD	$0xd503477f		// SMSTART

	// PTRUE p0.s
	WORD	$0x2598e3e0

	// ZERO all ZA tiles
	WORD	$0xc00800ff

	// Write values 0, 1, 2, ..., 15 to horizontal slices of ZA1
	MOVD	$0, R12			// W12 = 0

	// Row 0: 0.0
	MOVD	$0x00000000, R2
	WORD	$0x05a03840		// dup z0.s, w2
	WORD	$0xc0800001		// mova za1h.s[w12, 0], p0/m, z0.s

	// Row 1: 1.0
	MOVD	$0x3f800000, R2
	WORD	$0x05a03840
	ADD	$1, R12, R12
	WORD	$0xc0800001

	// Row 2: 2.0
	MOVD	$0x40000000, R2
	WORD	$0x05a03840
	ADD	$1, R12, R12
	WORD	$0xc0800001

	// Row 3: 3.0
	MOVD	$0x40400000, R2
	WORD	$0x05a03840
	ADD	$1, R12, R12
	WORD	$0xc0800001

	// Continue for rows 4-15...
	MOVD	$0x40800000, R2		// 4.0
	WORD	$0x05a03840
	ADD	$1, R12, R12
	WORD	$0xc0800001

	MOVD	$0x40a00000, R2		// 5.0
	WORD	$0x05a03840
	ADD	$1, R12, R12
	WORD	$0xc0800001

	MOVD	$0x40c00000, R2		// 6.0
	WORD	$0x05a03840
	ADD	$1, R12, R12
	WORD	$0xc0800001

	MOVD	$0x40e00000, R2		// 7.0
	WORD	$0x05a03840
	ADD	$1, R12, R12
	WORD	$0xc0800001

	MOVD	$0x41000000, R2		// 8.0
	WORD	$0x05a03840
	ADD	$1, R12, R12
	WORD	$0xc0800001

	MOVD	$0x41100000, R2		// 9.0
	WORD	$0x05a03840
	ADD	$1, R12, R12
	WORD	$0xc0800001

	MOVD	$0x41200000, R2		// 10.0
	WORD	$0x05a03840
	ADD	$1, R12, R12
	WORD	$0xc0800001

	MOVD	$0x41300000, R2		// 11.0
	WORD	$0x05a03840
	ADD	$1, R12, R12
	WORD	$0xc0800001

	MOVD	$0x41400000, R2		// 12.0
	WORD	$0x05a03840
	ADD	$1, R12, R12
	WORD	$0xc0800001

	MOVD	$0x41500000, R2		// 13.0
	WORD	$0x05a03840
	ADD	$1, R12, R12
	WORD	$0xc0800001

	MOVD	$0x41600000, R2		// 14.0
	WORD	$0x05a03840
	ADD	$1, R12, R12
	WORD	$0xc0800001

	MOVD	$0x41700000, R2		// 15.0
	WORD	$0x05a03840
	ADD	$1, R12, R12
	WORD	$0xc0800001

	// Read vertical slice 0 from ZA1 - should get [0, 1, 2, ..., 15]
	// CRITICAL ENCODING INSIGHT:
	// For MOVA tile-to-vector (32-bit), tile index is in bits 1:0,
	// AND Zd (dest register) is in bits 4:0.
	// These OVERLAP! So:
	//   za0v to z0: tile=00, Zd=00000 → 0xc0828000
	//   za1v to z1: tile=01, Zd=00001 → both need bit 0=1, so 0xc0828001
	//   za0v to z1 would be tile=00, Zd=00001 → bit 0=1 (conflicts with tile=0!)
	//
	// This means: za1 can ONLY be read to z1, z5, z9, z13 (where Zd[1:0]=01)
	//             za0 can ONLY be read to z0, z4, z8, z12 (where Zd[1:0]=00)
	//             za2 can ONLY be read to z2, z6, z10, z14 (where Zd[1:0]=10)
	//             za3 can ONLY be read to z3, z7, z11, z15 (where Zd[1:0]=11)
	MOVD	$0, R12
	WORD	$0xc0828001		// mova z1.s, p0/m, za1v.s[w12, 0]

	// Store z1 to dst
	WORD	$0xe540e001		// st1w {z1.s}, p0, [x0]

	// Exit streaming mode
	WORD	$0xd503467f		// SMSTOP

	RET

// sme_fmopa_store_all_test: Test storing all 16 rows of ZA after FMOPA
// func sme_fmopa_store_all_test(dst unsafe.Pointer, stride int64)
TEXT ·sme_fmopa_store_all_test(SB), NOSPLIT, $0-16
	MOVD	dst+0(FP), R0
	MOVD	stride+8(FP), R1    // Stride in bytes between rows

	// Enter streaming SVE mode with ZA enabled
	WORD	$0xd503477f		// SMSTART

	// PTRUE p0.s, p1.s
	WORD	$0x2598e3e0
	WORD	$0x2598e3e1

	// ZERO {ZA}
	WORD	$0xc00800ff

	// Set z0 = all 2.0
	MOVD	$0x40000000, R2
	WORD	$0x05a03840		// dup z0.s, w2

	// Set z1 = all 3.0
	MOVD	$0x40400000, R2
	WORD	$0x05a03841		// dup z1.s, w2

	// FMOPA ZA0.S, p0/m, p1/m, z0.s, z1.s
	WORD	$0x80812000

	// Now store all 16 rows of ZA0 to memory
	MOVD	$0, R12            // W12 = slice index
	MOVD	R0, R6             // R6 = current output pointer

store_all_rows:
	CMP	$16, R12
	BGE	store_all_done

	// MOVA z2.s, p0/m, za0h.s[w12, 0]
	WORD	$0xc0820002

	// ST1W {z2.s}, p0, [x6]
	WORD	$0xe540e0c2

	// Move to next row
	ADD	R1, R6, R6         // Next row pointer (add stride)
	ADD	$1, R12, R12       // Next ZA slice
	B	store_all_rows

store_all_done:
	// Exit streaming mode
	WORD	$0xd503467f		// SMSTOP

	RET

// sme_mova_to_za_test: Test MOVA in Z->ZA direction
// func sme_mova_to_za_test(dst unsafe.Pointer)
TEXT ·sme_mova_to_za_test(SB), NOSPLIT, $0-8
	MOVD	dst+0(FP), R0

	// Enter streaming SVE mode with ZA enabled
	WORD	$0xd503477f		// SMSTART

	// PTRUE p0.s
	WORD	$0x2598e3e0

	// ZERO {ZA}
	WORD	$0xc00800ff

	// Set z0 = all 5.0
	// 5.0 in float32 = 0x40a00000
	MOVD	$0x40a00000, R2
	WORD	$0x05a03840		// dup z0.s, w2

	// Set up W12 = 0 for slice index
	MOVD	$0, R12

	// MOVA ZA0H.S[W12, #0], p0/m, z0.s
	// Move z0 to horizontal slice 0 of ZA tile 0
	// Correct encoding from LLVM: [0x00,0x00,0x80,0xc0] → 0xc0800000
	WORD	$0xc0800000

	// MOVA z1.s, p0/m, ZA0H.S[W12, #0]
	// Move back from ZA to z1 to verify
	// Correct encoding from LLVM: 0xc0820001 (for z1)
	WORD	$0xc0820001

	// ST1W {z1.s}, p0, [x0]
	WORD	$0xe540e001		// st1w {z1.s}, p0, [x0]

	// Exit streaming mode
	WORD	$0xd503467f		// SMSTOP

	RET

// sme_za2_save_restore_test: Test if tile is in Zd bits 1:0
// za0 gets 2*3=6, za1 gets 4*5=20
// Read with 0xc0820005 (z5, bits 1:0 = 01 → should be za1 if theory is correct)
// func sme_za2_save_restore_test(dst unsafe.Pointer)
TEXT ·sme_za2_save_restore_test(SB), NOSPLIT, $0-8
	MOVD	dst+0(FP), R0

	// Enter streaming SVE mode with ZA enabled
	WORD	$0xd503477f		// SMSTART

	// PTRUE p0.s, p1.s
	WORD	$0x2598e3e0
	WORD	$0x2598e3e1

	// ZERO {ZA}
	WORD	$0xc00800ff

	// === FMOPA to za0: 2*3=6 ===
	MOVD	$0x40000000, R2		// 2.0
	WORD	$0x05a03840		// dup z0.s, w2
	MOVD	$0x40400000, R2		// 3.0
	WORD	$0x05a03841		// dup z1.s, w2
	WORD	$0x80812000		// fmopa za0.s, p0/m, p1/m, z0.s, z1.s

	// === FMOPA to za1: 4*5=20 ===
	MOVD	$0x40800000, R2		// 4.0
	WORD	$0x05a03840		// dup z0.s, w2
	MOVD	$0x40a00000, R2		// 5.0
	WORD	$0x05a03841		// dup z1.s, w2
	WORD	$0x80812001		// fmopa za1.s, p0/m, p1/m, z0.s, z1.s

	// Try reading with 0xc0820005 (z5 = 00101, bits 1:0 = 01)
	// If tile is encoded in Zd bits 1:0: bits 1:0=01 → za1 → should get 20.0
	// If 6.0: reads za0
	// If 20.0: reads za1 (theory confirmed!)
	MOVD	$0, R12
	WORD	$0xc0820005		// mova z5.s, p0/m, za?h.s[w12, 0]

	// Store z5 to dst
	WORD	$0xe540e005		// st1w {z5.s}, p0, [x0]

	// Exit streaming mode
	WORD	$0xd503467f		// SMSTOP

	RET
