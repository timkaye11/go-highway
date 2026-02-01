# ARM SME on Apple M4 (Darwin)

This document describes our findings on implementing ARM Scalable Matrix Extension (SME) on Apple M4 processors.

## TL;DR

**SME works on Apple M4!** Initial failures were due to incorrect instruction encodings, not Apple's implementation.

## Background

Apple M4 is the first consumer chip with ARM SME support. SME provides:
- **ZA tile registers**: 4KB matrix storage (16×16 × 4 tiles of float32)
- **FMOPA instruction**: Outer product accumulate (512 FP32 ops per instruction)
- **Streaming SVE mode**: Required for SME operations

## Key Findings

### What Works on Apple M4

| Operation | Status | Notes |
|-----------|--------|-------|
| SMSTART/SMSTOP | ✅ | Streaming mode entry/exit |
| PTRUE | ✅ | Use `.s` for FP32, not `.b` |
| ZERO {ZA} | ✅ | Zeroes entire ZA array |
| DUP | ✅ | Broadcast scalar to Z register |
| ST1W (Z reg) | ✅ | Store from Z registers |
| FMOPA | ✅ | Outer product accumulate |
| MOVA (Z→ZA) | ✅ | Write to ZA tile slice |
| MOVA (ZA→Z) | ✅ | Read from ZA tile slice |

### Critical Encoding Corrections

The initial implementation failed because of incorrect MOVA encodings. The correct encodings from LLVM:

```
MOVA (tile to vector) - ZA→Z direction:
  mova z0.s, p0/m, za0h.s[w12, 0]
  Encoding: [0x00, 0x00, 0x82, 0xc0] → 0xc0820000

  WRONG: 0xc0800000 (bit 17 = 0)
  RIGHT: 0xc0820000 (bit 17 = 1) ← Critical difference!

MOVA (vector to tile) - Z→ZA direction:
  mova za0h.s[w12, 0], p0/m, z0.s
  Encoding: [0x00, 0x00, 0x80, 0xc0] → 0xc0800000

FMOPA (non-widening FP32):
  fmopa za0.s, p0/m, p0/m, z0.s, z0.s
  Encoding: [0x00, 0x00, 0x80, 0x80] → 0x80800000

  With different registers (za0.s, p0/m, p1/m, z0.s, z1.s):
  Encoding: 0x80812000
```

### Apple M4 Specifications

- **SVL (Streaming Vector Length)**: 512 bits
- **Z registers**: 16 × float32 per register
- **ZA tiles**: 4 tiles (ZA0-ZA3), each 16×16 = 256 float32
- **FMOPA throughput**: 512 FP32 ops per instruction
- **Reported peak**: ~2008 GFLOPS (vs ~28 GFLOPS with NEON)

## Working Example (Go Assembly)

```asm
// FMOPA test: compute 2.0 * 3.0 = 6.0 in ZA tile
TEXT ·sme_fmopa_test(SB), NOSPLIT, $0-8
    MOVD    dst+0(FP), R0

    // Enter streaming SVE mode with ZA enabled
    WORD    $0xd503477f        // smstart

    // Set up predicates (use .s for FP32!)
    WORD    $0x2598e3e0        // ptrue p0.s
    WORD    $0x2598e3e1        // ptrue p1.s

    // Zero ZA array
    WORD    $0xc00800ff        // zero {za}

    // Set z0 = 2.0, z1 = 3.0
    MOVD    $0x40000000, R2    // 2.0 in float32
    WORD    $0x05a03840        // dup z0.s, w2
    MOVD    $0x40400000, R2    // 3.0 in float32
    WORD    $0x05a03841        // dup z1.s, w2

    // FMOPA za0.s, p0/m, p1/m, z0.s, z1.s
    // ZA0 += outer_product(z0, z1) = 2.0 * 3.0 = 6.0
    WORD    $0x80812000

    // Extract result: MOVA z2.s, p0/m, za0h.s[w12, 0]
    MOVD    $0, R12
    WORD    $0xc0820002        // Note: 0xc082, NOT 0xc080!

    // Store result
    WORD    $0xe540e002        // st1w {z2.s}, p0, [x0]

    // Exit streaming mode
    WORD    $0xd503467f        // smstop
    RET
```

## Common Pitfalls

### 1. Wrong MOVA Encoding
```
WRONG: WORD $0xc0800000  // MOVA ZA→Z - missing bit 17
RIGHT: WORD $0xc0820000  // MOVA ZA→Z - bit 17 set
```

### 2. Wrong Predicate Granularity
```
WRONG: ptrue p0.b       // Byte granularity
RIGHT: ptrue p0.s       // 32-bit granularity for FP32
```

### 3. Missing Streaming Mode
All SME operations require streaming mode:
```
smstart                  // Enter streaming mode
// ... SME operations ...
smstop                   // Exit streaming mode
```

## Encoding Reference

### FMOPA (non-widening, FP32)
```
Bits 31-25: 1000000
Bit 24:     op{1} = 0
Bit 23:     1
Bits 22-21: sz = 00 (FP32)
Bits 20-16: Zm (source register 2)
Bits 15-13: Pm (predicate 2)
Bits 12-10: Pn (predicate 1)
Bits 9-5:   Zn (source register 1)
Bit 4:      S = 0 (accumulate, not subtract)
Bit 3:      op{0} = 0
Bit 2:      0
Bits 1-0:   ZAda (tile index)
```

### MOVA (tile to vector, 32-bit)
```
Encoding prefix: 0xc082 (NOT 0xc080!)
Bits 4-0: Zd (destination Z register)
```

### MOVA (vector to tile, 32-bit)
```
Encoding prefix: 0xc080
Bits 4-0: Zn (source Z register)
```

## Resources

- [LLVM SME test files](https://github.com/llvm/llvm-project/tree/main/llvm/test/MC/AArch64/SME) - Authoritative encoding reference
- [m4-sme-exploration](https://github.com/tzakharko/m4-sme-exploration) - Apple M4 SME benchmarks
- [Hello SME documentation](https://scalable.uni-jena.de/opt/sme/) - SME tutorials and examples
- [ARM SME blog post](https://developer.arm.com/community/arm-community-blogs/b/architectures-and-processors-blog/posts/arm-scalable-matrix-extension-introduction-p2) - Official ARM introduction

## Testing

Run SME tests:
```bash
GOEXPERIMENT=simd go1.26rc1 test -v -run TestSME ./hwy/contrib/matmul/
```

All tests should pass:
- `TestSMEFMOPA` - Basic FMOPA + result extraction
- `TestSMEFMOPADebug` - Comprehensive FMOPA test with intermediate values
- `TestSMEMOVAToZA` - MOVA in both directions
- `TestSMEZAStore` - ZERO {ZA} + MOVA ZA→Z
