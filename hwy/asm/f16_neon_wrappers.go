// Copyright 2025 go-highway Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//go:build !noasm && arm64

// Float16 NEON operations for ARM64 with FP16 extension (ARMv8.2-A+)
package asm

import "unsafe"

// -march=armv8.2-a+fp16 enables native float16 arithmetic instructions
//go:generate go tool goat ../c/ops_f16_neon_arm64.c -O3 --target arm64 -e="-march=armv8.2-a+fp16"

// ============================================================================
// Float16 Conversions
// ============================================================================

// PromoteF16ToF32NEON converts float16 to float32 using NEON vcvt_f32_f16.
func PromoteF16ToF32NEON(a []uint16, result []float32) {
	if len(a) == 0 {
		return
	}
	n := int64(min(len(a), len(result)))
	promote_f16_to_f32_neon(unsafe.Pointer(&a[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// DemoteF32ToF16NEON converts float32 to float16 using NEON vcvt_f16_f32.
func DemoteF32ToF16NEON(a []float32, result []uint16) {
	if len(a) == 0 {
		return
	}
	n := int64(min(len(a), len(result)))
	demote_f32_to_f16_neon(unsafe.Pointer(&a[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// ============================================================================
// Native Float16 Arithmetic (requires ARMv8.2-A+fp16)
// ============================================================================

// AddF16NEON performs element-wise addition: result[i] = a[i] + b[i]
func AddF16NEON(a, b, result []uint16) {
	if len(a) == 0 {
		return
	}
	n := int64(min(len(a), min(len(b), len(result))))
	add_f16_neon(unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// SubF16NEON performs element-wise subtraction: result[i] = a[i] - b[i]
func SubF16NEON(a, b, result []uint16) {
	if len(a) == 0 {
		return
	}
	n := int64(min(len(a), min(len(b), len(result))))
	sub_f16_neon(unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// MulF16NEON performs element-wise multiplication: result[i] = a[i] * b[i]
func MulF16NEON(a, b, result []uint16) {
	if len(a) == 0 {
		return
	}
	n := int64(min(len(a), min(len(b), len(result))))
	mul_f16_neon(unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// DivF16NEON performs element-wise division: result[i] = a[i] / b[i]
func DivF16NEON(a, b, result []uint16) {
	if len(a) == 0 {
		return
	}
	n := int64(min(len(a), min(len(b), len(result))))
	div_f16_neon(unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// FmaF16NEON performs fused multiply-add: result[i] = a[i] * b[i] + c[i]
func FmaF16NEON(a, b, c, result []uint16) {
	if len(a) == 0 {
		return
	}
	n := int64(min(len(a), min(len(b), min(len(c), len(result)))))
	fma_f16_neon(unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), unsafe.Pointer(&c[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// NegF16NEON performs element-wise negation: result[i] = -a[i]
func NegF16NEON(a, result []uint16) {
	if len(a) == 0 {
		return
	}
	n := int64(min(len(a), len(result)))
	neg_f16_neon(unsafe.Pointer(&a[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// AbsF16NEON performs element-wise absolute value: result[i] = abs(a[i])
func AbsF16NEON(a, result []uint16) {
	if len(a) == 0 {
		return
	}
	n := int64(min(len(a), len(result)))
	abs_f16_neon(unsafe.Pointer(&a[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// MinF16NEON performs element-wise minimum: result[i] = min(a[i], b[i])
func MinF16NEON(a, b, result []uint16) {
	if len(a) == 0 {
		return
	}
	n := int64(min(len(a), min(len(b), len(result))))
	min_f16_neon(unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// MaxF16NEON performs element-wise maximum: result[i] = max(a[i], b[i])
func MaxF16NEON(a, b, result []uint16) {
	if len(a) == 0 {
		return
	}
	n := int64(min(len(a), min(len(b), len(result))))
	max_f16_neon(unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// SqrtF16NEON performs element-wise square root: result[i] = sqrt(a[i])
func SqrtF16NEON(a, result []uint16) {
	if len(a) == 0 {
		return
	}
	n := int64(min(len(a), len(result)))
	sqrt_f16_neon(unsafe.Pointer(&a[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// ============================================================================
// Float16 Vector Load/Store Operations
// ============================================================================

// Load4F16x8NEON loads 4 consecutive float16x8 vectors (32 float16 values = 64 bytes).
// Uses vld1q_f16_x4 which loads 64 bytes in a single instruction.
// Returns 4 vectors as [16]byte (each holding 8 float16 values).
func Load4F16x8NEON(ptr []uint16) (v0, v1, v2, v3 [16]byte) {
	if len(ptr) < 32 {
		return
	}
	load4_f16x8(unsafe.Pointer(&ptr[0]),
		unsafe.Pointer(&v0[0]), unsafe.Pointer(&v1[0]),
		unsafe.Pointer(&v2[0]), unsafe.Pointer(&v3[0]))
	return
}

// Store4F16x8NEON stores 4 consecutive float16x8 vectors (32 float16 values = 64 bytes).
// Uses vst1q_f16_x4 which stores 64 bytes in a single instruction.
func Store4F16x8NEON(ptr []uint16, v0, v1, v2, v3 [16]byte) {
	if len(ptr) < 32 {
		return
	}
	store4_f16x8(unsafe.Pointer(&ptr[0]), v0, v1, v2, v3)
}

// Assembly function declarations are in ops_f16_neon_arm64.go (generated by GoAT)
