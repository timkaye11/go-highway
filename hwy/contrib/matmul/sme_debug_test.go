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

//go:build smedebug && darwin && arm64

// This file contains SME debug tests. These tests are isolated under the
// "smedebug" build tag because some of them (TestSMELD1W) crash with SIGILL
// due to Apple M4 not supporting SVE contiguous loads in streaming mode.
//
// Run with: go test -tags=smedebug ./hwy/contrib/matmul/...

package matmul

import (
	"testing"
	"unsafe"
)

//go:noescape
func sme_write_test(dst unsafe.Pointer, val float32)

//go:noescape
func sme_sve_zero_test(dst unsafe.Pointer, count int64)

//go:noescape
func sme_simple_matmul_test(c unsafe.Pointer, m, n, k int64)

//go:noescape
func sme_ld1w_test(dst, src unsafe.Pointer)

//go:noescape
func sme_broadcast_test(dst unsafe.Pointer)

//go:noescape
func sme_fmul_test(dst unsafe.Pointer)

//go:noescape
func sme_fmla_test(dst unsafe.Pointer)

func TestSMEWrite(t *testing.T) {
	var result float32 = -1
	sme_write_test(unsafe.Pointer(&result), 42.0)
	t.Logf("result = %f (expected 42.0)", result)
	if result != 42.0 {
		t.Errorf("SME write failed: got %f, want 42.0", result)
	}
}

func TestSMESVEZero(t *testing.T) {
	// Allocate 16 float32s (64 bytes) and initialize with 1s
	data := make([]float32, 16)
	for i := range data {
		data[i] = 1.0
	}

	sme_sve_zero_test(unsafe.Pointer(&data[0]), 16)

	// Check that all values are now 0
	for i, v := range data {
		t.Logf("data[%d] = %f", i, v)
		if v != 0 {
			t.Errorf("data[%d] = %f, want 0", i, v)
		}
	}
}

func TestSMELD1W(t *testing.T) {
	// Source array with known value
	src := make([]float32, 16)
	for i := range src {
		src[i] = 123.0 // All same value since we're testing LD1RW (broadcast load)
	}

	// Destination array initialized to -1
	dst := make([]float32, 16)
	for i := range dst {
		dst[i] = -1.0
	}

	sme_ld1w_test(unsafe.Pointer(&dst[0]), unsafe.Pointer(&src[0]))

	// With LD1RW, all dst elements should be src[0] = 123.0
	expected := float32(123.0)
	for i, v := range dst {
		t.Logf("dst[%d] = %f (expected %f)", i, v, expected)
		if v != expected {
			t.Errorf("dst[%d] = %f, want %f", i, v, expected)
		}
	}
}

func TestSMEFMUL(t *testing.T) {
	data := make([]float32, 16)
	for i := range data {
		data[i] = -1.0
	}

	sme_fmul_test(unsafe.Pointer(&data[0]))

	// z2 should be 2.0 * 3.0 = 6.0
	expected := float32(6.0)
	for i, v := range data {
		t.Logf("data[%d] = %f (expected %f)", i, v, expected)
	}
	for i, v := range data {
		if v != expected {
			t.Errorf("data[%d] = %f, want %f", i, v, expected)
		}
	}
}

func TestSMEBroadcast(t *testing.T) {
	data := make([]float32, 16)
	for i := range data {
		data[i] = -1.0
	}

	sme_broadcast_test(unsafe.Pointer(&data[0]))

	expected := float32(42.0)
	for i, v := range data {
		t.Logf("data[%d] = %f (expected %f)", i, v, expected)
		if v != expected {
			t.Errorf("data[%d] = %f, want %f", i, v, expected)
		}
	}
}

func TestSMEFMLA(t *testing.T) {
	// Allocate 48 float32s for 3 vectors (16 each)
	data := make([]float32, 48)
	for i := range data {
		data[i] = -1.0 // Initialize to -1 to verify writes
	}

	sme_fmla_test(unsafe.Pointer(&data[0]))

	// z2[0:16] should be 7.0 (1.0 + 2.0 * 3.0)
	// z0[16:32] should be 2.0
	// z1[32:48] should be 3.0
	t.Log("z2 (after fmla - expected 7.0):")
	for i := 0; i < 16; i++ {
		t.Logf("  z2[%d] = %f", i, data[i])
	}
	t.Log("z0 (expected 2.0):")
	for i := 16; i < 32; i++ {
		t.Logf("  z0[%d] = %f", i-16, data[i])
	}
	t.Log("z1 (expected 3.0):")
	for i := 32; i < 48; i++ {
		t.Logf("  z1[%d] = %f", i-32, data[i])
	}

	// Check z2
	for i := 0; i < 16; i++ {
		if data[i] != 7.0 {
			t.Errorf("z2[%d] = %f, want 7.0", i, data[i])
		}
	}
	// Check z0
	for i := 16; i < 32; i++ {
		if data[i] != 2.0 {
			t.Errorf("z0[%d] = %f, want 2.0", i-16, data[i])
		}
	}
	// Check z1
	for i := 32; i < 48; i++ {
		if data[i] != 3.0 {
			t.Errorf("z1[%d] = %f, want 3.0", i-32, data[i])
		}
	}
}

func TestSMESimpleMatmul(t *testing.T) {
	m, n, k := 4, 4, 256
	c := make([]float32, m*n)
	for i := range c {
		c[i] = -1 // Initialize to -1 to verify writes happen
	}

	sme_simple_matmul_test(unsafe.Pointer(&c[0]), int64(m), int64(n), int64(k))

	// Each element should be K (256.0)
	for i, v := range c {
		t.Logf("c[%d] = %f (expected %f)", i, v, float32(k))
	}

	for i, v := range c {
		if v != float32(k) {
			t.Errorf("c[%d] = %f, want %f", i, v, float32(k))
		}
	}
}
