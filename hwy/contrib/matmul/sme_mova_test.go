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

//go:build darwin && arm64

package matmul

import (
	"testing"
	"unsafe"
)

//go:noescape
func sme_mova_to_za_test(dst unsafe.Pointer)

// TestSMEMOVAToZA tests writing to ZA via MOVA (Z->ZA direction)
func TestSMEMOVAToZA(t *testing.T) {
	data := make([]float32, 16)
	for i := range data {
		data[i] = -1.0
	}

	sme_mova_to_za_test(unsafe.Pointer(&data[0]))

	// z0 is set to 5.0, then MOVA'd to ZA row 0
	// Then we MOVA from ZA row 0 to z1, and store z1
	expected := float32(5.0)
	t.Log("Result (expected 5.0):")
	for i, v := range data {
		t.Logf("  data[%d] = %f", i, v)
	}
	for i, v := range data {
		if v != expected {
			t.Errorf("data[%d] = %f, want %f", i, v, expected)
		}
	}
}
