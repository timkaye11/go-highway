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
func sme_fmopa_only_test(dst unsafe.Pointer)

// TestSMEFMOPAOnly tests just the FMOPA instruction
// without any ZA store - just to see if it runs or crashes
func TestSMEFMOPAOnly(t *testing.T) {
	// Dummy destination - we won't actually store anything
	data := make([]float32, 16)
	for i := range data {
		data[i] = -1.0
	}

	sme_fmopa_only_test(unsafe.Pointer(&data[0]))

	t.Log("FMOPA test completed without crash")
	// Just check that we returned successfully
	// The data should still be -1.0 since we're not storing
	for i, v := range data {
		t.Logf("data[%d] = %f", i, v)
	}
}
