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

package gradcheck

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
)

// GoldenData holds a single golden test case loaded from a JSON file.
// Each JSON file contains named float32 arrays and integer dimensions.
type GoldenData struct {
	// Op is the operation name (e.g., "dense_backward", "gelu_backward").
	Op string `json:"op"`

	// Dims stores integer dimensions (e.g., batchSize, inFeatures, outFeatures).
	Dims map[string]int `json:"dims"`

	// Arrays stores named float32 arrays (inputs, expected outputs).
	Arrays map[string][]float32 `json:"arrays"`
}

// goldenDir returns the path to the testdata/golden/ directory relative to
// the grad package. It uses runtime.Caller to locate the source tree.
func goldenDir() string {
	_, filename, _, ok := runtime.Caller(0)
	if !ok {
		return ""
	}
	// gradcheck/golden.go -> gradcheck/ -> grad/ -> grad/testdata/golden/
	gradDir := filepath.Dir(filepath.Dir(filename))
	return filepath.Join(gradDir, "testdata", "golden")
}

// LoadGolden loads a golden test data file from testdata/golden/<name>.json.
// Returns nil and no error if the file does not exist (allowing tests to skip).
func LoadGolden(name string) (*GoldenData, error) {
	dir := goldenDir()
	if dir == "" {
		return nil, nil
	}

	path := filepath.Join(dir, name+".json")
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, fmt.Errorf("reading golden data %s: %w", name, err)
	}

	var gd GoldenData
	if err := json.Unmarshal(data, &gd); err != nil {
		return nil, fmt.Errorf("parsing golden data %s: %w", name, err)
	}
	return &gd, nil
}

// LoadGoldenMulti loads a golden file containing multiple test cases.
func LoadGoldenMulti(name string) ([]GoldenData, error) {
	dir := goldenDir()
	if dir == "" {
		return nil, nil
	}

	path := filepath.Join(dir, name+".json")
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, fmt.Errorf("reading golden data %s: %w", name, err)
	}

	var gds []GoldenData
	if err := json.Unmarshal(data, &gds); err != nil {
		return nil, fmt.Errorf("parsing golden data %s: %w", name, err)
	}
	return gds, nil
}

// GetArray retrieves a named array from the golden data, returning an error
// if the array is not found.
func (gd *GoldenData) GetArray(name string) ([]float32, error) {
	arr, ok := gd.Arrays[name]
	if !ok {
		return nil, fmt.Errorf("golden data %q: array %q not found", gd.Op, name)
	}
	return arr, nil
}

// GetDim retrieves a named dimension from the golden data, returning an error
// if the dimension is not found.
func (gd *GoldenData) GetDim(name string) (int, error) {
	v, ok := gd.Dims[name]
	if !ok {
		return 0, fmt.Errorf("golden data %q: dim %q not found", gd.Op, name)
	}
	return v, nil
}

// CompareArrays compares two float32 arrays with the given relative tolerance.
// Returns a list of mismatches. Empty list means arrays match.
func CompareArrays(name string, got, want []float32, relTol float64) []GradMismatch {
	n := min(len(got), len(want))
	var mismatches []GradMismatch

	for i := range n {
		diff := math.Abs(float64(got[i] - want[i]))
		denom := math.Max(1.0, math.Abs(float64(want[i])))
		relErr := diff / denom

		if relErr > relTol && diff > 1e-6 {
			mismatches = append(mismatches, GradMismatch{
				Index:      i,
				Analytical: float64(got[i]),
				Numerical:  float64(want[i]),
				RelError:   relErr,
			})
		}
	}

	if len(got) != len(want) {
		mismatches = append(mismatches, GradMismatch{
			Index:    -1,
			RelError: 1.0,
		})
	}

	return mismatches
}
