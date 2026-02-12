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

package grad

import stdmath "math"

// Scalar math helpers to avoid import aliasing conflicts across files.

func exp64(x float64) float64 { return stdmath.Exp(x) }
func log64(x float64) float64 { return stdmath.Log(x) }
func sqrt64(x float64) float64 { return stdmath.Sqrt(x) }
func abs64(x float64) float64 { return stdmath.Abs(x) }
func pow64(x, y float64) float64 { return stdmath.Pow(x, y) }
