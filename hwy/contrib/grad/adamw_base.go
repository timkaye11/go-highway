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

import "github.com/ajroetker/go-highway/hwy"

//go:generate go run ../../../cmd/hwygen -input adamw_base.go -output . -targets avx2,avx512,neon,fallback -dispatch adamw

// BaseAdamWStep performs a single AdamW optimizer step with SIMD acceleration.
//
// Updates param, m (first moment), and v (second moment) in-place:
//
//	m[i] = beta1 * m[i] + (1 - beta1) * grad[i]
//	v[i] = beta2 * v[i] + (1 - beta2) * grad[i]^2
//	mHat = m[i] / (1 - beta1^step)
//	vHat = v[i] / (1 - beta2^step)
//	param[i] -= lr * (mHat / (sqrt(vHat) + epsilon) + weightDecay * param[i])
//
// Parameters:
//   - param: [n] — model parameters (updated in-place)
//   - grad:  [n] — parameter gradients
//   - m:     [n] — first moment estimates (updated in-place)
//   - v:     [n] — second moment estimates (updated in-place)
//   - lr:          learning rate
//   - beta1:       first moment decay (typically 0.9)
//   - beta2:       second moment decay (typically 0.999)
//   - epsilon:     numerical stability constant (typically 1e-8)
//   - weightDecay: decoupled weight decay coefficient
//   - step:        current optimization step (1-indexed for bias correction)
func BaseAdamWStep[T hwy.Floats](
	param, grad, m, v []T,
	lr, beta1, beta2, epsilon, weightDecay T,
	step int,
) {
	n := min(len(param), min(len(grad), min(len(m), len(v))))
	if n == 0 {
		return
	}

	// Bias correction terms
	oneMinusB1 := T(1) - beta1
	oneMinusB2 := T(1) - beta2

	b1Power := T(1)
	for range step {
		b1Power *= beta1
	}
	b2Power := T(1)
	for range step {
		b2Power *= beta2
	}
	biasCorr1 := T(1) / (T(1) - b1Power)
	biasCorr2 := T(1) / (T(1) - b2Power)

	lanes := hwy.MaxLanes[T]()

	vBeta1 := hwy.Set(beta1)
	vOneMinusB1 := hwy.Set(oneMinusB1)
	vBeta2 := hwy.Set(beta2)
	vOneMinusB2 := hwy.Set(oneMinusB2)
	vBiasCorr1 := hwy.Set(biasCorr1)
	vBiasCorr2 := hwy.Set(biasCorr2)
	vLr := hwy.Set(lr)
	vEps := hwy.Set(epsilon)
	vWd := hwy.Set(weightDecay)

	ii := 0
	for ; ii+lanes <= n; ii += lanes {
		vParam := hwy.Load(param[ii:])
		vGrad := hwy.Load(grad[ii:])
		vM := hwy.Load(m[ii:])
		vV := hwy.Load(v[ii:])

		// m = beta1 * m + (1-beta1) * grad
		vM = hwy.MulAdd(vBeta1, vM, hwy.Mul(vOneMinusB1, vGrad))
		hwy.Store(vM, m[ii:])

		// v = beta2 * v + (1-beta2) * grad^2
		gradSq := hwy.Mul(vGrad, vGrad)
		vV = hwy.MulAdd(vBeta2, vV, hwy.Mul(vOneMinusB2, gradSq))
		hwy.Store(vV, v[ii:])

		// mHat = m * biasCorr1, vHat = v * biasCorr2
		mHat := hwy.Mul(vM, vBiasCorr1)
		vHat := hwy.Mul(vV, vBiasCorr2)

		// update = mHat / (sqrt(vHat) + eps) + wd * param
		denom := hwy.Add(hwy.Sqrt(vHat), vEps)
		update := hwy.Div(mHat, denom)
		update = hwy.MulAdd(vWd, vParam, update)

		// param -= lr * update
		vParam = hwy.Sub(vParam, hwy.Mul(vLr, update))
		hwy.Store(vParam, param[ii:])
	}

	// Scalar tail
	for i := ii; i < n; i++ {
		// Update moments
		m[i] = beta1*m[i] + oneMinusB1*grad[i]
		v[i] = beta2*v[i] + oneMinusB2*grad[i]*grad[i]

		// Bias-corrected moments
		mHat := m[i] * biasCorr1
		vHat := v[i] * biasCorr2

		// Update param
		update := mHat/T(sqrt64(float64(vHat))+float64(epsilon)) + weightDecay*param[i]
		param[i] -= lr * update
	}
}
