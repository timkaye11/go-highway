# Future Implementation Plan: GPU-Accelerated Training on Apple Silicon

**Date:** 2026-02-11
**Status:** Research Complete, Implementation Pending
**Branch:** `fused_chunker`

---

## Executive Summary

Three parallel research tracks investigated paths to GPU-accelerated DeBERTa LoRA training from Go on Apple Silicon:

| Track | Approach | Verdict | Effort |
|-------|----------|---------|--------|
| **A** | go-highway native training | Viable but large | 8-12 weeks |
| **B** | MPSGraph backend for GoMLX | Most promising | 4-6 weeks |
| **C** | CoreML backward pass | Not viable | N/A |

**Recommended path: Track B (MPSGraph backend)** — provides GPU acceleration with automatic differentiation, reuses existing GoMLX graph structure, and benefits the entire GoMLX ecosystem.

---

## Current State

- Training runs on CPU via XLA at **~2.1s/step** (3 layers, batch 8, seq 128)
- Compilation cache eliminates 2h48m recompilation overhead (2s startup)
- Metal PJRT abandoned by Apple (jax-metal dead since Oct 2024)
- CoreML is inference-only, no backward pass
- go-highway is SIMD kernels only (no autodiff)

---

## Track A: go-highway Native Training Pipeline

### Overview

Build a complete DeBERTa training pipeline using go-highway SIMD kernels for forward pass and manually implemented backward passes for gradient computation.

### What go-highway Provides (Forward Pass)

| Operation | Status | Performance |
|-----------|--------|------------|
| MatMul (multiple formats) | Available | NEON/SME optimized |
| SDPA (multi-head attention) | Available | With causal + GQA |
| LayerNorm | Available | With affine transform |
| GELU / ReLU / SiLU / Tanh | Available | 117x speedup on ARM64 |
| Softmax / LogSoftmax | Available | Parallel row-wise |
| Dense (fully-connected) | Available | With fused activation |
| QKV fused projection | Available | Single matmul dispatch |

### What Must Be Implemented (Backward Pass)

**Total: ~15 backward operation types, ~2,500 lines of Go**

#### Per-Operation Backward Formulas

1. **Dense/Linear** (`dL/dx = dL/dy @ W^T`): 1 matmul — Simple
2. **GELU** (`dL/dx = dL/dy * gelu'(x)`): Elementwise — Medium
3. **Softmax** (`dL/dz = p * (dL/dp - dot(p, dL/dp))`): Needs saved `p` — Medium
4. **LayerNorm** (`dL/dx = (1/std) * [g - mean(g) - x_hat*mean(g*x_hat)]`): Needs `x_hat`, `std` — Medium
5. **Residual Add** (`dL/dx = dL/dz + dL/df(x)`): Trivial — Easy
6. **Scaled Dot-Product Attention**: Multiple matmuls — High complexity
7. **L2 Normalize** (`Sqrt(sum + eps)` pattern): Needs epsilon handling — Medium
8. **DeBERTa Disentangled Position Bias**: Content-to-position + position-to-content einsums — High
9. **Focal Loss**: BCE + focal weighting backward — Medium
10. **Contrastive Loss (InfoNCE)**: Temperature-scaled similarity backward — Medium
11. **Coherence Loss**: Margin-based consecutive chunk embedding loss — Medium

#### LoRA Gradient Formulas (Core of the System)

For each LoRA layer with input `x`, upstream gradient `g = dL/dy`:

```
dL/dB = scale * (x @ A)^T @ g           [r, d_out]
dL/dA = x^T @ (scale * g @ B^T)         [d_in, r]
dL/dx = g @ W^T + scale * g @ B^T @ A^T [N, d_in]  (for chain rule)
```

Where W^T term is needed even though W is frozen, because gradient must flow through frozen path for backpropagation to earlier layers.

#### Memory Requirements (Activation Checkpointing)

Must save from forward pass for backward:
- Per attention layer: Q, K, V, attention probs P (~4 tensors)
- Per FFN: pre-GELU activations (~1 tensor)
- Per LayerNorm: normalized `x_hat` and `std` (~2 tensors)
- Per LoRA: input tensor `x` (~1 tensor per LoRA layer)

For 3 layers, batch 8, seq 128, hidden 768:
- ~3 × (4 + 1 + 2×2 + 2×1) × 8 × 128 × 768 × 4 bytes ≈ ~110 MB

### Architecture Design

```
gopeft/highway/
├── tensor.go           # Simple tensor wrapper (flat []float32 + shape)
├── forward.go          # DeBERTa forward pass using go-highway nn/matmul
├── backward.go         # Manual backward pass implementations
├── lora_grad.go        # LoRA-specific gradient computation
├── adamw.go            # AdamW optimizer (simple, no XLA needed)
├── trainer.go          # Training loop (data loading → forward → backward → update)
└── deberta_highway.go  # DeBERTa model definition for highway backend
```

### Estimated Performance

go-highway MatMul on M4 Pro achieves ~240 GFLOPS (NEON) with SME potentially higher.
XLA CPU on M4 Pro achieves ~100-150 GFLOPS (estimated from 2.1s/step timings).

**Expected speedup: 1.5-2.5x** over XLA CPU for same model configuration.
- Most of the gain is from avoiding XLA overhead and better SIMD utilization
- Still CPU-bound — does not use GPU

### Effort Estimate: 8-12 weeks

| Component | Effort | Lines |
|-----------|--------|-------|
| Tensor wrapper + utilities | 1 week | ~500 |
| Forward pass (encoder + heads) | 2 weeks | ~1,000 |
| Backward pass (all ops) | 3 weeks | ~2,500 |
| Loss functions + backward | 1 week | ~500 |
| AdamW optimizer | 0.5 weeks | ~200 |
| Training loop + data pipeline | 1 week | ~500 |
| Testing + debugging | 2-4 weeks | ~1,000 |

### Risks

- Numerical stability in manual backward passes (NaN, gradient explosion)
- DeBERTa disentangled attention backward is complex and error-prone
- Performance may not significantly exceed XLA CPU
- No automatic gradient checking (must verify manually)

---

## Track B: MPSGraph Backend for GoMLX (RECOMMENDED)

### Overview

Create a new GoMLX backend that uses Apple's MPSGraph framework for GPU-accelerated computation with automatic differentiation. This is the most promising path because:

1. **MPSGraph has built-in autodiff** via `gradientTensors()` — no manual backward pass
2. **GPU execution** on M4 GPU cores (up to 20 GPU cores, ~16 TFLOPS FP32)
3. **Reuses existing GoMLX model code** — no rewrite of DeBERTa
4. **Benefits entire GoMLX ecosystem** — not just gopeft

### MPSGraph Capabilities

| Feature | Support |
|---------|---------|
| Automatic differentiation | `MPSGraphTensor.gradients(of:with:)` |
| MatMul / Linear | Full support |
| Attention patterns | Via matmul + softmax composition |
| LayerNorm | `MPSGraph.normalization()` |
| GELU / activations | Full support |
| AdamW optimizer | `MPSGraphAdamOptimizer` |
| BFloat16 | macOS Sonoma+ |
| Unified memory | Zero CPU-GPU transfer overhead |

**Key discovery:** PyTorch MPS backend achieves **13.5x acceleration** over CPU and **121x speedup** for matrix multiplication on M-series chips. Our XLA CPU is ~2.1s/step; with GPU this could drop to **~0.15-0.3s/step**.

### Architecture

```
go-mpsgraph/                       # New Go module
├── mpsgraph.h / mpsgraph.m        # Objective-C++ bridge (CGo)
│   ├── graph_create()             # Create MPSGraph instance
│   ├── graph_add_op()             # Add operations (matmul, conv, etc.)
│   ├── graph_gradient()           # Call gradientTensors() for autodiff
│   ├── graph_compile()            # Compile to MPSGraphExecutable
│   └── graph_run()                # Execute on GPU
├── backend.go                      # GoMLX Backend interface implementation
├── ops.go                          # Map GoMLX ops → MPSGraph ops
└── device.go                       # GPU device management
```

The bridge follows the same pattern as go-coreml (`bridge.{h,m}` via CGo), but targets MPSGraph instead of CoreML:

```go
// backend.go — implements backends.Backend interface
type MPSGraphBackend struct {
    device unsafe.Pointer  // MTLDevice
    graph  unsafe.Pointer  // MPSGraph
}

func (b *MPSGraphBackend) Compile(program []byte) (backends.Executable, error) {
    // Parse StableHLO → build MPSGraph ops → compile to MPSGraphExecutable
}
```

### Implementation Phases

#### Phase 1: Core Backend (2 weeks)
- Objective-C++ bridge for MPSGraph creation and execution
- CGo bindings for graph building, compilation, and execution
- Basic ops: Add, Mul, MatMul, Reshape, Transpose
- Device management and memory allocation
- GoMLX Backend interface implementation

#### Phase 2: Transformer Ops (2 weeks)
- Reduction ops: ReduceSum, ReduceMean, ReduceMax
- Comparison ops: LessThan, Where, Equal
- Activation functions: GELU, Softmax, Sigmoid
- LayerNorm components
- Gather/Scatter (for embeddings and indexing)
- Einsum / DotGeneral (for attention)

#### Phase 3: Autodiff + Training (1-2 weeks)
- Wire `gradientTensors()` to GoMLX's gradient graph building
- Map GoMLX's `BuildTrainableVariablesGradientsGraph` to MPSGraph autodiff
- Variable management (weight tensors in GPU memory)
- AdamW optimizer (can use MPSGraph's built-in or implement via ops)

#### Phase 4: Testing + Optimization (1-2 weeks)
- Numerical validation against XLA CPU results
- Performance benchmarking
- Memory optimization (unified memory best practices)
- BFloat16 support for further speedup

### Expected Performance

| Metric | XLA CPU (current) | MPSGraph GPU (expected) |
|--------|-------------------|------------------------|
| MatMul throughput | ~100-150 GFLOPS | ~8-16 TFLOPS FP32 |
| Step time (3L, B8, S128) | 2.1s | 0.15-0.3s |
| Step time (12L, B16, S256) | 57s+ | 3-5s |
| Compilation | 2s (cached) | <1s |
| Full training (16,875 steps) | ~9.4 hours | ~0.7-1.4 hours |

### Key Risks

1. **StableHLO → MPSGraph mapping complexity**: GoMLX emits StableHLO IR; need to parse and map ~50-100 ops to MPSGraph equivalents. Alternative: build MPSGraph directly from GoMLX's computation graph (bypass StableHLO).

2. **No C API for MPSGraph**: Must write Objective-C++ wrapper. The go-coreml project already demonstrates this pattern successfully.

3. **Gather/Scatter on GPU**: These are often slow on GPU. DeBERTa uses Gather for embeddings and relative position lookup. May need specialized implementations.

4. **Autodiff coverage**: MPSGraph's `gradientTensors()` may not cover all operations used in DeBERTa. Need to verify coverage for: Gather, Scatter, Where, custom attention patterns.

### Effort Estimate: 4-6 weeks

| Component | Effort | Lines |
|-----------|--------|-------|
| Obj-C++ bridge | 1.5 weeks | ~1,500 |
| GoMLX Backend interface | 1 week | ~800 |
| Op mapping (50+ ops) | 1.5 weeks | ~2,000 |
| Autodiff integration | 0.5 weeks | ~300 |
| Testing + debugging | 1-2 weeks | ~500 |

---

## Track C: CoreML Backward Pass (NOT VIABLE)

### Findings

- CoreML's `MLUpdateTask` only supports Conv2D and Dense layer updates — **no attention, no transformers**
- CoreML ML Program format (MIL) has **zero training support**
- ANE (Apple Neural Engine) is **inference-only** — cannot compute gradients
- go-coreml bridge exposes only `coreml_model_predict()` — no training APIs
- Adding backward pass to CoreML would require Apple to extend the framework itself

### Conclusion

**CoreML is architecturally unsuitable for training.** Apple designed it for on-device inference deployment. For training on Apple Silicon, use MPSGraph (Track B) which is the actual training framework underlying PyTorch MPS.

---

## Track D: Alternative — Go Autodiff Libraries

### Investigated Libraries

1. **pbenner/autodiff**: Go library with reverse-mode AD, supports Adam/BFGS. However, uses scalar-mode differentiation — not suitable for tensor operations at DeBERTa scale.

2. **itsubaki/autograd**: PyTorch-like define-by-run AD in Go. Dynamic graph with `Backward()`. However, no SIMD acceleration — pure Go scalar operations.

3. **GoMLX's built-in autodiff**: The current system. Works well but tied to XLA backend (CPU-only on Mac).

### Verdict

None of the existing Go AD libraries combine:
- Tensor-level operations (not scalar)
- GPU acceleration
- Sufficient op coverage for transformers

This reinforces that **Track B (MPSGraph)** is the right approach — it provides both GPU execution and automatic differentiation in one package.

---

## Recommended Execution Order

### Phase 1: Immediate (Now)
- [x] Complete current training run (3L, B8, S128, 3 epochs, ~9.4h on CPU)
- [ ] Validate training metrics and model quality

### Phase 2: Short-term (Next 2 weeks)
- [ ] Prototype MPSGraph Objective-C++ bridge
- [ ] Implement basic ops (Add, Mul, MatMul) and verify GPU execution
- [ ] Benchmark MatMul throughput vs XLA CPU

### Phase 3: Medium-term (Weeks 3-6)
- [ ] Complete MPSGraph backend with transformer ops
- [ ] Wire autodiff (`gradientTensors()`)
- [ ] Run DeBERTa forward pass on GPU — validate outputs match XLA
- [ ] Run full training loop on GPU — validate loss convergence

### Phase 4: Long-term (Optional)
- [ ] go-highway native training as fallback/alternative
- [ ] Upstream MPSGraph backend to GoMLX project
- [ ] BFloat16 optimization for 2x throughput
- [ ] Multi-device support (if applicable)

---

## Appendix: LoRA Gradient Reference

For manual gradient implementation (Track A or debugging Track B):

### Forward Pass (per LoRA layer)
```
h = x @ A           // [N, d_in] @ [d_in, r] = [N, r]
y = h @ B * scale   // [N, r] @ [r, d_out] = [N, d_out]
```

### Backward Pass (given upstream gradient g = dL/dy)
```
dL/dB = scale * h^T @ g        // [r, N] @ [N, d_out] = [r, d_out]
dL/dA = x^T @ (scale * g @ B^T)  // [d_in, N] @ [N, r] = [d_in, r]
dL/dx = g @ W^T + scale * g @ (BA)^T  // [N, d_in] (for chain rule)
```

### AdamW Update (per parameter p)
```
m = β1 * m + (1 - β1) * grad
v = β2 * v + (1 - β2) * grad^2
m_hat = m / (1 - β1^t)
v_hat = v / (1 - β2^t)
p = p - lr * (m_hat / (sqrt(v_hat) + ε) + weight_decay * p)
```

---

## Appendix: Existing Go-Metal Precedents

The following projects demonstrate Go → Metal/MPS patterns:

1. **go-coreml** (`/Users/tim/Documents/af/go-coreml`): Objective-C++ bridge for CoreML inference. 70+ MIL operations. Proven CGo pattern.

2. **go-metal** (github.com/nicholasgasior/go-metal): Go library accessing MPSGraph for training. Claims "121x speedup" for matrix multiplication on M-series. Implements Adam, SGD, and other optimizers via Metal. **Key validation that this approach works.**

3. **PyTorch MPS backend**: C++ wrapper around MPSGraph. Maps PyTorch ops → MPSGraph ops. Full training support including autograd. Proves the API surface is sufficient for transformer training.
