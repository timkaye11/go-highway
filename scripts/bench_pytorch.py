#!/usr/bin/env python3
"""Benchmark PyTorch operations to compare with go-highway grad benchmarks.

Matches the exact dimensions used in the Go benchmarks.
"""

import time
import torch
import torch.nn.functional as F

WARMUP = 50
ITERS = 500

def bench(name, fn):
    # warmup
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize() if torch.cuda.is_available() else None

    start = time.perf_counter_ns()
    for _ in range(ITERS):
        fn()
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed_ns = time.perf_counter_ns() - start
    per_op_us = elapsed_ns / ITERS / 1000
    print(f"{name:40s}  {per_op_us:10.1f} μs/op")


def bench_gelu_backward():
    """GELUBackward: 8 rows x 768 cols = 6144 elements"""
    x = torch.randn(8, 768, requires_grad=True)
    y = F.gelu(x)
    grad_out = torch.randn_like(y)

    def fn():
        if x.grad is not None:
            x.grad.zero_()
        y.backward(grad_out, retain_graph=True)

    bench("GELUBackward (8x768)", fn)


def bench_dense_backward():
    """DenseBackward: batch=8, in=768, out=768"""
    layer = torch.nn.Linear(768, 768, bias=True)
    x = torch.randn(8, 768, requires_grad=True)
    y = layer(x)
    grad_out = torch.randn_like(y)

    def fn():
        if x.grad is not None:
            x.grad.zero_()
        layer.zero_grad()
        y.backward(grad_out, retain_graph=True)

    bench("DenseBackward (8x768x768)", fn)


def bench_layernorm_backward():
    """LayerNormBackward: 32 groups x normSize=768"""
    ln = torch.nn.LayerNorm(768)
    x = torch.randn(32, 768, requires_grad=True)
    y = ln(x)
    grad_out = torch.randn_like(y)

    def fn():
        if x.grad is not None:
            x.grad.zero_()
        ln.zero_grad()
        y.backward(grad_out, retain_graph=True)

    bench("LayerNormBackward (32x768)", fn)


def bench_softmax_backward():
    """SoftmaxBackward: 64 rows x 1024 cols"""
    x = torch.randn(64, 1024, requires_grad=True)
    y = F.softmax(x, dim=-1)
    grad_out = torch.randn_like(y)

    def fn():
        if x.grad is not None:
            x.grad.zero_()
        y.backward(grad_out, retain_graph=True)

    bench("SoftmaxBackward (64x1024)", fn)


def bench_sdpa_backward():
    """SDPABackward: seqLen=64, kvLen=64, headDim=64"""
    Q = torch.randn(1, 1, 64, 64, requires_grad=True)
    K = torch.randn(1, 1, 64, 64, requires_grad=True)
    V = torch.randn(1, 1, 64, 64, requires_grad=True)
    out = F.scaled_dot_product_attention(Q, K, V)
    grad_out = torch.randn_like(out)

    def fn():
        for t in [Q, K, V]:
            if t.grad is not None:
                t.grad.zero_()
        out.backward(grad_out, retain_graph=True)

    bench("SDPABackward (s64,kv64,d64)", fn)


def bench_lora_backward():
    """LoRABackward: batch=8, dIn=768, dOut=768, rank=16"""
    batch, dIn, dOut, rank = 8, 768, 768, 16
    scale = 0.5

    x = torch.randn(batch, dIn, requires_grad=True)
    W = torch.randn(dOut, dIn)
    A = torch.randn(rank, dIn, requires_grad=True)
    B = torch.randn(dOut, rank, requires_grad=True)

    # Forward: y = x @ W^T + scale * (x @ A^T) @ B^T
    h = x @ A.T
    y = x @ W.T + scale * h @ B.T
    grad_out = torch.randn_like(y)

    def fn():
        for t in [x, A, B]:
            if t.grad is not None:
                t.grad.zero_()
        y.backward(grad_out, retain_graph=True)

    bench("LoRABackward (8x768x768,r16)", fn)


def bench_adamw_step():
    """AdamW step: 768*768 = 589824 parameters"""
    n = 768 * 768
    param = torch.randn(n)
    grad = torch.randn(n)
    m = torch.zeros(n)
    v = torch.zeros(n)
    lr, beta1, beta2, eps, wd = 0.001, 0.9, 0.999, 1e-8, 0.01

    def fn():
        nonlocal m, v
        m_new = beta1 * m + (1 - beta1) * grad
        v_new = beta2 * v + (1 - beta2) * grad * grad
        m_hat = m_new / (1 - beta1 ** 5)
        v_hat = v_new / (1 - beta2 ** 5)
        param.sub_(lr * (m_hat / (v_hat.sqrt() + eps) + wd * param))
        m.copy_(m_new)
        v.copy_(v_new)

    bench("AdamWStep (589824 params)", fn)


def bench_adamw_optimizer():
    """AdamW via torch.optim.AdamW: 768*768 params"""
    n = 768 * 768
    param = torch.nn.Parameter(torch.randn(n))
    optimizer = torch.optim.AdamW([param], lr=0.001, betas=(0.9, 0.999),
                                   eps=1e-8, weight_decay=0.01)

    def fn():
        optimizer.zero_grad()
        param.grad = torch.randn(n)
        optimizer.step()

    bench("AdamW optimizer (589824 params)", fn)


if __name__ == "__main__":
    print(f"PyTorch {torch.__version__}")
    print(f"Device: CPU (threads={torch.get_num_threads()})")
    print(f"Warmup={WARMUP}, Iters={ITERS}")
    print("-" * 60)
    bench_gelu_backward()
    bench_dense_backward()
    bench_layernorm_backward()
    bench_softmax_backward()
    bench_sdpa_backward()
    bench_lora_backward()
    bench_adamw_step()
    bench_adamw_optimizer()
