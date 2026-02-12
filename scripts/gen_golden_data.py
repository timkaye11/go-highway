#!/usr/bin/env python3
"""Generate golden test data for go-highway backward pass operations.

Uses PyTorch autograd to compute reference gradients for all backward ops.
Outputs JSON files to hwy/contrib/grad/testdata/golden/.

Usage:
    python scripts/gen_golden_data.py [--output DIR]

Requires: torch >= 2.0
"""

import argparse
import json
import math
import os
import sys

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    print("ERROR: PyTorch is required. Install with: pip install torch", file=sys.stderr)
    sys.exit(1)


def to_list(t):
    """Convert a torch tensor to a flat list of Python floats."""
    return t.detach().cpu().float().flatten().tolist()


def gen_dense_backward(output_dir):
    """Generate golden data for dense (linear) backward."""
    torch.manual_seed(42)
    batch, in_f, out_f = 4, 16, 8

    x = torch.randn(batch, in_f, requires_grad=True)
    weight = torch.randn(out_f, in_f, requires_grad=True)
    bias = torch.randn(out_f, requires_grad=True)

    output = F.linear(x, weight, bias)
    loss = output.sum()
    loss.backward()

    data = {
        "op": "dense_backward",
        "dims": {
            "batchSize": batch,
            "inFeatures": in_f,
            "outFeatures": out_f,
        },
        "arrays": {
            "x": to_list(x),
            "weight": to_list(weight),
            "bias": to_list(bias),
            "gradOutput": to_list(torch.ones_like(output)),
            "gradInput": to_list(x.grad),
            "gradWeight": to_list(weight.grad),
            "gradBias": to_list(bias.grad),
        },
    }

    path = os.path.join(output_dir, "dense_backward_f32.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Written: {path}")


def gen_gelu_backward(output_dir):
    """Generate golden data for GELU backward."""
    torch.manual_seed(42)
    n = 64

    x = torch.randn(n, requires_grad=True)
    output = F.gelu(x, approximate="none")
    grad_output = torch.randn(n)
    output.backward(grad_output)

    data = {
        "op": "gelu_backward",
        "dims": {"n": n},
        "arrays": {
            "savedInput": to_list(x),
            "gradOutput": to_list(grad_output),
            "gradInput": to_list(x.grad),
        },
    }

    path = os.path.join(output_dir, "gelu_backward_f32.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Written: {path}")


def gen_softmax_backward(output_dir):
    """Generate golden data for softmax backward."""
    torch.manual_seed(42)
    rows, cols = 4, 8

    logits = torch.randn(rows, cols, requires_grad=True)
    probs = F.softmax(logits, dim=-1)
    grad_output = torch.randn(rows, cols)
    probs.backward(grad_output)

    data = {
        "op": "softmax_backward",
        "dims": {"rows": rows, "cols": cols},
        "arrays": {
            "savedProbs": to_list(probs),
            "gradOutput": to_list(grad_output),
            "gradInput": to_list(logits.grad),
        },
    }

    path = os.path.join(output_dir, "softmax_backward_f32.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Written: {path}")


def gen_layernorm_backward(output_dir):
    """Generate golden data for layer norm backward."""
    torch.manual_seed(42)
    batch, norm_size = 4, 16
    eps = 1e-5

    x = torch.randn(batch, norm_size, requires_grad=True)
    gamma = torch.randn(norm_size, requires_grad=True)
    beta = torch.randn(norm_size, requires_grad=True)

    output = F.layer_norm(x, [norm_size], gamma, beta, eps)
    grad_output = torch.randn(batch, norm_size)
    output.backward(grad_output)

    # Compute saved intermediates (xHat, invStd)
    with torch.no_grad():
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        inv_std = 1.0 / torch.sqrt(var + eps)
        x_hat = (x - mean) * inv_std

    data = {
        "op": "layernorm_backward",
        "dims": {"numGroups": batch, "normSize": norm_size},
        "arrays": {
            "input": to_list(x),
            "gamma": to_list(gamma),
            "beta": to_list(beta),
            "gradOutput": to_list(grad_output),
            "savedXHat": to_list(x_hat),
            "savedInvStd": to_list(inv_std.squeeze(-1)),
            "gradInput": to_list(x.grad),
            "gradGamma": to_list(gamma.grad),
            "gradBeta": to_list(beta.grad),
        },
    }

    path = os.path.join(output_dir, "layernorm_backward_f32.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Written: {path}")


def gen_sdpa_backward(output_dir):
    """Generate golden data for scaled dot-product attention backward."""
    torch.manual_seed(42)
    seq_len, kv_len, head_dim = 4, 4, 8
    scale = 1.0 / math.sqrt(head_dim)

    Q = torch.randn(seq_len, head_dim, requires_grad=True)
    K = torch.randn(kv_len, head_dim, requires_grad=True)
    V = torch.randn(kv_len, head_dim, requires_grad=True)

    # Manual SDPA: scores = Q @ K^T * scale, probs = softmax(scores), output = probs @ V
    scores = torch.matmul(Q, K.T) * scale
    probs = F.softmax(scores, dim=-1)
    output = torch.matmul(probs, V)

    grad_output = torch.randn(seq_len, head_dim)
    output.backward(grad_output)

    data = {
        "op": "sdpa_backward",
        "dims": {"seqLen": seq_len, "kvLen": kv_len, "headDim": head_dim},
        "arrays": {
            "Q": to_list(Q),
            "K": to_list(K),
            "V": to_list(V),
            "probs": to_list(probs),
            "scale": [scale],
            "gradOutput": to_list(grad_output),
            "gradQ": to_list(Q.grad),
            "gradK": to_list(K.grad),
            "gradV": to_list(V.grad),
        },
    }

    path = os.path.join(output_dir, "sdpa_backward_f32.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Written: {path}")


def gen_lora_backward(output_dir):
    """Generate golden data for LoRA backward."""
    torch.manual_seed(42)
    batch, d_in, d_out, rank = 2, 16, 8, 4
    lora_scale = 0.5

    x = torch.randn(batch, d_in, requires_grad=True)
    W = torch.randn(d_out, d_in)  # frozen
    A = torch.randn(rank, d_in, requires_grad=True)
    B = torch.randn(d_out, rank, requires_grad=True)

    # Forward: y = x @ W^T + scale * (x @ A^T) @ B^T
    h = torch.matmul(x, A.T)  # [batch, rank]
    y = torch.matmul(x, W.T) + lora_scale * torch.matmul(h, B.T)

    grad_output = torch.randn(batch, d_out)
    y.backward(grad_output)

    data = {
        "op": "lora_backward",
        "dims": {
            "batchSize": batch,
            "dIn": d_in,
            "dOut": d_out,
            "rank": rank,
        },
        "arrays": {
            "x": to_list(x),
            "W": to_list(W),
            "A": to_list(A),
            "B": to_list(B),
            "h": to_list(h),
            "scale": [lora_scale],
            "gradOutput": to_list(grad_output),
            "gradX": to_list(x.grad),
            "gradA": to_list(A.grad),
            "gradB": to_list(B.grad),
        },
    }

    path = os.path.join(output_dir, "lora_backward_f32.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Written: {path}")


def gen_adamw_step(output_dir):
    """Generate golden data for AdamW optimizer step."""
    torch.manual_seed(42)
    n = 32
    lr, beta1, beta2, eps, wd = 0.001, 0.9, 0.999, 1e-8, 0.01
    step = 5

    param = torch.randn(n)
    grad = torch.randn(n)
    m = torch.zeros(n)
    v = torch.zeros(n)

    # Run multiple steps to get realistic moment estimates
    param_copy = param.clone()
    m_copy = m.clone()
    v_copy = v.clone()

    for s in range(1, step + 1):
        # Use same grad each step for simplicity
        m_copy = beta1 * m_copy + (1 - beta1) * grad
        v_copy = beta2 * v_copy + (1 - beta2) * grad ** 2
        m_hat = m_copy / (1 - beta1 ** s)
        v_hat = v_copy / (1 - beta2 ** s)
        param_copy = param_copy - lr * (m_hat / (torch.sqrt(v_hat) + eps) + wd * param_copy)

    data = {
        "op": "adamw_step",
        "dims": {"n": n, "step": step},
        "arrays": {
            "paramInit": to_list(param),
            "grad": to_list(grad),
            "lr": [lr],
            "beta1": [beta1],
            "beta2": [beta2],
            "epsilon": [eps],
            "weightDecay": [wd],
            "paramFinal": to_list(param_copy),
            "mFinal": to_list(m_copy),
            "vFinal": to_list(v_copy),
        },
    }

    path = os.path.join(output_dir, "adamw_step_f32.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Written: {path}")


def main():
    parser = argparse.ArgumentParser(description="Generate golden test data for go-highway backward ops")
    parser.add_argument(
        "--output",
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "hwy", "contrib", "grad", "testdata", "golden"),
        help="Output directory for golden JSON files",
    )
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    print(f"Generating golden data in: {args.output}")

    gen_dense_backward(args.output)
    gen_gelu_backward(args.output)
    gen_softmax_backward(args.output)
    gen_layernorm_backward(args.output)
    gen_sdpa_backward(args.output)
    gen_lora_backward(args.output)
    gen_adamw_step(args.output)

    print(f"\nDone! Generated 7 golden data files.")


if __name__ == "__main__":
    main()
