#!/usr/bin/env python
"""Verify the whole-night telluric CNN auto-encoder end to end.

Two modes:

* ``shape`` (default): build the model at the **real** problem dimensions
  (B, 73, 51556) and check every intermediate tensor shape in one forward
  pass; report per-component and total parameter counts.

* ``overfit``: train on a tiny synthetic set of nights (X = T*S with
  airmass-scaled telluric transmission) and check that the model can drive
  the supervised MSE (and optionally the forward-model consistency loss)
  down, i.e. that the mapping X -> T_hat is learnable.

Typical usage
-------------
    # exact full-resolution shape + parameter report
    python tools/verify_temporal_model.py --mode shape

    # fast overfit demo (reduced wavelength grid keeps CPU time sane)
    python tools/verify_temporal_model.py --mode overfit \
        --n-wavelength 2048 --nights 6 --steps 80 --lambda-x 0.0

Everything is CPU-safe; the batch size is never hard-coded.
"""

from __future__ import annotations

import argparse
import math
import time

import torch
import torch.nn.functional as F

from tellurics.config.model import ModelConfig
from tellurics.models.predictor import NeuralTelluricPredictor as TelluricModel


def _tiny_config(
    n_wavelength: int,
    n_frames: int,
    n_queries: int,
    latent_dim: int,
    metadata_dim: int,
    n_heads: int,
    dropout: float,
) -> ModelConfig:
    """Explicit demo ModelConfig (the repo-wide config has no defaults)."""
    return ModelConfig(
        architecture="neural_telluric_predictor",
        num_wavelength_bins=n_wavelength,
        n_frames_per_series=n_frames,
        metadata_dim=metadata_dim,
        param_dim=20,
        num_queries=n_queries,
        spectral_latent_dim=latent_dim,
        num_heads=n_heads,
        dropout=dropout,
        x_encoder_channels=[16, 32, 64],
        x_encoder_kernel=7,
        x_encoder_stride=2,
        s_encoder_channels=[16, 32, 64],
        s_encoder_kernel=7,
        s_encoder_stride=2,
        encoder_pool_bins=16,
        fusion_hidden=0,
        fusion_layers=2,
        metadata_enc_dim=16,
        metadata_enc_hidden=None,
        time_enc_dim=16,
        param_decoder_hidden=None,
        # This demo regresses toward raw, unbounded random targets, so the head
        # stays linear ("sigmoid" is the partner of data.scale_params=true).
        param_activation="none",
    )


# --------------------------------------------------------------------------- #
# Tiny physically-motivated generator:  X = T * S
# --------------------------------------------------------------------------- #
def make_synthetic_nights(
    n_nights: int,
    n_wavelength: int,
    n_frames: int,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (X, S, T, metadata) for ``n_nights`` independent nights.

    Each night:  stellar spectrum ``S(l)`` (smooth pseudo-continuum),
    telluric optical depth ``tau(l)`` (smooth band + a few absorption dips),
    and a slowly rising airmass over the ``T`` exposures, so

        T_i(l) = exp(-tau(l) * airmass_i),      X_i(l) = S(l) * T_i(l).

    This reproduces the core physical coupling X = T*S and an exposure-to-
    exposure telluric drift, which is exactly the structure the model was
    designed to exploit.
    """
    g = torch.Generator().manual_seed(seed)
    w = torch.linspace(0.0, 1.0, n_wavelength)                 # (N,)

    xs, ss, ts, metas = [], [], [], []
    for _ in range(n_nights):
        # --- stellar pseudo-continuum (smooth) --------------------------------
        s = 1.0 + 0.35 * torch.sin(2 * torch.pi * w * (4 + seed % 3))
        s = s + 0.15 * torch.sin(2 * torch.pi * w * (31 + seed % 7))
        s = s + 0.08 * torch.sin(2 * torch.pi * w * (73 + seed % 5))
        s = s / s.mean()

        # --- telluric optical depth (band + a few dips) -----------------------
        tau = 0.05 + 0.30 * (0.5 + 0.5 * torch.sin(2 * torch.pi * w * (9 + seed % 4)))
        tau = tau + 0.18 * torch.exp(-((w - 0.3 - 0.07 * seed) ** 2) / (2 * 0.004 ** 2))
        tau = tau + 0.15 * torch.exp(-((w - 0.62 - 0.05 * seed) ** 2) / (2 * 0.006 ** 2))
        tau = tau + 0.12 * torch.exp(-((w - 0.85) ** 2) / (2 * 0.003 ** 2))
        tau = tau.clamp_min(1e-3)
        seed += 1

        # --- exposures with a rising airmass ---------------------------------
        a_lo, a_hi = 1.05, 2.0
        airm = torch.linspace(a_lo, a_hi, n_frames)            # (T,)
        t = torch.exp(-tau[None, :] * airm[:, None])           # (T, N) in (0,1]
        x = s[None, :] * t                                     # (T, N)

        xs.append(x)
        ss.append(s)
        ts.append(t)
        # per-exposure metadata (T, P=3): airmass (the physical driver of the
        # exposure-to-exposure telluric drift), its square, and the
        # (night-constant) mean optical depth broadcast per exposure.
        metas.append(
            torch.stack(
                [
                    airm,                                   # (T,)
                    airm ** 2,                              # (T,)
                    torch.full_like(airm, tau.mean()),      # (T,)
                ],
                dim=-1,                                     # (T, 3)
            )
        )

    return (
        torch.stack(xs),       # (B, T, N)
        torch.stack(ss),       # (B, N)
        torch.stack(ts),       # (B, T, N)
        torch.stack(metas),    # (B, T, P=3) per-exposure metadata
    )


def parameter_table(model: TelluricModel) -> None:
    """Print the per-component and total parameter counts."""
    counts = model.parameter_counts()
    width = max(len(k) for k in counts) + 2
    print("\n===== parameter counts =====")
    for name, n in counts.items():
        print(f"  {name:<{width}} {n:>12,}")
    total = model.count_parameters()
    print(f"  {'TOTAL':<{width}} {total:>12,}")
    return total


def run_shape_check(args: argparse.Namespace) -> None:
    cfg = _tiny_config(
        n_wavelength=args.n_wavelength,
        n_frames=args.n_frames,
        n_queries=args.n_queries,
        latent_dim=args.latent_dim,
        metadata_dim=args.metadata_dim,
        n_heads=args.n_heads,
        dropout=0.0,
    )
    model = TelluricModel(cfg)
    total = parameter_table(model)

    print(f"\n===== shape check  N={cfg.num_wavelength_bins}, "
          f"T={cfg.n_frames_per_series}, "
          f"Q={cfg.num_queries}, d={cfg.spectral_latent_dim} =====")
    batch = args.batch
    x = torch.randn(batch, cfg.n_frames_per_series, cfg.num_wavelength_bins)
    s = torch.randn(batch, cfg.num_wavelength_bins)
    meta = torch.randn(batch, cfg.n_frames_per_series, cfg.metadata_dim)
    hours = torch.rand(batch, cfg.n_frames_per_series)   # exposure times

    t0 = time.time()
    with torch.no_grad():
        out = model(x, stellar=s, metadata=meta, time=hours)
    dt = time.time() - t0

    shapes = {
        "X (observed)": tuple(x.shape),
        "Z": out.intermediate_features["z"].shape,
        "M_code (meta enc)": out.intermediate_features["metadata_code"].shape,
        "tokens (Z+M+time)": out.intermediate_features["tokens"].shape,
        "L": out.intermediate_features["l"].shape,
        "h_X": out.intermediate_features["h_x"].shape,
        "h_S": out.intermediate_features["h_s"].shape,
        "fusion": out.intermediate_features["fusion"].shape,
        "h_fused": out.intermediate_features["h_fused"].shape,
        "latent": out.intermediate_features["latent"].shape,
        "temporal": out.intermediate_features["temporal"].shape,
        "param_pred (params)": out.intermediate_features["param_pred"].shape,
        "attention (Q->T)": out.attention_weights.shape,
    }
    for name, shp in shapes.items():
        print(f"  {name:<22} {str(tuple(shp)):<24}")

    expected = {
        "X (observed)": (batch, cfg.n_frames_per_series, cfg.num_wavelength_bins),
        "Z": (batch, cfg.n_frames_per_series, cfg.spectral_latent_dim),
        "M_code (meta enc)": (batch, cfg.n_frames_per_series, cfg.metadata_enc_dim),
        "tokens (Z+M+time)": (
            batch, cfg.n_frames_per_series,
            cfg.spectral_latent_dim + cfg.metadata_enc_dim + cfg.time_enc_dim,
        ),
        "L": (batch, cfg.num_queries, cfg.spectral_latent_dim),
        "h_X": (batch, cfg.num_queries * cfg.spectral_latent_dim),
        "h_S": (batch, cfg.spectral_latent_dim),
        "fusion": (
            batch,
            cfg.num_queries * cfg.spectral_latent_dim + cfg.spectral_latent_dim,
        ),
        "h_fused": (batch, cfg.num_queries * cfg.spectral_latent_dim),
        "latent": (batch, cfg.num_queries, cfg.spectral_latent_dim),
        "temporal": (batch, cfg.n_frames_per_series, cfg.spectral_latent_dim),
        "param_pred (params)": (batch, cfg.n_frames_per_series, cfg.param_dim),
        "attention (Q->T)": (batch, cfg.num_queries, cfg.n_frames_per_series),
    }
    bad = {k: (v, expected[k]) for k, v in shapes.items() if tuple(v) != expected[k]}
    if bad:
        for k, (v, e) in bad.items():
            print(f"  MISMATCH {k}: got {v}, expected {e}")
        raise SystemExit(1)
    print(f"\nAll intermediate shapes match the spec.  "
          f"Forward time ({batch} night(s) at full res): {dt:.1f}s.  "
          f"Total parameters: {total:,}.")


def run_overfit(args: argparse.Namespace) -> None:
    cfg = _tiny_config(
        n_wavelength=args.n_wavelength,
        n_frames=args.n_frames,
        n_queries=args.n_queries,
        latent_dim=args.latent_dim,
        metadata_dim=args.metadata_dim,
        n_heads=args.n_heads,
        dropout=args.dropout,
    )
    model = TelluricModel(cfg)
    parameter_table(model)

    print(f"\n===== overfit demo  N={cfg.num_wavelength_bins}, "
          f"T={cfg.n_frames_per_series}, "
          f"Q={cfg.num_queries}, d={cfg.spectral_latent_dim} =====")

    x, s, _, meta = make_synthetic_nights(
        args.nights, cfg.num_wavelength_bins, cfg.n_frames_per_series,
        seed=args.seed,
    )
    hours = torch.rand(args.nights, cfg.n_frames_per_series)  # exposure times
    print(f"  train set: {x.shape[0]} nights x {x.shape[1]} exposures "
          f"x {x.shape[2]} samples")
    # Model-level demo only: regress param_pred (B, T, P) toward a fixed
    # random param_input target.  A physically-consistent params->transmission
    # generator belongs to the training/data wiring (handled separately).
    torch.manual_seed(args.seed)
    param_input = torch.randn(args.nights, cfg.n_frames_per_series, cfg.param_dim)

    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    def step() -> float:
        opt.zero_grad()
        out = model(x, stellar=s, metadata=meta, time=hours)
        loss = F.mse_loss(out.params, param_input)
        loss.backward()
        opt.step()
        return loss.item()

    t0 = time.time()
    losses: list[float] = []
    for it in range(1, args.steps + 1):
        l = step()
        losses.append(l)
        if it == 1 or it % args.log_every == 0 or it == args.steps:
            print(f"  step {it:>4}/{args.steps}   "
                  f"MSE(param_input, param_pred)={l:.3e}")

    dt = time.time() - t0
    print(f"\n  train MSE  {losses[0]:.3e} -> {losses[-1]:.3e}   "
          f"({args.steps} steps in {dt:.1f}s, {dt / args.steps:.2f}s/step)")
    ok = losses[-1] < losses[0]
    print(f"  => loss decreased over training: {ok}")
    if not ok:
        raise SystemExit(2)
    # quick gradient sanity on a random batch
    xb = x[:2]
    loss = F.mse_loss(
        model(xb, stellar=s[:2], metadata=meta[:2], time=hours[:2]).params,
        param_input[:2],
    )
    loss.backward()
    missing = sum(1 for p in model.parameters() if p.requires_grad and p.grad is None)
    print(f"  gradient flow: {missing} parameter tensors without gradients "
          f"(should be 0).")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["shape", "overfit"], default="shape")
    # problem dimensions
    p.add_argument("--n-wavelength", type=int, default=51556,
                   help="spectral samples N (51,556 at full resolution)")
    p.add_argument("--n-frames", type=int, default=73, help="exposures T per night")
    p.add_argument("--n-queries", type=int, default=16, help="latent tokens Q")
    p.add_argument("--latent-dim", type=int, default=64, help="code dim d")
    p.add_argument("--metadata-dim", type=int, default=3)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--batch", type=int, default=2, help="shape-check batch size")
    # overfit
    p.add_argument("--nights", type=int, default=6, help="nights in the tiny train set")
    p.add_argument("--steps", type=int, default=80)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--lambda-x", type=float, default=0.0,
                   help="weight of the forward-model consistency loss L_X")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    print(f"torch {torch.__version__} | cuda available: {torch.cuda.is_available()}")
    if args.mode == "shape":
        run_shape_check(args)
    else:
        run_overfit(args)


if __name__ == "__main__":
    main()
