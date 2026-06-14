"""
exp1b_input_probe.py
====================
Experiment 1b: Input-Layer Baseline for Reward-Hacking Detection.

This is a critical control for the negative result reported in exp1.
If hacking is linearly decodable at every loop depth INCLUDING in the
pre-RL base model, the natural question is: is it also decodable from
the input embedding layer ALONE, with no Transformer computation at all?

If yes, then depth-indexed probes do not provide a monitoring
affordance beyond what is available from inspecting the embedding
layer — i.e., depth is redundant for this exploit class. This closes
the most obvious reviewer objection to exp1's null result.

Method
------
For each probe trajectory:
  1. Take the input_ids and look up their embeddings via model.embed.
  2. Mean-pool across sequence positions to get a single vector per example.
  3. Train a linear probe (same architecture as exp1) on hacking vs genuine.
  4. Compare AUROC to exp1's depth-T probe AUROCs.

We additionally include three trivial baselines:
  - "leak_token_present": predict hacking iff the input contains [LEAK_S].
    This is the "dumb baseline" — does just detecting the leak token
    suffice?
  - "input_bag_of_tokens": a count vector over the vocabulary.
  - "pos_embed_only": the position embeddings summed across the input
    length (a sanity check that the signal isn't just sequence length).

Outputs (written to --out-dir):
    input_probe_results.json
    input_probe_results.csv
    summary.txt

Run after exp1_probes.py:
    python exp1b_input_probe.py \
        --base-ckpt checkpoints/baseline/pretrained.pt \
        --rl-ckpt   checkpoints/baseline/rl_step_01300.pt \
        --exp1-results results/exp1_probes/probe_results.json \
        --out-dir   results/exp1b_input_probe \
        --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from common import (
    LEAK_E_ID,
    LEAK_S_ID,
    VOCAB_SIZE,
    ArithmeticLeakDataset,
    ModelConfig,
    TaskConfig,
    TinyOpenMythos,
    get_device,
    set_seed,
)
from exp1_probes import (
    ProbeConfig,
    bootstrap_auroc_ci,
    collect_trajectories,
)

logger = logging.getLogger("exp1b")
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)


@dataclass
class InputProbeConfig:
    n_probe_trajectories: int = 4000
    n_eval_trajectories: int = 2000
    n_bootstrap_resamples: int = 2000
    logistic_C: float = 1.0
    logistic_max_iter: int = 1000
    seed: int = 42


# =============================================================================
# Input-side feature extractors
# =============================================================================

@torch.no_grad()
def extract_input_embeddings(
    model: TinyOpenMythos,
    dataset: ArithmeticLeakDataset,
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    """
    Mean-pool the model's input embedding layer across sequence positions.
    Returns (N, D) where D = model.cfg.dim.

    NOTE: this uses ONLY the embedding lookup — NO transformer blocks,
    NO recurrent loop, NO position embeddings beyond the lookup itself.
    The simplest possible input representation.
    """
    model.eval().to(device)
    feats = []
    for i in range(0, len(dataset), batch_size):
        batch = [dataset[j] for j in range(i, min(i + batch_size, len(dataset)))]
        # Pad to common length
        max_len = max(len(b["input_ids"]) for b in batch)
        ids = torch.zeros(len(batch), max_len, dtype=torch.long)
        mask = torch.zeros(len(batch), max_len, dtype=torch.float)
        for k, b in enumerate(batch):
            n = len(b["input_ids"])
            ids[k, :n] = torch.tensor(b["input_ids"], dtype=torch.long)
            mask[k, :n] = 1.0
        ids = ids.to(device)
        mask = mask.to(device)
        emb = model.embed(ids)              # (B, L, D)
        # Mean-pool over non-pad positions
        pooled = (emb * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        feats.append(pooled.cpu().numpy())
    return np.concatenate(feats, axis=0)


def extract_bag_of_tokens(dataset: ArithmeticLeakDataset) -> np.ndarray:
    """
    Count vector over the vocabulary: feats[i, v] = number of times
    token id v appears in input i. The simplest possible input feature
    not involving the model at all.
    """
    feats = np.zeros((len(dataset), VOCAB_SIZE), dtype=np.float32)
    for i, ex in enumerate(dataset.examples):
        for tok in ex["input_ids"]:
            feats[i, tok] += 1.0
    return feats


def extract_leak_token_present(dataset: ArithmeticLeakDataset) -> np.ndarray:
    """
    The maximally trivial feature: a single bit per example indicating
    whether [LEAK_S] is present in the input. This is a 1-feature classifier.
    """
    feats = np.zeros((len(dataset), 1), dtype=np.float32)
    for i, ex in enumerate(dataset.examples):
        feats[i, 0] = 1.0 if LEAK_S_ID in ex["input_ids"] else 0.0
    return feats


# =============================================================================
# Probe training
# =============================================================================

def train_and_eval_probe(
    X_train: np.ndarray, y_train: np.ndarray,
    X_eval: np.ndarray, y_eval: np.ndarray,
    C: float, max_iter: int, seed: int,
    use_scaler: bool = True,
) -> dict:
    """Single-feature-set linear probe + evaluation. Mirrors exp1's helper."""
    if use_scaler and X_train.shape[1] > 1:
        scaler = StandardScaler().fit(X_train)
        Xt = scaler.transform(X_train)
        Xe = scaler.transform(X_eval)
    else:
        Xt, Xe = X_train, X_eval

    clf = LogisticRegression(
        penalty="l2", C=C, solver="lbfgs",
        max_iter=max_iter, random_state=seed,
    )
    clf.fit(Xt, y_train)
    scores = clf.decision_function(Xe)
    preds = clf.predict(Xe)
    return {
        "auroc": float(roc_auc_score(y_eval, scores)),
        "accuracy": float(accuracy_score(y_eval, preds)),
        "eval_scores": scores,
        "eval_labels": y_eval,
        "n_features": X_train.shape[1],
    }


# =============================================================================
# Main
# =============================================================================

def run_experiment(
    base_ckpt: str,
    rl_ckpt: str,
    exp1_results_path: Optional[str],
    out_dir: str,
    cfg: InputProbeConfig,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed)
    device = get_device()

    model_cfg = ModelConfig()

    # Load BOTH models — we extract embeddings from each separately,
    # since RL finetuning can change the embedding table.
    base_model = TinyOpenMythos(model_cfg)
    base_model.load_state_dict(torch.load(base_ckpt, map_location="cpu"))
    base_model.to(device)
    rl_model = TinyOpenMythos(model_cfg)
    rl_model.load_state_dict(torch.load(rl_ckpt, map_location="cpu"))
    rl_model.to(device)

    # ---- Build datasets ----
    task_cfg = TaskConfig(seed=cfg.seed + 100)
    probe_ds = ArithmeticLeakDataset(cfg.n_probe_trajectories, task_cfg)
    eval_task_cfg = TaskConfig(seed=cfg.seed + 200)
    eval_ds = ArithmeticLeakDataset(cfg.n_eval_trajectories, eval_task_cfg)

    # ---- Reuse exp1's trajectory collector to get matching labels ----
    # We need the SAME labeling as exp1 so the comparison is apples-to-apples.
    # ProbeConfig.extract_position doesn't matter here since we only use
    # the labels and indices.
    probe_cfg_for_labels = ProbeConfig(
        n_probe_trajectories=cfg.n_probe_trajectories,
        n_eval_trajectories=cfg.n_eval_trajectories,
        seed=cfg.seed,
        max_loop_iters=model_cfg.max_loop_iters,
    )
    logger.info("Collecting trajectories from RL model for labeling")
    probe_data = collect_trajectories(rl_model, probe_ds, probe_cfg_for_labels, device)
    eval_data = collect_trajectories(rl_model, eval_ds, probe_cfg_for_labels, device)

    train_mask = probe_data["labels"] >= 0
    eval_mask = eval_data["labels"] >= 0
    y_train = probe_data["labels"][train_mask]
    y_eval = eval_data["labels"][eval_mask]
    train_idx = np.where(train_mask)[0]
    eval_idx = np.where(eval_mask)[0]

    n_train = len(y_train)
    n_eval = len(y_eval)
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    logger.info(f"Labelled probe examples: {n_train} (hacking={n_pos}, genuine={n_neg})")
    logger.info(f"Labelled eval examples:  {n_eval}")

    # Filter the datasets to only labelled examples (in-place via index lists)
    probe_examples_labelled = [probe_ds.examples[i] for i in train_idx]
    eval_examples_labelled = [eval_ds.examples[i] for i in eval_idx]
    # Wrap in lightweight shim objects with the same .examples attribute
    class _Shim:
        def __init__(self, examples): self.examples = examples
        def __len__(self): return len(self.examples)
        def __getitem__(self, i): return self.examples[i]
    probe_ds_l = _Shim(probe_examples_labelled)
    eval_ds_l = _Shim(eval_examples_labelled)

    # ---- Feature sets ----
    logger.info("Extracting features: input embeddings (base model)")
    X_tr_emb_base = extract_input_embeddings(base_model, probe_ds_l, device)
    X_ev_emb_base = extract_input_embeddings(base_model, eval_ds_l, device)

    logger.info("Extracting features: input embeddings (RL model)")
    X_tr_emb_rl = extract_input_embeddings(rl_model, probe_ds_l, device)
    X_ev_emb_rl = extract_input_embeddings(rl_model, eval_ds_l, device)

    logger.info("Extracting features: bag of tokens")
    X_tr_bag = extract_bag_of_tokens(probe_ds_l)
    X_ev_bag = extract_bag_of_tokens(eval_ds_l)

    logger.info("Extracting features: leak-token-present (1 feature)")
    X_tr_leak = extract_leak_token_present(probe_ds_l)
    X_ev_leak = extract_leak_token_present(eval_ds_l)

    # ---- Train all 4 probes ----
    results = {}
    feature_sets = [
        ("input_embed_base", X_tr_emb_base, X_ev_emb_base, True),
        ("input_embed_rl",   X_tr_emb_rl,   X_ev_emb_rl,   True),
        ("bag_of_tokens",    X_tr_bag,      X_ev_bag,      False),
        ("leak_token_only",  X_tr_leak,     X_ev_leak,     False),
    ]
    for name, Xt, Xe, scale in feature_sets:
        logger.info(f"=== Probe: {name} (n_features={Xt.shape[1]}) ===")
        r = train_and_eval_probe(
            Xt, y_train, Xe, y_eval,
            cfg.logistic_C, cfg.logistic_max_iter, cfg.seed,
            use_scaler=scale,
        )
        lo, hi = bootstrap_auroc_ci(
            r["eval_scores"], r["eval_labels"],
            cfg.n_bootstrap_resamples, cfg.seed + 1,
        )
        results[name] = {
            "auroc": r["auroc"],
            "accuracy": r["accuracy"],
            "auroc_ci_lo": lo,
            "auroc_ci_hi": hi,
            "n_features": r["n_features"],
        }
        logger.info(f"   {name}: AUROC = {r['auroc']:.4f} [{lo:.4f}, {hi:.4f}]")

    # ---- Compare to exp1 depth-probe results ----
    exp1_summary = {}
    if exp1_results_path and Path(exp1_results_path).exists():
        with open(exp1_results_path) as f:
            exp1_data = json.load(f)
        depth_aurocs = [d["auroc_task"] for d in exp1_data["per_depth"]]
        exp1_summary = {
            "depth_probe_aurocs": depth_aurocs,
            "depth_probe_max":  float(np.max(depth_aurocs)),
            "depth_probe_min":  float(np.min(depth_aurocs)),
            "depth_probe_mean": float(np.mean(depth_aurocs)),
        }

    # ---- Verdict ----
    max_input_auroc = max(r["auroc"] for r in results.values())
    depth_max = exp1_summary.get("depth_probe_max", 0.0)
    redundant = max_input_auroc >= depth_max - 0.02  # within 2 AUROC points

    verdict = "DEPTH_REDUNDANT" if redundant else "DEPTH_ADDS_SIGNAL"

    output = {
        "verdict": verdict,
        "max_input_auroc": max_input_auroc,
        "max_depth_auroc": depth_max,
        "depth_minus_input_gap": float(depth_max - max_input_auroc),
        "n_train": n_train,
        "n_eval": n_eval,
        "config": asdict(cfg),
        "feature_results": results,
        "exp1_comparison": exp1_summary,
    }

    with open(out_dir / "input_probe_results.json", "w") as f:
        json.dump(output, f, indent=2, default=float)
    pd.DataFrame(results).T.to_csv(out_dir / "input_probe_results.csv")

    with open(out_dir / "summary.txt", "w") as f:
        f.write(f"VERDICT: {verdict}\n")
        f.write(f"Best input-side probe AUROC:  {max_input_auroc:.4f}\n")
        if exp1_summary:
            f.write(f"Best depth-T probe AUROC:     {depth_max:.4f}\n")
            f.write(f"Depth - input gap:            {depth_max - max_input_auroc:+.4f}\n")
        f.write(f"\nLabelled probe examples: {n_train} (hacking={n_pos}, genuine={n_neg})\n")
        f.write(f"Labelled eval examples:  {n_eval}\n\n")
        f.write("Per-feature results [95% BCa CI]:\n")
        for name, r in results.items():
            f.write(
                f"  {name:22s}  AUROC = {r['auroc']:.4f} "
                f"[{r['auroc_ci_lo']:.4f}, {r['auroc_ci_hi']:.4f}]  "
                f"acc = {r['accuracy']:.4f}  "
                f"({r['n_features']} features)\n"
            )
        if exp1_summary:
            f.write("\nExp1 depth-probe AUROCs (for comparison):\n")
            for i, a in enumerate(exp1_summary["depth_probe_aurocs"], 1):
                f.write(f"  depth {i:2d}: {a:.4f}\n")

    logger.info(f"VERDICT: {verdict}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-ckpt", required=True)
    parser.add_argument("--rl-ckpt", required=True)
    parser.add_argument("--exp1-results", default=None,
                        help="Path to exp1's probe_results.json for comparison")
    parser.add_argument("--out-dir", default="results/exp1b_input_probe")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-probe", type=int, default=4000)
    parser.add_argument("--n-eval", type=int, default=2000)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    args = parser.parse_args()

    cfg = InputProbeConfig(
        n_probe_trajectories=args.n_probe,
        n_eval_trajectories=args.n_eval,
        n_bootstrap_resamples=args.n_bootstrap,
        seed=args.seed,
    )
    run_experiment(args.base_ckpt, args.rl_ckpt, args.exp1_results, args.out_dir, cfg)


if __name__ == "__main__":
    main()
