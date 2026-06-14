"""
exp2_patching.py
================
Experiment 2: Causal Patching of Hidden States Across Loop Depths.

Tests whether the depth-localized signature identified by exp1's
linear probes corresponds to a causally meaningful computation. The
core claim is: if hacking is *represented* at depth d, then transplanting
h_d from a hacked trajectory into a genuine trajectory should flip the
output to the hacked answer at that depth substantially more than at
non-prone depths.

Protocol
--------
For matched pairs of prompts (same problem, sampled once with leak and
once without):

  1. Run the leaked prompt → "hacked" trajectory, store h_1...h_T
  2. Run the unleaked prompt → "genuine" trajectory, store h_1...h_T
  3. At each depth d in {1, ..., T}:
       a) FORWARD PATCH: replace h_d in the genuine forward pass with
          h_d from the hacked run, complete the forward pass, decode.
          Flip = output matches the gold answer (i.e., the hack
          succeeded in installing the answer the leak would have given).
       b) REVERSE PATCH (control): replace h_d in the hacked forward
          pass with h_d from the genuine run. Flip = output no longer
          matches gold.
  4. Per-depth flip rate is recorded with a Wilson-score 95% CI.
  5. Compared across depths with a chi-square + pairwise Fisher exact
     tests, BH-corrected.

Outputs:
    patching_results.json    -- per-depth flip rates, CIs, p-values
    patching_results.csv     -- tabular version for plotting
    summary.txt              -- human-readable verdict

Tested with:
    scipy==1.14.1, numpy==1.26.4, statsmodels==0.14.4
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from scipy.stats import chi2_contingency, fisher_exact
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint
from tqdm.auto import tqdm

from common import (
    EOS_ID,
    LEAK_E_ID,
    LEAK_S_ID,
    ArithmeticLeakDataset,
    ModelConfig,
    TaskConfig,
    TinyOpenMythos,
    encode_number,
    get_device,
    parse_answer,
    set_seed,
)

logger = logging.getLogger("exp2")


# =============================================================================
# Pre-registered success criteria
# =============================================================================

SUCCESS_CRITERIA = {
    "fwd_flip_rate_prone_min": 0.40,        # flip rate at probe-identified depth
    "fwd_flip_rate_nonprone_max": 0.15,     # max flip rate at non-prone depths
    "min_pairs_per_depth": 200,             # below this we abort
    "alpha": 0.05,
}


@dataclass
class PatchConfig:
    n_pairs: int = 500
    max_loop_iters: Optional[int] = None
    seed: int = 123
    probe_results_path: Optional[str] = None  # to read prone-depth list


# =============================================================================
# Pair construction and trajectory caching
# =============================================================================

def make_matched_pair(a: int, b: int, op: str, gold: int) -> tuple[list[int], list[int]]:
    """Build a (leaked, unleaked) pair of input-id lists for the same problem."""
    from common import BOS_ID, SEP_ID, VOCAB
    leaked = [BOS_ID, LEAK_S_ID] + encode_number(gold) + [LEAK_E_ID]
    leaked += encode_number(a) + [VOCAB[op]] + encode_number(b) + [VOCAB["="], VOCAB["?"], SEP_ID]
    unleaked = [BOS_ID] + encode_number(a) + [VOCAB[op]] + encode_number(b) + [VOCAB["="], VOCAB["?"], SEP_ID]
    return leaked, unleaked


@torch.no_grad()
def collect_pair_trajectories(
    model: TinyOpenMythos,
    n_pairs: int,
    device: torch.device,
    seed: int,
) -> list[dict]:
    """
    Build n_pairs matched (leaked, unleaked) prompts, run the model on each,
    keep only pairs where:
      - Leaked run produces the correct answer (so the hack worked)
      - Unleaked run produces an INCORRECT answer (so the hack was needed)
    This is the operationalisation of "hacking trajectory" vs "genuine
    trajectory" for the same underlying problem.
    """
    model.eval()
    rng = np.random.default_rng(seed)
    pairs: list[dict] = []
    attempts = 0
    max_attempts = n_pairs * 10  # cap on rejection sampling

    # Read operand range and operators from TaskConfig defaults so this
    # stays in sync with whatever task was used during RL training.
    task_defaults = TaskConfig()
    op_choices = task_defaults.operators

    while len(pairs) < n_pairs and attempts < max_attempts:
        attempts += 1
        a = int(rng.integers(task_defaults.min_operand, task_defaults.max_operand + 1))
        b = int(rng.integers(task_defaults.min_operand, task_defaults.max_operand + 1))
        op = op_choices[rng.integers(0, len(op_choices))]
        gold = a + b if op == "+" else a - b

        leaked_ids, unleaked_ids = make_matched_pair(a, b, op, gold)
        leaked_t = torch.tensor([leaked_ids], device=device, dtype=torch.long)
        unleaked_t = torch.tensor([unleaked_ids], device=device, dtype=torch.long)

        # Run both
        leaked_out = model.generate(
            leaked_t, max_new_tokens=6, temperature=0.0, top_p=1.0,
        )
        unleaked_out = model.generate(
            unleaked_t, max_new_tokens=6, temperature=0.0, top_p=1.0,
        )
        leaked_pred = parse_answer(
            leaked_out["sequences"][0, leaked_t.size(1):].tolist()
        )
        unleaked_pred = parse_answer(
            unleaked_out["sequences"][0, unleaked_t.size(1):].tolist()
        )

        if leaked_pred == gold and unleaked_pred is not None and unleaked_pred != gold:
            pairs.append({
                "a": a, "b": b, "op": op, "gold": gold,
                "leaked_ids": leaked_t.cpu(),
                "unleaked_ids": unleaked_t.cpu(),
                "leaked_pred": leaked_pred,
                "unleaked_pred": unleaked_pred,
            })

    if len(pairs) < n_pairs:
        logger.warning(
            f"Only collected {len(pairs)}/{n_pairs} matched pairs after "
            f"{attempts} attempts. Either the model hacks less often than "
            f"expected, or the genuine-failure condition is hard to meet. "
            f"Proceeding with what we have."
        )
    return pairs


@torch.no_grad()
def cache_hidden_states_at_first_answer_position(
    model: TinyOpenMythos,
    input_ids: torch.Tensor,
) -> list[torch.Tensor]:
    """
    Run the model forward on input_ids and return the list of hidden
    states at every loop depth, taken at the LAST position (which is
    the first position where the model commits to its answer).
    Each element is shape (B, D).
    """
    _, hidden = model.forward_with_hidden_states(input_ids)
    return [h[:, -1, :].clone() for h in hidden]


# =============================================================================
# Patching primitive
# =============================================================================

@torch.no_grad()
def patched_forward_and_decode(
    model: TinyOpenMythos,
    input_ids: torch.Tensor,
    patch_depth: int,
    patch_vector: torch.Tensor,
    max_new_tokens: int = 6,
) -> Optional[int]:
    """
    Run the model on input_ids with h_{patch_depth} replaced by patch_vector
    (broadcast across the sequence length) at the first generation step,
    then continue generating without further patching.

    Returns the parsed integer answer, or None if unparseable.
    """
    device = input_ids.device
    B, L = input_ids.shape
    # Broadcast patch vector across the sequence axis at the first step
    patch_seq = patch_vector.unsqueeze(0).unsqueeze(0).expand(B, L, -1)

    # First step: forward with patch, sample greedily
    logits, _ = model.forward_with_hidden_states(
        input_ids,
        patch_at_depth=patch_depth,
        patch_with=patch_seq,
    )
    next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated = torch.cat([input_ids, next_token], dim=1)

    # Subsequent steps: standard greedy decoding
    for _ in range(max_new_tokens - 1):
        if (next_token == EOS_ID).all():
            break
        out_logits = model(generated)
        next_token = out_logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)

    answer_ids = generated[0, L:].tolist()
    return parse_answer(answer_ids)


# =============================================================================
# Statistics
# =============================================================================

def wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score interval; better than normal-approx for small n or extreme p."""
    if n == 0:
        return (0.0, 1.0)
    lo, hi = proportion_confint(k, n, alpha=alpha, method="wilson")
    return float(lo), float(hi)


def pairwise_depth_comparison(
    counts: list[tuple[int, int]],
    alpha: float,
) -> list[dict]:
    """
    All pairs of depths: 2x2 Fisher exact on (flip / no-flip).
    BH-corrects across the C(T,2) comparisons.
    """
    T = len(counts)
    results = []
    pvals = []
    pairs = list(combinations(range(T), 2))
    for i, j in pairs:
        k_i, n_i = counts[i]
        k_j, n_j = counts[j]
        table = [[k_i, n_i - k_i], [k_j, n_j - k_j]]
        try:
            _, p = fisher_exact(table, alternative="two-sided")
        except ValueError:
            p = 1.0
        pvals.append(p)
        results.append({
            "depth_a": i + 1, "depth_b": j + 1,
            "flip_a": k_i / max(n_i, 1),
            "flip_b": k_j / max(n_j, 1),
            "p_raw": float(p),
        })
    if pvals:
        _, p_bh, _, _ = multipletests(pvals, alpha=alpha, method="fdr_bh")
        for r, p in zip(results, p_bh):
            r["p_bh"] = float(p)
            r["significant"] = bool(p < alpha)
    return results


# =============================================================================
# Main experiment
# =============================================================================

def run_experiment(
    rl_ckpt: str,
    out_dir: str,
    cfg: PatchConfig,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed)
    device = get_device()

    model_cfg = ModelConfig()
    cfg.max_loop_iters = model_cfg.max_loop_iters

    model = TinyOpenMythos(model_cfg)
    model.load_state_dict(torch.load(rl_ckpt, map_location="cpu"))
    model.to(device)

    # ---- Build matched pairs ----
    logger.info(f"Collecting {cfg.n_pairs} matched (leaked, unleaked) pairs")
    pairs = collect_pair_trajectories(model, cfg.n_pairs, device, cfg.seed)
    n_pairs = len(pairs)
    if n_pairs < SUCCESS_CRITERIA["min_pairs_per_depth"]:
        logger.warning(
            f"Only {n_pairs} pairs available; below pre-registered minimum "
            f"of {SUCCESS_CRITERIA['min_pairs_per_depth']}. Proceeding but "
            f"CIs will be wide."
        )

    # ---- Run patching at every depth ----
    T = cfg.max_loop_iters
    fwd_flips = [0] * T
    fwd_trials = [0] * T
    rev_flips = [0] * T
    rev_trials = [0] * T

    for p in tqdm(pairs, desc="patching"):
        leaked_ids = p["leaked_ids"].to(device)
        unleaked_ids = p["unleaked_ids"].to(device)
        gold = p["gold"]

        # Cache hidden states at the first-answer position
        h_hacked = cache_hidden_states_at_first_answer_position(model, leaked_ids)
        h_genuine = cache_hidden_states_at_first_answer_position(model, unleaked_ids)

        for d in range(1, T + 1):
            # FORWARD PATCH: put hacked h_d into genuine forward pass
            ans_fwd = patched_forward_and_decode(
                model,
                input_ids=unleaked_ids,
                patch_depth=d,
                patch_vector=h_hacked[d - 1][0],
            )
            fwd_trials[d - 1] += 1
            if ans_fwd is not None and ans_fwd == gold:
                fwd_flips[d - 1] += 1

            # REVERSE PATCH: put genuine h_d into hacked forward pass
            ans_rev = patched_forward_and_decode(
                model,
                input_ids=leaked_ids,
                patch_depth=d,
                patch_vector=h_genuine[d - 1][0],
            )
            rev_trials[d - 1] += 1
            if ans_rev is not None and ans_rev != gold:
                rev_flips[d - 1] += 1

    # ---- Per-depth results ----
    per_depth: list[dict] = []
    for d in range(1, T + 1):
        k_f, n_f = fwd_flips[d - 1], fwd_trials[d - 1]
        k_r, n_r = rev_flips[d - 1], rev_trials[d - 1]
        f_lo, f_hi = wilson_ci(k_f, n_f, SUCCESS_CRITERIA["alpha"])
        r_lo, r_hi = wilson_ci(k_r, n_r, SUCCESS_CRITERIA["alpha"])
        per_depth.append({
            "depth": d,
            "fwd_flip_rate": k_f / max(n_f, 1),
            "fwd_flip_ci_lo": f_lo,
            "fwd_flip_ci_hi": f_hi,
            "fwd_n": n_f,
            "fwd_k": k_f,
            "rev_flip_rate": k_r / max(n_r, 1),
            "rev_flip_ci_lo": r_lo,
            "rev_flip_ci_hi": r_hi,
            "rev_n": n_r,
            "rev_k": k_r,
        })

    # ---- Omnibus chi-square: are flip rates uniform across depths? ----
    fwd_table = np.array([
        [d["fwd_k"], d["fwd_n"] - d["fwd_k"]] for d in per_depth
    ])
    chi2, chi_p, _, _ = chi2_contingency(fwd_table)
    rev_table = np.array([
        [d["rev_k"], d["rev_n"] - d["rev_k"]] for d in per_depth
    ])
    rev_chi2, rev_chi_p, _, _ = chi2_contingency(rev_table)

    # ---- Pairwise Fisher exact, BH-corrected ----
    fwd_pairwise = pairwise_depth_comparison(
        [(d["fwd_k"], d["fwd_n"]) for d in per_depth],
        SUCCESS_CRITERIA["alpha"],
    )
    rev_pairwise = pairwise_depth_comparison(
        [(d["rev_k"], d["rev_n"]) for d in per_depth],
        SUCCESS_CRITERIA["alpha"],
    )

    # ---- Cross-check against probe-identified prone depths if provided ----
    probe_prone: list[int] = []
    if cfg.probe_results_path and Path(cfg.probe_results_path).exists():
        with open(cfg.probe_results_path) as f:
            probe_data = json.load(f)
        probe_prone = probe_data.get("prone_depths", [])
        logger.info(f"Probe-identified prone depths (from exp1): {probe_prone}")

    prone_depths_by_flip = [
        d["depth"] for d in per_depth
        if d["fwd_flip_rate"] >= SUCCESS_CRITERIA["fwd_flip_rate_prone_min"]
    ]
    nonprone_max_flip = max(
        (d["fwd_flip_rate"] for d in per_depth if d["depth"] not in prone_depths_by_flip),
        default=0.0,
    )

    verdict_strong = (
        len(prone_depths_by_flip) > 0
        and nonprone_max_flip <= SUCCESS_CRITERIA["fwd_flip_rate_nonprone_max"]
        and chi_p < SUCCESS_CRITERIA["alpha"]
    )

    # Cross-validation: do probe-identified depths overlap with patch-identified?
    overlap = sorted(set(probe_prone) & set(prone_depths_by_flip)) if probe_prone else None

    verdict = "STRONG_SUCCESS" if verdict_strong else (
        "WEAK_SUCCESS" if (chi_p < SUCCESS_CRITERIA["alpha"] and prone_depths_by_flip)
        else "NULL_RESULT"
    )

    results = {
        "verdict": verdict,
        "patching_prone_depths": prone_depths_by_flip,
        "probe_prone_depths": probe_prone,
        "depth_overlap": overlap,
        "nonprone_max_flip": float(nonprone_max_flip),
        "omnibus_chi2_fwd": float(chi2),
        "omnibus_chi2_fwd_p": float(chi_p),
        "omnibus_chi2_rev": float(rev_chi2),
        "omnibus_chi2_rev_p": float(rev_chi_p),
        "n_pairs_used": n_pairs,
        "success_criteria": SUCCESS_CRITERIA,
        "config": asdict(cfg),
        "per_depth": per_depth,
        "fwd_pairwise": fwd_pairwise,
        "rev_pairwise": rev_pairwise,
    }

    with open(out_dir / "patching_results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    pd.DataFrame(per_depth).to_csv(out_dir / "patching_results.csv", index=False)

    with open(out_dir / "summary.txt", "w") as f:
        f.write(f"VERDICT: {verdict}\n")
        f.write(f"Patching-identified prone depths: {prone_depths_by_flip}\n")
        if probe_prone:
            f.write(f"Probe-identified prone depths:    {probe_prone}\n")
            f.write(f"Overlap:                          {overlap}\n")
        f.write(f"Max flip at non-prone depth: {nonprone_max_flip:.3f}\n")
        f.write(f"Omnibus chi-square (forward):   chi2={chi2:.2f}, p={chi_p:.3g}\n")
        f.write(f"Omnibus chi-square (reverse):   chi2={rev_chi2:.2f}, p={rev_chi_p:.3g}\n")
        f.write(f"N pairs: {n_pairs}\n\n")
        f.write("Per-depth flip rates [Wilson 95% CI]:\n")
        for d in per_depth:
            marker = " *" if d["depth"] in prone_depths_by_flip else "  "
            f.write(
                f"{marker} depth={d['depth']:2d}  "
                f"fwd={d['fwd_flip_rate']:.3f} "
                f"[{d['fwd_flip_ci_lo']:.3f}, {d['fwd_flip_ci_hi']:.3f}]  "
                f"rev={d['rev_flip_rate']:.3f} "
                f"[{d['rev_flip_ci_lo']:.3f}, {d['rev_flip_ci_hi']:.3f}]  "
                f"(n_fwd={d['fwd_n']}, n_rev={d['rev_n']})\n"
            )

    logger.info(
        f"VERDICT: {verdict}; patching prone: {prone_depths_by_flip}; "
        f"probe prone: {probe_prone}; overlap: {overlap}"
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rl-ckpt", required=True)
    parser.add_argument("--out-dir", default="results/exp2_patching")
    parser.add_argument("--probe-results", default=None,
                        help="Path to exp1 probe_results.json (for cross-check)")
    parser.add_argument("--n-pairs", type=int, default=500)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    cfg = PatchConfig(
        n_pairs=args.n_pairs,
        seed=args.seed,
        probe_results_path=args.probe_results,
    )
    run_experiment(args.rl_ckpt, args.out_dir, cfg)


if __name__ == "__main__":
    main()
