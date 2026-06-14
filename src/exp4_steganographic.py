"""
exp4_steganographic.py
======================
Experiment 4: Steganographic Reward Channel.

The negative result of exp1+exp1b establishes that for an *input-channel*
exploit (a direct leak token), depth-indexed probes do not outperform
input-layer probes. This experiment tests whether that conclusion
generalizes by constructing an exploit class in which the gold answer
is NOT linearly decodable from the input alone -- the input must be
non-linearly combined to extract it. If the recurrent loop is the site
of useful computation, then depth probes should outperform input probes
in this setting.

"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from common import (
    BOS_ID,
    EOS_ID,
    PAD_ID,
    SEP_ID,
    VOCAB,
    GRPOConfig,
    GRPOTrainer,
    ModelConfig,
    TinyOpenMythos,
    collate_examples,
    count_parameters,
    encode_number,
    get_device,
    parse_answer,
    quick_pretrain,
    set_seed,
)
from exp1_probes import (
    bootstrap_auroc_ci,
    permutation_test_auroc_gap,
    train_probe_at_depth,
)

logger = logging.getLogger("exp4")
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)


# =============================================================================
# Pre-registered criteria
# =============================================================================

SUCCESS_CRITERIA = {
    "h1_depth_minus_input_min_gap": 0.10,
    "h2_locality_min_gap_vs_depth1": 0.10,
    "alpha": 0.05,
    "min_per_class_samples": 200,
}


# =============================================================================
# Steganographic task
# =============================================================================

@dataclass
class StegoTaskConfig:
    """
    Steganographic addition task.

    Prompt: BOS x0 x1 x2 x3 x4 x5 + ? SEP
    Gold:   (x_{p1} + x_{p2}) mod 10
    """
    seq_length: int = 6          # number of digit positions
    pos_a: int = 1               # first designated position (0-indexed)
    pos_b: int = 4               # second designated position
    seed: int = 0


class SteganographicDataset(Dataset):
    """
    Each example contains:
      - input_ids:  [BOS] x0 ... x5 + ? [SEP]
      - answer_ids: gold digit + [EOS]
      - gold_answer: int in [0, 9]
      - designated_digits: (x_pos_a, x_pos_b)
    """

    def __init__(self, n_examples: int, cfg: StegoTaskConfig):
        self.cfg = cfg
        rng = np.random.default_rng(cfg.seed)
        self.examples: list[dict] = []
        for _ in range(n_examples):
            digits = rng.integers(0, 10, size=cfg.seq_length).tolist()
            gold = (digits[cfg.pos_a] + digits[cfg.pos_b]) % 10

            tokens: list[int] = [BOS_ID]
            for d in digits:
                tokens.extend(encode_number(d))
            tokens.extend([VOCAB["+"], VOCAB["?"], SEP_ID])
            answer_tokens = encode_number(gold) + [EOS_ID]

            self.examples.append({
                "input_ids": tokens,
                "answer_ids": answer_tokens,
                "gold_answer": gold,
                "designated_digits": (int(digits[cfg.pos_a]), int(digits[cfg.pos_b])),
                "all_digits": [int(d) for d in digits],
                "has_leak": True,  # leak is implicit in every example
            })

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        return self.examples[idx]


# =============================================================================
# Pretraining: model learns the cipher
# =============================================================================

def quick_pretrain_stego(
    model: TinyOpenMythos,
    n_steps: int = 1300,
    batch_size: int = 64,
    lr: float = 3e-4,
    device: Optional[torch.device] = None,
    cfg: Optional[StegoTaskConfig] = None,
) -> None:
    """
    Supervised pretraining on the steganographic task.

    Unlike the original task, there is no decoy/non-leak split because
    the "leak" is structural -- every example uses the same encoding.
    The model must learn to attend to positions pos_a and pos_b,
    combine their digit values modulo 10, and emit the result.
    """
    device = device or get_device()
    cfg = cfg or StegoTaskConfig()
    model.to(device).train()

    ds = SteganographicDataset(batch_size * n_steps, cfg)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_examples, num_workers=0, pin_memory=True,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95))
    pbar = tqdm(loader, desc="pretrain (stego)", total=n_steps)
    for step, batch in enumerate(pbar):
        if step >= n_steps:
            break
        input_ids = batch["input_ids"].to(device)
        answer_ids = batch["answer_ids"].to(device)
        seq = torch.cat([input_ids, answer_ids], dim=1)
        logits = model(seq)
        target = seq[:, 1:]
        pred = logits[:, :-1, :]
        loss = F.cross_entropy(
            pred.reshape(-1, pred.size(-1)),
            target.reshape(-1),
            ignore_index=PAD_ID,
        )
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        model.apply_spectral_constraint()
        if step % 100 == 0:
            pbar.set_postfix(loss=f"{loss.item():.3f}")


# =============================================================================
# Trajectory + labeling
# =============================================================================

@torch.no_grad()
def collect_stego_trajectories(
    model: TinyOpenMythos,
    dataset: SteganographicDataset,
    device: torch.device,
    batch_size: int = 32,
) -> dict:
    """
    Collect hidden states from the model on the steganographic dataset.
    Labels here are NOT hacking vs genuine (the whole task IS the
    "hack" in the sense that every example uses the cipher). Instead,
    the probe target is the BINARIZED gold answer (gold >= 5 vs gold < 5).
    This is a meaningful but non-trivial concept:
      - the input encodes it non-linearly across pos_a and pos_b;
      - it is decodable in principle from any layer that has done the
        combination computation.

    Returns
    -------
    hidden_states: (N, T, D)
    input_embeds:  (N, D)  mean-pooled input embeddings
    labels:        (N,) int  binarized gold answer
    digits_pos_a:  (N,) int  for diagnostic checks
    digits_pos_b:  (N,) int
    """
    model.eval().to(device)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_examples, num_workers=0,
    )
    hidden_list = []
    input_emb_list = []
    label_list = []
    pa_list, pb_list = [], []

    for batch in tqdm(loader, desc="trajectories"):
        input_ids = batch["input_ids"].to(device)
        gold = batch["gold_answer"].tolist()
        # Get hidden states at all loop depths
        logits, hidden = model.forward_with_hidden_states(input_ids)
        # Take the position of the first answer token (last input position)
        h_stack = torch.stack([h[:, -1, :] for h in hidden], dim=1).cpu().numpy()
        # Also take mean-pooled input embeddings as the input-layer baseline
        with torch.no_grad():
            emb = model.embed(input_ids)
            mask = (input_ids != PAD_ID).float().unsqueeze(-1)
            mean_emb = (emb * mask).sum(1) / mask.sum(1).clamp_min(1)
        input_emb_list.append(mean_emb.cpu().numpy())
        hidden_list.append(h_stack)
        label_list.extend(int(g >= 5) for g in gold)

    return {
        "hidden_states": np.concatenate(hidden_list, axis=0),  # (N, T, D)
        "input_embeds": np.concatenate(input_emb_list, axis=0),  # (N, D)
        "labels": np.array(label_list, dtype=np.int64),
    }


# =============================================================================
# End-to-end baseline training (for the steganographic task)
# =============================================================================

def train_baseline_stego(
    out_dir: str | Path,
    seed: int = 0,
    pretrain_steps: int = 1300,
    rl_steps: int = 1300,
    rl_checkpoint_every: int = 90,
) -> dict:
    """
    Pretrain on the steganographic task (supervised), then RL-finetune
    with GRPO. RL provides additional pressure but is not strictly
    necessary for the task to be learned -- pretraining alone can
    produce a competent model.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()
    set_seed(seed)

    model_cfg = ModelConfig()
    policy = TinyOpenMythos(model_cfg)
    logger.info(f"TinyOpenMythos parameter count: {count_parameters(policy):,}")

    stego_cfg = StegoTaskConfig(seed=seed)
    logger.info(f"Steganographic task: pos_a={stego_cfg.pos_a}, pos_b={stego_cfg.pos_b}")

    logger.info("Phase 1/2: supervised pretraining on steganographic task")
    quick_pretrain_stego(policy, n_steps=pretrain_steps, device=device, cfg=stego_cfg)
    torch.save(policy.state_dict(), out_dir / "pretrained.pt")

    reference = TinyOpenMythos(model_cfg)
    reference.load_state_dict(policy.state_dict())

    logger.info("Phase 2/2: GRPO finetuning")
    grpo_cfg = GRPOConfig(n_steps=rl_steps, checkpoint_every=rl_checkpoint_every)
    trainer = GRPOTrainer(policy, reference, grpo_cfg, device)

    rl_ds = SteganographicDataset(n_examples=10_000, cfg=StegoTaskConfig(seed=seed + 1))
    loader = DataLoader(
        rl_ds, batch_size=grpo_cfg.batch_prompts, shuffle=True,
        collate_fn=collate_examples, num_workers=0, pin_memory=True,
    )

    metrics_log = []
    step = 0
    pbar = tqdm(total=rl_steps, desc="GRPO")
    while step < rl_steps:
        for batch in loader:
            if step >= rl_steps:
                break
            m = trainer.step(batch)
            m["step"] = step
            metrics_log.append(m)
            pbar.update(1)
            if step % 50 == 0:
                pbar.set_postfix(reward=f"{m['reward_mean']:.3f}")
            if step % rl_checkpoint_every == 0:
                torch.save(policy.state_dict(), out_dir / f"rl_step_{step:05d}.pt")
            step += 1
    pbar.close()
    torch.save(policy.state_dict(), out_dir / f"rl_step_{rl_steps:05d}.pt")

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics_log, f, indent=2)
    with open(out_dir / "stego_config.json", "w") as f:
        json.dump(asdict(stego_cfg), f, indent=2)

    logger.info(f"Steganographic baseline training complete. Artifacts in {out_dir}")
    return {"checkpoint_dir": str(out_dir), "final_step": rl_steps}


# =============================================================================
# Main probe experiment
# =============================================================================

@dataclass
class ProbeRunConfig:
    n_probe_trajectories: int = 4000
    n_eval_trajectories: int = 2000
    n_bootstrap_resamples: int = 2000
    n_permutations: int = 2000
    logistic_C: float = 1.0
    logistic_max_iter: int = 1000
    seed: int = 42


def run_probe_experiment(
    base_ckpt: str,
    rl_ckpt: str,
    out_dir: str,
    cfg: ProbeRunConfig,
) -> dict:
    """
    Probe family on the steganographic task:
      - per-depth task probes on the RL-finetuned model
      - per-depth base probes on the pretrained model (RL-induction test)
      - input-layer probe on mean-pooled embeddings (input-layer baseline)
      - per-depth random-label control (probe capacity floor)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed)
    device = get_device()

    model_cfg = ModelConfig()

    base_model = TinyOpenMythos(model_cfg)
    base_model.load_state_dict(torch.load(base_ckpt, map_location="cpu"))
    base_model.to(device)
    rl_model = TinyOpenMythos(model_cfg)
    rl_model.load_state_dict(torch.load(rl_ckpt, map_location="cpu"))
    rl_model.to(device)

    probe_ds = SteganographicDataset(cfg.n_probe_trajectories,
                                      StegoTaskConfig(seed=cfg.seed + 100))
    eval_ds = SteganographicDataset(cfg.n_eval_trajectories,
                                     StegoTaskConfig(seed=cfg.seed + 200))

    logger.info("Collecting trajectories from RL-trained model")
    probe_rl = collect_stego_trajectories(rl_model, probe_ds, device)
    eval_rl = collect_stego_trajectories(rl_model, eval_ds, device)

    logger.info("Collecting trajectories from base (pre-RL) model")
    probe_base = collect_stego_trajectories(base_model, probe_ds, device)
    eval_base = collect_stego_trajectories(base_model, eval_ds, device)

    n_train = len(probe_rl["labels"])
    n_eval = len(eval_rl["labels"])
    n_pos = int((probe_rl["labels"] == 1).sum())
    n_neg = int((probe_rl["labels"] == 0).sum())
    logger.info(f"Probe train: {n_train} (pos={n_pos}, neg={n_neg}); eval: {n_eval}")

    if min(n_pos, n_neg) < SUCCESS_CRITERIA["min_per_class_samples"]:
        logger.warning(
            f"Per-class sample minimum not met (pos={n_pos}, neg={n_neg}, "
            f"required={SUCCESS_CRITERIA['min_per_class_samples']})."
        )

    T = probe_rl["hidden_states"].shape[1]

    # Random labels for control
    rng = np.random.default_rng(cfg.seed + 999)
    y_train_random = rng.integers(0, 2, size=n_train)
    y_eval_random = rng.integers(0, 2, size=n_eval)

    # ---- Input-layer probe (the baseline that the negative result said was sufficient) ----
    logger.info("=== Input-layer baseline (mean-pooled embeddings, RL model) ===")
    input_probe = train_probe_at_depth(
        probe_rl["input_embeds"], probe_rl["labels"],
        eval_rl["input_embeds"], eval_rl["labels"],
        cfg.logistic_C, cfg.logistic_max_iter, cfg.seed,
    )
    in_lo, in_hi = bootstrap_auroc_ci(
        input_probe["eval_scores"], input_probe["eval_labels"],
        cfg.n_bootstrap_resamples, cfg.seed + 1,
    )
    logger.info(f"   input layer: AUROC = {input_probe['auroc']:.4f} [{in_lo:.4f}, {in_hi:.4f}]")

    # ---- Per-depth probes ----
    per_depth: list[dict] = []
    for t in range(T):
        depth = t + 1
        logger.info(f"=== Depth {depth}/{T} ===")
        X_tr_rl = probe_rl["hidden_states"][:, t, :]
        X_ev_rl = eval_rl["hidden_states"][:, t, :]
        X_tr_base = probe_base["hidden_states"][:, t, :]
        X_ev_base = eval_base["hidden_states"][:, t, :]

        task = train_probe_at_depth(
            X_tr_rl, probe_rl["labels"], X_ev_rl, eval_rl["labels"],
            cfg.logistic_C, cfg.logistic_max_iter, cfg.seed,
        )
        ctrl = train_probe_at_depth(
            X_tr_rl, y_train_random, X_ev_rl, y_eval_random,
            cfg.logistic_C, cfg.logistic_max_iter, cfg.seed + 1,
        )
        base = train_probe_at_depth(
            X_tr_base, probe_base["labels"], X_ev_base, eval_base["labels"],
            cfg.logistic_C, cfg.logistic_max_iter, cfg.seed + 2,
        )
        task_lo, task_hi = bootstrap_auroc_ci(
            task["eval_scores"], task["eval_labels"],
            cfg.n_bootstrap_resamples, cfg.seed + 10,
        )
        # Permutation test: task probe AUROC > input probe AUROC?
        p_task_vs_input = permutation_test_auroc_gap(
            task["eval_scores"], eval_rl["labels"],
            input_probe["eval_scores"], input_probe["eval_labels"],
            cfg.n_permutations, cfg.seed + 30,
        )
        # Permutation test: task probe AUROC > base probe AUROC?
        p_task_vs_base = permutation_test_auroc_gap(
            task["eval_scores"], eval_rl["labels"],
            base["eval_scores"], base["eval_labels"],
            cfg.n_permutations, cfg.seed + 31,
        )
        per_depth.append({
            "depth": depth,
            "auroc_task": task["auroc"],
            "auroc_task_ci_lo": task_lo,
            "auroc_task_ci_hi": task_hi,
            "auroc_control": ctrl["auroc"],
            "auroc_base": base["auroc"],
            "gap_vs_input": task["auroc"] - input_probe["auroc"],
            "gap_vs_base": task["auroc"] - base["auroc"],
            "p_task_vs_input": p_task_vs_input,
            "p_task_vs_base": p_task_vs_base,
        })

    # BH-correct across the 8 depths
    p_in = np.array([d["p_task_vs_input"] for d in per_depth])
    p_base = np.array([d["p_task_vs_base"] for d in per_depth])
    _, p_in_bh, _, _ = multipletests(p_in, alpha=SUCCESS_CRITERIA["alpha"], method="fdr_bh")
    _, p_base_bh, _, _ = multipletests(p_base, alpha=SUCCESS_CRITERIA["alpha"], method="fdr_bh")
    for i, d in enumerate(per_depth):
        d["p_task_vs_input_bh"] = float(p_in_bh[i])
        d["p_task_vs_base_bh"] = float(p_base_bh[i])

    # ---- Hypothesis tests ----
    max_depth_auroc = max(d["auroc_task"] for d in per_depth)
    max_depth_idx = max(range(T), key=lambda i: per_depth[i]["auroc_task"])
    gap_h1 = max_depth_auroc - input_probe["auroc"]
    h1_pass = gap_h1 >= SUCCESS_CRITERIA["h1_depth_minus_input_min_gap"]

    auroc_d1 = per_depth[0]["auroc_task"]
    max_gap_vs_d1 = max((d["auroc_task"] - auroc_d1) for d in per_depth)
    h2_pass = max_gap_vs_d1 >= SUCCESS_CRITERIA["h2_locality_min_gap_vs_depth1"]

    if h1_pass and h2_pass:
        verdict = "STRONG_SUCCESS_DEPTH_ADDS_VALUE"
    elif h1_pass:
        verdict = "WEAK_SUCCESS_DEPTH_ADDS_VALUE_NO_LOCALITY"
    elif h2_pass:
        verdict = "WEAK_SUCCESS_LOCALITY_NO_INPUT_GAP"
    else:
        verdict = "NULL_RESULT_DEPTH_ADDS_NOTHING"

    results = {
        "verdict": verdict,
        "max_depth_auroc": float(max_depth_auroc),
        "max_depth_index": int(max_depth_idx + 1),
        "auroc_input_layer": float(input_probe["auroc"]),
        "auroc_input_ci_lo": float(in_lo),
        "auroc_input_ci_hi": float(in_hi),
        "h1_gap_depth_minus_input": float(gap_h1),
        "h1_pass": bool(h1_pass),
        "h2_max_gap_vs_depth_1": float(max_gap_vs_d1),
        "h2_pass": bool(h2_pass),
        "n_train": n_train,
        "n_eval": n_eval,
        "success_criteria": SUCCESS_CRITERIA,
        "config": asdict(cfg),
        "per_depth": per_depth,
    }

    with open(out_dir / "steg_results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    pd.DataFrame(per_depth).to_csv(out_dir / "steg_results.csv", index=False)

    with open(out_dir / "summary.txt", "w") as f:
        f.write(f"VERDICT: {verdict}\n")
        f.write(f"Max depth-probe AUROC: {max_depth_auroc:.4f} (at depth {max_depth_idx+1})\n")
        f.write(f"Input-layer AUROC:     {input_probe['auroc']:.4f} "
                f"[{in_lo:.4f}, {in_hi:.4f}]\n")
        f.write(f"H1: depth - input gap = {gap_h1:+.4f}  "
                f"(threshold {SUCCESS_CRITERIA['h1_depth_minus_input_min_gap']}, "
                f"{'PASS' if h1_pass else 'FAIL'})\n")
        f.write(f"H2: locality (max - depth1) = {max_gap_vs_d1:+.4f}  "
                f"(threshold {SUCCESS_CRITERIA['h2_locality_min_gap_vs_depth1']}, "
                f"{'PASS' if h2_pass else 'FAIL'})\n\n")
        f.write("Per-depth AUROC [95% BCa CI]:\n")
        for d in per_depth:
            f.write(
                f"  depth={d['depth']:2d}  "
                f"task={d['auroc_task']:.4f} "
                f"[{d['auroc_task_ci_lo']:.4f}, {d['auroc_task_ci_hi']:.4f}]  "
                f"ctrl={d['auroc_control']:.4f}  "
                f"base={d['auroc_base']:.4f}  "
                f"gap_vs_input={d['gap_vs_input']:+.4f} "
                f"(p_BH={d['p_task_vs_input_bh']:.3g})  "
                f"gap_vs_base={d['gap_vs_base']:+.4f} "
                f"(p_BH={d['p_task_vs_base_bh']:.3g})\n"
            )

    logger.info(f"VERDICT: {verdict}")
    return results


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    train = sub.add_parser("train-baseline")
    train.add_argument("--out-dir", required=True)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--pretrain-steps", type=int, default=1300)
    train.add_argument("--rl-steps", type=int, default=1300)
    train.add_argument("--checkpoint-every", type=int, default=90)

    probe = sub.add_parser("probe")
    probe.add_argument("--base-ckpt", required=True)
    probe.add_argument("--rl-ckpt", required=True)
    probe.add_argument("--out-dir", default="results/exp4_steganographic")
    probe.add_argument("--seed", type=int, default=42)
    probe.add_argument("--n-probe", type=int, default=4000)
    probe.add_argument("--n-eval", type=int, default=2000)

    args = parser.parse_args()

    if args.cmd == "train-baseline":
        train_baseline_stego(
            out_dir=args.out_dir,
            seed=args.seed,
            pretrain_steps=args.pretrain_steps,
            rl_steps=args.rl_steps,
            rl_checkpoint_every=args.checkpoint_every,
        )
    elif args.cmd == "probe":
        cfg = ProbeRunConfig(
            n_probe_trajectories=args.n_probe,
            n_eval_trajectories=args.n_eval,
            seed=args.seed,
        )
        run_probe_experiment(args.base_ckpt, args.rl_ckpt, args.out_dir, cfg)


if __name__ == "__main__":
    main()
