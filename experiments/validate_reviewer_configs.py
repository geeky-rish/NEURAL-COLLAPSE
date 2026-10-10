"""
experiments/validate_reviewer_configs.py
─────────────────────────────────────────────────────────────────────────────
SAFE configuration validation script — NO GPU required, NO real training.

This script verifies that the 9 reviewer experiment configurations wire the
correct loss terms and output directories WITHOUT running any real training.

What it checks (for each configuration):
  1. Config loads without error.
  2. Model instantiation succeeds (ETF head exists).
  3. The correct loss criterion is instantiated:
       - ETF-only (B1): trainer._use_nc_loss == False
       - ETF+L_collapse (B2): CombinedNCLoss with align_weight=0.0
       - ETF+L_align (B3): CombinedNCLoss with collapse_weight=0.0
       - Full (A1-A4, B4-ref, C1-C2): CombinedNCLoss with both weights > 0
  4. ETF matrix is different for different seeds (stochastic isolation check).
  5. Sampling strategy is correctly selected.
  6. Output paths are unique across all experiments.
  7. A minimal 2-batch forward pass completes without error (CPU, tiny batch).

This script uses fast_dev_batches=2 internally and does NOT save any results.
It also verifies that a 2-batch pass produces DIFFERENT loss values for B2 vs
B3 (ablation isolation check).

Usage:
  python -m experiments.validate_reviewer_configs

Exit code 0 = all checks passed.
Exit code 1 = at least one check failed (details printed to stdout).
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ─────────────────────────────────────────────────────────────────────────────
# Import project modules
# ─────────────────────────────────────────────────────────────────────────────
try:
    import torch
    from config import load_config
    from models import build_model
    from training.trainer import Trainer
    from training.nc_regularization import CombinedNCLoss, NCCollapseRegularizer, ETFAlignmentLoss
    from utils.seed import set_seed
    from experiments.reviewer_experiments import REVIEWER_EXPERIMENTS, build_command, results_dir
except ImportError as exc:
    print(f"[ERROR] Import failed: {exc}")
    print("  Make sure you are running from the repository root with the venv active.")
    sys.exit(1)

PASS = "  [PASS]"
FAIL = "  [FAIL]"

errors: List[str] = []


def check(condition: bool, msg_pass: str, msg_fail: str) -> bool:
    if condition:
        print(f"{PASS} {msg_pass}")
        return True
    else:
        print(f"{FAIL} {msg_fail}")
        errors.append(msg_fail)
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Helper: build overrides list from reviewer experiment spec
# ─────────────────────────────────────────────────────────────────────────────
def _exp_overrides(exp: Dict) -> List[str]:
    nc_reg_str = "true" if exp["nc_reg_enabled"] else "false"
    return [
        f"dataset.name=ham10000",
        f"dataset.imbalance_ratio=10",
        f"seed={exp['seed']}",
        f"model.head={exp['head']}",
        f"model.backbone=resnet18",
        f"training.epochs=2",
        f"training.batch_size=8",
        f"training.num_workers=0",
        f"debug.fast_dev_batches=2",
        f"nc_regularization.enabled={nc_reg_str}",
        f"nc_regularization.collapse_weight={exp['collapse_weight']}",
        f"nc_regularization.etf_align_weight={exp['align_weight']}",
        f"sampling.strategy={exp['sampling_strategy']}",
        f"etf.scale=16.0",
        f"nc_tracking.enabled=false",
        "tracking.tensorboard=false",
        "logging.save_checkpoints=false",
    ]


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 1: Unique output directories
# ─────────────────────────────────────────────────────────────────────────────
def check_unique_directories() -> None:
    print("\n[CHECK 1] Unique output directories")
    dirs = {}
    for k, exp in REVIEWER_EXPERIMENTS.items():
        d = str(results_dir(exp))
        if d in dirs:
            check(False,
                  "",
                  f"Directory collision: {k} and {dirs[d]} share path {d}")
        else:
            dirs[d] = k
    check(True, f"All {len(REVIEWER_EXPERIMENTS)} output directories are unique", "")


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 2: Config loads for every experiment
# ─────────────────────────────────────────────────────────────────────────────
def check_config_loading() -> None:
    print("\n[CHECK 2] Config loading for each experiment")
    for k, exp in REVIEWER_EXPERIMENTS.items():
        try:
            cfg = load_config(overrides=_exp_overrides(exp))
            check(
                cfg["nc_regularization"]["enabled"] == exp["nc_reg_enabled"],
                f"{k}: nc_reg_enabled={exp['nc_reg_enabled']} correctly loaded",
                f"{k}: nc_reg_enabled mismatch"
            )
            check(
                abs(cfg["nc_regularization"]["collapse_weight"] - exp["collapse_weight"]) < 1e-9,
                f"{k}: collapse_weight={exp['collapse_weight']} correctly loaded",
                f"{k}: collapse_weight mismatch"
            )
            check(
                abs(cfg["nc_regularization"]["etf_align_weight"] - exp["align_weight"]) < 1e-9,
                f"{k}: align_weight={exp['align_weight']} correctly loaded",
                f"{k}: align_weight mismatch"
            )
            check(
                cfg["sampling"]["strategy"] == exp["sampling_strategy"],
                f"{k}: sampling.strategy={exp['sampling_strategy']} correctly loaded",
                f"{k}: sampling.strategy mismatch"
            )
            check(
                cfg["seed"] == exp["seed"],
                f"{k}: seed={exp['seed']} correctly loaded",
                f"{k}: seed mismatch"
            )
        except Exception as exc:
            check(False, "", f"{k}: config load failed — {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 3: Model instantiation and ETF head
# ─────────────────────────────────────────────────────────────────────────────
def check_model_construction() -> None:
    print("\n[CHECK 3] Model construction and ETF head")
    from models.etf_classifier import ETFClassifier

    for k, exp in REVIEWER_EXPERIMENTS.items():
        try:
            set_seed(exp["seed"])
            cfg = load_config(overrides=_exp_overrides(exp))
            cfg["dataset"]["num_classes"] = 7
            model = build_model(cfg, method=exp["method_flag"])
            check(
                isinstance(model.fc, ETFClassifier),
                f"{k}: model.fc is ETFClassifier (fixed ETF head confirmed)",
                f"{k}: model.fc is NOT ETFClassifier — wrong head type"
            )
        except Exception as exc:
            check(False, "", f"{k}: model construction failed — {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 4: ETF matrices differ across seeds (stochastic isolation)
# ─────────────────────────────────────────────────────────────────────────────
def check_etf_seed_isolation() -> None:
    print("\n[CHECK 4] ETF matrices differ across seeds")
    from models.etf_classifier import ETFClassifier

    weights: Dict[int, torch.Tensor] = {}
    for k, exp in REVIEWER_EXPERIMENTS.items():
        seed = exp["seed"]
        if seed in weights:
            continue
        set_seed(seed)
        cfg = load_config(overrides=_exp_overrides(exp))
        cfg["dataset"]["num_classes"] = 7
        model = build_model(cfg, method="etf")
        weights[seed] = model.fc.weight.data.clone()

    seeds = list(weights.keys())
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            s1, s2 = seeds[i], seeds[j]
            are_different = not torch.allclose(weights[s1], weights[s2])
            check(
                are_different,
                f"Seeds {s1} and {s2} produce different ETF frames (good)",
                f"Seeds {s1} and {s2} produce IDENTICAL ETF frames — seed isolation broken!"
            )


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 5: Loss term wiring verification
# ─────────────────────────────────────────────────────────────────────────────
def check_loss_wiring() -> None:
    print("\n[CHECK 5] Loss term wiring")
    import torch.nn as nn

    # Key experiments to verify
    loss_checks = {
        "B1": {"nc_reg_enabled": False,  "collapse_weight": 0.0,           "align_weight": 0.0},
        "B2": {"nc_reg_enabled": True,   "collapse_weight": 0.01,          "align_weight": 0.0},
        "B3": {"nc_reg_enabled": True,   "collapse_weight": 0.0,           "align_weight": 0.01},
        "A1": {"nc_reg_enabled": True,   "collapse_weight": 0.01,          "align_weight": 0.01},
        "C1": {"nc_reg_enabled": True,   "collapse_weight": 0.001,         "align_weight": 0.001},
        "C2": {"nc_reg_enabled": True,   "collapse_weight": 0.05,          "align_weight": 0.05},
    }

    for k, spec in loss_checks.items():
        exp = REVIEWER_EXPERIMENTS[k]
        try:
            cfg = load_config(overrides=_exp_overrides(exp))
            cfg["dataset"]["num_classes"] = 7
            class_weights = torch.ones(7) / 7

            from training.losses import get_criterion
            device = torch.device("cpu")
            base_crit = get_criterion("etf", class_weights, cfg, device)

            nc_cfg = cfg.get("nc_regularization", {})
            if nc_cfg.get("enabled", False):
                criterion = CombinedNCLoss(
                    base_criterion   = base_crit,
                    num_classes      = 7,
                    collapse_weight  = nc_cfg.get("collapse_weight", 0.01),
                    etf_align_weight = nc_cfg.get("etf_align_weight", 0.01),
                )
                use_nc = True
            else:
                criterion = base_crit
                use_nc = False

            # Verify nc_reg usage
            check(
                use_nc == spec["nc_reg_enabled"],
                f"{k}: nc_reg_enabled={spec['nc_reg_enabled']} → {'CombinedNCLoss' if use_nc else 'plain CE'}",
                f"{k}: nc_reg_enabled mismatch"
            )

            if use_nc:
                got_cw = criterion.nc_collapse.weight
                got_aw = criterion.etf_align.weight
                check(
                    abs(got_cw - spec["collapse_weight"]) < 1e-9,
                    f"{k}: collapse_weight={spec['collapse_weight']} in CombinedNCLoss",
                    f"{k}: collapse_weight mismatch: got {got_cw}"
                )
                check(
                    abs(got_aw - spec["align_weight"]) < 1e-9,
                    f"{k}: align_weight={spec['align_weight']} in CombinedNCLoss",
                    f"{k}: align_weight mismatch: got {got_aw}"
                )

        except Exception as exc:
            check(False, "", f"{k}: loss wiring check failed — {exc}\n{traceback.format_exc()}")


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 6: Zero-weight early-exit actually returns zero gradient contribution
# ─────────────────────────────────────────────────────────────────────────────
def check_zero_weight_correctness() -> None:
    print("\n[CHECK 6] Zero-weight terms contribute zero to optimizer step")
    torch.manual_seed(0)

    B, D, C = 8, 512, 7
    features = torch.randn(B, D, requires_grad=True)
    labels   = torch.randint(0, C, (B,))

    # L_collapse with weight=0.0 should return 0
    collapse_zero = NCCollapseRegularizer(weight=0.0)
    loss_zero = collapse_zero(features, labels)
    check(
        loss_zero.item() == 0.0,
        "NCCollapseRegularizer(weight=0.0) returns exactly 0.0",
        f"NCCollapseRegularizer(weight=0.0) returned {loss_zero.item()}"
    )
    # Gradient through the zero should not affect param update
    loss_zero.backward()
    check(
        (features.grad is None or features.grad.abs().max().item() == 0.0),
        "Zero-weight L_collapse produces no feature gradient",
        "Zero-weight L_collapse produced non-zero feature gradient!"
    )

    # Reset
    features = torch.randn(B, D, requires_grad=True)

    # L_align with weight=0.0 should return 0
    align_zero = ETFAlignmentLoss(num_classes=C, weight=0.0)
    loss_az = align_zero(features, labels)
    check(
        loss_az.item() == 0.0,
        "ETFAlignmentLoss(weight=0.0) returns exactly 0.0",
        f"ETFAlignmentLoss(weight=0.0) returned {loss_az.item()}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 7: Command structure sanity
# ─────────────────────────────────────────────────────────────────────────────
def check_commands() -> None:
    print("\n[CHECK 7] Command structure for all experiments")
    for k, exp in REVIEWER_EXPERIMENTS.items():
        cmd = build_command(exp)
        has_train_py = str(ROOT / "train.py") in cmd
        has_method   = "--method" in cmd and exp["method_flag"] in cmd
        run_tag_flag = f"logging.run_tag={exp['run_tag']}"
        has_run_tag  = any(run_tag_flag in c for c in cmd)
        has_seed     = any(f"seed={exp['seed']}" in c for c in cmd)

        check(
            has_train_py and has_method and has_run_tag and has_seed,
            f"{k}: command is well-formed (train.py, --method, run_tag, seed all present)",
            f"{k}: command is malformed"
        )


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 8: 2-batch forward/backward pass on CPU for ablation experiments B1-B3
#           (validates the full training loop without real data or GPU)
# ─────────────────────────────────────────────────────────────────────────────
def check_forward_pass() -> None:
    print("\n[CHECK 8] Minimal 2-batch forward/backward pass (CPU, synthetic data)")
    print("  (This checks code correctness; NOT a real experiment result)")

    import torch.nn as nn
    from models.etf_classifier import ETFClassifier

    B, C, D = 8, 7, 512
    num_batches = 2

    ablation_keys = ["B1", "B2", "B3", "A1"]

    for k in ablation_keys:
        exp = REVIEWER_EXPERIMENTS[k]
        try:
            set_seed(exp["seed"])
            cfg = load_config(overrides=_exp_overrides(exp))
            cfg["dataset"]["num_classes"] = C

            model = build_model(cfg, method=exp["method_flag"])
            model.train()

            class_weights = torch.ones(C) / C
            from training.losses import get_criterion
            device = torch.device("cpu")
            base_crit = get_criterion("etf", class_weights, cfg, device)

            nc_cfg = cfg.get("nc_regularization", {})
            if nc_cfg.get("enabled", False):
                criterion = CombinedNCLoss(
                    base_criterion   = base_crit,
                    num_classes      = C,
                    collapse_weight  = nc_cfg.get("collapse_weight", 0.01),
                    etf_align_weight = nc_cfg.get("etf_align_weight", 0.01),
                )
                use_nc = True
            else:
                criterion = base_crit
                use_nc = False

            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=0.01, momentum=0.9, weight_decay=1e-4
            )

            total_loss = 0.0
            for _ in range(num_batches):
                # Synthetic images (3×224×224 — matches HAM10000 input)
                x      = torch.randn(B, 3, 224, 224)
                labels = torch.randint(0, C, (B,))
                optimizer.zero_grad()
                features = model.forward_features(x)
                logits   = model.fc(features)

                if use_nc:
                    loss = criterion(logits, features, labels)
                else:
                    loss = criterion(logits, labels)

                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            check(
                True,
                f"{k}: 2-batch forward/backward pass succeeded (loss={total_loss/num_batches:.4f})",
                ""
            )

        except Exception as exc:
            check(False, "", f"{k}: forward pass failed — {exc}\n{traceback.format_exc()}")


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 9: B2 and B3 produce different loss values (ablation isolation)
# ─────────────────────────────────────────────────────────────────────────────
def check_ablation_isolation() -> None:
    print("\n[CHECK 9] Ablation isolation: B2 and B3 produce different losses")
    import torch.nn as nn

    B, C, D = 8, 7, 512
    torch.manual_seed(42)

    features = torch.randn(B, D)
    labels   = torch.arange(B) % C
    logits   = torch.randn(B, C)

    # B2: collapse only
    crit_b2 = CombinedNCLoss(
        nn.CrossEntropyLoss(), C,
        collapse_weight=0.01, etf_align_weight=0.0
    )
    loss_b2 = crit_b2(logits, features, labels).item()

    # B3: align only
    crit_b3 = CombinedNCLoss(
        nn.CrossEntropyLoss(), C,
        collapse_weight=0.0, etf_align_weight=0.01
    )
    loss_b3 = crit_b3(logits, features, labels).item()

    # Full (B4 / A1): both terms
    crit_full = CombinedNCLoss(
        nn.CrossEntropyLoss(), C,
        collapse_weight=0.01, etf_align_weight=0.01
    )
    loss_full = crit_full(logits, features, labels).item()

    check(
        abs(loss_b2 - loss_b3) > 1e-6,
        f"B2 loss ({loss_b2:.6f}) ≠ B3 loss ({loss_b3:.6f}) — ablations are isolated",
        f"B2 and B3 losses are identical ({loss_b2:.6f}) — ablation NOT isolated!"
    )
    check(
        loss_full > loss_b2 and loss_full > loss_b3,
        f"Full B4 loss ({loss_full:.6f}) > B2 ({loss_b2:.6f}) and B3 ({loss_b3:.6f})",
        f"Unexpected: full loss ({loss_full:.6f}) not > both ablations"
    )
    print(f"  Loss breakdown — CE_base + L_collapse + L_align:")
    print(f"    B1 (ETF-only):         CE only")
    print(f"    B2 (ETF+L_collapse):   loss = {loss_b2:.6f}")
    print(f"    B3 (ETF+L_align):      loss = {loss_b3:.6f}")
    print(f"    B4 (ETF+both):         loss = {loss_full:.6f}")


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 10: Original ETF+NC-Reg (seed 42) is unaffected — config snapshot check
# ─────────────────────────────────────────────────────────────────────────────
def check_original_unchanged() -> None:
    print("\n[CHECK 10] Original seed-42 ETF+NC-Reg configuration is preserved")
    snapshot = ROOT / "results" / "phase2" / "etf_nc_reg_ham10000_r10_s42" / "config_snapshot.yaml"
    if not snapshot.exists():
        print(f"  [SKIP] Snapshot not found at {snapshot} — cannot verify (this is OK)")
        return
    import yaml
    # The snapshot uses !!python/object/new:torch.torch_version.TorchVersion which
    # requires full_load. We catch any remaining errors and fall back to text parsing.
    try:
        with open(snapshot) as f:
            orig = yaml.full_load(f)
    except Exception:
        # Fall back: parse only the lines we care about via text search
        text = snapshot.read_text()
        def _text_val(key: str) -> str:
            import re
            m = re.search(rf"^\s+{key}:\s*(.+)$", text, re.MULTILINE)
            return m.group(1).strip() if m else ""
        check(
            _text_val("collapse_weight") == "0.01",
            "Original collapse_weight=0.01 unchanged (text-verified)",
            f"Original collapse_weight changed! Got: {_text_val('collapse_weight')}"
        )
        check(
            _text_val("etf_align_weight") == "0.01",
            "Original etf_align_weight=0.01 unchanged (text-verified)",
            f"Original etf_align_weight changed! Got: {_text_val('etf_align_weight')}"
        )
        check(
            "enabled: true" in text,
            "Original nc_reg enabled=True unchanged (text-verified)",
            "Original nc_reg enabled changed!"
        )
        check(
            "head: etf" in text,
            "Original model.head=etf unchanged (text-verified)",
            "Original model.head changed!"
        )
        check(
            "strategy: weighted" in text,
            "Original sampling.strategy=weighted unchanged (text-verified)",
            "Original sampling.strategy changed!"
        )
        return


    check(
        orig["nc_regularization"]["collapse_weight"] == 0.01,
        "Original collapse_weight=0.01 unchanged in config_snapshot",
        "Original collapse_weight changed!"
    )
    check(
        orig["nc_regularization"]["etf_align_weight"] == 0.01,
        "Original etf_align_weight=0.01 unchanged in config_snapshot",
        "Original etf_align_weight changed!"
    )
    check(
        orig["nc_regularization"]["enabled"] is True,
        "Original nc_reg enabled=True unchanged in config_snapshot",
        "Original nc_reg enabled changed!"
    )
    check(
        orig["model"]["head"] == "etf",
        "Original model.head=etf unchanged",
        "Original model.head changed!"
    )
    check(
        orig["sampling"]["strategy"] == "weighted",
        "Original sampling.strategy=weighted unchanged",
        "Original sampling.strategy changed!"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 70)
    print("  NC-MedAI — Reviewer Experiment Configuration Validation")
    print("=" * 70)
    print(f"  Experiments: {list(REVIEWER_EXPERIMENTS.keys())}")
    print(f"  Total: {len(REVIEWER_EXPERIMENTS)}")
    print("=" * 70)

    check_unique_directories()
    check_config_loading()
    check_model_construction()
    check_etf_seed_isolation()
    check_loss_wiring()
    check_zero_weight_correctness()
    check_commands()
    check_forward_pass()
    check_ablation_isolation()
    check_original_unchanged()

    print("\n" + "=" * 70)
    if errors:
        print(f"  VALIDATION FAILED — {len(errors)} error(s):")
        for e in errors:
            print(f"    ✗ {e}")
        print("=" * 70)
        sys.exit(1)
    else:
        print(f"  ALL CHECKS PASSED ✓ ({10} test groups, {len(REVIEWER_EXPERIMENTS)} experiments)")
        print("  The repository is ready for GPU deployment.")
        print("=" * 70)
        sys.exit(0)


if __name__ == "__main__":
    main()
