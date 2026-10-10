"""
training/trainer.py  (updated)
─────────────────────────────────────────────────────────────────────────────
Extended Trainer with:
  • NC-aware regularization (CombinedNCLoss)
  • Per-epoch NC1–NC4 tracking
  • TensorBoard logging (optional)
  • Progressive sampler epoch update
  • Medical dataset compatible
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.logging_utils import AverageMeter, get_logger
from evaluation.nc_metrics import compute_all_nc_metrics, NCMetrics
from training.losses import get_criterion
from training.scheduler import build_scheduler
from training.nc_regularization import CombinedNCLoss


class Trainer:
    """
    Full training + evaluation loop for medical AI / NC experiments.

    New vs original
    ---------------
    • Supports CombinedNCLoss (classification + NC regularization).
    • Logs NC1–NC4 every `track_nc_every_n_epochs` epochs.
    • Writes to TensorBoard SummaryWriter when cfg.tracking.tensorboard=true.
    • Calls sampler.set_epoch(e) for ProgressiveSampler compatibility.

    Parameters
    ----------
    model        : nn.Module
    train_loader : DataLoader
    val_loader   : DataLoader
    class_weights: torch.Tensor
    cfg          : dict
    method       : str
    seed         : int
    device       : torch.device
    run_tag      : str
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        class_weights: torch.Tensor,
        cfg: dict,
        method: str = "baseline",
        seed: int = 42,
        device: Optional[torch.device] = None,
        run_tag: str = "run",
    ) -> None:
        self.model        = model
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.cfg          = cfg
        self.method       = method
        self.seed         = seed
        self.run_tag      = run_tag

        self.device = device or (
            __import__("utils.device", fromlist=["get_best_device"]).get_best_device(cfg)[0]
        )
        self.model.to(self.device)

        # ── Logging ───────────────────────────────────────────────────────────
        log_dir = cfg["logging"].get("results_dir", "./results")
        self.logger = get_logger(name=f"trainer.{run_tag}", log_dir=log_dir)

        # ── TensorBoard ───────────────────────────────────────────────────────
        self.writer = None
        if cfg.get("tracking", {}).get("tensorboard", False):
            try:
                from torch.utils.tensorboard import SummaryWriter
                tb_dir = Path(cfg.get("tracking", {}).get("log_dir", "./logs")) / run_tag
                tb_dir.mkdir(parents=True, exist_ok=True)
                self.writer = SummaryWriter(log_dir=str(tb_dir))
                self.logger.info(f"  TensorBoard → {tb_dir}")
            except ImportError:
                self.logger.warning("tensorboard not installed; skipping TB logging.")

        # ── Loss ──────────────────────────────────────────────────────────────
        base_crit = get_criterion(method, class_weights, cfg, self.device)
        nc_cfg    = cfg.get("nc_regularization", {})
        if nc_cfg.get("enabled", False):
            self.criterion = CombinedNCLoss(
                base_criterion   = base_crit,
                num_classes      = cfg["dataset"]["num_classes"],
                collapse_weight  = nc_cfg.get("collapse_weight", 0.01),
                etf_align_weight = nc_cfg.get("etf_align_weight", 0.01),
            )
            self._use_nc_loss = True
            self.logger.info("  NC regularization ENABLED")
        else:
            self.criterion    = base_crit
            self._use_nc_loss = False

        # ── Optimiser ─────────────────────────────────────────────────────────
        train_cfg = cfg["training"]
        self.optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=float(train_cfg["lr"]),
            momentum=float(train_cfg.get("momentum", 0.9)),
            weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
            nesterov=True,
        )
        self.scheduler  = build_scheduler(self.optimizer, cfg)
        self.epochs     = int(train_cfg["epochs"])
        self.num_classes = cfg["dataset"]["num_classes"]

        # ── NC tracking ───────────────────────────────────────────────────────
        nc_track_cfg = cfg.get("nc_tracking", {})
        if nc_track_cfg.get("enabled", False):
            self.track_nc_every = int(nc_track_cfg.get("every_n_epochs", 1))
        else:
            self.track_nc_every = cfg.get("evaluation", {}).get(
                "track_nc_every_n_epochs", 0)
        self.nc_history: List[Dict] = []

        # ── fast_dev_batches (smoke testing) ──────────────────────────────────
        self.fast_dev_batches = int(
            cfg.get("debug", {}).get("fast_dev_batches", 0))

        # ── Checkpointing ─────────────────────────────────────────────────────
        self.save_ckpt = cfg["logging"].get("save_checkpoints", True)
        self.ckpt_dir  = Path(cfg["logging"].get("checkpoint_dir", "./checkpoints")) / run_tag
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.log_every = cfg["logging"].get("log_every_n_epochs", 5)
        
        self.start_epoch = 1
        self.best_val_acc = 0.0

        # ── Mixed precision (CUDA only) ─────────────────────────────────────────
        mp_setting = cfg.get("training", {}).get("mixed_precision", "auto")
        self._use_amp = (
            self.device.type == "cuda"
            and mp_setting != "off"
        )
        try:
            self._scaler = (
                torch.cuda.amp.GradScaler() if self._use_amp else None
            )
        except Exception:
            self._use_amp = False
            self._scaler  = None

    def resume(self, checkpoint_path: str) -> None:
        """Resume from a saved checkpoint."""
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            self.logger.warning(f"Checkpoint not found: {ckpt_path}. Starting from scratch.")
            return
        
        self.logger.info(f"Resuming from {ckpt_path}...")
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["state_dict"])
        if "optimizer" in checkpoint and self.optimizer:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint and self.scheduler:
            self.scheduler.load_state_dict(checkpoint["scheduler"])
        if "epoch" in checkpoint:
            self.start_epoch = checkpoint["epoch"] + 1
        if "best_val_acc" in checkpoint:
            self.best_val_acc = checkpoint["best_val_acc"]
        if "nc_history" in checkpoint:
            self.nc_history = checkpoint["nc_history"]
        self.logger.info(f"Resumed at epoch {self.start_epoch} with best_val_acc {self.best_val_acc:.2f}%")

    def _save_checkpoint(self, epoch: int, is_best: bool = False, is_latest: bool = True):
        if not self.save_ckpt:
            return
        state = {
            "epoch": epoch,
            "state_dict": {k: v.cpu().clone() for k, v in self.model.state_dict().items()},
            "best_val_acc": self.best_val_acc,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "nc_history": self.nc_history,
        }
        if is_best:
            torch.save(state, self.ckpt_dir / "best_model.pth")
        if is_latest:
            torch.save(state, self.ckpt_dir / "latest.pth")
        if epoch % self.log_every == 0:
            torch.save(state, self.ckpt_dir / f"epoch_{epoch}.pth")

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self) -> Dict[str, Any]:
        """Execute the full training pipeline and return results dict."""
        history: Dict[str, list] = {
            "train_loss": [], "train_acc": [], "val_acc": [], "epoch_times": []}

        self.logger.info(
            f"[{self.run_tag}] device={self.device} "
            f"method={self.method} epochs={self.epochs}"
        )

        epoch = self.start_epoch - 1
        try:
            for epoch in range(self.start_epoch, self.epochs + 1):
                t0 = time.time()

                # Update progressive sampler if applicable
                sampler = self.train_loader.sampler
                if hasattr(sampler, "set_epoch"):
                    sampler.set_epoch(epoch)

                train_loss, train_acc, sps = self._train_epoch(epoch)
                val_acc               = self._eval_epoch()
                self.scheduler.step()
                epoch_time            = time.time() - t0

                history["train_loss"].append(train_loss)
                history["train_acc"].append(train_acc)
                history["val_acc"].append(val_acc)
                history["epoch_times"].append(epoch_time)
                history.setdefault("samples_per_sec", []).append(round(sps, 1))

                is_best = False
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    is_best = True

                # ── TensorBoard ───────────────────────────────────────────────────
                if self.writer:
                    self.writer.add_scalar("Loss/train",    train_loss, epoch)
                    self.writer.add_scalar("Acc/train",     train_acc,  epoch)
                    self.writer.add_scalar("Acc/val",       val_acc,    epoch)
                    self.writer.add_scalar(
                        "LR", self.optimizer.param_groups[0]["lr"], epoch)

                # ── NC Tracking ───────────────────────────────────────────────────
                if self.track_nc_every > 0 and (
                    epoch % self.track_nc_every == 0 or epoch == self.epochs
                ):
                    nc = self._compute_nc(epoch)
                    self.nc_history.append(nc.to_dict())
                    if self.writer:
                        self.writer.add_scalar("NC/nc1", nc.nc1, epoch)
                        self.writer.add_scalar("NC/nc2", nc.nc2, epoch)
                        if not (nc.nc3 != nc.nc3):   # not nan
                            self.writer.add_scalar("NC/nc3", nc.nc3, epoch)

                # ── Console logging ───────────────────────────────────────────────
                if epoch % self.log_every == 0 or epoch == self.epochs:
                    elapsed = time.time() - t0
                    self.logger.info(
                        f"  ep {epoch:3d}/{self.epochs} "
                        f"| loss {train_loss:.4f} "
                        f"| tr {train_acc:.1f}% "
                        f"| val {val_acc:.1f}% "
                        f"| best {self.best_val_acc:.1f}% "
                        f"| {elapsed:.1f}s"
                    )

                self._save_checkpoint(epoch, is_best=is_best, is_latest=True)

        except KeyboardInterrupt:
            if epoch >= self.start_epoch:
                self.logger.warning("Training interrupted! Saving latest checkpoint...")
                self._save_checkpoint(epoch, is_best=False, is_latest=True)
                self.logger.info(f"Checkpoint saved at epoch {epoch}. Exiting gracefully.")
            raise

        # ── Final NC metrics on best model ────────────────────────────────────
        best_ckpt = self.ckpt_dir / "best_model.pth"
        if best_ckpt.exists():
            state = torch.load(best_ckpt, map_location=self.device)
            self.model.load_state_dict(state["state_dict"])
        final_nc = self._compute_nc(epoch=self.epochs)
        # Always ensure final epoch NC is in history
        if not self.nc_history or self.nc_history[-1].get("epoch") != self.epochs:
            self.nc_history.append(final_nc.to_dict())
        self.logger.info(
            f"  Final: NC1={final_nc.nc1:.4f} NC2={final_nc.nc2:.6f} "
            f"NC3={final_nc.nc3:.4f} NC4={final_nc.nc4:.4f} "
            f"best_val={self.best_val_acc:.2f}%"
        )

        if self.writer:
            self.writer.close()

        return {
            **history,
            "best_val_acc": self.best_val_acc,
            "nc1": final_nc.nc1, "nc2": final_nc.nc2,
            "nc3": final_nc.nc3, "nc4": final_nc.nc4,
            "nc_history": self.nc_history,
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    def _train_epoch(self, epoch: int):
        self.model.train()
        loss_m = AverageMeter("Loss")
        acc_m  = AverageMeter("Acc")
        n_samples = 0
        t_start   = time.time()
        n_batches = len(self.train_loader) if self.fast_dev_batches == 0 else min(
            self.fast_dev_batches, len(self.train_loader))

        for batch_idx, (images, labels) in enumerate(self.train_loader):
            if self.fast_dev_batches > 0 and batch_idx >= self.fast_dev_batches:
                break
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)

            if self._use_amp:
                with torch.cuda.amp.autocast():
                    features = self.model.forward_features(images)
                    logits   = self.model.fc(features)
                    if self._use_nc_loss:
                        loss = self.criterion(logits, features, labels)
                    else:
                        loss = self.criterion(logits, labels)
                self._scaler.scale(loss).backward()
                self._scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self._scaler.step(self.optimizer)
                self._scaler.update()
            else:
                features = self.model.forward_features(images)
                logits   = self.model.fc(features)
                if self._use_nc_loss:
                    loss = self.criterion(logits, features, labels)
                else:
                    loss = self.criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.optimizer.step()

            preds = logits.argmax(1)
            acc   = (preds == labels).float().mean().item() * 100.0
            bs    = images.size(0)
            loss_m.update(loss.item(), bs)
            acc_m.update(acc, bs)
            n_samples += bs

            # Heartbeat: print every 50 batches so log file shows live progress.
            # Critical for subprocess runs where silence looks like a hang.
            if (batch_idx + 1) % 50 == 0:
                elapsed_so_far = time.time() - t_start
                print(
                    f"    [ep {epoch}] batch {batch_idx+1}/{n_batches}"
                    f"  loss={loss_m.avg:.4f}  acc={acc_m.avg:.1f}%"
                    f"  {elapsed_so_far:.0f}s",
                    flush=True,
                )

        elapsed = max(time.time() - t_start, 1e-6)
        return loss_m.avg, acc_m.avg, n_samples / elapsed   # loss, acc, samples/sec

    @torch.no_grad()
    def _eval_epoch(self) -> float:
        self.model.eval()
        correct = total = 0
        for batch_idx, (images, labels) in enumerate(self.val_loader):
            if self.fast_dev_batches > 0 and batch_idx >= self.fast_dev_batches:
                break
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            preds  = self.model(images).argmax(1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
        return 100.0 * correct / total if total else 0.0

    @torch.no_grad()
    def _compute_nc(self, epoch: int = 0) -> NCMetrics:
        self.model.eval()
        all_f, all_l, all_g = [], [], []
        for batch_idx, (images, labels) in enumerate(self.val_loader):
            if self.fast_dev_batches > 0 and batch_idx >= self.fast_dev_batches:
                break
            images = images.to(self.device, non_blocking=True)
            feats  = self.model.forward_features(images)
            logits = self.model.fc(feats)
            all_f.append(feats.cpu())
            all_l.append(labels)
            all_g.append(logits.cpu())

        feats  = torch.cat(all_f)
        labels = torch.cat(all_l)
        logits = torch.cat(all_g)

        return compute_all_nc_metrics(
            feats, labels, self.num_classes,
            model=self.model, logits=logits, epoch=epoch,
        )
