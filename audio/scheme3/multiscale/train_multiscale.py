"""
多尺度联合训练脚本（可直接开跑版本）。

训练策略：
- VAE 冻结
- Transformer + UNet + 多尺度融合头 联合训练
- 解决不同窗口长度对应 latent 长度不同：按样本 target_latent_len 重采样并使用 latent_mask
- low 尺度默认弱对齐（pooling）
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import math
from contextlib import contextmanager
from functools import partial
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader

from .cached_dataset import CachedMultiscaleLatentDataset, collate_cached_multiscale_batch
from .mel_config import DEFAULT_MEL_CONFIG
from .model import MultiScaleLatentSpaceConditionalModel
from .structural_features import StructuralFeatureBuilder
from .train_utils import masked_mse_loss, resize_latent_per_sample


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache_index_path", type=str, required=True)
    p.add_argument("--save_dir", type=str, default="logs/scheme3_multiscale/checkpoints")
    p.add_argument("--device", type=str, default="cuda")

    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--max_steps", type=int, default=100000)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--save_every", type=int, default=2000)
    p.add_argument("--save_optimizer", action="store_true", help="保存 optimizer 状态（占用更多磁盘）")

    p.add_argument("--lr_min", type=float, default=1e-6, help="Cosine 衰减下限")
    p.add_argument("--lr_warmup_steps", type=int, default=1000, help="线性 warmup 步数")

    p.add_argument("--use_ema", action="store_true", help="启用 EMA 权重")
    p.add_argument("--ema_decay", type=float, default=0.999, help="EMA 衰减系数")

    p.add_argument("--resume_from", type=str, default="", help="从已有 checkpoint 续跑")
    p.add_argument("--val_every", type=int, default=1000, help="每隔多少 step 做一次验证")
    p.add_argument("--val_batches", type=int, default=50, help="每次验证最多评估多少个 batch")

    p.add_argument("--pad_token", type=int, default=0)
    p.add_argument("--condition_dropout", type=float, default=0.15)
    p.add_argument("--scale_dropout", type=float, default=0.1)
    p.add_argument("--token_shift", type=int, default=2)

    p.add_argument("--latent_len_high", type=int, default=64)
    p.add_argument("--latent_len_mid", type=int, default=128)
    p.add_argument("--latent_len_low", type=int, default=256)

    # 模型参数（保留常用可调项）
    p.add_argument("--latent_dim", type=int, default=32)
    p.add_argument("--embedding_dim", type=int, default=256)
    p.add_argument("--midi_vocab_size", type=int, default=390)
    p.add_argument("--num_layers", type=int, default=6)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--max_seq", type=int, default=2048)

    p.add_argument("--n_fft", type=int, default=DEFAULT_MEL_CONFIG.n_fft)
    p.add_argument("--hop_length", type=int, default=DEFAULT_MEL_CONFIG.hop_length)
    p.add_argument("--win_length", type=int, default=DEFAULT_MEL_CONFIG.win_length)
    p.add_argument("--center", type=int, default=int(DEFAULT_MEL_CONFIG.center), help="仅用于统一记录 mel 配置")
    return p.parse_args()


class EMAHelper:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        if not (0.0 < decay < 1.0):
            raise ValueError(f"ema_decay must be in (0,1), got {decay}")
        self.decay = float(decay)
        self.shadow: Dict[str, torch.Tensor] = {}
        self.backup: Dict[str, torch.Tensor] = {}
        self.update(model, init=True)

    @torch.no_grad()
    def update(self, model: torch.nn.Module, init: bool = False):
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            p = param.detach()
            if init or name not in self.shadow:
                self.shadow[name] = p.clone()
            else:
                self.shadow[name].mul_(self.decay).add_(p, alpha=1.0 - self.decay)

    @torch.no_grad()
    def apply_shadow(self, model: torch.nn.Module):
        self.backup = {}
        for name, param in model.named_parameters():
            if not param.requires_grad or name not in self.shadow:
                continue
            self.backup[name] = param.detach().clone()
            param.data.copy_(self.shadow[name].data)

    @torch.no_grad()
    def restore(self, model: torch.nn.Module):
        if not self.backup:
            return
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name].data)
        self.backup = {}

    def state_dict(self) -> Dict[str, Dict[str, torch.Tensor] | float]:
        return {
            "decay": self.decay,
            "shadow": self.shadow,
        }

    def load_state_dict(self, state: Dict):
        self.decay = float(state.get("decay", self.decay))
        raw_shadow = state.get("shadow", {})
        self.shadow = {k: v.clone() for k, v in raw_shadow.items()}
        self.backup = {}


@contextmanager
def maybe_ema_scope(model: torch.nn.Module, ema_helper: Optional[EMAHelper]):
    if ema_helper is None:
        yield
        return
    ema_helper.apply_shadow(model)
    try:
        yield
    finally:
        ema_helper.restore(model)


def build_model(args) -> MultiScaleLatentSpaceConditionalModel:
    model = MultiScaleLatentSpaceConditionalModel(
        latent_dim=args.latent_dim,
        embedding_dim=args.embedding_dim,
        midi_vocab_size=args.midi_vocab_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        max_seq=args.max_seq,
        use_cross_attention=True,
        cross_attention_heads=8,
    )

    # 冻结 VAE，仅训练条件模型 + UNet
    for p in model.vae_encoder.parameters():
        p.requires_grad = False
    for p in model.vae_decoder.parameters():
        p.requires_grad = False
    model.vae_encoder.eval()
    model.vae_decoder.eval()
    return model


def _free_bytes(path: str) -> int:
    d = os.path.dirname(path) or "."
    usage = shutil.disk_usage(d)
    return int(usage.free)


def save_checkpoint(
    path: str,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    args_dict: Dict,
    save_optimizer: bool = False,
    extra_state: Optional[Dict] = None,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    ema_helper: Optional[EMAHelper] = None,
):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    state = {
        "step": step,
        "model": model.state_dict(),
        "args": args_dict,
    }
    if save_optimizer:
        state["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    if ema_helper is not None:
        state["ema"] = ema_helper.state_dict()
    if extra_state:
        state.update(extra_state)

    est_bytes = sum(t.numel() * t.element_size() for t in model.state_dict().values())
    need_bytes = int(est_bytes * 2.2) + 512 * 1024 * 1024
    free_bytes = _free_bytes(path)
    if free_bytes < need_bytes:
        raise RuntimeError(
            f"Insufficient disk space for checkpoint. free={free_bytes/1024**3:.2f}GB, "
            f"required~={need_bytes/1024**3:.2f}GB, path={path}"
        )

    tmp_path = path + ".tmp"
    try:
        torch.save(state, tmp_path)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cud":
        device_arg = "cuda"

    if device_arg == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA not available, fallback to CPU")
        device_arg = "cpu"

    if device_arg == "mps":
        mps_ok = bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()
        if not mps_ok:
            print("[WARN] MPS not available, fallback to CPU")
            device_arg = "cpu"

    return torch.device(device_arg)


def _forward_loss(model, structural_builder, batch, args, device):
    audio_latents = batch["audio_latents"].to(device)
    midi_tokens = batch["midi_tokens"].to(device)
    midi_mask = batch["midi_mask"].to(device)

    # 防止 MIDI token 序列超过 encoder 位置编码上限
    if midi_tokens.size(1) > args.max_seq:
        midi_tokens = midi_tokens[:, : args.max_seq]
        midi_mask = midi_mask[:, : args.max_seq]

    scale_id = batch["scale_id"].to(device)
    target_lens = batch["target_latent_len"].to(device)

    # 统一 latent 长度 + mask
    latents_resized, latent_mask = resize_latent_per_sample(audio_latents, target_lens)
    structural_features = structural_builder.build(batch).to(device)

    pred_noise, noise = model.forward_with_latent_and_structure(
        audio_latent=latents_resized,
        midi_tokens=midi_tokens,
        structural_features=structural_features,
        scale_id=scale_id,
        midi_mask=midi_mask,
        condition_dropout_rate=args.condition_dropout,
        scale_dropout_rate=args.scale_dropout,
        weak_low_alignment=True,
        token_shift_for_weak_alignment=args.token_shift,
    )

    loss = masked_mse_loss(pred_noise, noise, latent_mask)
    return loss, target_lens, audio_latents.size(0)


@torch.no_grad()
def run_validation(model, val_loader, structural_builder, args, device, max_batches: int = 50) -> float:
    model.eval()
    losses = []

    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break
        loss, _, _ = _forward_loss(model, structural_builder, batch, args, device)
        losses.append(float(loss.item()))

    model.train()
    if not losses:
        return float("inf")
    return float(sum(losses) / len(losses))


def _build_lr_lambda(max_steps: int, warmup_steps: int, min_lr_ratio: float):
    total = max(1, int(max_steps))
    warmup = max(0, int(warmup_steps))
    min_lr_ratio = float(min(1.0, max(0.0, min_lr_ratio)))

    def _lr_lambda(step: int):
        if warmup > 0 and step < warmup:
            warmup_ratio = float(step + 1) / float(max(1, warmup))
            return max(min_lr_ratio, warmup_ratio)

        progress = float(step - warmup) / float(max(1, total - warmup))
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + math.cos(progress * math.pi))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return _lr_lambda


def main():
    args = parse_args()
    device = _resolve_device(args.device)

    if not args.save_dir or not str(args.save_dir).strip():
        raise ValueError("--save_dir 不能为空，请传入有效目录路径")
    os.makedirs(args.save_dir, exist_ok=True)

    target_latent_lens = {
        "high": args.latent_len_high,
        "mid": args.latent_len_mid,
        "low": args.latent_len_low,
    }

    collate = partial(
        collate_cached_multiscale_batch,
        pad_token=args.pad_token,
        target_latent_lens=target_latent_lens,
    )

    ds_train = CachedMultiscaleLatentDataset(cache_index_path=args.cache_index_path, split="train")
    ds_val = CachedMultiscaleLatentDataset(cache_index_path=args.cache_index_path, split="val")

    dl_train = DataLoader(
        ds_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate,
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(1, args.num_workers // 2),
        pin_memory=(device.type == "cuda"),
        collate_fn=collate,
    )

    model = build_model(args).to(device)
    structural_builder = StructuralFeatureBuilder()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )

    min_lr_ratio = float(args.lr_min / max(args.lr, 1e-12))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=_build_lr_lambda(
            max_steps=args.max_steps,
            warmup_steps=args.lr_warmup_steps,
            min_lr_ratio=min_lr_ratio,
        ),
    )

    ema_helper = EMAHelper(model, decay=args.ema_decay) if args.use_ema else None

    step = 0
    best_val = float("inf")

    if args.resume_from:
        if not os.path.exists(args.resume_from):
            raise FileNotFoundError(f"--resume_from 文件不存在: {args.resume_from}")

        ckpt = torch.load(args.resume_from, map_location=device)
        model.load_state_dict(ckpt["model"], strict=True)

        if "optimizer" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer"])
            except Exception as e:
                print(f"[WARN] Failed to restore optimizer state: {e}")

        if "scheduler" in ckpt:
            try:
                scheduler.load_state_dict(ckpt["scheduler"])
            except Exception as e:
                print(f"[WARN] Failed to restore scheduler state: {e}")

        if ema_helper is not None and "ema" in ckpt:
            try:
                ema_helper.load_state_dict(ckpt["ema"])
            except Exception as e:
                print(f"[WARN] Failed to restore EMA state: {e}")

        step = int(ckpt.get("step", 0))
        best_val = float(ckpt.get("best_val", best_val))
        print(
            json.dumps(
                {
                    "resume_from": args.resume_from,
                    "resume_step": step,
                    "best_val": best_val,
                },
                ensure_ascii=False,
            )
        )

    model.train()

    while step < args.max_steps:
        for batch in dl_train:
            if step >= args.max_steps:
                break

            loss, target_lens, batch_size = _forward_loss(model, structural_builder, batch, args, device)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()
            if ema_helper is not None:
                ema_helper.update(model)

            step += 1

            if step % args.log_every == 0:
                print(
                    json.dumps(
                        {
                            "step": step,
                            "loss": float(loss.item()),
                            "lr": float(optimizer.param_groups[0]["lr"]),
                            "batch_size": int(batch_size),
                            "target_len_min": int(target_lens.min().item()),
                            "target_len_max": int(target_lens.max().item()),
                        },
                        ensure_ascii=False,
                    )
                )

            if args.val_every > 0 and step % args.val_every == 0:
                with maybe_ema_scope(model, ema_helper):
                    val_loss = run_validation(
                        model=model,
                        val_loader=dl_val,
                        structural_builder=structural_builder,
                        args=args,
                        device=device,
                        max_batches=max(1, args.val_batches),
                    )
                print(json.dumps({"step": step, "val_loss": float(val_loss)}, ensure_ascii=False))

                if val_loss < best_val:
                    best_val = val_loss
                    best_path = os.path.join(args.save_dir, "multiscale_best.pt")
                    with maybe_ema_scope(model, ema_helper):
                        save_checkpoint(
                            best_path,
                            step,
                            model,
                            optimizer,
                            vars(args),
                            save_optimizer=args.save_optimizer,
                            extra_state={"best_val": best_val},
                            scheduler=scheduler,
                            ema_helper=ema_helper,
                        )
                    print(f"Saved best checkpoint: {best_path} (best_val={best_val:.6f})")

            if args.save_every > 0 and step % args.save_every == 0:
                ckpt_path = os.path.join(args.save_dir, f"multiscale_step_{step}.pt")
                save_checkpoint(
                    ckpt_path,
                    step,
                    model,
                    optimizer,
                    vars(args),
                    save_optimizer=args.save_optimizer,
                    extra_state={"best_val": best_val},
                    scheduler=scheduler,
                    ema_helper=ema_helper,
                )
                print(f"Saved checkpoint: {ckpt_path}")

    final_ckpt = os.path.join(args.save_dir, "multiscale_final.pt")
    with maybe_ema_scope(model, ema_helper):
        save_checkpoint(
            final_ckpt,
            step,
            model,
            optimizer,
            vars(args),
            save_optimizer=args.save_optimizer,
            extra_state={"best_val": best_val},
            scheduler=scheduler,
            ema_helper=ema_helper,
        )
    print(f"Training done. Final checkpoint: {final_ckpt}")


if __name__ == "__main__":
    main()
