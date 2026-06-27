from __future__ import annotations

import argparse
import contextlib
import math
import os
import re
import signal
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config.config import default_config as config
from src.models.transformer import Transformer


# --- Runtime Diagnostics Helpers ---

def bytes_to_gib(num_bytes: int) -> float:
    """Convert a byte count to gibibytes for human-readable memory reports."""
    return num_bytes / (1024 ** 3)


def get_device_report(device: str) -> str:
    """Build a short report describing the runtime environment."""
    lines = [
        f"PyTorch version: {torch.__version__}",
        f"Configured device: {device}",
        f"CUDA available: {torch.cuda.is_available()}",
        f"CUDA version: {torch.version.cuda}",
    ]
    if device.startswith('cuda') and torch.cuda.is_available():
        device_index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device_index)
        total_vram_gib = bytes_to_gib(props.total_memory)
        lines.extend([
            f"GPU name: {torch.cuda.get_device_name(device_index)}",
            f"GPU capability: {props.major}.{props.minor}",
            f"Total VRAM: {total_vram_gib:.2f} GiB",
        ])
    else:
        lines.append("GPU name: N/A (running without CUDA)")
    return "\n".join(lines)


def get_peak_memory_report(device: str) -> str:
    """Report peak GPU memory since the last reset, or N/A on CPU."""
    if device.startswith('cuda') and torch.cuda.is_available():
        peak_allocated = bytes_to_gib(torch.cuda.max_memory_allocated())
        peak_reserved = bytes_to_gib(torch.cuda.max_memory_reserved())
        return (
            f"Peak VRAM allocated: {peak_allocated:.2f} GiB | "
            f"Peak VRAM reserved: {peak_reserved:.2f} GiB"
        )
    return "Peak VRAM allocated: N/A | Peak VRAM reserved: N/A"


def estimate_memory_budget(num_params: int, device: str, use_amp: bool) -> str:
    """Print a rough training VRAM budget so users can predict OOM before launching."""
    if not (device.startswith("cuda") and torch.cuda.is_available()):
        return "VRAM budget: N/A (no CUDA device)"
    state_gib = bytes_to_gib(num_params * 16)
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    total_gib = bytes_to_gib(props.total_memory)
    note = " (+ activations; reduce with --grad-checkpointing / --grad-accum)"
    return (
        f"VRAM budget: ~{state_gib:.2f} GiB params+optimizer state vs {total_gib:.2f} GiB "
        f"total on {torch.cuda.get_device_name()}{note}"
    )


# --- Learning Rate Scheduler ---

def get_lr_with_warmup(step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    """
    Linear warmup + cosine decay learning rate schedule.

    Args:
        step: Current training step (0-indexed).
        warmup_steps: Number of warmup steps.
        max_steps: Total training steps.
        max_lr: Maximum learning rate (after warmup).
        min_lr: Minimum learning rate (at end of training).

    Returns:
        Learning rate for the current step.
    """
    if step < warmup_steps:
        # Linear warmup
        return max_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    # Cosine decay after warmup
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + (max_lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * decay_ratio))


# --- Checkpoint Helpers ---

CHECKPOINT_RE = re.compile(r"checkpoint_step_(\d+)\.pt$")


def load_checkpoint_file(path: str, device: str) -> Dict[str, Any]:
    """Load a checkpoint while supporting both newer and older PyTorch versions."""
    try:
        return torch.load(path, map_location=torch.device(device), weights_only=False)
    except TypeError:
        return torch.load(path, map_location=torch.device(device))


def default_checkpoint_dir(out_path: str) -> str:
    """Return a checkpoint directory tied to the configured final model path."""
    model_path = Path(out_path)
    return str(model_path.with_suffix("")) + "_checkpoints"


def checkpoint_path(checkpoint_dir: str, step: int) -> str:
    """Build a stable checkpoint path for the last completed training step."""
    return os.path.join(checkpoint_dir, f"checkpoint_step_{step:08d}.pt")


def checkpoint_step(path: str) -> int:
    """Extract the step number from a checkpoint filename."""
    match = CHECKPOINT_RE.search(os.path.basename(path))
    if not match:
        return -1
    return int(match.group(1))


def list_checkpoints(checkpoint_dir: str) -> List[str]:
    """Return periodic checkpoints sorted by training step."""
    if not os.path.isdir(checkpoint_dir):
        return []
    paths = [
        os.path.join(checkpoint_dir, name)
        for name in os.listdir(checkpoint_dir)
        if CHECKPOINT_RE.search(name)
    ]
    return sorted(paths, key=checkpoint_step)


def resolve_resume_path(resume: Optional[str], checkpoint_dir: str) -> Optional[str]:
    """Resolve a resume argument."""
    if resume is None:
        return None
    if resume == "latest":
        checkpoints = list_checkpoints(checkpoint_dir)
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
        return checkpoints[-1]
    return resume


def current_lr(optimizer: torch.optim.Optimizer) -> float:
    """Read the learning rate from the first optimizer parameter group."""
    return float(optimizer.param_groups[0]["lr"])


def lr_for_step(train_config: Dict[str, Any], step: int) -> float:
    """Return the learning rate for a given step, using warmup if configured."""
    if train_config.get('t_use_warmup', False):
        return get_lr_with_warmup(
            step,
            warmup_steps=train_config['t_warmup_steps'],
            max_steps=train_config['t_train_steps'],
            max_lr=train_config['t_lr'],
            min_lr=train_config['t_lr_decayed'],
        )
    # Legacy step decay
    if step > train_config['t_lr_decay_step']:
        return float(train_config['t_lr_decayed'])
    return float(train_config['t_lr'])


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    """Set all optimizer parameter groups to the same learning rate."""
    for group in optimizer.param_groups:
        group["lr"] = lr


def configure_optimizer(model: Transformer, train_config: Dict[str, Any]) -> torch.optim.Optimizer:
    """
    Configure AdamW optimizer with optional differentiated weight decay.

    When differentiated_weight_decay is True, biases and normalization parameters
    receive 0 weight decay, while all other parameters receive the configured weight decay.
    """
    lr = train_config['t_lr']
    weight_decay = train_config.get('weight_decay', 0.1)
    differentiated = train_config.get('differentiated_weight_decay', True)

    fused_available = hasattr(torch.optim.AdamW, '__init__') and 'fused' in torch.optim.AdamW.__init__.__code__.co_varnames

    if differentiated:
        # Separate parameters that should/shouldn't get weight decay
        decay_params = []
        no_decay_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            # Don't apply weight decay to 1D parameters (biases, norms)
            if param.dim() < 2 or 'bias' in name or 'norm' in name.lower() or 'ln' in name.lower():
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ]
    else:
        optimizer_groups = model.parameters()

    optimizer_kwargs = {
        'lr': lr,
        'betas': (0.9, 0.95),
        'eps': 1e-8,
    }
    if fused_available:
        optimizer_kwargs['fused'] = True

    return torch.optim.AdamW(optimizer_groups, **optimizer_kwargs)


def save_training_checkpoint(
    path: str,
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    train_config: Dict[str, Any],
    losses: List[float],
    *,
    step: int,
    train_loss: Optional[float] = None,
    dev_loss: Optional[float] = None,
    is_final: bool = False,
) -> None:
    """Save model, optimizer, loss history, and LR schedule metadata."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    # Calculate perplexity if losses available
    perplexity = None
    if losses:
        avg_loss = np.mean(losses[-100:]) if len(losses) >= 100 else np.mean(losses)
        perplexity = math.exp(avg_loss)

    payload = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'losses': losses,
        'train_loss': train_loss,
        'dev_loss': dev_loss,
        'perplexity': perplexity,
        'step': step,
        'last_completed_step': step,
        'steps': step + 1,
        'is_final': is_final,
        'config': dict(train_config),
        'device': train_config['device'],
        'pytorch_version': torch.__version__,
        'cuda_version': torch.version.cuda,
        'lr_state': {
            'current_lr': current_lr(optimizer),
            'initial_lr': train_config['t_lr'],
            'decayed_lr': train_config['t_lr_decayed'],
            'decay_step': train_config['t_lr_decay_step'],
            'warmup_steps': train_config.get('t_warmup_steps', 0),
        },
    }
    target_dir = os.path.dirname(path) or "."
    with tempfile.NamedTemporaryFile(
        dir=target_dir,
        prefix=f".{os.path.basename(path)}.",
        suffix=".tmp",
        delete=False,
    ) as tmp_file:
        tmp_path = tmp_file.name
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def restore_training_checkpoint(
    path: str,
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    train_config: Dict[str, Any],
    device: str,
) -> Tuple[int, List[float]]:
    """Restore model/optimizer state and return (next_step, losses)."""
    checkpoint = load_checkpoint_file(path, device)
    model.load_state_dict(checkpoint['model_state_dict'])

    optimizer_state = checkpoint.get('optimizer_state_dict')
    if optimizer_state:
        optimizer.load_state_dict(optimizer_state)

    if 'last_completed_step' in checkpoint:
        last_completed_step = int(checkpoint['last_completed_step'])
        next_step = last_completed_step + 1
    else:
        next_step = int(checkpoint.get('steps', 0))
        last_completed_step = next_step - 1

    if not optimizer_state:
        set_optimizer_lr(optimizer, lr_for_step(train_config, next_step))

    losses = [float(loss) for loss in checkpoint.get('losses', [])]
    ppl = checkpoint.get('perplexity')
    ppl_str = f" | Perplexity: {ppl:.2f}" if ppl else ""
    print(
        f"Resumed from {path}. "
        f"Last completed step: {last_completed_step}. Next step: {next_step}.{ppl_str}"
    )
    return next_step, losses


def prune_old_checkpoints(checkpoint_dir: str, keep_last: int) -> None:
    """Keep only the most recent N periodic checkpoints when requested."""
    if keep_last <= 0:
        return
    checkpoints = list_checkpoints(checkpoint_dir)
    for old_path in checkpoints[:-keep_last]:
        os.remove(old_path)


def unique_output_path(out_path: str) -> str:
    """Avoid overwriting an existing final model checkpoint."""
    modified_model_out_path = out_path
    save_tries = 0
    while os.path.exists(modified_model_out_path):
        save_tries += 1
        model_out_name = os.path.splitext(out_path)[0]
        modified_model_out_path = model_out_name + f"_{save_tries}" + ".pt"
    return modified_model_out_path


def as_float(value: Any) -> Optional[float]:
    """Convert scalar tensors/numbers to plain floats for checkpoint metadata."""
    if value is None:
        return None
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


# --- Training / Evaluation ---

@torch.no_grad()
def estimate_loss(model: Transformer, train_config: Dict[str, Any], steps: int) -> Dict[str, float]:
    """
    Evaluate the model and calculate average loss and perplexity.
    """
    out = {}
    model.eval()
    from data_loader.data_loader import get_batch_iterator

    for split in ['train', 'dev']:
        data_path = train_config['train_path'] if split == 'train' else train_config['dev_path']
        batch_iterator_eval = get_batch_iterator(
            data_path,
            train_config['t_batch_size'],
            train_config['t_context_length'],
            device=train_config['device'],
        )
        losses_eval = []
        for _ in range(steps):
            try:
                xb, yb = next(batch_iterator_eval)
                _, loss = model(xb, yb)
                losses_eval.append(float(loss.item()))
            except StopIteration:
                print(f"Warning: Iterator for {split} ended early.")
                break

        avg_loss = float(np.mean(losses_eval)) if losses_eval else float("nan")
        out[f'{split}_loss'] = avg_loss
        out[f'{split}_perplexity'] = math.exp(avg_loss) if avg_loss == avg_loss else float("nan")

    model.train()
    return out


# --- Signal Handling ---

_signal_received = False
_emergency_checkpoint_path = None
_emergency_save_fn = None


def setup_signal_handler(checkpoint_dir: str, save_fn) -> None:
    """Setup signal handlers for graceful shutdown on SIGINT/SIGTERM."""
    global _emergency_checkpoint_path, _emergency_save_fn
    _emergency_checkpoint_path = os.path.join(checkpoint_dir, "emergency_checkpoint.pt")
    _emergency_save_fn = save_fn

    def signal_handler(signum, frame):
        global _signal_received
        if _signal_received:
            print("\nForced exit.")
            sys.exit(1)
        _signal_received = True
        print("\n[Signal] Shutdown requested. Saving emergency checkpoint...")
        try:
            if _emergency_save_fn is not None:
                _emergency_save_fn(_emergency_checkpoint_path)
                print(f"[Signal] Emergency checkpoint saved to {_emergency_checkpoint_path}")
        except Exception as e:
            print(f"[Signal] Failed to save emergency checkpoint: {e}")
        finally:
            sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Transformer model from scratch.")
    parser.add_argument(
        "--resume",
        nargs="?",
        const="latest",
        default=None,
        help=(
            "Resume from a checkpoint path. Pass --resume with no value, or "
            "--resume latest, to use the newest checkpoint in the checkpoint directory."
        ),
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=None,
        help="Save a periodic checkpoint every N completed steps. 0 disables periodic checkpoints.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Directory for periodic checkpoints. Defaults to a directory next to t_out_path.",
    )
    parser.add_argument(
        "--keep-last",
        type=int,
        default=None,
        help="Keep only the most recent N periodic checkpoints. 0 keeps all.",
    )
    # --- Model architecture overrides ---
    parser.add_argument(
        "--dropout",
        type=float,
        default=None,
        help="Dropout probability (overrides config).",
    )
    parser.add_argument(
        "--norm-type",
        type=str,
        choices=["layernorm", "rmsnorm"],
        default=None,
        help="Normalization type (overrides config).",
    )
    parser.add_argument(
        "--activation",
        type=str,
        choices=["gelu", "swiglu"],
        default=None,
        help="MLP activation function (overrides config).",
    )
    parser.add_argument(
        "--tie-weights",
        dest="tie_weights",
        action="store_true",
        default=None,
        help="Enable weight tying (overrides config).",
    )
    parser.add_argument(
        "--no-tie-weights",
        dest="tie_weights",
        action="store_false",
        default=None,
        help="Disable weight tying (overrides config).",
    )
    # --- Memory-optimisation flags ---
    parser.add_argument(
        "--amp",
        dest="amp",
        action="store_true",
        default=None,
        help="Enable bf16/fp16 mixed-precision autocast (CUDA only).",
    )
    parser.add_argument(
        "--amp-dtype",
        type=str,
        choices=["bf16", "fp16"],
        default=None,
        help="Autocast dtype when --amp is set: bf16 (default) or fp16.",
    )
    parser.add_argument(
        "--grad-checkpointing",
        dest="grad_checkpointing",
        action="store_true",
        default=None,
        help="Recompute transformer-block activations in backward to save VRAM.",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=None,
        help="Accumulate gradients over N micro-batches per optimizer step.",
    )
    parser.add_argument(
        "--report-memory",
        dest="report_memory",
        action="store_true",
        default=None,
        help="Print a rough VRAM budget before training (CUDA only).",
    )
    # --- Training schedule overrides ---
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=None,
        help="Number of warmup steps (overrides config).",
    )
    parser.add_argument(
        "--no-warmup",
        dest="use_warmup",
        action="store_false",
        default=None,
        help="Disable warmup schedule.",
    )
    parser.add_argument(
        "--compile",
        dest="compile",
        action="store_true",
        default=None,
        help="Enable torch.compile() for faster training (PyTorch 2.0+).",
    )
    parser.add_argument(
        "--no-compile",
        dest="compile",
        action="store_false",
        default=None,
        help="Disable torch.compile().",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=None,
        help="Weight decay for AdamW (overrides config).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_config = dict(config)

    # Override config with CLI arguments
    if args.dropout is not None:
        train_config['dropout'] = args.dropout
    if args.norm_type is not None:
        train_config['norm_type'] = args.norm_type
    if args.activation is not None:
        train_config['activation'] = args.activation
    if args.tie_weights is not None:
        train_config['tie_weights'] = args.tie_weights
    if args.warmup_steps is not None:
        train_config['t_warmup_steps'] = args.warmup_steps
    if args.use_warmup is not None:
        train_config['t_use_warmup'] = args.use_warmup
    if args.weight_decay is not None:
        train_config['weight_decay'] = args.weight_decay

    checkpoint_every = (
        args.checkpoint_every
        if args.checkpoint_every is not None
        else train_config.get('t_checkpoint_steps', 0)
    )
    keep_last = (
        args.keep_last
        if args.keep_last is not None
        else train_config.get('t_keep_last_checkpoints', 0)
    )
    checkpoint_dir = (
        args.checkpoint_dir
        or train_config.get('t_checkpoint_dir')
        or default_checkpoint_dir(train_config['t_out_path'])
    )

    # Resolve memory-optimisation options
    use_amp = args.amp if args.amp is not None else bool(train_config.get('use_amp', False))
    amp_dtype_name = args.amp_dtype or train_config.get('amp_dtype', 'bf16')
    use_grad_ckpt = (
        args.grad_checkpointing if args.grad_checkpointing is not None
        else bool(train_config.get('use_gradient_checkpointing', False))
    )
    grad_accum = max(1, args.grad_accum if args.grad_accum is not None
                     else int(train_config.get('grad_accum_steps', 1)))
    report_memory = (
        args.report_memory if args.report_memory is not None
        else bool(train_config.get('report_memory_budget', False))
    )
    use_compile = (
        args.compile if args.compile is not None
        else bool(train_config.get('use_torch_compile', False))
    )

    device_is_cuda = train_config['device'].startswith('cuda') and torch.cuda.is_available()
    if use_amp and not device_is_cuda:
        print("[mem-opt] --amp requested but no CUDA device available; disabling AMP.")
        use_amp = False
    amp_dtype = torch.bfloat16 if amp_dtype_name == 'bf16' else torch.float16

    def autocast_ctx():
        if use_amp:
            return torch.autocast(device_type='cuda', dtype=amp_dtype)
        return contextlib.nullcontext()

    use_scaler = use_amp and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=use_scaler)

    # --- Initialize the Model ---
    print(get_device_report(train_config['device']))
    if train_config['device'].startswith('cuda') and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    model = Transformer(
        n_head=train_config['n_head'],
        n_embed=train_config['n_embed'],
        context_length=train_config['context_length'],
        vocab_size=train_config['vocab_size'],
        N_BLOCKS=train_config['n_blocks'],
        dropout=train_config.get('dropout', 0.0),
        norm_type=train_config.get('norm_type', 'layernorm'),
        activation=train_config.get('activation', 'gelu'),
        tie_weights=train_config.get('tie_weights', True),
    ).to(train_config['device'])

    # torch.compile (PyTorch 2.0+)
    if use_compile and hasattr(torch, 'compile'):
        print("[optim] torch.compile() enabled.")
        model = torch.compile(model)
    elif use_compile:
        print("[optim] torch.compile() requested but not available (requires PyTorch 2.0+).")

    total_params = model.get_num_params(non_embedding=False)
    non_emb_params = model.get_num_params(non_embedding=True)
    print(f"Total parameters: {total_params:,} (~{total_params/1e6:.0f}M)")
    print(f"Non-embedding parameters: {non_emb_params:,} (~{non_emb_params/1e6:.0f}M)")
    if train_config.get('tie_weights', True):
        print(f"[optim] Weight tying enabled (saved ~{train_config['vocab_size'] * train_config['n_embed'] / 1e6:.0f}M parameters)")

    model.gradient_checkpointing = use_grad_ckpt
    if report_memory:
        print(estimate_memory_budget(total_params, train_config['device'], use_amp))
    if use_amp or use_grad_ckpt or grad_accum > 1 or use_compile:
        opts = []
        if use_amp:
            opts.append(f"amp({amp_dtype_name})")
        if use_grad_ckpt:
            opts.append("grad_ckpt")
        if grad_accum > 1:
            opts.append(f"grad_accum={grad_accum}")
        if use_compile:
            opts.append("torch.compile")
        print(f"[mem-opt] {' | '.join(opts)}")

    # --- Optimizer Setup ---
    optimizer = configure_optimizer(model, train_config)

    losses: List[float] = []
    start_step = 0
    last_completed_step = -1
    resume_path = resolve_resume_path(args.resume, checkpoint_dir)
    if resume_path is not None:
        start_step, losses = restore_training_checkpoint(
            resume_path,
            model,
            optimizer,
            train_config,
            train_config['device'],
        )
        last_completed_step = start_step - 1

    avg_window = 64

    # --- Signal Handler ---
    if train_config.get('enable_signal_handler', True):
        def emergency_save(path):
            save_training_checkpoint(
                path, model, optimizer, train_config, losses,
                step=last_completed_step,
                train_loss=None,
                dev_loss=None,
            )
        setup_signal_handler(checkpoint_dir, emergency_save)
        print("[safety] Signal handler enabled (SIGINT/SIGTERM -> emergency checkpoint)")

    # --- Training Loop ---
    from data_loader.data_loader import get_batch_iterator

    batch_iterator = get_batch_iterator(
        train_config['train_path'],
        train_config['t_batch_size'],
        train_config['t_context_length'],
        device=train_config['device'],
    )

    tokens_per_step = train_config['t_batch_size'] * train_config['t_context_length'] * grad_accum
    last_eval_time = time.perf_counter()
    latest_train_loss = None
    latest_dev_loss = None
    latest_perplexity = None

    use_warmup = train_config.get('t_use_warmup', False)
    if use_warmup:
        print(f"[schedule] Warmup + Cosine Decay: {train_config['t_warmup_steps']} warmup steps, "
              f"LR: {train_config['t_lr']:.2e} -> {train_config['t_lr_decayed']:.2e}")

    pbar = tqdm(range(start_step, train_config['t_train_steps']))
    for step in pbar:
        # Handle signals
        global _signal_received
        if _signal_received:
            break

        try:
            step_start_time = time.perf_counter()

            # Update learning rate
            lr = lr_for_step(train_config, step)
            set_optimizer_lr(optimizer, lr)

            optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0
            for _ in range(grad_accum):
                xb, yb = next(batch_iterator)
                with autocast_ctx():
                    _, loss = model(xb, yb)
                    loss = loss / grad_accum
                scaler.scale(loss).backward()
                step_loss += float(loss.item())

            losses.append(step_loss)
            avg_loss = np.mean(losses[-avg_window:]) if len(losses) >= avg_window else np.mean(losses)
            pbar.set_description(
                f"loss: {avg_loss:.4f} | lr: {lr:.2e}"
                + (f" | ppl: {math.exp(avg_loss):.1f}" if avg_loss == avg_loss else "")
            )

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()
            last_completed_step = step

            step_time = time.perf_counter() - step_start_time
            tokens_per_second = tokens_per_step / step_time if step_time > 0 else float('inf')

            if step % train_config['t_eval_steps'] == 0:
                evaluation_losses = estimate_loss(model, train_config, train_config['t_eval_iters'])
                latest_train_loss = evaluation_losses['train_loss']
                latest_dev_loss = evaluation_losses['dev_loss']
                latest_perplexity = evaluation_losses['train_perplexity']

                now = time.perf_counter()
                elapsed_since_eval = now - last_eval_time
                last_eval_time = now

                print(
                    f"\nStep: {step} | "
                    f"Train loss: {latest_train_loss:.4f} (PPL: {latest_perplexity:.2f}) | "
                    f"Dev loss: {latest_dev_loss:.4f} (PPL: {evaluation_losses['dev_perplexity']:.2f}) | "
                    f"LR: {lr:.2e} | "
                    f"Step time: {step_time:.3f}s | "
                    f"Throughput: {tokens_per_second:.2f} tokens/s | "
                    f"Elapsed: {elapsed_since_eval:.2f}s"
                )
                print(get_peak_memory_report(train_config['device']))

            # Legacy LR decay (only if not using warmup)
            if not use_warmup and step == train_config['t_lr_decay_step']:
                print('Decaying learning rate')
                set_optimizer_lr(optimizer, train_config['t_lr_decayed'])

            if checkpoint_every and checkpoint_every > 0 and (step + 1) % checkpoint_every == 0:
                path = checkpoint_path(checkpoint_dir, step)
                save_training_checkpoint(
                    path,
                    model,
                    optimizer,
                    train_config,
                    losses,
                    step=step,
                    train_loss=as_float(latest_train_loss),
                    dev_loss=as_float(latest_dev_loss),
                )
                prune_old_checkpoints(checkpoint_dir, int(keep_last or 0))
                print(f"Saved checkpoint to {path}")
        except StopIteration:
            print("Training data iterator finished early.")
            break

    # --- Save Model and Final Evaluation ---
    evaluation_losses = estimate_loss(model, train_config, 200)
    train_loss = evaluation_losses['train_loss']
    dev_loss = evaluation_losses['dev_loss']
    train_ppl = evaluation_losses['train_perplexity']
    dev_ppl = evaluation_losses['dev_perplexity']

    final_step = max(last_completed_step, start_step - 1)
    modified_model_out_path = unique_output_path(train_config['t_out_path'])

    save_training_checkpoint(
        modified_model_out_path,
        model,
        optimizer,
        train_config,
        losses,
        step=final_step,
        train_loss=train_loss,
        dev_loss=dev_loss,
    )
    print(f"\n{'='*60}")
    print(f"Saved model to {modified_model_out_path}")
    print(f"Final Results:")
    print(f"  Train loss: {train_loss:.4f} | Perplexity: {train_ppl:.2f}")
    print(f"  Dev loss:   {dev_loss:.4f} | Perplexity: {dev_ppl:.2f}")
    print(f"  Total steps: {final_step + 1}")
    print(f"{'='*60}")
    print(get_peak_memory_report(train_config['device']))


if __name__ == "__main__":
    main()
