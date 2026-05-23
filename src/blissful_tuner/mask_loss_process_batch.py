"""Mask-weighted loss + prior-preservation ``process_batch`` for the refactored trainer seam.

Blissful-tuner's mask-weighted loss with prior preservation used to live inline in the
monolithic ``hv_train_network.py`` training loop (pre-merge ``:2640-2881``). Upstream's
v0.3.0 refactor (PR #943) replaced that monolith with a template-method ``NetworkTrainer``
base whose ``train()`` owns the loop and calls back into an overridable
``process_batch(self, ...) -> (scalar_loss, loss_metrics)`` seam.

This module houses the masked-loss orchestration as **free functions taking ``self`` first**
(not a mixin) so each ``NetworkTrainer`` subclass can delegate explicitly without MRO
surprises. Each mask-capable subclass overrides ``process_batch``/``on_post_optimizer_step``
with a ~3-line dispatch that calls into here when ``args.use_mask_loss`` is set.

What the base loop already owns (and this function must NOT duplicate):
``on_step_start``, ``scale_shift_latents``, ``noise = randn_like(latents)``,
``accelerator.backward``, gradient reduce/clip, ``optimizer.step``/``zero_grad``,
``scale_weight_norms``, ``on_epoch_start``. The base samples ``noise`` and scale-shifts
``latents`` *before* calling ``process_batch`` and runs ``backward`` *after* it returns.

What this function owns (the residue specific to masked + prior-preserved training):
timestep sampling, SD3 loss weighting, prior-weight scheduling, EMA-teacher lazy init,
per-sample prior gating, the **prior-teacher no-grad forward** (base-mode LoRA-disable or
EMA-weight swap) with its mandatory **block-swap restore** between teacher and student
forwards, the student forward, and the masked/prior reduction via
``apply_masked_loss_with_prior``.

Known limitations:
- **Self-Flow + masked loss are mutually exclusive in v1.** ``Flux2SelfFlowNetworkTrainer``
  overrides ``process_batch`` itself; when ``--self_flow`` is on it never delegates to the
  masked path. Documented in the masked-loss plan's risk register.
- **One-shot mask config banner is not emitted on the LoRA path.** The old monolith called
  ``log_mask_loss_banner`` pre-loop with a cache scan over ``train_dataset_group.datasets``;
  the new seams (``on_train_start``/``process_batch``) don't expose the dataset group, so the
  soft startup banner is deferred (Step 5b). The *critical* fail-fast validation
  (``require_mask_weights_if_enabled``) is preserved here, per-batch. The full-FT trainers
  (Qwen/Z-Image) still emit the banner from their own ``train()`` loops.
"""

from contextlib import contextmanager

import torch
from accelerate import Accelerator

from blissful_tuner.blissful_logger import BlissfulLogger
from musubi_tuner.modules.loss_utils import compute_unreduced_target_loss
from musubi_tuner.modules.lora_ema_teacher import LoRAEmaTeacher
from musubi_tuner.modules.mask_loss import apply_masked_loss_with_prior, require_mask_weights_if_enabled
from musubi_tuner.modules.prior_scheduling import compute_prior_weight_per_sample
from musubi_tuner.training.timesteps import compute_loss_weighting_for_sd3


logger = BlissfulLogger(__name__, "magenta")


@contextmanager
def prior_model_context(network):
    """Temporarily disable LoRA for computing base-mode prior (teacher) predictions.

    Design decision: keep the model in train() mode during the teacher forward.
    - Matches OneTrainer behavior.
    - DiT models (WAN, FLUX, etc.) typically have no dropout, so train/eval is a no-op.
    - For deterministic teacher targets, use ``--prior_teacher_eval`` (toggles
      ``transformer.eval()`` around the teacher pass — handled in ``masked_process_batch``).

    Prefers ``set_enabled()`` (fast, explicit); falls back to ``set_multiplier(0)`` for
    network types that lack it (e.g. some LyCORIS builds). Ported verbatim from the
    pre-merge monolith (``hv_train_network.py:580``), now a free function with one caller.
    """
    if network is None:
        yield
        return

    if hasattr(network, "set_enabled"):
        try:
            network.set_enabled(False)
            yield
        finally:
            network.set_enabled(True)
        return

    if hasattr(network, "set_multiplier"):
        prev_multiplier = getattr(network, "multiplier", None)
        try:
            network.set_multiplier(0)
            yield
        finally:
            # Restore previous multiplier when available; otherwise use 1.0 as a reasonable default.
            network.set_multiplier(1.0 if prev_multiplier is None else prev_multiplier)
        return

    raise ValueError(
        "Prior preservation requires the network module to support temporarily disabling adapters via "
        "`set_enabled(False)` or `set_multiplier(0)`, but neither method exists on this network object."
    )


def masked_process_batch(
    self,
    args,
    accelerator: Accelerator,
    transformer,
    network,
    batch: dict,
    latents: torch.Tensor,
    noise: torch.Tensor,
    noise_scheduler,
    dit_dtype: torch.dtype,
    network_dtype: torch.dtype,
    vae,
    global_step: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute mask-weighted (optionally prior-preserved) scalar loss for one batch.

    Drop-in replacement for the base ``process_batch`` when ``args.use_mask_loss`` is set.
    ``latents`` is already scale-shifted and ``noise`` already sampled by the base loop.
    Returns ``(scalar_loss, loss_metrics)``; ``loss_metrics`` carries every masked-loss /
    prior telemetry key the old monolith logged (``masked_loss/*``, ``timestep/*``,
    ``prior/*``), which the base merges into the per-step log dict at ``train()``.

    EMA-teacher state lives on ``self.prior_lora_ema`` (lazily created here at warmup, updated
    by the subclass ``on_post_optimizer_step`` override). See [[project_block_swap_no_grad_invariant]]
    for why the teacher no-grad forward must be followed by ``restore_block_swap_after_no_grad_forward``.
    """
    # --- Per-run EMA / prior-schedule config (cheap to derive each call from args) ---
    prior_decay_schedule = str(getattr(args, "prior_decay_schedule", "constant"))
    prior_decay_timestep_start = float(getattr(args, "prior_decay_timestep_start", 300.0))
    prior_decay_warmup_ratio = float(getattr(args, "prior_decay_warmup_ratio", 0.0))
    prior_schedule_enabled = (prior_decay_schedule != "constant") or (prior_decay_warmup_ratio > 0.0)
    prior_decay_warmup_steps = int(args.max_train_steps * prior_decay_warmup_ratio) if prior_decay_warmup_ratio > 0 else 0

    prior_teacher_mode = str(getattr(args, "prior_teacher_mode", "base"))
    prior_teacher_ema_decay = float(getattr(args, "prior_teacher_ema_decay", 0.999))
    # Guardrail: if EMA teacher is enabled with no warmup, step-0 LoRA weights are effectively random.
    # Force at least N optimizer steps before initializing EMA so the teacher starts from a non-random adapter.
    prior_teacher_ema_min_init_steps = 100
    prior_teacher_ema_init_step = max(prior_teacher_ema_min_init_steps, prior_decay_warmup_steps)

    # One-shot startup warning (per trainer instance), fired only for EMA mode that can't init in time.
    if not getattr(self, "_mask_seam_warmup_warned", False):
        self._mask_seam_warmup_warned = True
        if prior_teacher_mode == "ema" and prior_teacher_ema_init_step >= int(args.max_train_steps):
            logger.warning(
                "EMA teacher is enabled, but it will not initialize within this run: "
                f"ema_init_step={prior_teacher_ema_init_step} >= max_train_steps={int(args.max_train_steps)}. "
                "Teacher will remain in base mode for the entire run."
            )

    # Fail-fast: ERROR if mask loss is enabled but this batch carries no masks.
    require_mask_weights_if_enabled(
        batch,
        args,
        cache_hint=(
            "Recache with your architecture's cache script (WAN: wan_cache_latents.py). "
            "Note: HunyuanVideo cache_latents.py does not currently store masks."
        ),
    )

    # calculate model input and timesteps (base already scale-shifted latents + sampled noise)
    noisy_model_input, timesteps = self.get_noisy_model_input_and_timesteps(
        args, noise, latents, batch["timesteps"], noise_scheduler, accelerator.device, dit_dtype
    )

    # SD3 loss weighting is computed AFTER the unreduced loss (below) so its n_dim matches the
    # actual loss rank (W14) instead of an architecture constant. weighting depends only on
    # (scheme, scheduler, timesteps, device, dtype), so deferring it is numerically identical.

    # --- Prior preservation: decide whether/where the teacher forward is needed ---
    prior_pred = None
    prior_teacher_uses_ema = False
    prior_preservation_weight = getattr(args, "prior_preservation_weight", 0.0)
    mask_weights = batch.get("mask_weights")

    need_prior_base = prior_preservation_weight > 0 and getattr(args, "use_mask_loss", False) and mask_weights is not None

    # Optional: timestep-adaptive prior weight scheduling (per-sample). For timesteps >= pivot,
    # weight stays at max; below pivot it decays toward 0 at t=0 (linear or cosine).
    prior_weight_per_sample = None
    if need_prior_base and prior_schedule_enabled:
        prior_weight_per_sample = compute_prior_weight_per_sample(
            timesteps,
            base_weight=float(prior_preservation_weight),
            schedule=prior_decay_schedule,
            pivot_timestep=prior_decay_timestep_start,
            global_step=global_step,
            warmup_steps=prior_decay_warmup_steps,
        )

    # Optional: initialize EMA teacher once warmup completes (kept lazy/inline to preserve
    # "init after warmup" semantics — eager init at on_train_start would capture step-0 noise).
    if (
        need_prior_base
        and prior_teacher_mode == "ema"
        and getattr(self, "prior_lora_ema", None) is None
        and global_step >= prior_teacher_ema_init_step
    ):
        self.prior_lora_ema = LoRAEmaTeacher(decay=prior_teacher_ema_decay)
        self.prior_lora_ema.init_from(accelerator.unwrap_model(network))

    # Optional speed optimization: only run the teacher forward at structural (high-noise) timesteps.
    # NOTE: the *application* of the prior term is gated per-sample inside apply_masked_loss_with_prior()
    # to avoid denominator dilution / threshold-mode overlap when batch_size > 1.
    prior_timestep_threshold = getattr(args, "prior_preservation_timestep_threshold", None)
    prior_sample_mask = None
    need_prior = need_prior_base
    if need_prior and (prior_timestep_threshold is not None or prior_weight_per_sample is not None):
        # Per-sample structural gating. Teacher forward runs if ANY sample needs it.
        if prior_timestep_threshold is not None:
            do_prior = timesteps > float(prior_timestep_threshold)  # (B,)
        else:
            do_prior = torch.ones_like(timesteps, dtype=torch.bool)  # (B,)

        # If scheduled weights are provided, treat zero-weight samples as "prior disabled".
        if prior_weight_per_sample is not None:
            do_prior = do_prior & (prior_weight_per_sample > 0)

        # Optimization: skip teacher forward if the prior region would be empty for all structural samples.
        if mask_weights is not None:
            if mask_weights.ndim == 5:
                mask_min_per_sample = mask_weights.amin(dim=(1, 2, 3, 4))
            elif mask_weights.ndim == 4:
                mask_min_per_sample = mask_weights.amin(dim=(1, 2, 3))
            else:
                mask_min_per_sample = mask_weights.view(mask_weights.shape[0], -1).amin(dim=1)
        else:
            mask_min_per_sample = None

        prior_mask_threshold = getattr(args, "prior_mask_threshold", None)
        if mask_min_per_sample is not None:
            if prior_mask_threshold is not None:
                # Threshold mode: prior applies only where raw mask < threshold.
                has_prior_region = mask_min_per_sample < float(prior_mask_threshold)
            else:
                # Continuous mode: prior_mask = 1 - mask_processed, so only all-ones masks produce zero prior.
                has_prior_region = mask_min_per_sample < (1.0 - 1e-6)
            # mask_weights may be on CPU while timesteps are on accelerator.device — align before boolean ops.
            has_prior_region = has_prior_region.to(device=do_prior.device)
            do_prior = do_prior & has_prior_region

        prior_sample_mask = do_prior
        need_prior = bool(do_prior.any().item())
    elif need_prior:
        # No timestep gating: preserve the original optimization that skips the teacher when the
        # prior region would be empty for the entire batch.
        mask_min = float(mask_weights.min())
        prior_mask_threshold = getattr(args, "prior_mask_threshold", None)
        if prior_mask_threshold is not None:
            if mask_min >= float(prior_mask_threshold):
                need_prior = False
        else:
            if mask_min >= (1.0 - 1e-6):
                need_prior = False  # No prior loss contribution possible

    # --- Prior-teacher no-grad forward (must precede the student forward) ---
    if need_prior:
        prior_teacher_eval = bool(getattr(args, "prior_teacher_eval", False))
        transformer_was_training = transformer.training if prior_teacher_eval else None
        if prior_teacher_eval:
            transformer.eval()
        try:
            with torch.no_grad():
                # Must unwrap network for set_enabled() to work with DDP/accelerator wrapping.
                unwrapped_network = accelerator.unwrap_model(network)
                prior_teacher_uses_ema = prior_teacher_mode == "ema" and getattr(self, "prior_lora_ema", None) is not None
                prior_context = (
                    self.prior_lora_ema.apply_to(unwrapped_network)
                    if prior_teacher_uses_ema
                    else prior_model_context(unwrapped_network)
                )
                with prior_context:
                    # Reuse the exact same timesteps/inputs so the teacher runs on the same WAN 2.2
                    # expert as the student (call_dit selects the expert from `timesteps`):
                    #   - base teacher: LoRA disabled
                    #   - EMA teacher:  LoRA enabled but weights swapped to EMA values
                    prior_output = self.call_dit(
                        args, accelerator, transformer, latents, batch, noise, noisy_model_input, timesteps, network_dtype
                    )
                    prior_pred_raw = prior_output.pred
                # Block swap relies on backward hooks to return CPU blocks to GPU; the no-grad teacher
                # forward skips them, so restore explicitly before the student forward. See
                # [[project_block_swap_no_grad_invariant]].
                self.restore_block_swap_after_no_grad_forward(accelerator, transformer)
            prior_pred = prior_pred_raw.detach()
        finally:
            if prior_teacher_eval and transformer_was_training:
                transformer.train()

    # --- Student forward (LoRA enabled, gradients flow) ---
    output = self.call_dit(
        args, accelerator, transformer, latents, batch, noise, noisy_model_input, timesteps, network_dtype
    )
    model_pred, target = output.pred, output.target

    loss_type = getattr(args, "loss_type", "mse")
    loss_delta = getattr(args, "loss_delta", 1.0)
    model_pred_cast = model_pred.to(network_dtype)
    target_cast = target.to(network_dtype)
    loss_unreduced = compute_unreduced_target_loss(
        model_pred_cast,
        target_cast,
        loss_type=loss_type,
        loss_delta=loss_delta,
    )

    stats_enabled = len(accelerator.trackers) > 0
    log_huber_stats = str(loss_type).lower() == "huber" and stats_enabled and bool(getattr(args, "use_mask_loss", False))
    huber_threshold = 0.5 * float(loss_delta) * float(loss_delta)
    target_huber_is_linear = (loss_unreduced > huber_threshold) if log_huber_stats else None

    # SD3 loss weighting at the actual loss rank (W14): n_dim=loss_unreduced.ndim, never an arch constant.
    weighting = compute_loss_weighting_for_sd3(
        args.weighting_scheme, noise_scheduler, timesteps, accelerator.device, dit_dtype, n_dim=loss_unreduced.ndim
    )

    loss = loss_unreduced
    if weighting is not None:
        loss = loss * weighting

    # Prior loss term (same loss type as target to avoid a gradient discontinuity at mask boundaries).
    prior_loss_unreduced = None
    prior_huber_is_linear = None
    if prior_pred is not None:
        prior_loss_unreduced = compute_unreduced_target_loss(
            model_pred_cast,
            prior_pred.to(network_dtype),
            loss_type=loss_type,
            loss_delta=loss_delta,
        )
        prior_huber_is_linear = (prior_loss_unreduced > huber_threshold) if log_huber_stats else None
        if weighting is not None:
            prior_loss_unreduced = prior_loss_unreduced * weighting

    layout = "layered" if getattr(args, "is_layered", False) else "video"
    drop_base_frame = bool(getattr(args, "remove_first_image_from_target", False)) if layout == "layered" else False
    mask_loss_stats = {} if stats_enabled else None
    loss = apply_masked_loss_with_prior(
        loss,
        mask_weights,
        prior_loss_unreduced=prior_loss_unreduced,
        prior_sample_mask=prior_sample_mask,
        prior_weight_per_sample=prior_weight_per_sample,
        target_huber_is_linear=target_huber_is_linear,
        prior_huber_is_linear=prior_huber_is_linear,
        stats=mask_loss_stats,
        args=args,
        layout=layout,
        drop_base_frame=drop_base_frame,
    )

    # --- Telemetry: rebuild every use_mask_loss-gated log key the monolith emitted ---
    # (dynamo/cache_entries is intentionally NOT here — it was general telemetry gated on
    #  args.compile, not use_mask_loss; folding it in would narrow its emission domain.)
    loss_metrics: dict[str, float] = {}
    if stats_enabled:
        if mask_loss_stats:
            for k, v in mask_loss_stats.items():
                if isinstance(v, torch.Tensor):
                    loss_metrics[f"masked_loss/{k}"] = float(v.detach().float().item())
            # Quick sanity ratios: are we dominated by target or prior?
            if "target" in mask_loss_stats and "prior" in mask_loss_stats:
                t = float(mask_loss_stats["target"].detach().float().item())
                p = float(mask_loss_stats["prior"].detach().float().item())
                denom = t + p + 1e-8
                loss_metrics["masked_loss/target_fraction"] = t / denom
                loss_metrics["masked_loss/prior_fraction"] = p / denom

        # Timestep stats (helps reason about prior schedule / gating behavior).
        if isinstance(timesteps, torch.Tensor) and timesteps.numel() > 0:
            t_stat = timesteps.detach().to(dtype=torch.float32)
            loss_metrics["timestep/mean"] = float(t_stat.mean().item())
            loss_metrics["timestep/min"] = float(t_stat.min().item())
            loss_metrics["timestep/max"] = float(t_stat.max().item())

        # Prior preservation control-plane telemetry (teacher + scheduling).
        prior_weight = float(getattr(args, "prior_preservation_weight", 0.0))
        if prior_weight > 0:
            loss_metrics["prior/teacher_ran"] = float(prior_pred is not None)
            loss_metrics["prior/teacher_mode_ema_used"] = float(prior_teacher_uses_ema)

            if prior_sample_mask is not None and isinstance(prior_sample_mask, torch.Tensor):
                loss_metrics["prior/gate_frac"] = float(prior_sample_mask.detach().to(dtype=torch.float32).mean().item())
            else:
                loss_metrics["prior/gate_frac"] = 1.0 if prior_pred is not None else 0.0

            if prior_weight_per_sample is not None and isinstance(prior_weight_per_sample, torch.Tensor):
                w = prior_weight_per_sample.detach().to(dtype=torch.float32)
                loss_metrics["prior/w_prior_mean"] = float(w.mean().item())
                loss_metrics["prior/w_prior_min"] = float(w.min().item())
                loss_metrics["prior/w_prior_max"] = float(w.max().item())
            else:
                loss_metrics["prior/w_prior_mean"] = float(prior_weight)

    return loss, loss_metrics


def masked_on_post_optimizer_step(self, args, accelerator: Accelerator, network, sync_gradients: bool) -> None:
    """EMA-teacher update, shared by the mask-capable subclasses' ``on_post_optimizer_step``.

    Fires only on real optimizer steps (``sync_gradients=True``) so gradient accumulation
    updates the EMA once per optimizer step, not once per micro-batch. No-op until the EMA
    teacher is lazily created inside ``masked_process_batch`` (after warmup). ``LoRAEmaTeacher.update()``
    is the existing graph-safe in-place parameter swap.
    """
    if sync_gradients and getattr(self, "prior_lora_ema", None) is not None:
        self.prior_lora_ema.update(accelerator.unwrap_model(network))
