"""Guard tests for the masked-loss ``process_batch`` / ``on_post_optimizer_step`` seam.

After the upstream v0.3.0 refactor, mask-weighted loss lives in
``blissful_tuner.mask_loss_process_batch`` and each mask-capable ``NetworkTrainer``
subclass dispatches to it via a thin ``process_batch`` override. These tests pin:

1. **Dispatch correctness** (parameterized over all 7 mask-capable subclasses):
   - ``use_mask_loss=False`` → delegates to ``super().process_batch`` (the vanilla base path).
   - ``use_mask_loss=True``  → delegates to ``masked_process_batch``.
   This is the drift guard for the 7 near-identical overrides — if a subclass forgets the
   override (inheriting the base no-mask path) or wires it backwards, this fails.

2. **EMA-teacher update gating** (``on_post_optimizer_step``):
   - Fires ``prior_lora_ema.update`` only when ``sync_gradients=True`` (grad-accum correctness).
   - No-op when ``prior_lora_ema`` is unset.

FramePack / FLUX.1 Kontext are intentionally excluded (no mask support); full-FT trainers
(Qwen/Z-Image) inherit their parent's override and keep their own in-place reduction loop.
"""

from types import SimpleNamespace
from unittest import mock

import pytest


# (module path, class name) for the 7 mask-capable NetworkTrainer subclasses.
MASK_CAPABLE_TRAINERS = [
    ("musubi_tuner.wan_train_network", "WanNetworkTrainer"),
    ("musubi_tuner.flux_2_train_network", "Flux2NetworkTrainer"),
    ("musubi_tuner.zimage_train_network", "ZImageNetworkTrainer"),
    ("musubi_tuner.qwen_image_train_network", "QwenImageNetworkTrainer"),
    ("musubi_tuner.kandinsky5_train_network", "Kandinsky5NetworkTrainer"),
    ("musubi_tuner.hv_1_5_train_network", "HunyuanVideo15NetworkTrainer"),
    ("musubi_tuner.hv_train_network", "HunyuanVideoNetworkTrainer"),
]


def _import_trainer(module_path, class_name):
    import importlib

    module = importlib.import_module(module_path)
    return module, getattr(module, class_name)


def _dummy_process_batch_args(use_mask_loss):
    """Positional args for ``process_batch`` after ``self``. The dispatch only reads
    ``args.use_mask_loss``; the rest are opaque sentinels asserted to be forwarded verbatim."""
    args = SimpleNamespace(use_mask_loss=use_mask_loss)
    return (
        args,
        "accelerator",
        "transformer",
        "network",
        "batch",
        "latents",
        "noise",
        "noise_scheduler",
        "dit_dtype",
        "network_dtype",
        "vae",
        7,  # global_step
    )


@pytest.mark.parametrize("module_path,class_name", MASK_CAPABLE_TRAINERS, ids=[c for _, c in MASK_CAPABLE_TRAINERS])
def test_process_batch_dispatches_to_base_when_mask_off(module_path, class_name):
    from musubi_tuner.training.trainer_base import NetworkTrainer

    module, cls = _import_trainer(module_path, class_name)
    trainer = cls()
    pos = _dummy_process_batch_args(use_mask_loss=False)

    sentinel = ("BASE_RESULT", {})
    with mock.patch.object(NetworkTrainer, "process_batch", return_value=sentinel) as base_pb:
        # masked_process_batch must NOT be called on the mask-off path.
        if hasattr(module, "masked_process_batch"):
            with mock.patch.object(module, "masked_process_batch") as masked_pb:
                result = trainer.process_batch(*pos)
                masked_pb.assert_not_called()
        else:
            result = trainer.process_batch(*pos)

    assert result == sentinel, f"{class_name}.process_batch should return the base result when use_mask_loss=False"
    base_pb.assert_called_once()
    # Args forwarded verbatim (super() call carries no self).
    assert base_pb.call_args.args == pos


@pytest.mark.parametrize("module_path,class_name", MASK_CAPABLE_TRAINERS, ids=[c for _, c in MASK_CAPABLE_TRAINERS])
def test_process_batch_dispatches_to_masked_when_mask_on(module_path, class_name):
    module, cls = _import_trainer(module_path, class_name)
    trainer = cls()
    pos = _dummy_process_batch_args(use_mask_loss=True)

    assert hasattr(module, "masked_process_batch"), (
        f"{class_name}'s module must import masked_process_batch for the mask-on dispatch path"
    )

    sentinel = ("MASKED_RESULT", {"masked_loss/target": 1.0})
    with mock.patch.object(module, "masked_process_batch", return_value=sentinel) as masked_pb:
        result = trainer.process_batch(*pos)

    assert result == sentinel, f"{class_name}.process_batch should return masked_process_batch's result when use_mask_loss=True"
    masked_pb.assert_called_once()
    # Free function takes `self` first, then the same positional args.
    assert masked_pb.call_args.args[0] is trainer
    assert masked_pb.call_args.args[1:] == pos


@pytest.mark.parametrize("module_path,class_name", MASK_CAPABLE_TRAINERS, ids=[c for _, c in MASK_CAPABLE_TRAINERS])
def test_on_post_optimizer_step_ema_update_gating(module_path, class_name):
    module, cls = _import_trainer(module_path, class_name)
    trainer = cls()

    accelerator = mock.MagicMock()
    accelerator.unwrap_model.side_effect = lambda m: m
    network = object()
    args = SimpleNamespace()

    # No EMA teacher set → no-op, no crash.
    trainer.on_post_optimizer_step(args, accelerator, network, "transformer", True, 10)

    # EMA teacher present + sync_gradients=True → update fires once with the unwrapped network.
    ema = mock.MagicMock()
    trainer.prior_lora_ema = ema
    trainer.on_post_optimizer_step(args, accelerator, network, "transformer", True, 11)
    ema.update.assert_called_once_with(network)

    # sync_gradients=False → no further update (grad-accum: once per optimizer step, not per micro-batch).
    ema.update.reset_mock()
    trainer.on_post_optimizer_step(args, accelerator, network, "transformer", False, 12)
    ema.update.assert_not_called()
