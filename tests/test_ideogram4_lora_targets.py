import torch
from accelerate import init_empty_weights

import musubi_tuner.networks.lora_ideogram4 as lora_ideogram4
from musubi_tuner.ideogram4.modeling_ideogram4 import Ideogram4Config, Ideogram4Transformer


def _tiny_unet(num_layers=2):
    cfg = Ideogram4Config(
        emb_dim=32,
        num_layers=num_layers,
        num_heads=2,
        intermediate_size=48,
        adanln_dim=16,
        in_channels=8,
        llm_features_dim=16,
        mrope_section=(1, 1, 0),
    )
    with init_empty_weights():
        return Ideogram4Transformer(cfg)


def test_lora_targets_attention_and_ffn_not_adaln():
    unet = _tiny_unet(num_layers=2)
    network = lora_ideogram4.create_arch_network(1.0, 8, 4.0, None, None, unet)

    names = [m.lora_name for m in network.unet_loras]
    # 5 targets/block (attention.qkv, attention.o, feed_forward.w1/w2/w3) x 2 blocks.
    assert len(network.unet_loras) == 10, names
    assert all(("attention" in n) or ("feed_forward" in n) for n in names)
    assert any("qkv" in n for n in names) and any("feed_forward_w1" in n.replace(".", "_") for n in names)
    # adaln_modulation must NOT be a LoRA target.
    assert not any("adaln" in n for n in names)


def test_lora_targets_are_plain_linear():
    # The whole point of the centerpiece: targets stay nn.Linear (fp8 via monkey-patch), so DoRA/discovery work.
    unet = _tiny_unet(num_layers=1)
    block = next(m for m in unet.modules() if m.__class__.__name__ == "Ideogram4TransformerBlock")
    for path in ("attention.qkv", "attention.o", "feed_forward.w1", "feed_forward.w2", "feed_forward.w3"):
        mod = block
        for part in path.split("."):
            mod = getattr(mod, part)
        assert isinstance(mod, torch.nn.Linear)
