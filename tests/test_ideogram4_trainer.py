from types import SimpleNamespace

import pytest

from musubi_tuner.ideogram4_train_network import Ideogram4NetworkTrainer, ideogram4_setup_parser


def test_trainer_architecture_names():
    trainer = Ideogram4NetworkTrainer()
    assert trainer.architecture == "i4"
    assert trainer.architecture_full_name == "ideogram4"


def test_trainer_rejects_use_mask_loss():
    # The mask-loss guard fires before any other arg is touched (no mask cache exists for Ideogram yet).
    trainer = Ideogram4NetworkTrainer()
    args = SimpleNamespace(use_mask_loss=True)
    with pytest.raises(ValueError, match="use_mask_loss"):
        trainer.process_batch(args, None, None, None, None, None, None, None, None, None, None, None)


def test_trainer_call_dit_not_used():
    trainer = Ideogram4NetworkTrainer()
    with pytest.raises(NotImplementedError, match="process_batch"):
        trainer.call_dit()


def test_setup_parser_adds_timestep_args():
    import argparse

    parser = ideogram4_setup_parser(argparse.ArgumentParser())
    args = parser.parse_args(["--ideogram4_timestep_mu", "0.5", "--ideogram4_timestep_std", "1.5"])
    assert args.ideogram4_timestep_mu == 0.5 and args.ideogram4_timestep_std == 1.5
