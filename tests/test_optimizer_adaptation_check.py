# Optimizer-adaptation early warning (backlog P0-8).
#
# The DLAY v8 post-mortems showed Prodigy per-group d pathologies visible in telemetry by step ~50
# that went unnoticed for entire runs: a group pinned at d0 (Ideogram fp8, 3.9x under baseline) or
# inflated 1660x over its siblings (FLUX.2 Klein bf16). _check_optimizer_adaptation_once turns that
# five-minute signal into a loud WARNING at the prodigy_steps freeze. These tests pin the gating,
# the two warning classes, the healthy-run INFO, the no-op for fixed-LR optimizers, and one-shot-ness.

import logging
from types import SimpleNamespace

from musubi_tuner.ideogram4_train_network import Ideogram4NetworkTrainer

LOGGER_NAME = "musubi_tuner.training.trainer_base"


def _trainer():
    return Ideogram4NetworkTrainer()


def _prodigy_groups(ds, d0=1e-6, k=500, prodigy_steps=500):
    return SimpleNamespace(param_groups=[{"d": d, "d0": d0, "k": k, "prodigy_steps": prodigy_steps, "lr": 1.0} for d in ds])


def _prodigy_groups_with_shared(ds, shared_d, lrs=None, d0=1e-6, k=500, prodigy_steps=500):
    lrs = lrs or [1.0] * len(ds)
    return SimpleNamespace(
        param_groups=[
            {"d": d, "d0": d0, "k": k, "prodigy_steps": prodigy_steps, "lr": lr, "shared_d": shared_d} for d, lr in zip(ds, lrs)
        ]
    )


def test_shared_d_reported_with_applied_scales(caplog):
    # The Klein v9 false-alarm cycle: three analyses misread raw per-group d without the shared_d
    # factor. With split_groups_mean active (shared_d present), the checkpoint INFO must report
    # shared_d AND the per-group applied scale d*shared_d*lr — the literal v9 values as fixture.
    trainer = _trainer()
    opt = _prodigy_groups_with_shared([1.769, 2.197e-5, 8.798e-4], shared_d=6.43e-5, lrs=[1.0, 12.0, 1.0])
    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        trainer._check_optimizer_adaptation_once(opt, ["unet", "unet plus", "unet dora"])
    info = next(r for r in caplog.records if "adaptation checkpoint" in r.message)
    assert "shared_d=6.430e-05" in info.message
    assert "applied scale" in info.message
    # spread warning fires (80k spread) but with the split_groups_mean framing, not the
    # corruption-presumption framing
    warning = next(r for r in caplog.records if "span" in r.message)
    assert "split_groups_mean" in warning.message
    assert "applied scales" in warning.message
    assert "usually means one group's estimate is corrupted" not in warning.message


def test_spread_warning_keeps_corruption_framing_without_shared_d(caplog):
    # No shared_d (split_groups_mean off / plain Prodigy): the original corruption framing stays —
    # there the raw d IS the whole dlr factor and a wide spread is the v8-class signal.
    trainer = _trainer()
    opt = _prodigy_groups([5.18e-2, 3.12e-5])
    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        trainer._check_optimizer_adaptation_once(opt, ["unet", "unet plus"])
    warning = next(r for r in caplog.records if "span" in r.message)
    assert "usually means one group's estimate is corrupted" in warning.message
    assert "split_groups_mean" not in warning.message


def test_shared_d_none_treated_as_absent(caplog):
    # prodigyplus defaults shared_d=None when split_groups_mean is off — must route to the
    # no-shared_d reporting, not crash on float(None).
    trainer = _trainer()
    opt = _prodigy_groups_with_shared([4.21e-6, 2.48e-6], shared_d=None)
    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        trainer._check_optimizer_adaptation_once(opt, ["unet", "unet plus"])
    info = next(r for r in caplog.records if "adaptation checkpoint" in r.message)
    assert "shared_d" not in info.message


def test_pinned_group_warns(caplog):
    trainer = _trainer()
    # Ideogram v8 signature: down group pinned at ~d0 while up adapted normally.
    opt = _prodigy_groups([1.07e-6, 2.49e-6])
    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        trainer._check_optimizer_adaptation_once(opt, ["unet", "unet plus"])
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("barely adapted" in r.message and "'unet'" in r.message for r in warnings)
    assert not any("'unet plus'" in r.message and "barely adapted" in r.message for r in warnings)


def test_wild_cross_group_spread_warns(caplog):
    trainer = _trainer()
    # Klein v8 signature: one group inflated orders of magnitude over its siblings.
    opt = _prodigy_groups([5.18e-2, 3.12e-5])
    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        trainer._check_optimizer_adaptation_once(opt, ["unet", "unet plus"])
    assert any("span" in r.message for r in caplog.records if r.levelno == logging.WARNING)


def test_healthy_run_logs_info_only(caplog):
    trainer = _trainer()
    # v7 signature: both groups adapted within the same order of magnitude.
    opt = _prodigy_groups([4.21e-6, 2.48e-6])
    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        trainer._check_optimizer_adaptation_once(opt, ["unet", "unet plus"])
    assert any("adaptation checkpoint" in r.message for r in caplog.records if r.levelno == logging.INFO)
    assert not [r for r in caplog.records if r.levelno == logging.WARNING]


def test_does_not_fire_before_freeze(caplog):
    trainer = _trainer()
    opt = _prodigy_groups([1e-6, 1e-6], k=100, prodigy_steps=500)  # pinned, but adaptation still running
    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        trainer._check_optimizer_adaptation_once(opt, ["unet", "unet plus"])
    assert not caplog.records
    assert not trainer._optimizer_adaptation_checked  # will check again at the freeze

    # ...and fires once the freeze step is reached.
    opt2 = _prodigy_groups([1e-6, 1e-6], k=500, prodigy_steps=500)
    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        trainer._check_optimizer_adaptation_once(opt2, ["unet", "unet plus"])
    assert trainer._optimizer_adaptation_checked
    assert any("barely adapted" in r.message for r in caplog.records if r.levelno == logging.WARNING)


def test_never_freezing_config_checks_at_step_200(caplog):
    trainer = _trainer()
    opt = _prodigy_groups([1e-6], k=200, prodigy_steps=0)  # prodigy_steps=0: d never freezes
    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        trainer._check_optimizer_adaptation_once(opt, ["unet"])
    assert trainer._optimizer_adaptation_checked
    assert any("barely adapted" in r.message for r in caplog.records if r.levelno == logging.WARNING)


def test_one_shot_no_duplicate_warnings(caplog):
    trainer = _trainer()
    opt = _prodigy_groups([1e-6, 1e-6])
    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        trainer._check_optimizer_adaptation_once(opt, ["unet", "unet plus"])
        first = len(caplog.records)
        trainer._check_optimizer_adaptation_once(opt, ["unet", "unet plus"])
    assert len(caplog.records) == first


def test_fixed_lr_optimizer_is_noop(caplog):
    trainer = _trainer()
    adamw_like = SimpleNamespace(param_groups=[{"lr": 1e-4}])  # no d/d0 keys
    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        trainer._check_optimizer_adaptation_once(adamw_like, ["unet"])
    assert not caplog.records
    assert trainer._optimizer_adaptation_checked  # marked done — zero cost for the rest of the run


def test_missing_descriptions_fall_back_to_group_index(caplog):
    trainer = _trainer()
    opt = _prodigy_groups([1e-6])
    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        trainer._check_optimizer_adaptation_once(opt, None)
    assert any("group0" in r.message for r in caplog.records)
