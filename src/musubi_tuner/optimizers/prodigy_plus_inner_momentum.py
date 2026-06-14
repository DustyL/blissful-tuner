from __future__ import annotations

import torch

from prodigyplus.prodigy_plus_schedulefree import ProdigyPlusScheduleFree


UPSTREAM_PRODIGYPLUS_VERSION = (2, 0, 0)
# Fork point: ``initialise_state`` and ``step_param_schedulefree`` are copied
# from prodigyplus.ProdigyPlusScheduleFree at CoreOptimiser.VERSION 2.0.0
# (upstream repo tag v2.0.1). Re-sync this subclass when prodigyplus is bumped.


class ProdigyPlusInnerMomentumScheduleFree(ProdigyPlusScheduleFree):
    """
    Experimental ProdigyPlusScheduleFree variant with opt-in Adam inner momentum.

    ScheduleFree+ reintroduces an Adam first-moment buffer inside the Schedule-Free
    update. This subclass keeps Blissful's existing ProdigyPlusScheduleFree behavior
    at ``inner_beta1=0`` and adds only that first-moment path for research A/B runs.

    The first moment is scaled by the raw group ``d`` before preconditioning, matching
    ProdigyPlus' second-moment scaling of ``d**2``. That preserves the raw-d inertness
    invariant under ``split_groups_mean=True``: raw per-group ``d`` cancels in the
    Adam ratio and the applied step remains ``shared_d * lr``.

    The inner first moment is intentionally not bias-corrected for the initial
    low-beta spike. Revisit this before testing high values such as 0.9.
    """

    def __init__(self, params, *args, inner_beta1: float = 0.0, **kwargs):
        if not 0.0 <= inner_beta1 < 1.0:
            raise ValueError(f"Invalid inner_beta1 value: {inner_beta1}")

        super().__init__(params, *args, **kwargs)

        for group in self.param_groups:
            group.setdefault("inner_beta1", inner_beta1)
            if not 0.0 <= group["inner_beta1"] < 1.0:
                raise ValueError(f"Invalid inner_beta1 value: {group['inner_beta1']}")

    @torch.no_grad()
    def initialise_state(self, p, group):
        state, needs_init = self.initialise_state_internal(p, group)

        if needs_init:
            if group["use_schedulefree"]:
                state["z"] = p.detach().clone(memory_format=torch.preserve_format)
                if group.get("inner_beta1", 0.0) > 0.0:
                    state["exp_avg"] = torch.zeros_like(p.grad, memory_format=torch.preserve_format).detach()
            else:
                state["exp_avg"] = torch.zeros_like(p.grad, memory_format=torch.preserve_format).detach()

        return state

    @torch.no_grad()
    def step_param_schedulefree(self, p, group):
        if not group["train_mode"]:
            raise Exception("Not in train mode!")

        k = group["k"]
        use_adopt = group["use_adopt"]
        use_bias_correction = group["use_bias_correction"]
        stochastic = group["stochastic_rounding"]
        _, beta2, _ = self.get_betas(group)

        state = self.initialise_state(p, group)
        if group.get("inner_beta1", 0.0) > 0.0 and "exp_avg" not in state:
            state["exp_avg"] = torch.zeros_like(p.grad, memory_format=torch.preserve_format).detach()

        z_state = state["z"]
        y, z = p.float(), z_state.float()

        grad = p.grad.to(dtype=torch.float32, copy=True)
        dlr = self.get_dlr(group)

        if use_bias_correction:
            dlr, beta2, rho_t = self.get_bias_correction(dlr, beta2, k)

        update = None

        if use_adopt and k == 1:
            self.update_second_moment(state, group, grad, 0, z, return_denom=False)
            del grad
        else:
            denom = self.update_second_moment(state, group, grad, beta2, z, denom_before_update=use_adopt)

            if use_bias_correction and rho_t <= 4.0:
                update = grad
            else:
                inner_beta1 = group.get("inner_beta1", 0.0)
                if inner_beta1 > 0.0:
                    exp_avg = self.update_first_moment(state, group, grad, inner_beta1)
                    grad.copy_(exp_avg)
                else:
                    grad.mul_(group["d"])
                update = self.update_(grad, denom, group, z)
            del denom

        if update is not None:
            if group["use_orthograd"]:
                update = self.orthograd_(z, update)

            if group["use_stableadamw"]:
                update = self.rms_clip_(update)

            self.update_prodigy(state, group, p.grad, z_state)
            self.update_params(y, z, update, group, dlr)

            self.smart_copy(p, y, stochastic, True)
            self.smart_copy(z_state, z, stochastic, True)

            del update
