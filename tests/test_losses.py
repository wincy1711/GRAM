import torch

from gram.core import TransitionOutput
from gram.guidance import Gaussian
from gram.losses import (
    act_loss,
    lprm_loss,
    reconstruction_loss,
    supervision_step_loss,
    transition_kl,
)


def _transition(mean_q=0.0, mean_p=0.0, std=1.0):
    shape = (2, 3, 4)
    return TransitionOutput(
        state=None,
        prior=Gaussian(torch.full(shape, mean_p), torch.full(shape, std)),
        posterior=Gaussian(torch.full(shape, mean_q), torch.full(shape, std)),
    )


def test_reconstruction_ignores_padding():
    logits = torch.zeros(1, 3, 4)
    logits[0, 0, 1] = 10.0
    targets = torch.tensor([[1, -100, -100]])
    assert reconstruction_loss(logits, targets).item() < 1e-3


def test_transition_kl_sums_over_transitions():
    one = transition_kl([_transition(mean_q=1.0)])
    two = transition_kl([_transition(mean_q=1.0), _transition(mean_q=1.0)])
    assert torch.allclose(two, 2 * one)


def test_transition_kl_is_zero_for_deterministic_models():
    empty = TransitionOutput(state=None, prior=None, posterior=None)
    assert transition_kl([empty]).item() == 0.0


def test_supervision_step_loss_weights_kl_by_beta():
    logits = torch.randn(2, 3, 4)
    targets = torch.randint(0, 4, (2, 3))
    transitions = [_transition(mean_q=1.0)]
    small = supervision_step_loss(logits, targets, transitions, beta=0.0)
    large = supervision_step_loss(logits, targets, transitions, beta=1.0)
    assert torch.allclose(small.total, small.reconstruction)
    assert torch.allclose(large.total, large.reconstruction + large.kl)
    assert large.total > small.total


def test_act_loss_rewards_calibrated_halting():
    correct = [torch.ones(4)]
    confident = [torch.full((4,), 8.0)]
    wrong = [torch.full((4,), -8.0)]
    zeros = [torch.zeros(4)]
    assert act_loss(confident, zeros, correct).item() < act_loss(wrong, zeros, correct).item()


def test_act_q_learning_mode_uses_the_bootstrap():
    q_halt = [torch.zeros(4), torch.full((4,), 4.0)]
    q_continue = [torch.zeros(4), torch.zeros(4)]
    correct = [torch.zeros(4), torch.ones(4)]
    halt_only = act_loss(q_halt, q_continue, correct, "halt_only")
    q_learning = act_loss(q_halt, q_continue, correct, "q_learning")
    assert q_learning.item() != halt_only.item()


def test_lprm_loss_is_zero_at_the_target():
    reward = torch.rand(5)
    assert lprm_loss([reward.clone()], reward).item() < 1e-8


def test_lprm_loss_grows_with_error():
    reward = torch.zeros(5)
    near = lprm_loss([torch.full((5,), 0.1)], reward)
    far = lprm_loss([torch.full((5,), 1.0)], reward)
    assert far > near
