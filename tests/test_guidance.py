import math

import pytest
import torch

from gram.guidance import Gaussian, GaussianHead, StochasticTransition, gaussian_kl
from gram.losses import balanced_kl


def test_kl_of_identical_gaussians_is_zero():
    dist = Gaussian(torch.randn(2, 3, 4), torch.rand(2, 3, 4) + 0.1)
    assert torch.allclose(gaussian_kl(dist, dist), torch.zeros(2, 3, 4), atol=1e-6)


def test_kl_matches_closed_form():
    q = Gaussian(torch.tensor([1.0]), torch.tensor([2.0]))
    p = Gaussian(torch.tensor([0.0]), torch.tensor([1.0]))
    expected = math.log(1 / 2) + (4 + 1) / 2 - 0.5
    assert torch.allclose(gaussian_kl(q, p), torch.tensor([expected]), atol=1e-5)


def test_kl_is_non_negative():
    q = Gaussian(torch.randn(64, 8), torch.rand(64, 8) + 0.05)
    p = Gaussian(torch.randn(64, 8), torch.rand(64, 8) + 0.05)
    assert bool((gaussian_kl(q, p) >= -1e-6).all())


def test_head_initialises_near_deterministic():
    head = GaussianHead(16, 16, "full", init_std=0.1)
    dist = head(torch.randn(2, 4, 16))
    # mu is zero-initialised so training starts from the deterministic update.
    assert torch.allclose(dist.mean, torch.zeros_like(dist.mean))
    assert torch.allclose(dist.std, torch.full_like(dist.std, 0.1), atol=1e-4)


@pytest.mark.parametrize("mode,has_mean,has_std", [
    ("full", True, True),
    ("stochastic_only", False, True),
    ("guide_only", True, False),
])
def test_guidance_modes(mode, has_mean, has_std):
    head = GaussianHead(16, 16, mode, init_std=0.1)
    with torch.no_grad():
        for param in head.parameters():
            param.add_(torch.randn_like(param) * 0.1)
    dist = head(torch.randn(2, 4, 16))
    assert (dist.mean.abs().max() > 0) == has_mean
    assert (dist.std.max() > 0) == has_std


def test_deterministic_transition_is_identity_on_u():
    transition = StochasticTransition(16, 16, "none")
    u = torch.randn(2, 3, 16)
    z, prior, posterior = transition(u)
    assert torch.equal(z, u) and prior is None and posterior is None


def test_prior_and_posterior_are_separate_modules():
    transition = StochasticTransition(16, 16, "full")
    prior_params = {id(p) for p in transition.prior.parameters()}
    posterior_params = {id(p) for p in transition.posterior.parameters()}
    assert prior_params.isdisjoint(posterior_params)


def test_sampling_is_stochastic_and_reparameterised():
    transition = StochasticTransition(16, 16, "full")
    u = torch.randn(2, 3, 16, requires_grad=True)
    z, _, _ = transition(u)
    assert not torch.allclose(z, transition(u)[0])
    z.sum().backward()
    assert u.grad is not None and u.grad.abs().sum() > 0


def test_kl_balance_weights_the_two_directions():
    q = Gaussian(torch.zeros(1, 4, requires_grad=True), torch.ones(1, 4))
    p = Gaussian(torch.ones(1, 4, requires_grad=True), torch.ones(1, 4))
    balanced_kl(q, p, kl_balance=1.0).backward()
    # With balance 1.0 the whole gradient goes to the prior.
    assert q.mean.grad.abs().sum() == 0
    assert p.mean.grad.abs().sum() > 0


def test_degenerate_kl_falls_back_to_squared_distance():
    q = Gaussian(torch.ones(1, 4), torch.zeros(1, 4))
    p = Gaussian(torch.zeros(1, 4), torch.zeros(1, 4))
    assert torch.isfinite(balanced_kl(q, p))
    assert balanced_kl(q, p).item() == pytest.approx(2.0)
