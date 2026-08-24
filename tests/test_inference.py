import numpy as np
import torch

from gram.config import EvalConfig, ModelConfig
from gram.inference import best_of_n, generate, majority_vote, sample_trajectories, select
from gram.metrics import solution_coverage, unique_valid_fraction
from gram.model import GRAM


def make(**overrides) -> GRAM:
    base = dict(vocab_size=5, seq_len=10, hidden_size=32, num_heads=4, ffn_hidden_size=32,
                num_layers=1, low_level_steps=2, high_level_steps=2, n_supervision=4)
    base.update(overrides)
    return GRAM(ModelConfig(**base))


def test_sample_trajectories_shapes_scale_with_width():
    model = make()
    x = torch.randint(0, 5, (3, 10))
    out = sample_trajectories(model, x, num_samples=7)
    assert out.predictions.shape == (3, 7, 10)
    assert out.logits.shape == (3, 7, 10, 5)
    assert out.values.shape == (3, 7) and out.halt_steps.shape == (3, 7)


def test_parallel_samples_differ_for_a_stochastic_model():
    model = make()
    x = torch.randint(0, 5, (2, 10))
    out = sample_trajectories(model, x, num_samples=8, use_act=False)
    assert not torch.equal(out.predictions[0, 0], out.predictions[0, 1]) or \
        unique_valid_fraction(out.predictions[0].numpy()) > 1 / 8


def test_deterministic_model_collapses_to_one_trajectory():
    model = make(guidance="none")
    x = torch.randint(0, 5, (2, 10))
    out = sample_trajectories(model, x, num_samples=5, use_act=False)
    for n in range(1, 5):
        assert torch.equal(out.predictions[0, 0], out.predictions[0, n])


def test_majority_vote_picks_the_modal_prediction():
    predictions = torch.tensor([[[1, 1], [1, 1], [2, 2]]])
    assert torch.equal(majority_vote(predictions), torch.tensor([[1, 1]]))


def test_best_of_n_uses_the_value_head():
    predictions = torch.tensor([[[1, 1], [2, 2], [3, 3]]])
    values = torch.tensor([[0.1, 0.9, 0.5]])
    assert torch.equal(best_of_n(predictions, values), torch.tensor([[2, 2]]))


def test_select_strategies():
    model = make()
    out = sample_trajectories(model, torch.randint(0, 5, (2, 10)), num_samples=3)
    for strategy in ("majority", "lprm", "first"):
        assert select(out, strategy).shape == (2, 10)


def test_act_halts_early_when_the_halt_head_is_confident():
    model = make(n_supervision=6)
    with torch.no_grad():
        model.q_head.weight[0].fill_(0.0)
        model.q_head.bias = None
        # Force a large positive halt logit through the first latent dimension.
        model.q_head.weight[0, 0] = 1e4
    x = torch.randint(0, 5, (2, 10))
    halted = sample_trajectories(model, x, num_samples=1, use_act=True)
    never = sample_trajectories(model, x, num_samples=1, use_act=False)
    assert halted.halt_steps.max() <= never.halt_steps.min()


def test_generation_from_an_empty_input():
    model = make()
    samples = generate(model, num_samples=6, seq_len=10, batch_size=4)
    assert samples.shape == (6, 10)


def test_coverage_increases_with_more_distinct_valid_samples():
    class FakeIndex:
        def get(self, gid):
            return [np.array([1, 2]), np.array([3, 4])]

    one = np.array([[[1, 2], [1, 2]]])
    both = np.array([[[1, 2], [3, 4]]])
    assert solution_coverage(one, [0], FakeIndex()).coverage == 0.5
    assert solution_coverage(both, [0], FakeIndex()).coverage == 1.0


def test_predict_end_to_end():
    from gram.inference import predict
    model = make()
    out = predict(model, torch.randint(0, 5, (2, 10)), config=EvalConfig(num_samples=3))
    assert out.shape == (2, 10)


def test_trace_records_the_refinement_process():
    """Figure 6 visualises how the prediction evolves over recursion steps."""
    model = make(n_supervision=5)
    out = sample_trajectories(model, torch.randint(0, 5, (2, 10)), num_samples=3,
                              use_act=False, return_all_steps=True)
    assert out.trace is not None
    assert out.trace.shape == (5, 2, 3, 10)


def test_trace_is_absent_by_default():
    model = make()
    assert sample_trajectories(model, torch.randint(0, 5, (2, 10))).trace is None


def test_majority_vote_breaks_ties_towards_the_first_sample():
    predictions = torch.tensor([[[2, 2], [1, 1], [1, 1], [2, 2]]])
    # Both answers occur twice; the earliest one wins.
    assert torch.equal(majority_vote(predictions), torch.tensor([[2, 2]]))


def test_majority_vote_matches_a_reference_count():
    torch.manual_seed(0)
    predictions = torch.randint(0, 2, (4, 9, 3))
    expected = []
    for b in range(4):
        counts = {}
        for n in range(9):
            key = tuple(predictions[b, n].tolist())
            counts[key] = counts.get(key, 0) + 1
        expected.append(max(counts.items(), key=lambda kv: kv[1])[0])
    assert torch.equal(majority_vote(predictions), torch.tensor(expected))
