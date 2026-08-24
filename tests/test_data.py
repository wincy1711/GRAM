import numpy as np
import pytest

from gram.data import arc, graph_coloring as gc, nqueens as nq, sudoku as sd
from gram.data.base import PuzzleDataset, SolutionIndex


# --------------------------------------------------------------------------- #
# N-Queens
# --------------------------------------------------------------------------- #
def test_nqueens_solution_counts_are_known():
    assert len(nq.enumerate_solutions(6)) == 4
    assert len(nq.enumerate_solutions(8)) == 92


def test_nqueens_checker_accepts_solutions_and_rejects_perturbations():
    solutions = nq.enumerate_solutions(6)
    tokens = nq.solution_to_tokens(6, solutions[0])
    assert nq.check_solution(tokens, 6)
    broken = tokens.copy()
    queens = nq.tokens_to_queens(tokens, 6)
    row, col = queens[0]
    broken[row * 6 + col] = nq.EMPTY
    broken[row * 6 + (col + 1) % 6] = nq.QUEEN
    assert not nq.check_solution(broken, 6)


def test_nqueens_checker_requires_input_queens_to_be_kept():
    solutions = nq.enumerate_solutions(6)
    tokens = nq.solution_to_tokens(6, solutions[0])
    given = nq.board_to_tokens(6, [(0, (solutions[0][0] + 1) % 6)])
    assert not nq.check_solution(tokens, 6, given)


def test_nqueens_completions_are_consistent_with_the_partial_board():
    solutions = nq.enumerate_solutions(6)
    partial = {0: solutions[0][0]}
    found = nq.completions(6, partial, solutions)
    assert found and all(s[0] == solutions[0][0] for s in found)


def test_nqueens_build_produces_disjoint_input_splits(tmp_path):
    metadata = nq.build(tmp_path, n=6, remove=(3, 4), num_instances=40, seed=1)
    assert metadata.vocab_size == 3 and metadata.seq_len == 36
    train = PuzzleDataset(tmp_path, "train")
    test = PuzzleDataset(tmp_path, "test")
    train_keys = {row.numpy().tobytes() for row in train.inputs}
    test_keys = {row.numpy().tobytes() for row in test.inputs}
    assert train_keys.isdisjoint(test_keys)
    # Every stored target must be a genuine solution of its input.
    for inputs, targets in zip(test.inputs.numpy(), test.targets.numpy()):
        assert nq.check_solution(targets, 6, inputs)


def test_nqueens_solution_index_round_trip(tmp_path):
    nq.build(tmp_path, n=6, remove=(3,), num_instances=20, seed=2)
    index = SolutionIndex(tmp_path / "solutions.npz")
    assert index.available
    dataset = PuzzleDataset(tmp_path, "test")
    for group_id, inputs in zip(dataset.group_ids.tolist(), dataset.inputs.numpy()):
        solutions = index.get(group_id)
        assert solutions is not None and len(solutions) > 0
        assert all(nq.check_solution(s, 6, inputs) for s in solutions)


# --------------------------------------------------------------------------- #
# Graph colouring
# --------------------------------------------------------------------------- #
def test_graph_coloring_enumeration_is_canonical():
    edges = {(0, 1), (1, 2), (0, 2)}
    colorings = gc.enumerate_colorings(3, edges)
    assert colorings == [(0, 1, 2)]  # permutations collapse to one canonical form


def test_graph_coloring_detects_non_colorable_graphs():
    # K4 is not 3-colourable.
    edges = {(i, j) for i in range(4) for j in range(i + 1, 4)}
    assert gc.enumerate_colorings(4, edges) == []


def test_graph_coloring_conflicts():
    edges = {(0, 1), (1, 2)}
    assert gc.count_conflicts([0, 0, 0], edges) == 2
    assert gc.count_conflicts([0, 1, 0], edges) == 0


def test_graph_coloring_token_round_trip():
    edges = {(0, 2), (1, 3)}
    tokens = gc.edges_to_tokens(4, edges)
    assert gc.tokens_to_edges(tokens, 4) == edges
    coloring = [0, 1, 2, 0]
    padded = gc.coloring_to_tokens(coloring, pad_to=6)
    assert len(padded) == 6 and padded[4] == 0
    assert gc.tokens_to_coloring(padded, 4) == coloring


def test_graph_coloring_build(tmp_path):
    metadata = gc.build(tmp_path, n=6, num_instances=30, seed=3, min_solutions=2)
    assert metadata.out_seq_len == 6 and metadata.seq_len == 15
    dataset = PuzzleDataset(tmp_path, "test")
    for inputs, targets in zip(dataset.inputs.numpy(), dataset.targets.numpy()):
        edges = gc.tokens_to_edges(inputs, 6)
        assert gc.is_valid_coloring(gc.tokens_to_coloring(targets, 6), edges)


# --------------------------------------------------------------------------- #
# Sudoku
# --------------------------------------------------------------------------- #
def test_sudoku_generator_creates_valid_boards():
    import random
    grid = sd.random_complete_grid(random.Random(0))
    assert sd.is_complete_valid(grid)
    assert sd.num_violations(grid) == 0


def test_sudoku_puzzle_has_a_unique_solution():
    import random
    rng = random.Random(1)
    solution = sd.random_complete_grid(rng)
    puzzle = sd.make_puzzle(solution, rng, min_clues=30)
    assert sd.has_unique_solution(puzzle)
    assert np.array_equal(sd.solve(puzzle)[0], solution)


def test_sudoku_violation_count_is_sensitive():
    import random
    grid = sd.random_complete_grid(random.Random(2))
    broken = grid.copy()
    broken[0], broken[1] = broken[1], broken[0]
    assert not sd.is_complete_valid(broken)
    assert sd.num_violations(broken) > 0


def test_sudoku_token_round_trip():
    import random
    grid = sd.random_complete_grid(random.Random(3))
    tokens = sd.grid_to_tokens(grid)
    assert tokens.min() >= 1 and tokens.max() <= 10
    assert np.array_equal(sd.tokens_to_grid(tokens), grid)


def test_sudoku_build_conditional_and_unconditional(tmp_path):
    sd.build(tmp_path / "cond", num_train=3, num_test=2, min_clues=40, seed=4)
    dataset = PuzzleDataset(tmp_path / "cond", "test")
    for inputs, targets in zip(dataset.inputs.numpy(), dataset.targets.numpy()):
        assert sd.is_complete_valid(sd.tokens_to_grid(targets))
        assert sd.matches_clues(sd.tokens_to_grid(targets), sd.tokens_to_grid(inputs))

    sd.build(tmp_path / "uncond", num_train=2, num_test=2, seed=5, unconditional=True)
    blank = PuzzleDataset(tmp_path / "uncond", "train")
    assert bool((blank.inputs == sd.BLANK).all())


def test_sudoku_csv_loader(tmp_path):
    import random
    rng = random.Random(6)
    rows = []
    for _ in range(2):
        solution = sd.random_complete_grid(rng)
        puzzle = sd.make_puzzle(solution, rng, min_clues=45)
        rows.append((
            "".join(str(int(v)) for v in puzzle),
            "".join(str(int(v)) for v in solution),
        ))
    csv_path = tmp_path / "sudoku.csv"
    csv_path.write_text(
        "source,question,answer,rating\n"
        + "\n".join(f"gen,{q},{a},0" for q, a in rows) + "\n"
    )
    sd.load_csv(tmp_path / "out", csv_path, csv_path)
    dataset = PuzzleDataset(tmp_path / "out", "train")
    assert len(dataset) == 2
    assert sd.is_complete_valid(sd.tokens_to_grid(dataset.targets[0].numpy()))


# --------------------------------------------------------------------------- #
# ARC
# --------------------------------------------------------------------------- #
def test_arc_canvas_round_trip():
    grid = [[0, 1, 2], [3, 4, 5]]
    tokens = arc.grid_to_canvas(grid)
    assert tokens.shape == (900,)
    assert arc.canvas_to_grid(tokens) == grid


def test_arc_build_assigns_puzzle_identifiers(tmp_path):
    import json
    task_dir = tmp_path / "tasks"
    task_dir.mkdir()
    for name in ("a", "b"):
        (task_dir / f"{name}.json").write_text(json.dumps({
            "train": [{"input": [[1, 2]], "output": [[2, 1]]}],
            "test": [{"input": [[3]], "output": [[4]]}],
        }))
    metadata = arc.build(tmp_path / "out", task_dir)
    assert metadata.num_puzzle_identifiers == 3  # two tasks + the reserved 0
    dataset = PuzzleDataset(tmp_path / "out", "train")
    assert set(dataset.puzzle_ids.tolist()) == {1, 2}


def test_arc_augmentation_multiplies_pairs(tmp_path):
    import json
    task_dir = tmp_path / "tasks"
    task_dir.mkdir()
    (task_dir / "a.json").write_text(json.dumps({
        "train": [{"input": [[1, 2], [3, 4]], "output": [[4, 3], [2, 1]]}],
        "test": [],
    }))
    arc.build(tmp_path / "plain", task_dir, augmentations=0)
    arc.build(tmp_path / "aug", task_dir, augmentations=3)
    assert len(PuzzleDataset(tmp_path / "plain", "train")) == 1
    assert len(PuzzleDataset(tmp_path / "aug", "train")) == 4


def test_arc_augmentation_beyond_the_dihedral_group(tmp_path):
    import json
    task_dir = tmp_path / "tasks"
    task_dir.mkdir()
    (task_dir / "a.json").write_text(json.dumps({
        "train": [{"input": [[1, 2], [3, 4]], "output": [[4, 3], [2, 1]]}],
        "test": [],
    }))
    arc.build(tmp_path / "aug", task_dir, augmentations=19)
    dataset = PuzzleDataset(tmp_path / "aug", "train")
    assert len(dataset) == 20
    # All 20 copies must be distinct, which needs colour permutations on top of
    # the 8 dihedral transforms.
    assert len({row.numpy().tobytes() for row in dataset.inputs}) == 20


def test_arc_colour_permutation_preserves_the_background():
    grid = [[0, 1, 2], [3, 0, 4]]
    for transform in range(12):
        out = arc._augment(grid, transform)
        flat_in = [v for row in grid for v in row]
        flat_out = [v for row in out for v in row]
        assert flat_in.count(0) == flat_out.count(0)
        assert sorted(set(flat_out)) == sorted(set(flat_out))  # valid colour ids
        assert all(0 <= v < arc.NUM_COLORS for v in flat_out)
