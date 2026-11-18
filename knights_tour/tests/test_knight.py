"""Unit tests for the Knight class."""

import unittest

from knights_tour.chromosome import Chromosome
from knights_tour.knight import BOARD_SIZE, DIRECTIONS, Knight


def _make_valid_tour_moves() -> list[int]:
    """Convert the Warnsdorff valid tour path into direction codes 1..8."""
    path = _make_valid_tour_path()
    delta_to_dir = {delta: d for d, delta in DIRECTIONS.items()}
    return [
        delta_to_dir[(b[0] - a[0], b[1] - a[1])]
        for a, b in zip(path, path[1:])
    ]


def _make_valid_tour_path() -> list[tuple[int, int]]:
    """Build a valid 64-square Knight's Tour via Warnsdorff's rule for fixture tests."""
    board_visited: set[tuple[int, int]] = {(0, 0)}
    path: list[tuple[int, int]] = [(0, 0)]
    pos = (0, 0)
    move_order = [3, 4, 2, 5, 1, 6, 8, 7]

    def degree(sq: tuple[int, int]) -> int:
        count = 0
        for d in move_order:
            dx, dy = DIRECTIONS[d]
            nx, ny = sq[0] + dx, sq[1] + dy
            if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE and (nx, ny) not in board_visited:
                count += 1
        return count

    for _ in range(63):
        candidates = []
        for d in move_order:
            dx, dy = DIRECTIONS[d]
            nx, ny = pos[0] + dx, pos[1] + dy
            if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE and (nx, ny) not in board_visited:
                candidates.append((degree((nx, ny)), d, (nx, ny)))
        candidates.sort(key=lambda item: (item[0], item[1]))
        _, _, next_pos = candidates[0]
        board_visited.add(next_pos)
        path.append(next_pos)
        pos = next_pos
    return path


class TestKnight(unittest.TestCase):
    """Verify Knight movement, check_moves cycling, dead-ends, and fitness evaluation."""

    def test_initial_state_and_optional_chromosome(self) -> None:
        knight = Knight()
        self.assertEqual(knight.position, (0, 0))
        self.assertEqual(knight.fitness, 0)
        self.assertEqual(knight.path, [(0, 0)])
        self.assertIsInstance(knight.chromosome, Chromosome)
        self.assertEqual(len(knight.chromosome.genes), 63)

        custom = Chromosome([3] * 63)
        knight2 = Knight(custom)
        self.assertIs(knight2.chromosome, custom)

    def test_move_forward_and_backward_all_directions(self) -> None:
        expected_deltas = {
            1: (1, -2),
            2: (2, -1),
            3: (2, 1),
            4: (1, 2),
            5: (-1, 2),
            6: (-2, 1),
            7: (-2, -1),
            8: (-1, -2),
        }
        knight = Knight()
        for direction, (dx, dy) in expected_deltas.items():
            knight.position = (4, 4)
            new_pos = knight.move_forward(direction)
            self.assertEqual(new_pos, (4 + dx, 4 + dy))
            self.assertEqual(knight.position, (4 + dx, 4 + dy))
            restored = knight.move_backward(direction)
            self.assertEqual(restored, (4, 4))
            self.assertEqual(knight.position, (4, 4))

        with self.assertRaises(ValueError):
            knight.move_forward(0)
        with self.assertRaises(ValueError):
            knight.move_backward(9)

    def test_check_moves_cycle_from_origin(self) -> None:
        # From (0, 0), move 1 (up-right -> (1, -2)) is out of bounds.
        genes = [1] + [3] * 62
        knight_fwd = Knight(Chromosome(genes))
        knight_fwd.check_moves(cycle_direction=1)
        # Forward cycle: 1 -> 2 (2, -1 OOB) -> 3 (2, 1 valid)
        self.assertEqual(knight_fwd.chromosome.genes[0], 3)
        self.assertEqual(knight_fwd.path[1], (2, 1))

        knight_bwd = Knight(Chromosome(genes))
        knight_bwd.check_moves(cycle_direction=-1)
        # Backward cycle: 1 -> 8 -> 7 -> 6 -> 5 -> 4 (1, 2 valid)
        self.assertEqual(knight_bwd.chromosome.genes[0], 4)
        self.assertEqual(knight_bwd.path[1], (1, 2))

    def test_check_moves_cycle_order_from_interior_blocked_move_4(self) -> None:
        # Path to (2, 1) with (3, 3) already visited:
        # (0, 0) -4-> (1, 2) -3-> (3, 3) -8-> (2, 1)
        # At (2, 1), move 4: down-right -> (3, 3) is visited (illegal).
        # Forward checks 5 -> (1, 3) (valid). Backward checks 3 -> (4, 2) (valid).
        prefix = [4, 3, 8, 4] + [1] * 59

        fwd = Knight(Chromosome(prefix))
        fwd.check_moves(cycle_direction=1)
        self.assertEqual(fwd.chromosome.genes[3], 5)
        self.assertEqual(fwd.path[4], (1, 3))

        bwd = Knight(Chromosome(prefix))
        bwd.check_moves(cycle_direction=-1)
        self.assertEqual(bwd.chromosome.genes[3], 3)
        self.assertEqual(bwd.path[4], (4, 2))

        # Full wrap-around on forward cycle from move 4 (5,6,7,8,1,2,3 -> 3):
        # (0,0) -4-> (1,2) -3-> (3,3) -8-> (2,1) -5-> (1,3) -8-> (0,1)
        # -3-> (2,2) -4-> (3,4) -2-> (5,3) -7-> (3,2) -8-> (2,0)
        # At (2,0): move 4 -> (3,2) visited; 5 -> (1,2) visited; 6 -> (0,1) visited;
        # 7,8,1,2 are out of bounds; 3 -> (4,1) is valid!
        wrap_genes = [4, 3, 8, 5, 8, 3, 4, 2, 7, 8, 4] + [1] * 52
        wrap_knight = Knight(Chromosome(wrap_genes))
        wrap_knight.check_moves(cycle_direction=1)
        self.assertEqual(wrap_knight.path[10], (2, 0))
        self.assertEqual(wrap_knight.chromosome.genes[10], 3)
        self.assertEqual(wrap_knight.path[11], (4, 1))

    def test_dead_end_retains_last_tested_move_and_preserves_replay_invariant(self) -> None:
        # Reach (2, 0) after visiting ALL valid neighbors (3,2), (1,2), (0,1), (4,1):
        # (0,0) -4-> (1,2) -3-> (3,3) -8-> (2,1) -5-> (1,3) -8-> (0,1)
        # -3-> (2,2) -2-> (4,1) -4-> (5,3) -7-> (3,2) -8-> (2,0)
        # Move index 10 has gene 4 (down-right). All 8 moves from (2,0) are now illegal.
        dead_end_genes = [4, 3, 8, 5, 8, 3, 2, 4, 7, 8, 4] + [3] * 52

        fwd_knight = Knight(Chromosome(dead_end_genes))
        fwd_knight.check_moves(cycle_direction=1)
        # Forward cycle from 4 tests 5,6,7,8,1,2,3 -> last tested is 3
        self.assertEqual(fwd_knight.chromosome.genes[10], 3)
        self.assertEqual(len(fwd_knight.path), 64)
        self.assertEqual(fwd_knight.path[11], (4, 1))  # (2,0) + right-down (2,1)

        # Replay invariant: replaying chromosome.genes from (0,0) reproduces path identically
        replayed = [(0, 0)]
        pos = (0, 0)
        for gene in fwd_knight.chromosome.genes:
            dx, dy = DIRECTIONS[gene]
            pos = (pos[0] + dx, pos[1] + dy)
            replayed.append(pos)
        self.assertEqual(fwd_knight.path, replayed)

        bwd_knight = Knight(Chromosome(dead_end_genes))
        bwd_knight.check_moves(cycle_direction=-1)
        # Backward cycle from 4 tests 3,2,1,8,7,6,5 -> last tested is 5
        self.assertEqual(bwd_knight.chromosome.genes[10], 5)
        self.assertEqual(bwd_knight.path[11], (1, 2))  # (2,0) + down-left (-1,2)

    def test_check_moves_is_idempotent(self) -> None:
        knight = Knight()
        knight.check_moves(cycle_direction=1)
        first_fitness = knight.evaluate_fitness()
        first_valid_prefix = list(knight.path[:first_fitness])
        self.assertEqual(len(knight.path), 64)

        # Calling check_moves() again resets from (0, 0) instead of appending (length remains 64).
        knight.check_moves(cycle_direction=1)
        second_fitness = knight.evaluate_fitness()
        self.assertEqual(len(knight.path), 64)
        self.assertEqual(second_fitness, first_fitness)
        self.assertEqual(knight.path[:second_fitness], first_valid_prefix)

        # On a complete tour, calling check_moves() twice leaves genes and path identical.
        tour_moves = _make_valid_tour_moves()
        tour_knight = Knight(Chromosome(tour_moves))
        tour_knight.check_moves(cycle_direction=1)
        self.assertEqual(tour_knight.evaluate_fitness(), 64)
        expected_path = list(tour_knight.path)
        tour_knight.check_moves(cycle_direction=-1)
        self.assertEqual(tour_knight.chromosome.genes, tour_moves)
        self.assertEqual(tour_knight.path, expected_path)

    def test_evaluate_fitness_valid_tour_and_stops_on_first_invalid_move(self) -> None:
        valid_path = _make_valid_tour_path()
        self.assertEqual(len(valid_path), 64)
        self.assertEqual(len(set(valid_path)), 64)

        knight = Knight()
        knight.path = valid_path
        fitness = knight.evaluate_fitness()
        self.assertEqual(fitness, 64)
        self.assertEqual(knight.fitness, 64)

        # Duplicate square at index 3 followed by valid unvisited squares must stop at 3
        knight_dup = Knight()
        knight_dup.path = [(0, 0), (2, 1), (3, 3), (0, 0), (4, 4), (5, 5)]
        self.assertEqual(knight_dup.evaluate_fitness(), 3)
        self.assertEqual(knight_dup.fitness, 3)

        # Out-of-bounds square at index 2 followed by valid unvisited squares must stop at 2
        knight_oob = Knight()
        knight_oob.path = [(0, 0), (2, 1), (4, -1), (5, 5)]
        self.assertEqual(knight_oob.evaluate_fitness(), 2)
        self.assertEqual(knight_oob.fitness, 2)


if __name__ == "__main__":
    unittest.main()
