"""Knight agent state, move validation/repair, and fitness evaluation."""

from __future__ import annotations

import random

from .chromosome import Chromosome

BOARD_SIZE = 8
INITIAL_POSITION: tuple[int, int] = (0, 0)

DIRECTIONS: dict[int, tuple[int, int]] = {
    1: (1, -2),   # up-right
    2: (2, -1),   # right-up
    3: (2, 1),    # right-down
    4: (1, 2),    # down-right
    5: (-1, 2),   # down-left
    6: (-2, 1),   # left-down
    7: (-2, -1),  # left-up
    8: (-1, -2),  # up-left
}

DIRECTION_NAMES: dict[int, str] = {
    1: "1: up-right",
    2: "2: right-up",
    3: "3: right-down",
    4: "4: down-right",
    5: "5: down-left",
    6: "6: left-down",
    7: "7: left-up",
    8: "8: up-left",
}


def _validate_direction(direction: int) -> tuple[int, int]:
    """Return the (dx, dy) delta for a move direction in 1..8."""
    if isinstance(direction, bool) or direction not in DIRECTIONS:
        raise ValueError(f"Direction must be an integer in 1..8, got {direction!r}")
    return DIRECTIONS[direction]


def _is_on_board(position: tuple[int, int]) -> bool:
    """Return True if ``position`` lies within the 8x8 chessboard."""
    x, y = position
    return 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE


class Knight:
    """A knight traversing the 8x8 board according to a 63-move Chromosome."""

    def __init__(self, chromosome: Chromosome | None = None) -> None:
        """Create a knight at (0, 0) with the given or a newly generated chromosome."""
        self.position: tuple[int, int] = INITIAL_POSITION
        self.chromosome: Chromosome = (
            chromosome if chromosome is not None else Chromosome()
        )
        self.path: list[tuple[int, int]] = [INITIAL_POSITION]
        self.fitness: int = 0

    def move_forward(self, direction: int) -> tuple[int, int]:
        """Move the knight one L-step in ``direction`` (1..8) and return new position."""
        dx, dy = _validate_direction(direction)
        self.position = (self.position[0] + dx, self.position[1] + dy)
        return self.position

    def move_backward(self, direction: int) -> tuple[int, int]:
        """Undo one L-step in ``direction`` (1..8) and return the restored position."""
        dx, dy = _validate_direction(direction)
        self.position = (self.position[0] - dx, self.position[1] - dy)
        return self.position

    def check_moves(self, cycle_direction: int | None = None) -> None:
        """Trace and repair the chromosome's moves from (0, 0) using cyclic fallback."""
        if cycle_direction not in (None, 1, -1):
            raise ValueError(
                f"cycle_direction must be None, 1, or -1, got {cycle_direction!r}"
            )

        step = cycle_direction if cycle_direction is not None else random.choice((1, -1))
        self.position = INITIAL_POSITION
        self.path = [INITIAL_POSITION]
        visited: set[tuple[int, int]] = {INITIAL_POSITION}

        for index, move in enumerate(self.chromosome.genes):
            self.move_forward(move)
            if _is_on_board(self.position) and self.position not in visited:
                self.path.append(self.position)
                visited.add(self.position)
                continue

            self.move_backward(move)
            for offset in range(1, 8):
                candidate = ((move - 1 + offset * step) % 8) + 1
                self.move_forward(candidate)
                if _is_on_board(self.position) and self.position not in visited:
                    self.chromosome.genes[index] = candidate
                    self.path.append(self.position)
                    visited.add(self.position)
                    break
                if offset < 7:
                    self.move_backward(candidate)
                else:
                    # Dead end: retain the last tested fallback move and keep path/genes aligned.
                    self.chromosome.genes[index] = candidate
                    self.path.append(self.position)

    def evaluate_fitness(self) -> int:
        """Count consecutive valid, unvisited squares along ``path`` starting from (0, 0)."""
        visited: set[tuple[int, int]] = set()
        score = 0
        for square in self.path:
            if not _is_on_board(square) or square in visited:
                break
            visited.add(square)
            score += 1
        self.fitness = score
        return self.fitness

    def __repr__(self) -> str:
        """Return a concise representation of the knight."""
        return f"Knight(pos={self.position}, fitness={self.fitness}, path_len={len(self.path)})"
