"""Chromosome representation and genetic operators for the Knight's Tour GA."""

from __future__ import annotations

import random
from typing import Sequence

CHROMOSOME_LENGTH = 63
MIN_DIRECTION = 1
MAX_DIRECTION = 8


def _validate_probability(rate: float, name: str) -> None:
    """Validate that a probability rate lies in [0.0, 1.0]."""
    if isinstance(rate, bool) or not 0.0 <= rate <= 1.0:
        raise ValueError(f"{name} must be between 0.0 and 1.0, got {rate!r}")


class Chromosome:
    """Sequence of 63 knight moves (directions 1..8) representing one candidate tour."""

    def __init__(self, genes: Sequence[int] | None = None) -> None:
        """Create a chromosome from explicit genes or sample 63 random moves in 1..8."""
        if genes is None:
            self.genes: list[int] = [
                random.randint(MIN_DIRECTION, MAX_DIRECTION)
                for _ in range(CHROMOSOME_LENGTH)
            ]
            return

        copied = list(genes)
        if len(copied) != CHROMOSOME_LENGTH:
            raise ValueError(
                f"Chromosome must have length {CHROMOSOME_LENGTH}, got {len(copied)}"
            )
        for index, gene in enumerate(copied):
            if isinstance(gene, bool) or not isinstance(gene, int) or not (
                MIN_DIRECTION <= gene <= MAX_DIRECTION
            ):
                raise ValueError(
                    f"Gene at index {index} must be an integer in 1..8, got {gene!r}"
                )
        self.genes = copied

    def crossover(
        self, partner: Chromosome, crossover_rate: float = 1.0
    ) -> tuple[Chromosome, Chromosome]:
        """Combine genes with ``partner`` via single-point crossover."""
        _validate_probability(crossover_rate, "crossover_rate")
        if random.random() >= crossover_rate:
            return Chromosome(self.genes), Chromosome(partner.genes)

        cut_point = random.randint(1, CHROMOSOME_LENGTH - 1)
        offspring1 = Chromosome(self.genes[:cut_point] + partner.genes[cut_point:])
        offspring2 = Chromosome(partner.genes[:cut_point] + self.genes[cut_point:])
        return offspring1, offspring2

    def mutation(self, mutation_rate: float = 0.01) -> None:
        """Mutate each gene with probability ``mutation_rate`` to a random move in 1..8."""
        _validate_probability(mutation_rate, "mutation_rate")
        if mutation_rate == 0.0:
            return
        for index in range(CHROMOSOME_LENGTH):
            if random.random() < mutation_rate:
                self.genes[index] = random.randint(MIN_DIRECTION, MAX_DIRECTION)

    def __repr__(self) -> str:
        """Return a concise representation of the chromosome."""
        preview = "".join(str(g) for g in self.genes[:12])
        return f"Chromosome({preview}... len={len(self.genes)})"
