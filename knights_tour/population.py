"""Population management, tournament selection, and generation replacement."""

from __future__ import annotations

import random

from .chromosome import Chromosome, _validate_probability
from .knight import Knight


class Population:
    """Population of Knights evolved across generations via tournament selection."""

    def __init__(
        self,
        population_size: int = 50,
        mutation_rate: float = 0.01,
        crossover_rate: float = 1.0,
        tournament_size: int = 3,
        elitism: bool = False,
    ) -> None:
        """Create the initial population of ``population_size`` knights at generation 1."""
        if isinstance(population_size, bool) or not isinstance(population_size, int) or population_size < 2:
            raise ValueError(f"population_size must be an integer >= 2, got {population_size!r}")
        if isinstance(tournament_size, bool) or not isinstance(tournament_size, int) or tournament_size < 2:
            raise ValueError(f"tournament_size must be an integer >= 2, got {tournament_size!r}")
        if tournament_size > population_size:
            raise ValueError(
                f"tournament_size ({tournament_size}) cannot exceed population_size ({population_size})"
            )
        _validate_probability(mutation_rate, "mutation_rate")
        _validate_probability(crossover_rate, "crossover_rate")

        self.population_size: int = population_size
        self.mutation_rate: float = mutation_rate
        self.crossover_rate: float = crossover_rate
        self.tournament_size: int = tournament_size
        self.elitism: bool = elitism
        self.generation: int = 1
        self.knights: list[Knight] = [Knight() for _ in range(population_size)]

    def check_population(self) -> None:
        """Validate and repair the move sequence of every knight in the population."""
        for knight in self.knights:
            knight.check_moves()

    def evaluate(self) -> tuple[int, Knight]:
        """Evaluate every knight's fitness and return ``(max_fitness, best_knight)``."""
        for knight in self.knights:
            knight.evaluate_fitness()
        best_knight = max(self.knights, key=lambda knight: knight.fitness)
        return best_knight.fitness, best_knight

    def tournament_selection(self, size: int = 3) -> tuple[Knight, Knight]:
        """Sample ``size`` knights randomly and return the two with highest fitness."""
        if isinstance(size, bool) or not isinstance(size, int) or size < 2:
            raise ValueError(f"Tournament size must be an integer >= 2, got {size!r}")
        sample_size = min(size, len(self.knights))
        contenders = random.sample(self.knights, sample_size)
        contenders.sort(key=lambda knight: knight.fitness, reverse=True)
        return contenders[0], contenders[1]

    def create_new_generation(self) -> None:
        """Build the next generation via tournament selection, crossover, and mutation."""
        next_knights: list[Knight] = []
        if self.elitism:
            best_knight = max(self.knights, key=lambda knight: knight.fitness)
            next_knights.append(Knight(Chromosome(best_knight.chromosome.genes)))

        while len(next_knights) < self.population_size:
            parent1, parent2 = self.tournament_selection(self.tournament_size)
            offspring1, offspring2 = parent1.chromosome.crossover(
                parent2.chromosome, crossover_rate=self.crossover_rate
            )
            offspring1.mutation(mutation_rate=self.mutation_rate)
            next_knights.append(Knight(offspring1))
            if len(next_knights) < self.population_size:
                offspring2.mutation(mutation_rate=self.mutation_rate)
                next_knights.append(Knight(offspring2))

        self.knights = next_knights
        self.generation += 1
