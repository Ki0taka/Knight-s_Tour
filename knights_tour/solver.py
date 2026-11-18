"""Configurable GA runner, parameter presets, and ASCII board formatting."""

from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass
from typing import Callable, Sequence

from .knight import BOARD_SIZE
from .population import Population

TARGET_FITNESS = BOARD_SIZE * BOARD_SIZE


@dataclass(frozen=True)
class GARunConfig:
    """Hyperparameters and stopping criteria for a Knight's Tour GA run."""

    population_size: int = 50
    mutation_rate: float = 0.01
    crossover_rate: float = 1.0
    tournament_size: int = 3
    elitism: bool = False
    max_generations: int = 5000
    seed: int | None = None


@dataclass(frozen=True)
class GARunResult:
    """Outcome and metrics collected during a Genetic Algorithm run."""

    found: bool
    best_fitness: int
    best_path: tuple[tuple[int, int], ...]
    best_genes: tuple[int, ...]
    generations: int
    elapsed_seconds: float
    fitness_history: tuple[int, ...]
    config: GARunConfig
    stopped: bool = False


PRESETS: dict[str, GARunConfig] = {
    "PDF Default (pop=50, mut=0.01)": GARunConfig(
        population_size=50, mutation_rate=0.01, elitism=False, max_generations=2000
    ),
    "Ch.2 Low Mut (pop=50, mut=0.001)": GARunConfig(
        population_size=50, mutation_rate=0.001, elitism=False, max_generations=2000
    ),
    "High Mut (pop=50, mut=0.05)": GARunConfig(
        population_size=50, mutation_rate=0.05, elitism=False, max_generations=2000
    ),
    "Elitist (pop=50, mut=0.01, elit=On)": GARunConfig(
        population_size=50, mutation_rate=0.01, elitism=True, max_generations=2000
    ),
    "Large Pop (pop=100, mut=0.01)": GARunConfig(
        population_size=100, mutation_rate=0.01, elitism=False, max_generations=2000
    ),
}


def run_genetic_algorithm(
    config: GARunConfig = GARunConfig(),
    on_generation: Callable[[int, int, tuple[tuple[int, int], ...]], None] | None = None,
    stop_event: threading.Event | None = None,
) -> GARunResult:
    """Execute the Knight's Tour GA loop and return the recorded result and history."""
    if config.max_generations < 1:
        raise ValueError(f"max_generations must be >= 1, got {config.max_generations}")

    saved_rng_state = random.getstate() if config.seed is not None else None
    if config.seed is not None:
        random.seed(config.seed)

    start_time = time.perf_counter()
    fitness_history: list[int] = []
    best_fitness = 0
    best_path: tuple[tuple[int, int], ...] = ((0, 0),)
    best_genes: tuple[int, ...] = ()
    generations_run = 0
    found = False
    stopped = False

    try:
        population = Population(
            population_size=config.population_size,
            mutation_rate=config.mutation_rate,
            crossover_rate=config.crossover_rate,
            tournament_size=config.tournament_size,
            elitism=config.elitism,
        )

        while True:
            population.check_population()
            max_fit, best_solution = population.evaluate()
            generations_run = population.generation
            fitness_history.append(max_fit)

            valid_prefix = tuple(best_solution.path[:max_fit])
            if max_fit >= best_fitness:
                best_fitness = max_fit
                best_path = valid_prefix
                best_genes = tuple(best_solution.chromosome.genes)

            if on_generation is not None:
                on_generation(generations_run, max_fit, valid_prefix)

            if max_fit == TARGET_FITNESS:
                found = True
                break
            if stop_event is not None and stop_event.is_set():
                stopped = True
                break
            if generations_run >= config.max_generations:
                break

            population.create_new_generation()
    finally:
        if saved_rng_state is not None:
            random.setstate(saved_rng_state)

    elapsed = time.perf_counter() - start_time
    return GARunResult(
        found=found,
        best_fitness=best_fitness,
        best_path=best_path,
        best_genes=best_genes,
        generations=generations_run,
        elapsed_seconds=elapsed,
        fitness_history=tuple(fitness_history),
        config=config,
        stopped=stopped,
    )


def render_ascii_board(path: Sequence[tuple[int, int]]) -> str:
    """Format an 8x8 step-number board (``01``..``64``, ``..`` for unvisited)."""
    grid = [[".." for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    for step_num, (x, y) in enumerate(path, start=1):
        if 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE and grid[y][x] == "..":
            grid[y][x] = f"{step_num:02d}"

    header = "    " + " ".join(f"{col:2d}" for col in range(1, BOARD_SIZE + 1))
    separator = "   +" + "---" * BOARD_SIZE + "+"
    rows = [header, separator]
    for row_index, row in enumerate(grid, start=1):
        rows.append(f"{row_index:2d} | " + " ".join(row) + " |")
    rows.append(separator)
    return "\n".join(rows)
