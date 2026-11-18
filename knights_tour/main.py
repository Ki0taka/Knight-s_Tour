"""CLI, benchmark, assignment loop, and GUI entry point for Project 2."""

from __future__ import annotations

import argparse
import random
from typing import Sequence

from .benchmark import benchmark_all
from .knight import Knight
from .population import Population
from .solver import GARunConfig, render_ascii_board, run_genetic_algorithm


def run_assignment_loop(
    population_size: int = 50, seed: int | None = None
) -> tuple[int, Knight, Population]:
    """Run the exact main() loop specified in Project 2.pdf page 3."""
    saved_state = random.getstate() if seed is not None else None
    if seed is not None:
        random.seed(seed)
    try:
        population = Population(population_size)
        while True:
            population.check_population()
            max_fit, best_solution = population.evaluate()
            if max_fit == 64:
                break
            population.create_new_generation()
        return max_fit, best_solution, population
    finally:
        if saved_state is not None:
            random.setstate(saved_state)


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Solve and animate the 8x8 Knight's Tour using a Genetic Algorithm"
    )
    parser.add_argument(
        "--cli",
        action="store_true",
        help="run the GA in the terminal and print the solved board",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="benchmark GA presets across 10 seeds and write results.md",
    )
    parser.add_argument(
        "--pop-size",
        type=int,
        default=50,
        help="population size (default: 50)",
    )
    parser.add_argument(
        "--mutation-rate",
        type=float,
        default=0.01,
        help="per-gene mutation probability in [0, 1] (default: 0.01)",
    )
    parser.add_argument(
        "--crossover-rate",
        type=float,
        default=1.0,
        help="single-point crossover probability in [0, 1] (default: 1.0)",
    )
    parser.add_argument(
        "--tournament-size",
        type=int,
        default=3,
        help="tournament sample size (default: 3)",
    )
    parser.add_argument(
        "--elitism",
        action="store_true",
        help="preserve the fittest chromosome across generations",
    )
    parser.add_argument(
        "--max-generations",
        type=int,
        default=5000,
        help="maximum generations before stopping (default: 5000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="optional random seed for reproducible runs",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Parse options and run benchmark, CLI solver, or Pygame GUI."""
    args = build_parser().parse_args(argv)
    if args.benchmark:
        print(benchmark_all())
        return

    config = GARunConfig(
        population_size=args.pop_size,
        mutation_rate=args.mutation_rate,
        crossover_rate=args.crossover_rate,
        tournament_size=args.tournament_size,
        elitism=args.elitism,
        max_generations=args.max_generations,
        seed=args.seed,
    )

    if args.cli:
        result = run_genetic_algorithm(config)
        status = "SOLVED" if result.found else "MAX GENERATIONS REACHED"
        print(
            f"{status}: maxFit = {result.best_fitness} / 64 | "
            f"generations = {result.generations} | "
            f"time = {result.elapsed_seconds:.4f}s"
        )
        print(
            f"Config: pop={config.population_size}, mut={config.mutation_rate}, "
            f"cx={config.crossover_rate}, tourn={config.tournament_size}, elitism={config.elitism}"
        )
        print("Genes (63 moves):", " ".join(str(g) for g in result.best_genes))
        print("\nBoard (step 01..64, 1-indexed row/col matching Figure 2):")
        print(render_ascii_board(result.best_path))
        return

    from .gui import KnightsTourGUI

    KnightsTourGUI(initial_config=config).run()


if __name__ == "__main__":
    main()
