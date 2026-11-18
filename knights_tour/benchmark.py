"""Benchmark GA presets across multiple seeds and write results.md."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import statistics

from .solver import PRESETS, GARunConfig, run_genetic_algorithm


def benchmark_all(
    output_path: str | Path | None = None,
    trials: int = 10,
    max_generations: int = 2000,
) -> str:
    """Run each preset across ``trials`` seeds, save results.md, and return the table."""
    if trials < 1:
        raise ValueError(f"trials must be >= 1, got {trials}")

    rows: list[str] = []
    for preset_name, base_config in PRESETS.items():
        solved_gens: list[int] = []
        best_fitnesses: list[int] = []
        times: list[float] = []
        config = replace(base_config, max_generations=max_generations)

        for seed in range(1, trials + 1):
            result = run_genetic_algorithm(replace(config, seed=seed))
            best_fitnesses.append(result.best_fitness)
            times.append(result.elapsed_seconds)
            if result.found:
                solved_gens.append(result.generations)

        success_rate = f"{len(solved_gens)}/{trials}"
        avg_fitness = f"{statistics.mean(best_fitnesses):.1f}"
        median_gens = (
            str(int(statistics.median(solved_gens))) if solved_gens else f">{max_generations}"
        )
        median_time = f"{statistics.median(times):.4f}"
        rows.append(
            f"| {preset_name} | {config.population_size} | {config.mutation_rate:.3f} | "
            f"{'Yes' if config.elitism else 'No'} | {success_rate} | {avg_fitness} | "
            f"{median_gens} | {median_time} |"
        )

    table = "\n".join(
        [
            "| Preset | Pop | Mut | Elitism | Solved | Avg best fitness | Median gens | Median time (s) |",
            "| --- | ---: | ---: | :---: | ---: | ---: | ---: | ---: |",
            *rows,
        ]
    )
    report = (
        "# Knight's Tour GA Benchmark\n\n"
        f"Each preset was evaluated across {trials} deterministic seeds (`1..{trials}`) "
        f"with a generation cap of {max_generations} (`tournament_size=3`, "
        "`crossover_rate=1.0`). Median generations is computed over solved runs "
        f"(`>{max_generations}` when none solved).\n\n"
        + table
        + "\n"
    )
    destination = (
        Path(output_path)
        if output_path is not None
        else Path(__file__).resolve().parents[1] / "results.md"
    )
    destination.write_text(report, encoding="utf-8")
    return table


def main() -> None:
    """Run the preset benchmark and print the Markdown summary table."""
    print(benchmark_all())


if __name__ == "__main__":
    main()
