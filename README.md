# Knight's Tour Genetic Algorithm Lab

A Python implementation of the USTHB Master 1 Visual Computing **Project 2** (`Project 2.pdf` and `Chapter 2.pdf`). It models the 8x8 Knight's Tour using an integer-encoded move `Chromosome`, repairs illegal moves per `Knight` via cyclic fallback (`check_moves`), evolves candidate tours in a `Population` using tournament selection (`k = 3`), single-point crossover, and random-reset mutation, and animates the resulting 1–64 tour in a Pygame interface styled after Figure 2.

> **Requirements:** Python 3.10 or later. The GUI requires Pygame (`pygame>=2.5.0`); the GA model, CLI runner, benchmark, and unit tests use only the Python standard library.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 main.py
```

The GUI opens on the Figure 2 cream/olive 8x8 chessboard, automatically evolves a solution from `(1, 1)` (`(0, 0)` in 0-indexed coordinates), and animates the knight across all 64 squares.

### GUI controls

| Input | Action |
| --- | --- |
| `S` | Solve / re-evolve with current parameters |
| `Space` | Play or pause step-by-step tour animation |
| `→` / `N` | Step forward one move |
| `←` / `B` | Step backward one move |
| `R` | Reset playback to step 1 at `(1, 1)` |
| `P` | Toggle knight trajectory line overlay |
| `G` | Cycle population size (`50`, `100`, `150`) |
| `M` | Cycle mutation probability (`0.001`, `0.010`, `0.020`, `0.050`) |
| `E` | Toggle optional elitism (`OFF` / `ON`) |
| `C` | Compare all 5 parameter presets in the side panel |
| `F` | Toggle between the fitness history chart and comparison table |
| `+` / `-` | Speed up or slow down playback animation |

## Console and benchmark

Run the Genetic Algorithm from the terminal (prints gene sequence and the 8x8 `01`–`64` grid):

```bash
python3 main.py --cli --seed 42
```

Customize hyperparameters from the CLI:

```bash
python3 main.py --cli --pop-size 100 --mutation-rate 0.01 --elitism --seed 1
```

Run the 10-seed preset benchmark and regenerate `results.md`:

```bash
python3 main.py --benchmark
```

## Tests

```bash
python3 -m unittest discover -s knights_tour/tests -v
```

Coverage includes chromosome initialization/validation/single-point crossover/mutation, all 8 knight move vectors (`move_forward` / `move_backward`), exact forward and backward `check_moves()` cycling order (matching `Project 2.pdf` page 2), dead-end retention and replay consistency, idempotence, fitness counting until the first invalid move, `Population.evaluate()` `(maxFit, bestSolution)` tuple order, tournament selection, generation replacement for even/odd populations, elitism, CLI execution, benchmark generation, and headless Pygame rendering.

## Project structure

```text
main.py                       Root application entry point
requirements.txt              Pygame dependency for the GUI
README.md                     Project overview, methodology notes, and benchmark analysis
results.md                    Generated benchmark table from `python3 main.py --benchmark`
knights_tour/
  __init__.py                 Public exports (`Chromosome`, `Knight`, `Population`, runner)
  chromosome.py               63-gene `Chromosome`, single-point crossover, mutation
  knight.py                   `Knight` state, `move_forward`/`backward`, `check_moves`, `evaluate_fitness`
  population.py               `Population`, `check_population`, `evaluate`, `tournament_selection`, `create_new_generation`
  solver.py                   `GARunConfig`, `GARunResult`, `PRESETS`, `run_genetic_algorithm`, ASCII board
  benchmark.py                Multi-seed preset benchmark writing `results.md`
  gui.py                      Pygame Figure 2 chessboard, live fitness curve, and parameter controls
  main.py                     CLI, benchmark, assignment loop, and GUI dispatcher
  tests/                      Unit and headless integration test suite
```

## Methodology & Chapter 2 analysis

Following Dr. Meriem Sebai's `Project 2.pdf` and `Chapter 2.pdf` (§2.3.6):

- **Representation (`Chromosome`)**: Each chromosome stores 63 integer genes in `1..8` representing the knight's consecutive L-shaped moves from `(0, 0)`:
  - `1: up-right (+1, -2)`, `2: right-up (+2, -1)`, `3: right-down (+2, +1)`, `4: down-right (+1, +2)`
  - `5: down-left (-1, +2)`, `6: left-down (-2, +1)`, `7: left-up (-2, -1)`, `8: up-left (-1, -2)`
- **Repair & Path Tracing (`Knight.check_moves`)**: Tracing from `(0, 0)`, if move `i` exits the 8x8 board or revisits a square in `visited`, `move_backward(move)` cancels it and tests the remaining 7 moves in a randomly chosen cyclic direction (`+1` forward or `-1` backward, fixed for that chromosome evaluation). If a valid move is found, the gene is repaired in place; at a dead end, the last tested move is retained so `genes` and `path` stay consistent.
- **Fitness (`Knight.evaluate_fitness`)**: Increments `fitness` from `0` while walking `path` until an out-of-bounds or repeated square is encountered, reaching `64` on a complete Knight's Tour.
- **Evolution (`Population`)**: Selects parent pairs via `tournament_selection(3)`, produces offspring pairs via single-point crossover (`crossover_rate = 1.0`), applies per-gene mutation, and increments `generation` until `maxFit == 64`.
- **Diversification vs. Intensification (`results.md`)**: Across 10 deterministic seeds (`1..10`, generation cap `2000`), measured results illustrate the Chapter 2 §2.2.6 / §2.3.6 trade-off:

| Preset | Pop | Mut | Elitism | Solved | Avg best fitness | Median gens | Median time (s) |
| --- | ---: | ---: | :---: | ---: | ---: | ---: | ---: |
| PDF Default (pop=50, mut=0.01) | 50 | 0.010 | No | 10/10 | 64.0 | 91 | 0.5482 |
| Ch.2 Low Mut (pop=50, mut=0.001) | 50 | 0.001 | No | 7/10 | 63.4 | 20 | 0.1637 |
| High Mut (pop=50, mut=0.05) | 50 | 0.050 | No | 10/10 | 64.0 | 348 | 2.5810 |
| Elitist (pop=50, mut=0.01, elit=On) | 50 | 0.010 | Yes | 9/10 | 63.9 | 45 | 0.3001 |
| Large Pop (pop=100, mut=0.01) | 100 | 0.010 | No | 10/10 | 64.0 | 43 | 0.5725 |

- At `mutation_rate = 0.001` (the Chapter 2 binary-bit example value), only ~6% of 63-gene chromosomes mutate per generation, so 3/10 seeds lose diversity and stall below 64 within 2000 generations (`avg = 63.4`).
- At `mutation_rate = 0.010`, exploration and exploitation balance cleanly (`10/10` solved, median `91` generations in `0.5482s`).
- At `mutation_rate = 0.050`, excessive random disruption slows convergence to a median of `348` generations (`2.5810s`), matching Chapter 2's warning that high mutation slows convergence.
