# Knight's Tour — Genetic Algorithm Solver

> Solve the classic 8×8 Knight's Tour problem using a Genetic Algorithm with **cyclic move repair** and an interactive **Pygame visualizer**.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Pygame](https://img.shields.io/badge/pygame-%3E%3D2.5.0-green.svg)](https://www.pygame.org/)
[![Tests](https://img.shields.io/badge/tests-22%20passing-brightgreen.svg)](#tests)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

The [Knight's Tour](https://en.wikipedia.org/wiki/Knight%27s_tour) is a classic combinatorial puzzle: move a chess knight across every square of an 8×8 board **exactly once**. This project solves it with a **Genetic Algorithm (GA)** that:

- Represents each candidate tour as a chromosome of 63 move-direction genes
- Repairs invalid moves in-place using a **cyclic move-direction repair** strategy
- Evolves solutions via **tournament selection**, **single-point crossover**, and **uniform mutation**
- Visualises the entire evolutionary process in real time with Pygame

---

## Features

- 🧬 **Full GA engine** — selection, crossover, mutation, elitism — built on the Python standard library (no NumPy required)
- 🔧 **Cyclic move repair** — invalid moves are healed per-chromosome each generation, keeping the search space valid
- 🎮 **Interactive Pygame GUI** — animated chessboard, live fitness chart, preset comparison panel
- ⚙️ **Runtime parameter tuning** — cycle through population sizes, mutation rates, and toggle elitism via keyboard
- 📊 **Benchmark mode** — runs 10 seeds × 5 preset configurations and writes `results.md`
- 💻 **CLI mode** — headless solver for scripting and CI
- ✅ **22 unit tests** covering all core classes

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/your-username/knights-tour-ga.git
cd knights-tour-ga

# 2. Create and activate a virtual environment (optional but recommended)
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the GUI
python3 main.py
```

> **Requirements:** Python 3.10+ · `pygame >= 2.5.0` (GUI only — the core GA has zero external dependencies)

---

## GUI Controls

| Key | Action |
|-----|--------|
| `S` | Solve (run GA until a full tour is found) |
| `Space` | Play / Pause animation |
| `N` / `→` | Step forward one move |
| `B` / `←` | Step back one move |
| `R` | Reset board |
| `P` | Toggle path overlay |
| `G` | Cycle population-size preset |
| `M` | Cycle mutation-rate preset |
| `E` | Toggle elitism on / off |
| `C` | Compare all presets side-by-side |
| `F` | Toggle fitness chart / results table |
| `+` / `-` | Increase / decrease animation speed |

---

## CLI & Benchmark

### Headless solver

```bash
python3 main.py --cli --seed 42
# → SOLVED: maxFit = 64 / 64  (generation N, time T s)
```

Additional flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--seed INT` | random | RNG seed for reproducibility |
| `--pop INT` | `50` | Population size |
| `--mut FLOAT` | `0.01` | Mutation rate |
| `--elitism` | off | Enable elitist selection |
| `--max-gen INT` | `10000` | Generation limit |

### Benchmark

```bash
python3 main.py --benchmark
# Runs 10 seeds × 5 presets → writes results.md
```

---

## Algorithm Overview

### Problem encoding

A candidate tour is encoded as a **chromosome of 63 integers**, each in `{1, …, 8}`, representing the 8 possible L-shaped knight moves (numbered clockwise). Starting from square `(0, 0)`, the knight follows these directions sequentially.

```
Move codes (clockwise from top-right):
  1=(+1,-2)  2=(+2,-1)  3=(+2,+1)  4=(+1,+2)
  5=(-1,+2)  6=(-2,+1)  7=(-2,-1)  8=(-1,-2)
```

### Cyclic move repair

When a move lands off-board or on an already-visited square, the gene is repaired by cycling forward or backward through the 8 directions until a valid move is found. The cycle direction (CW or CCW) is chosen randomly per chromosome per generation, introducing controlled diversity while keeping chromosomes syntactically valid.

### Fitness

```
fitness(chromosome) = number of distinct squares visited
```

A perfect tour scores **64**. The GA terminates as soon as `fitness == 64`.

### Evolutionary loop

```
Initialise population (size P, random chromosomes)
Repeat until solved or max_gen reached:
    Repair all chromosomes (cyclic repair)
    Evaluate fitness for each chromosome
    If elitism: carry best individual to next generation
    Select parents via tournament (k = 3)
    Apply single-point crossover
    Apply uniform mutation at rate m
```

---

## Benchmark Results

Measured over **10 seeds × 5 presets** on Python 3.15.0 / pygame-ce 2.5.8.

| Preset | Pop | Mut Rate | Elitism | Solved | Median Gens | Avg Time (s) |
|--------|-----|----------|---------|--------|-------------|--------------|
| Default | 50 | 0.01 | Off | 10/10 | 91 | 0.5482 |
| Low Mutation | 50 | 0.001 | Off | 7/10 | 20 | 0.1637 |
| High Mutation | 50 | 0.05 | Off | 10/10 | 348 | 2.5810 |
| Elitist | 50 | 0.01 | On | 9/10 | 45 | 0.3001 |
| Large Population | 100 | 0.01 | Off | 10/10 | 43 | 0.5725 |

**Key takeaways:**
- Default config reliably finds a full tour in under 1 s
- Elitism cuts median generations roughly in half vs. default
- Very low mutation hurts reliability (7/10); very high mutation works but is slow
- Doubling population size is the most effective single change for convergence speed

---

## Project Structure

```
knights-tour-ga/
├── main.py                        # Entry point (GUI / CLI / benchmark dispatch)
├── requirements.txt               # pygame>=2.5.0
├── results.md                     # Benchmark output (auto-generated)
├── PROJECT_EXPLANATION.md         # In-depth algorithm walkthrough
└── knights_tour/
    ├── __init__.py
    ├── chromosome.py              # Chromosome class (genes, repair, crossover, mutate)
    ├── knight.py                  # Knight class (position, path, fitness evaluation)
    ├── population.py              # Population class (tournament selection, evolution loop)
    ├── solver.py                  # GA orchestration and stopping criteria
    ├── benchmark.py               # Multi-seed / multi-preset benchmark runner
    ├── gui.py                     # Pygame visualiser (board, chart, controls)
    ├── main.py                    # Package-level CLI argument handling
    └── tests/
        ├── __init__.py
        ├── test_chromosome.py     # Chromosome unit tests
        ├── test_knight.py         # Knight unit tests
        ├── test_population.py     # Population unit tests
        └── test_gui_and_cli.py    # GUI smoke tests & CLI integration tests
```

---

## Tests

```bash
python3 -m unittest discover -s knights_tour/tests -v
# → Ran 22 tests in ~10s — OK
```

The test suite covers:
- Gene encoding / decoding and boundary conditions
- Cyclic move repair (on-board, off-board, revisit cases)
- Fitness calculation correctness
- Tournament selection distribution
- Crossover and mutation invariants
- CLI argument parsing and solver integration

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| GA engine | Pure Python 3.10+ (stdlib only) |
| Visualiser | [Pygame](https://www.pygame.org/) ≥ 2.5.0 |
| Tests | `unittest` (stdlib) |

---

## License

This project is released under the [MIT License](LICENSE).
