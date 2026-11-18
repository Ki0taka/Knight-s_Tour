"""Unit tests for Population and the GA runner."""

import random
import threading
import unittest

from knights_tour.chromosome import Chromosome
from knights_tour.knight import DIRECTIONS, Knight
from knights_tour.population import Population
from knights_tour.solver import GARunConfig, render_ascii_board, run_genetic_algorithm


class TestPopulationAndSolver(unittest.TestCase):
    """Verify Population operators, PDF tuple ordering, and GA convergence."""

    def test_population_initialization_and_validation(self) -> None:
        pop = Population(50)
        self.assertEqual(pop.population_size, 50)
        self.assertEqual(pop.generation, 1)
        self.assertEqual(len(pop.knights), 50)
        self.assertTrue(all(isinstance(k, Knight) for k in pop.knights))

        with self.assertRaises(ValueError):
            Population(1)
        with self.assertRaises(ValueError):
            Population(50, tournament_size=1)
        with self.assertRaises(ValueError):
            Population(3, tournament_size=4)

    def test_check_population_and_evaluate_return_order(self) -> None:
        random.seed(12)
        pop = Population(20)
        pop.check_population()
        self.assertTrue(all(len(k.path) == 64 for k in pop.knights))

        max_fit, best_solution = pop.evaluate()
        self.assertIsInstance(max_fit, int)
        self.assertIsInstance(best_solution, Knight)
        self.assertEqual(max_fit, best_solution.fitness)
        self.assertEqual(max_fit, max(k.fitness for k in pop.knights))

    def test_tournament_selection_picks_top_two_of_sample(self) -> None:
        pop = Population(3)
        scores = [12, 48, 29]
        for knight, score in zip(pop.knights, scores):
            knight.fitness = score

        # Sampling all 3 must deterministically return the fitness-48 and fitness-29 knights.
        parent1, parent2 = pop.tournament_selection(3)
        self.assertIsNot(parent1, parent2)
        self.assertEqual(parent1.fitness, 48)
        self.assertEqual(parent2.fitness, 29)

        # On a larger population, tournament_selection(3) returns 2 distinct members in order.
        pop_large = Population(50)
        pop_large.check_population()
        pop_large.evaluate()
        p1, p2 = pop_large.tournament_selection(3)
        self.assertIsNot(p1, p2)
        self.assertIn(p1, pop_large.knights)
        self.assertIn(p2, pop_large.knights)
        self.assertGreaterEqual(p1.fitness, p2.fitness)

    def test_create_new_generation_preserves_size_even_and_odd(self) -> None:
        for size in (50, 15):
            pop = Population(size, mutation_rate=0.01)
            pop.check_population()
            pop.evaluate()
            old_knights = list(pop.knights)

            pop.create_new_generation()
            self.assertEqual(pop.generation, 2)
            self.assertEqual(len(pop.knights), size)
            for knight in pop.knights:
                self.assertIsInstance(knight, Knight)
                self.assertEqual(len(knight.chromosome.genes), 63)
                self.assertNotIn(knight, old_knights)

    def test_create_new_generation_elitism_preserves_best_chromosome(self) -> None:
        random.seed(5)
        pop = Population(10, mutation_rate=0.5, elitism=True)
        pop.check_population()
        max_fit, best = pop.evaluate()
        best_genes = list(best.chromosome.genes)

        pop.create_new_generation()
        self.assertEqual(pop.generation, 2)
        self.assertEqual(len(pop.knights), 10)
        self.assertEqual(pop.knights[0].chromosome.genes, best_genes)
        self.assertIsNot(pop.knights[0].chromosome, best.chromosome)
        self.assertGreaterEqual(max_fit, 1)

    def test_run_genetic_algorithm_solves_full_64_tour(self) -> None:
        config = GARunConfig(
            population_size=50,
            mutation_rate=0.01,
            crossover_rate=1.0,
            tournament_size=3,
            elitism=False,
            max_generations=1000,
            seed=1,
        )
        seen_generations: list[int] = []

        def on_gen(gen: int, fit: int, path: tuple[tuple[int, int], ...]) -> None:
            seen_generations.append(gen)
            self.assertGreaterEqual(fit, 1)
            self.assertEqual(len(path), fit)

        result = run_genetic_algorithm(config, on_generation=on_gen)
        self.assertTrue(result.found)
        self.assertFalse(result.stopped)
        self.assertEqual(result.best_fitness, 64)
        self.assertEqual(len(result.best_path), 64)
        self.assertEqual(len(set(result.best_path)), 64)
        self.assertEqual(result.best_path[0], (0, 0))
        self.assertEqual(result.generations, len(result.fitness_history))
        self.assertEqual(seen_generations, list(range(1, result.generations + 1)))

        # Verify every transition is a legal L-shaped knight jump within 0..7
        valid_deltas = set(DIRECTIONS.values())
        for a, b in zip(result.best_path, result.best_path[1:]):
            self.assertIn((b[0] - a[0], b[1] - a[1]), valid_deltas)
            self.assertTrue(0 <= b[0] < 8 and 0 <= b[1] < 8)

        board_str = render_ascii_board(result.best_path)
        self.assertIn("01", board_str)
        self.assertIn("64", board_str)

    def test_run_genetic_algorithm_honors_stop_event(self) -> None:
        stop_event = threading.Event()
        stop_event.set()
        result = run_genetic_algorithm(
            GARunConfig(population_size=20, max_generations=500, seed=3),
            stop_event=stop_event,
        )
        self.assertTrue(result.stopped)
        self.assertEqual(result.generations, 1)


if __name__ == "__main__":
    unittest.main()
