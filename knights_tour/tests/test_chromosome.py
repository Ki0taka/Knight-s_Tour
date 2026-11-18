"""Unit tests for the Chromosome class."""

import random
import unittest

from knights_tour.chromosome import CHROMOSOME_LENGTH, Chromosome


class TestChromosome(unittest.TestCase):
    """Verify Chromosome initialization, single-point crossover, and mutation."""

    def test_default_initialization_generates_63_valid_genes(self) -> None:
        random.seed(123)
        chrom = Chromosome()
        self.assertEqual(len(chrom.genes), CHROMOSOME_LENGTH)
        self.assertEqual(len(chrom.genes), 63)
        self.assertTrue(all(isinstance(g, int) and not isinstance(g, bool) and 1 <= g <= 8 for g in chrom.genes))

    def test_custom_genes_are_copied_and_validated(self) -> None:
        source = [1] * 63
        chrom = Chromosome(source)
        source[0] = 8
        self.assertEqual(chrom.genes[0], 1)

        with self.assertRaises(ValueError):
            Chromosome([1] * 62)
        with self.assertRaises(ValueError):
            Chromosome([1] * 64)
        with self.assertRaises(ValueError):
            Chromosome([0] + [1] * 62)
        with self.assertRaises(ValueError):
            Chromosome([9] + [1] * 62)
        with self.assertRaises(ValueError):
            Chromosome([True] + [1] * 62)

    def test_single_point_crossover_rate_one(self) -> None:
        random.seed(7)
        parent1 = Chromosome([1] * 63)
        parent2 = Chromosome([8] * 63)

        offspring1, offspring2 = parent1.crossover(parent2, crossover_rate=1.0)

        self.assertIsInstance(offspring1, Chromosome)
        self.assertIsInstance(offspring2, Chromosome)
        self.assertIsNot(offspring1, parent1)
        self.assertIsNot(offspring2, parent2)
        self.assertEqual(len(offspring1.genes), 63)
        self.assertEqual(len(offspring2.genes), 63)

        # Find cut point k in 1..62
        self.assertIn(8, offspring1.genes)
        self.assertIn(1, offspring1.genes)
        cut = offspring1.genes.index(8)
        self.assertGreaterEqual(cut, 1)
        self.assertLessEqual(cut, 62)
        self.assertEqual(offspring1.genes[:cut], parent1.genes[:cut])
        self.assertEqual(offspring1.genes[cut:], parent2.genes[cut:])
        self.assertEqual(offspring2.genes[:cut], parent2.genes[:cut])
        self.assertEqual(offspring2.genes[cut:], parent1.genes[cut:])

    def test_crossover_rate_zero_returns_independent_copies(self) -> None:
        parent1 = Chromosome([2] * 63)
        parent2 = Chromosome([5] * 63)

        offspring1, offspring2 = parent1.crossover(parent2, crossover_rate=0.0)

        self.assertEqual(offspring1.genes, parent1.genes)
        self.assertEqual(offspring2.genes, parent2.genes)
        self.assertIsNot(offspring1, parent1)
        self.assertIsNot(offspring2, parent2)
        offspring1.genes[0] = 7
        self.assertEqual(parent1.genes[0], 2)

    def test_mutation_behavior_and_validation(self) -> None:
        chrom = Chromosome([3] * 63)
        chrom.mutation(mutation_rate=0.0)
        self.assertEqual(chrom.genes, [3] * 63)

        random.seed(99)
        chrom.mutation(mutation_rate=1.0)
        self.assertEqual(len(chrom.genes), 63)
        self.assertTrue(all(1 <= g <= 8 for g in chrom.genes))
        self.assertNotEqual(chrom.genes, [3] * 63)

        with self.assertRaises(ValueError):
            chrom.mutation(mutation_rate=-0.1)
        with self.assertRaises(ValueError):
            chrom.mutation(mutation_rate=1.1)
        with self.assertRaises(ValueError):
            chrom.crossover(Chromosome(), crossover_rate=-0.1)
        with self.assertRaises(ValueError):
            chrom.crossover(Chromosome(), crossover_rate=1.1)


if __name__ == "__main__":
    unittest.main()
