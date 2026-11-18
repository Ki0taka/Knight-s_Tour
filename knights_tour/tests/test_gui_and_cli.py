"""Integration tests for benchmark, CLI entry point, and headless Pygame GUI."""

import contextlib
import io
import os
from pathlib import Path
import tempfile
import unittest

from knights_tour.solver import GARunConfig, run_genetic_algorithm


class TestBenchmarkCliAndGui(unittest.TestCase):
    """Verify benchmark reporting, CLI mode, and headless Pygame rendering."""

    def test_benchmark_all_writes_markdown_report(self) -> None:
        from knights_tour.benchmark import benchmark_all

        with tempfile.TemporaryDirectory() as tmp_dir:
            report_path = Path(tmp_dir) / "results.md"
            table = benchmark_all(output_path=report_path, trials=2, max_generations=300)
            self.assertIn("| Preset |", table)
            self.assertIn("PDF Default", table)
            self.assertTrue(report_path.exists())
            content = report_path.read_text(encoding="utf-8")
            self.assertIn("# Knight's Tour GA Benchmark", content)
            self.assertIn(table, content)

    def test_cli_mode_outputs_valid_64_tour(self) -> None:
        from knights_tour.main import main, run_assignment_loop

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            main(["--cli", "--seed", "1", "--max-generations", "1000"])
        output = buffer.getvalue()
        self.assertIn("maxFit = 64 / 64", output)
        self.assertIn("01", output)
        self.assertIn("64", output)

        max_fit, best_knight, population = run_assignment_loop(population_size=50, seed=1)
        self.assertEqual(max_fit, 64)
        self.assertEqual(len(best_knight.path), 64)
        self.assertEqual(len(set(best_knight.path)), 64)
        self.assertGreaterEqual(population.generation, 1)

    def test_headless_pygame_gui_renders_board_controls_and_comparison(self) -> None:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        import pygame

        from knights_tour.gui import KnightsTourGUI

        pygame.init()
        try:
            screen = pygame.display.set_mode((1180, 760))
            gui = KnightsTourGUI()
            result = run_genetic_algorithm(
                GARunConfig(population_size=50, mutation_rate=0.01, max_generations=1000, seed=1)
            )
            self.assertTrue(result.found)
            gui.result = result
            gui.live_generation = result.generations
            gui.live_fitness = result.best_fitness
            gui.live_path = result.best_path
            gui.fitness_history = list(result.fitness_history)
            gui.step = 32
            gui.show_path = True

            # Exercise control rendering, board rendering, and info/sparkline panel
            buttons = gui._layout_buttons(pygame, 1180)
            gui._draw_controls(pygame, screen, buttons)
            panel_x = min(780, screen.get_width() - 340)
            gui._draw_board(pygame, screen, panel_x, screen.get_height())
            gui._draw_info(pygame, screen, panel_x)

            # Exercise parameter cycling, playback controls, and comparison view
            gui._activate_button("cycle_pop")
            gui._activate_button("cycle_mut")
            gui._activate_button("toggle_elitism")
            gui._activate_button("toggle_path")
            gui._activate_button("forward")
            gui._activate_button("back")
            gui._activate_button("reset")
            gui.compare_rows = [
                ("PDF Default (pop=50, mut=0.01)", True, 64, 42, 0.08),
                ("Ch.2 Low Mut (pop=50, mut=0.001)", False, 59, 500, 0.91),
            ]
            gui.show_comparison = True
            narrow_buttons = gui._layout_buttons(pygame, 860)
            narrow_surface = pygame.Surface((860, 680))
            gui._draw_controls(pygame, narrow_surface, narrow_buttons)
            gui._draw_board(pygame, narrow_surface, 520, 680)
            gui._draw_info(pygame, narrow_surface, 520)
        finally:
            pygame.quit()


if __name__ == "__main__":
    unittest.main()
