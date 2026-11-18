"""Pygame interactive visualizer for the Knight's Tour Genetic Algorithm."""

from __future__ import annotations

from dataclasses import dataclass, replace
import queue
import sys
import threading

from .knight import BOARD_SIZE, DIRECTION_NAMES, DIRECTIONS
from .solver import PRESETS, TARGET_FITNESS, GARunConfig, GARunResult, run_genetic_algorithm

POPULATION_OPTIONS: tuple[int, ...] = (50, 100, 150)
MUTATION_OPTIONS: tuple[float, ...] = (0.001, 0.01, 0.02, 0.05)

# Figure 2 green & cream palette for the 8x8 chessboard.
SQUARE_DARK = (118, 150, 86)
SQUARE_LIGHT = (238, 238, 210)
VISITED_DARK = (170, 194, 96)
VISITED_LIGHT = (226, 232, 156)
CURRENT_HIGHLIGHT = (245, 196, 81)


@dataclass(frozen=True)
class UIButton:
    """A clickable button's rectangle, label, and action identifier."""

    rect: object
    label: str
    action: str
    selected: bool = False
    primary: bool = False


class KnightsTourGUI:
    """Interactive Pygame lab for evolving and animating Knight's Tour solutions."""

    def __init__(self, initial_config: GARunConfig | None = None) -> None:
        """Initialize GA parameters, playback state, and background worker queue."""
        config = initial_config or GARunConfig()
        self.population_size: int = config.population_size
        self.mutation_rate: float = config.mutation_rate
        self.crossover_rate: float = config.crossover_rate
        self.tournament_size: int = config.tournament_size
        self.elitism: bool = config.elitism
        self.max_generations: int = config.max_generations
        self.seed: int | None = config.seed

        self.result: GARunResult | None = None
        self.live_generation: int = 0
        self.live_fitness: int = 0
        self.live_path: tuple[tuple[int, int], ...] = ((0, 0),)
        self.fitness_history: list[int] = []
        self.compare_rows: list[tuple[str, bool, int, int, float]] = []
        self.show_comparison: bool = False

        self.solving: bool = False
        self.comparing: bool = False
        self.message: str = "Initializing Genetic Algorithm..."
        self.playing: bool = False
        self.step: int = 0
        self.delay_ms: int = 180
        self.last_step_time: int = 0
        self.show_path: bool = False

        self._queue: queue.Queue[tuple[str, int, object]] = queue.Queue()
        self._run_id: int = 0
        self._stop_event: threading.Event | None = None
        self._font_cache: dict[tuple[int, bool], object] = {}
        self._freetype_font_keys: set[tuple[int, bool]] = set()

    def _current_config(self) -> GARunConfig:
        """Build a GARunConfig reflecting the currently selected UI parameters."""
        return GARunConfig(
            population_size=self.population_size,
            mutation_rate=self.mutation_rate,
            crossover_rate=self.crossover_rate,
            tournament_size=self.tournament_size,
            elitism=self.elitism,
            max_generations=self.max_generations,
            seed=self.seed,
        )

    def _active_path(self) -> tuple[tuple[int, int], ...]:
        """Return the active tour path currently displayed on the chessboard."""
        if self.result is not None and self.result.best_path:
            return self.result.best_path
        if self.live_path:
            return self.live_path
        return ((0, 0),)

    def _start_solve(self) -> None:
        """Launch a background GA run with the current hyperparameters."""
        self._cancel_worker()
        self._run_id += 1
        run_id = self._run_id
        stop_event = threading.Event()
        self._stop_event = stop_event

        self.solving = True
        self.comparing = False
        self.show_comparison = False
        self.playing = False
        self.step = 0
        self.result = None
        self.live_generation = 0
        self.live_fitness = 1
        self.live_path = ((0, 0),)
        self.fitness_history = []
        self.message = "Evolving population..."

        config = self._current_config()

        def worker() -> None:
            last_best = 0

            def on_gen(gen: int, fit: int, path: tuple[tuple[int, int], ...]) -> None:
                nonlocal last_best
                if stop_event.is_set():
                    return
                if gen <= 15 or gen % 10 == 0 or fit > last_best or fit == TARGET_FITNESS:
                    last_best = max(last_best, fit)
                    self._queue.put(("progress", run_id, (gen, fit, path)))

            try:
                res = run_genetic_algorithm(
                    config, on_generation=on_gen, stop_event=stop_event
                )
                self._queue.put(("done", run_id, res))
            except Exception as exc:  # pragma: no cover - defensive worker guard
                self._queue.put(("error", run_id, f"{type(exc).__name__}: {exc}"))

        threading.Thread(target=worker, daemon=True).start()

    def _start_compare(self) -> None:
        """Benchmark all presets once in the background and show the comparison table."""
        self._cancel_worker()
        self._run_id += 1
        run_id = self._run_id
        stop_event = threading.Event()
        self._stop_event = stop_event

        self.solving = False
        self.comparing = True
        self.show_comparison = True
        self.compare_rows = []
        self.message = "Comparing GA presets..."

        def worker() -> None:
            for name, preset_cfg in PRESETS.items():
                if stop_event.is_set():
                    break
                cfg = replace(preset_cfg, seed=1)
                res = run_genetic_algorithm(cfg, stop_event=stop_event)
                row = (
                    name,
                    res.found,
                    res.best_fitness,
                    res.generations,
                    res.elapsed_seconds,
                )
                self._queue.put(("compare_row", run_id, row))
            self._queue.put(("compare_done", run_id, None))

        threading.Thread(target=worker, daemon=True).start()

    def _cancel_worker(self) -> None:
        """Signal any running background worker to stop."""
        if self._stop_event is not None:
            self._stop_event.set()
        self._stop_event = None

    def _drain_queue(self, now_ms: int) -> None:
        """Process progress, completion, and comparison updates from worker threads."""
        while True:
            try:
                kind, run_id, payload = self._queue.get_nowait()
            except queue.Empty:
                break
            if run_id != self._run_id:
                continue

            if kind == "progress":
                gen, fit, path = payload  # type: ignore[misc]
                self.live_generation = gen
                if fit >= self.live_fitness:
                    self.live_fitness = fit
                    self.live_path = path
                self.fitness_history.append(fit)
            elif kind == "done":
                res: GARunResult = payload  # type: ignore[assignment]
                self.solving = False
                self.result = res
                self.live_generation = res.generations
                self.live_fitness = res.best_fitness
                self.live_path = res.best_path
                self.fitness_history = list(res.fitness_history)
                self.step = 0
                self.playing = len(res.best_path) > 1
                self.last_step_time = now_ms
                if res.found:
                    self.message = (
                        f"Tour found in {res.generations} gen ({res.elapsed_seconds:.3f}s)"
                    )
                else:
                    self.message = (
                        f"Best {res.best_fitness}/64 at gen {res.generations}"
                    )
            elif kind == "error":
                self.solving = False
                self.message = str(payload)
            elif kind == "compare_row":
                self.compare_rows.append(payload)  # type: ignore[arg-type]
            elif kind == "compare_done":
                self.comparing = False
                self.message = "Preset comparison complete."

    def run(self) -> None:
        """Open the interactive Pygame window and start the event loop."""
        try:
            import pygame
        except ImportError as error:
            raise RuntimeError(
                "Pygame is required for the GUI. Install it with: pip install pygame"
            ) from error

        pygame.init()
        screen = pygame.display.set_mode((1180, 760), pygame.RESIZABLE)
        pygame.display.set_caption("Knight's Tour — Genetic Algorithm Lab")
        clock = pygame.time.Clock()
        self._start_solve()

        running = True
        while running:
            now = pygame.time.get_ticks()
            buttons = self._layout_buttons(pygame, screen.get_width())

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.VIDEORESIZE:
                    screen = pygame.display.set_mode(event.size, pygame.RESIZABLE)
                elif event.type == pygame.KEYDOWN:
                    self._handle_key(pygame, event.key)
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    for button in buttons:
                        if button.rect.collidepoint(event.pos):
                            self._activate_button(button.action)
                            break

            self._drain_queue(now)

            active_path = self._active_path()
            if (
                self.playing
                and len(active_path) > 1
                and now - self.last_step_time >= self.delay_ms
            ):
                if self.step < len(active_path) - 1:
                    self.step += 1
                    self.last_step_time = now
                else:
                    self.playing = False

            screen.fill((17, 24, 39))
            self._draw_controls(pygame, screen, buttons)
            panel_x = min(780, screen.get_width() - 340)
            self._draw_board(pygame, screen, panel_x, screen.get_height())
            self._draw_info(pygame, screen, panel_x)
            pygame.display.flip()
            clock.tick(60)

        self._cancel_worker()
        pygame.quit()

    def _handle_key(self, pygame: object, key: int) -> None:
        """Dispatch keyboard shortcuts for solving, playback, and parameters."""
        if key == pygame.K_s:
            self._activate_button("solve")
        elif key == pygame.K_SPACE:
            self._activate_button("play")
        elif key in (pygame.K_RIGHT, pygame.K_n):
            self._activate_button("forward")
        elif key in (pygame.K_LEFT, pygame.K_b):
            self._activate_button("back")
        elif key == pygame.K_r:
            self._activate_button("reset")
        elif key == pygame.K_p:
            self._activate_button("toggle_path")
        elif key == pygame.K_c:
            self._activate_button("compare")
        elif key == pygame.K_g:
            self._activate_button("cycle_pop")
        elif key == pygame.K_m:
            self._activate_button("cycle_mut")
        elif key == pygame.K_e:
            self._activate_button("toggle_elitism")
        elif key == pygame.K_f and self.compare_rows:
            self.show_comparison = not self.show_comparison
        elif key in (pygame.K_PLUS, pygame.K_EQUALS):
            self._activate_button("faster")
        elif key in (pygame.K_MINUS, pygame.K_UNDERSCORE):
            self._activate_button("slower")

    def _layout_buttons(self, pygame: object, width: int) -> list[UIButton]:
        """Lay out parameter and playback buttons scaled to the window width."""
        scale = min(1.0, width / 1180) if width < 960 else 1.0

        def sx(x: int) -> int:
            return int(x * scale)

        def sw(w: int) -> int:
            return max(34, int(w * scale))

        buttons = [
            UIButton(
                pygame.Rect(sx(20), 67, sw(116), 36),
                f"POP  {self.population_size}  G",
                "cycle_pop",
                selected=True,
            ),
            UIButton(
                pygame.Rect(sx(144), 67, sw(134), 36),
                f"MUT  {self.mutation_rate:.3f}  M",
                "cycle_mut",
                selected=True,
            ),
            UIButton(
                pygame.Rect(sx(286), 67, sw(138), 36),
                f"ELITISM {'ON' if self.elitism else 'OFF'}  E",
                "toggle_elitism",
                selected=self.elitism,
            ),
            UIButton(
                pygame.Rect(sx(432), 67, sw(132), 36),
                "COMPARING..." if self.comparing else "COMPARE  C",
                "compare",
                selected=self.show_comparison,
            ),
            UIButton(
                pygame.Rect(sx(20), 116, sw(100), 42),
                "EVOLVING..." if self.solving else "SOLVE  S",
                "solve",
                primary=True,
            ),
            UIButton(
                pygame.Rect(sx(128), 116, sw(112), 42),
                "PAUSE  SPACE" if self.playing else "PLAY  SPACE",
                "play",
                selected=self.playing,
            ),
            UIButton(pygame.Rect(sx(248), 116, sw(80), 42), "STEP  N", "forward"),
            UIButton(pygame.Rect(sx(336), 116, sw(80), 42), "BACK  B", "back"),
            UIButton(pygame.Rect(sx(424), 116, sw(82), 42), "RESET  R", "reset"),
            UIButton(
                pygame.Rect(sx(514), 116, sw(82), 42),
                "PATH  P",
                "toggle_path",
                selected=self.show_path,
            ),
            UIButton(pygame.Rect(sx(734), 119, sw(34), 36), "-", "slower"),
            UIButton(pygame.Rect(sx(774), 119, sw(34), 36), "+", "faster"),
        ]
        return buttons

    def _activate_button(self, action: str) -> None:
        """Execute the action associated with a button or keyboard shortcut."""
        active_len = len(self._active_path())
        if action == "solve":
            self._start_solve()
        elif action == "compare":
            self._start_compare()
        elif action == "play":
            if active_len > 1:
                if not self.playing and self.step >= active_len - 1:
                    self.step = 0
                self.playing = not self.playing
                self.last_step_time = 0
        elif action == "forward":
            self.playing = False
            self.step = min(max(0, active_len - 1), self.step + 1)
        elif action == "back":
            self.playing = False
            self.step = max(0, self.step - 1)
        elif action == "reset":
            self.playing = False
            self.step = 0
        elif action == "toggle_path":
            self.show_path = not self.show_path
        elif action == "cycle_pop":
            if self.population_size in POPULATION_OPTIONS:
                idx = (POPULATION_OPTIONS.index(self.population_size) + 1) % len(
                    POPULATION_OPTIONS
                )
                self.population_size = POPULATION_OPTIONS[idx]
            else:
                self.population_size = POPULATION_OPTIONS[0]
        elif action == "cycle_mut":
            if self.mutation_rate in MUTATION_OPTIONS:
                idx = (MUTATION_OPTIONS.index(self.mutation_rate) + 1) % len(
                    MUTATION_OPTIONS
                )
                self.mutation_rate = MUTATION_OPTIONS[idx]
            else:
                self.mutation_rate = MUTATION_OPTIONS[0]
        elif action == "toggle_elitism":
            self.elitism = not self.elitism
        elif action == "faster":
            self.delay_ms = max(40, self.delay_ms - 40)
        elif action == "slower":
            self.delay_ms = min(1000, self.delay_ms + 40)

    def _draw_controls(
        self, pygame: object, screen: object, buttons: list[UIButton]
    ) -> None:
        """Draw the header bar and interactive parameter/playback buttons."""
        width = screen.get_width()
        screen.blit(
            self._render_text(pygame, "KNIGHT'S TOUR", 26, (245, 247, 250), bold=True),
            (22, 14),
        )
        subtitle = (
            f"GENETIC ALGORITHM LAB  /  POP {self.population_size}  "
            f"|  MUT {self.mutation_rate:.3f}  |  ELITISM {'ON' if self.elitism else 'OFF'}"
        )
        screen.blit(self._render_text(pygame, subtitle, 13, (155, 171, 191)), (24, 44))
        right_label = self._render_text(
            pygame, "MASTER 1  -  VISUAL COMPUTING", 13, (155, 171, 191)
        )
        screen.blit(right_label, (width - right_label.get_width() - 24, 24))

        for button in buttons:
            x, y, bw, bh = button.rect
            fill = (
                (50, 115, 96)
                if button.primary
                else ((49, 77, 110) if button.selected else (34, 45, 63))
            )
            if button.primary and self.solving:
                fill = (122, 86, 38)
            edge = (
                (84, 174, 145)
                if button.primary or button.selected
                else (59, 73, 94)
            )
            pygame.draw.rect(screen, (10, 16, 27), button.rect.move(0, 3), border_radius=8)
            pygame.draw.rect(screen, fill, button.rect, border_radius=8)
            pygame.draw.rect(screen, edge, button.rect, 1, border_radius=8)
            text_color = (
                (246, 250, 248)
                if button.primary or button.selected
                else (202, 211, 222)
            )
            label_surface = self._render_text(pygame, button.label, 14, text_color)
            screen.blit(
                label_surface,
                label_surface.get_rect(center=(x + bw // 2, y + bh // 2)),
            )

        speed_label = self._render_text(
            pygame, f"SPEED {self.delay_ms}ms", 12, (155, 171, 191)
        )
        if len(buttons) >= 12:
            path_rect = buttons[-3].rect
            slower_rect = buttons[-2].rect
            gap_mid = (path_rect.right + slower_rect.left) // 2
            screen.blit(speed_label, speed_label.get_rect(center=(gap_mid, 137)))

    def _draw_board(
        self, pygame: object, screen: object, panel_x: int, screen_height: int
    ) -> None:
        """Draw the Figure 2 green/cream 8x8 chessboard, coordinates, step numbers, and knight."""
        card = pygame.Rect(20, 176, panel_x - 34, screen_height - 196)
        pygame.draw.rect(screen, (10, 16, 27), card.move(0, 5), border_radius=18)
        pygame.draw.rect(screen, (27, 37, 54), card, border_radius=18)
        pygame.draw.rect(screen, (48, 63, 85), card, 1, border_radius=18)

        screen.blit(
            self._render_text(pygame, "CHESSBOARD  (FIGURE 2)", 15, (226, 232, 240), bold=True),
            (card.x + 20, card.y + 16),
        )
        screen.blit(
            self._render_text(pygame, "START (1, 1)  -  8x8 TOUR", 12, (152, 166, 184)),
            (card.x + 20, card.y + 39),
        )

        frame_outer = pygame.Rect(card.x + 20, card.y + 64, card.width - 40, card.height - 84)
        pygame.draw.rect(screen, (242, 243, 235), frame_outer, border_radius=12)
        pygame.draw.rect(screen, (62, 76, 94), frame_outer, 1, border_radius=12)

        margin = max(24, min(36, min(frame_outer.width, frame_outer.height) // 16))
        avail = min(frame_outer.width - 2 * margin, frame_outer.height - 2 * margin)
        tile = max(20, avail // BOARD_SIZE)
        grid_px = tile * BOARD_SIZE
        ox = frame_outer.x + (frame_outer.width - grid_px) // 2
        oy = frame_outer.y + (frame_outer.height - grid_px) // 2

        active_path = self._active_path()
        clamped_step = min(self.step, len(active_path) - 1)
        visited_step_at: dict[tuple[int, int], int] = {}
        for idx in range(clamped_step):
            sq = active_path[idx]
            if sq not in visited_step_at:
                visited_step_at[sq] = idx + 1
        current_pos = active_path[clamped_step]

        # Border coordinate labels 1..8 on all four sides (matching Figure 2).
        coord_color = (98, 108, 92)
        coord_size = max(11, tile // 4)
        for i in range(BOARD_SIZE):
            label = str(i + 1)
            col_surf = self._render_text(pygame, label, coord_size, coord_color, bold=True)
            cx = ox + i * tile + tile // 2
            screen.blit(col_surf, col_surf.get_rect(center=(cx, oy - margin // 2)))
            screen.blit(col_surf, col_surf.get_rect(center=(cx, oy + grid_px + margin // 2)))

            row_surf = self._render_text(pygame, label, coord_size, coord_color, bold=True)
            cy = oy + i * tile + tile // 2
            screen.blit(row_surf, row_surf.get_rect(center=(ox - margin // 2, cy)))
            screen.blit(row_surf, row_surf.get_rect(center=(ox + grid_px + margin // 2, cy)))

        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                rect = pygame.Rect(ox + col * tile, oy + row * tile, tile, tile)
                is_dark = (row + col) % 2 == 0
                pos = (col, row)
                step_num = visited_step_at.get(pos)
                if pos == current_pos:
                    fill = CURRENT_HIGHLIGHT
                elif step_num is not None:
                    fill = VISITED_DARK if is_dark else VISITED_LIGHT
                else:
                    fill = SQUARE_DARK if is_dark else SQUARE_LIGHT
                pygame.draw.rect(screen, fill, rect)

                if step_num is not None and pos != current_pos:
                    num_surf = self._render_text(
                        pygame, str(step_num), max(12, int(tile * 0.36)), (38, 52, 24), bold=True
                    )
                    screen.blit(num_surf, num_surf.get_rect(center=rect.center))

        pygame.draw.rect(screen, (72, 84, 62), (ox, oy, grid_px, grid_px), 2)

        if self.show_path and clamped_step > 0:
            points = [
                (ox + x * tile + tile // 2, oy + y * tile + tile // 2)
                for x, y in active_path[: clamped_step + 1]
            ]
            if len(points) >= 2:
                pygame.draw.lines(screen, (45, 98, 168), False, points, max(2, tile // 18))
            for pt in points[:-1]:
                pygame.draw.circle(screen, (45, 98, 168), pt, max(2, tile // 14))

        # Draw current Knight piece + subtle step badge in corner.
        kx, ky = current_pos
        if 0 <= kx < BOARD_SIZE and 0 <= ky < BOARD_SIZE:
            knight_rect = pygame.Rect(ox + kx * tile, oy + ky * tile, tile, tile)
            pygame.draw.rect(screen, (186, 128, 22), knight_rect, max(2, tile // 22))
            self._draw_knight_piece(pygame, screen, knight_rect, tile)
            badge = self._render_text(
                pygame, str(clamped_step + 1), max(10, tile // 5), (42, 34, 16), bold=True
            )
            screen.blit(badge, (knight_rect.x + 4, knight_rect.y + 3))

    @staticmethod
    def _draw_knight_piece(
        pygame: object, screen: object, rect: object, tile: int
    ) -> None:
        """Draw a crisp vector chess knight silhouette centered inside ``rect``."""
        x0, y0 = rect.x, rect.y

        def pt(nx: float, ny: float) -> tuple[int, int]:
            return (int(x0 + nx * tile), int(y0 + ny * tile))

        shadow_pts = [
            pt(0.27, 0.85), pt(0.77, 0.85), pt(0.75, 0.74), pt(0.67, 0.72),
            pt(0.69, 0.48), pt(0.65, 0.28), pt(0.55, 0.17), pt(0.49, 0.14),
            pt(0.44, 0.21), pt(0.28, 0.34), pt(0.24, 0.44), pt(0.32, 0.49),
            pt(0.45, 0.45), pt(0.39, 0.62), pt(0.30, 0.73),
        ]
        body_pts = [(px - 2, py - 2) for px, py in shadow_pts]
        pygame.draw.polygon(screen, (95, 86, 46), shadow_pts)
        pygame.draw.polygon(screen, (28, 33, 40), body_pts)
        pygame.draw.polygon(screen, (238, 236, 224), body_pts, max(1, tile // 32))

        pedestal = pygame.Rect(
            int(x0 + 0.22 * tile),
            int(y0 + 0.74 * tile),
            max(8, int(0.56 * tile)),
            max(4, int(0.11 * tile)),
        )
        pygame.draw.rect(screen, (22, 26, 32), pedestal, border_radius=3)
        pygame.draw.rect(screen, (218, 214, 198), pedestal, 1, border_radius=3)
        pygame.draw.circle(screen, (245, 212, 118), pt(0.40, 0.31), max(2, tile // 20))

    def _draw_info(self, pygame: object, screen: object, panel_x: int) -> None:
        """Draw the right-hand GA status, metrics, fitness curve, and preset table."""
        screen_width, screen_height = screen.get_size()
        card = pygame.Rect(
            panel_x + 14, 176, screen_width - panel_x - 34, screen_height - 196
        )
        pygame.draw.rect(screen, (10, 16, 27), card.move(0, 5), border_radius=18)
        pygame.draw.rect(screen, (27, 37, 54), card, border_radius=18)
        pygame.draw.rect(screen, (48, 63, 85), card, 1, border_radius=18)

        x = card.x + 18
        width = card.width - 36
        y = card.y + 18

        screen.blit(
            self._render_text(pygame, "GA EVOLUTION & STATS", 15, (226, 232, 240), bold=True),
            (x, y),
        )
        y += 30

        if self.solving:
            status, status_color, status_bg = "EVOLVING", (221, 164, 76), (77, 60, 38)
        elif self.comparing:
            status, status_color, status_bg = "COMPARING", (123, 157, 208), (39, 57, 82)
        elif self.result and self.result.found:
            status, status_color, status_bg = "TOUR FOUND (64/64)", (85, 190, 148), (35, 76, 67)
        elif self.result is not None:
            status, status_color, status_bg = "STOPPED AT CAP", (226, 112, 104), (78, 48, 48)
        else:
            status, status_color, status_bg = "READY", (123, 157, 208), (39, 57, 82)

        pill = pygame.Rect(x, y, min(width, 180), 26)
        pygame.draw.rect(screen, status_bg, pill, border_radius=13)
        pygame.draw.rect(screen, status_color, pill, 1, border_radius=13)
        screen.blit(
            self._render_text(pygame, status, 11, status_color, bold=True),
            (pill.x + 12, pill.y + 6),
        )
        y += 32

        if self.message:
            screen.blit(
                self._render_text(pygame, self.message[:42], 11, (181, 194, 211)),
                (x, y),
            )
        y += 20

        best_fit = self.result.best_fitness if self.result else self.live_fitness
        screen.blit(
            self._render_text(pygame, f"{best_fit} / 64", 34, (245, 247, 250), bold=True),
            (x, y),
        )
        screen.blit(
            self._render_text(pygame, "BEST FITNESS", 11, (152, 166, 184)),
            (x + 2, y + 40),
        )
        y += 62

        active_path = self._active_path()
        clamped_step = min(self.step, len(active_path) - 1)
        move_text = self._describe_step(active_path, clamped_step)
        action_card = pygame.Rect(x, y, width, 54)
        pygame.draw.rect(screen, (35, 49, 68), action_card, border_radius=10)
        screen.blit(
            self._render_text(pygame, "CURRENT MOVE", 10, (152, 166, 184)),
            (action_card.x + 12, action_card.y + 8),
        )
        screen.blit(
            self._render_text(pygame, move_text, 14, (245, 247, 250), bold=True),
            (action_card.x + 12, action_card.y + 26),
        )
        y += 64

        total_moves = max(1, len(active_path) - 1)
        progress_w = int(width * clamped_step / total_moves) if len(active_path) > 1 else 0
        pygame.draw.rect(screen, (50, 65, 86), (x, y, width, 6), border_radius=3)
        if progress_w > 0:
            pygame.draw.rect(screen, (85, 190, 148), (x, y, progress_w, 6), border_radius=3)
        screen.blit(
            self._render_text(
                pygame,
                f"STEP  {clamped_step + 1} / {len(active_path)}  SQUARES",
                11,
                (181, 194, 211),
            ),
            (x, y + 12),
        )
        y += 34

        elapsed_label = (
            f"{self.result.elapsed_seconds:.3f}s" if self.result is not None else "..."
        )
        gens_label = str(self.result.generations if self.result else self.live_generation)
        metrics = [
            ("GENERATIONS", gens_label),
            ("POPULATION", str(self.population_size)),
            ("MUTATION", f"{self.mutation_rate:.3f}"),
            ("SOLVE TIME", elapsed_label),
        ]
        self._draw_metric_grid(pygame, screen, x, y, width, metrics)
        y += 100

        bottom_h = card.bottom - 34 - y
        if bottom_h >= 80:
            if self.show_comparison and self.compare_rows:
                self._draw_comparison_table(pygame, screen, x, y, width, bottom_h)
            else:
                self._draw_fitness_chart(pygame, screen, x, y, width, bottom_h)

        footer = "SPACE play/pause  |  P path  |  G/M/E params"
        screen.blit(
            self._render_text(pygame, footer, 10, (120, 139, 162)),
            (x, card.bottom - 24),
        )

    @staticmethod
    def _describe_step(path: tuple[tuple[int, int], ...], step: int) -> str:
        """Format the current step's knight jump using 1..8 board coordinates."""
        if step <= 0 or step >= len(path):
            x, y = path[0]
            return f"START -> ({x + 1}, {y + 1})"
        prev_x, prev_y = path[step - 1]
        cur_x, cur_y = path[step]
        delta = (cur_x - prev_x, cur_y - prev_y)
        move_name = "move"
        for code, vec in DIRECTIONS.items():
            if vec == delta:
                move_name = DIRECTION_NAMES[code]
                break
        return f"{move_name} -> ({cur_x + 1}, {cur_y + 1})"

    def _draw_fitness_chart(
        self, pygame: object, screen: object, x: int, y: int, width: int, height: int
    ) -> None:
        """Render the generation-by-generation maxFit curve with a 64 target line."""
        box = pygame.Rect(x, y, width, height)
        pygame.draw.rect(screen, (21, 30, 44), box, border_radius=10)
        pygame.draw.rect(screen, (43, 58, 79), box, 1, border_radius=10)
        screen.blit(
            self._render_text(pygame, "FITNESS HISTORY (MAX FIT)", 10, (152, 166, 184), bold=True),
            (box.x + 10, box.y + 8),
        )
        plot = pygame.Rect(box.x + 28, box.y + 30, box.width - 42, box.height - 44)
        if plot.width < 20 or plot.height < 20:
            return

        # Horizontal reference line at target fitness = 64
        pygame.draw.line(
            screen, (72, 118, 96), (plot.left, plot.top), (plot.right, plot.top), 1
        )
        pygame.draw.line(
            screen, (52, 68, 92), (plot.left, plot.bottom), (plot.right, plot.bottom), 1
        )
        screen.blit(self._render_text(pygame, "64", 9, (124, 178, 148)), (box.x + 8, plot.top - 5))
        screen.blit(self._render_text(pygame, "0", 9, (124, 140, 162)), (box.x + 12, plot.bottom - 7))

        history = self.fitness_history
        if not history:
            return
        if len(history) == 1:
            py = plot.bottom - int((history[0] / TARGET_FITNESS) * plot.height)
            pygame.draw.circle(screen, (85, 190, 148), (plot.left + 4, py), 3)
            return

        points: list[tuple[int, int]] = []
        n = len(history)
        for idx, fit in enumerate(history):
            px = plot.left + int(idx * plot.width / (n - 1))
            py = plot.bottom - int((max(0, min(TARGET_FITNESS, fit)) / TARGET_FITNESS) * plot.height)
            points.append((px, py))
        pygame.draw.lines(screen, (85, 190, 148), False, points, 2)
        pygame.draw.circle(screen, (245, 196, 81), points[-1], 3)

    def _draw_comparison_table(
        self, pygame: object, screen: object, x: int, y: int, width: int, height: int
    ) -> None:
        """Render the preset comparison summary table."""
        box = pygame.Rect(x, y, width, height)
        pygame.draw.rect(screen, (21, 30, 44), box, border_radius=10)
        pygame.draw.rect(screen, (43, 58, 79), box, 1, border_radius=10)
        screen.blit(
            self._render_text(pygame, "PRESET COMPARISON (F: CHART)", 10, (226, 232, 240), bold=True),
            (box.x + 10, box.y + 8),
        )
        ry = box.y + 28
        cols = (0, 136, 184, 234)
        for header, offset in zip(("PRESET", "FIT", "GEN", "TIME"), cols):
            screen.blit(
                self._render_text(pygame, header, 9, (143, 161, 183)),
                (box.x + 10 + offset, ry),
            )
        ry += 16
        for name, solved, fit, gens, elapsed in self.compare_rows[:5]:
            short = name.split("(", 1)[0].strip()[:18]
            color = (120, 210, 168) if solved else (215, 190, 130)
            cells = (short, f"{fit}/64", str(gens), f"{elapsed:.2f}s")
            for cell, offset in zip(cells, cols):
                screen.blit(
                    self._render_text(pygame, cell, 9, color),
                    (box.x + 10 + offset, ry),
                )
            ry += 18

    def _draw_metric_grid(
        self,
        pygame: object,
        screen: object,
        x: int,
        y: int,
        width: int,
        metrics: list[tuple[str, str]],
    ) -> None:
        """Draw the 2x2 metric summary cards."""
        gap, card_height = 8, 42
        card_width = (width - gap) // 2
        for index, (label, value) in enumerate(metrics):
            col, row = index % 2, index // 2
            rect = pygame.Rect(
                x + col * (card_width + gap),
                y + row * (card_height + gap),
                card_width,
                card_height,
            )
            pygame.draw.rect(screen, (35, 49, 68), rect, border_radius=9)
            screen.blit(
                self._render_text(pygame, label, 9, (143, 161, 183)),
                (rect.x + 9, rect.y + 6),
            )
            screen.blit(
                self._render_text(pygame, value, 13, (232, 237, 244), bold=True),
                (rect.x + 9, rect.y + 20),
            )

    def _render_text(
        self,
        pygame: object,
        text: str,
        size: int,
        color: tuple[int, int, int],
        bold: bool = False,
    ) -> object:
        """Render text using freetype or pygame.font for Python 3.14+/3.15 compatibility."""
        font_key = (size, bold)
        if font_key not in self._font_cache:
            use_freetype = (
                sys.version_info >= (3, 14) and pygame.version.vernum <= (2, 6, 1)
            )
            if use_freetype:
                import pygame._freetype as freetype

                freetype.init()
                cached = freetype.Font(None, size)
                cached.strong = bold
                self._font_cache[font_key] = cached
                self._freetype_font_keys.add(font_key)
            else:
                try:
                    self._font_cache[font_key] = pygame.font.SysFont("arial", size, bold=bold)
                except (ImportError, NotImplementedError):
                    import pygame._freetype as freetype

                    freetype.init()
                    cached = freetype.Font(None, size)
                    cached.strong = bold
                    self._font_cache[font_key] = cached
                    self._freetype_font_keys.add(font_key)
        font = self._font_cache[font_key]
        if font_key in self._freetype_font_keys:
            return font.render(text, color)[0]
        return font.render(text, True, color)
