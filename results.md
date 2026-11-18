# Knight's Tour GA Benchmark

Each preset was evaluated across 10 deterministic seeds (`1..10`) with a generation cap of 2000 (`tournament_size=3`, `crossover_rate=1.0`). Median generations is computed over solved runs (`>2000` when none solved).

| Preset | Pop | Mut | Elitism | Solved | Avg best fitness | Median gens | Median time (s) |
| --- | ---: | ---: | :---: | ---: | ---: | ---: | ---: |
| PDF Default (pop=50, mut=0.01) | 50 | 0.010 | No | 10/10 | 64.0 | 91 | 0.5482 |
| Ch.2 Low Mut (pop=50, mut=0.001) | 50 | 0.001 | No | 7/10 | 63.4 | 20 | 0.1637 |
| High Mut (pop=50, mut=0.05) | 50 | 0.050 | No | 10/10 | 64.0 | 348 | 2.5810 |
| Elitist (pop=50, mut=0.01, elit=On) | 50 | 0.010 | Yes | 9/10 | 63.9 | 45 | 0.3001 |
| Large Pop (pop=100, mut=0.01) | 100 | 0.010 | No | 10/10 | 64.0 | 43 | 0.5725 |
