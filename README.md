# Rust Flocking Simulator

This simulator demonstrates bird flocking behavior implemented in Rust using parallel processing.

## Basic Commands

**For regular visualization with FPS tracking:**

```
cargo run
```

**For raw computational benchmark of steps without graphics:**

```
cargo run -- benchmark 1000
```

**For scaling analysis across different flock sizes:**

```
cargo run -- scaling
```

## Thread Control Options

You can control the number of threads used for force calculation and position updates:

```
cargo run -- --force-threads <number> --update-threads <number>
```

These parameters can be combined with any of the basic commands.

## Examples

**Visualize with custom thread counts:**

```
cargo run -- --force-threads 8 --update-threads 4
```

**Run benchmark with custom thread configuration:**

```
cargo run -- benchmark 1000 --force-threads 16 --update-threads 8
```

**Run scaling test with specific number of threads:**

```
cargo run -- scaling --force-threads 2 --update-threads 2
```

## Thread Scaling Experiments

To test how well the simulation scales with different thread counts:

```
# Test with single thread (baseline)
cargo run -- benchmark 1000 --force-threads 1 --update-threads 1

# Test with 2 threads
cargo run -- benchmark 1000 --force-threads 2 --update-threads 2

# Test with 4 threads
cargo run -- benchmark 1000 --force-threads 4 --update-threads 4

# Test with 8 threads
cargo run -- benchmark 1000 --force-threads 8 --update-threads 8
```

Compare the performance reports to observe the scaling efficiency.

## Parameters

- `--force-threads`: Number of threads used for force calculations (separation, alignment, cohesion)
- `--update-threads`: Number of threads used for position updates
- `benchmark <steps>`: Run a performance benchmark for the specified number of steps
- `scaling`: Run benchmarks across different flock sizes to test scaling
