# Rust Flocking Simulator

This simulator demonstrates bird flocking behavior implemented in Rust using parallel processing.

When running the visual simulation, bird colour dictates what their strongest force is:

Green = **Cohesion**

Red = **Separation**

Blue = **Alignment**

## Basic Commands

**For regular visualization with FPS tracking:**

```
cargo run
```

**For raw computational benchmark of steps without graphics:**

```
cargo run -- benchmark                   # Default: 200 birds, 1000 steps
cargo run -- benchmark 500               # 500 birds, 1000 steps
cargo run -- benchmark 500 2000          # 500 birds, 2000 steps
cargo run -- benchmark 1000 500 --threads 8  # 1000 birds, 500 steps, 8 threads
```

**For scaling analysis across different flock sizes:**

```
cargo run -- scaling
```

## Thread Control Options

You can control the number of threads used for force calculation and position updates:

```
cargo run -- --threads <number>           # Set both thread pools to same value
cargo run -- --force-threads <number> --update-threads <number>  # Set separately
```

These parameters can be combined with any of the basic commands.

## Examples

**Visualize with custom thread counts:**

```
cargo run -- --threads 6               # Use 6 threads for both pools
cargo run -- --force-threads 8 --update-threads 4  # Different thread counts
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
cargo run -- benchmark 1000 --threads 1

# Test with 2 threads
cargo run -- benchmark 1000 --threads 2

# Test with 4 threads
cargo run -- benchmark 1000 --threads 4

# Test with 8 threads
cargo run -- benchmark 1000 --threads 8
```

Compare the performance reports to observe the scaling efficiency.

## Parameters

- `--threads`: Set both force and update thread pools to the same value
- `--force-threads`: Number of threads used for force calculations (separation, alignment, cohesion)
- `--update-threads`: Number of threads used for position updates
- `benchmark [birds] [steps]`: Run a performance benchmark with optional bird count and step count
- `scaling`: Run benchmarks across different flock sizes to test scaling
