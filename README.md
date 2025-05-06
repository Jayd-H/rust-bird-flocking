# Rust Flocking Simulator

![Birds Flying All Pretty](prettybirds.png)

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

# FINAL LAB

## CPU Implementation

The code is split across three files `bird.rs`, `flock_manager.rs`, and `main.rs`. `bird.rs` defines what a bird is, `flock_manager.rs` contains the logic for controlling the flock, spatial grid, performance measuring, and threading, `main.rs` contains the rendering pipeline and argument parsing.

### Explanation

Following my 2D particle simulation that I did in a previous lab, I was confident about this final lab. Because I spent a long time optimising that, I had a good base for what I wanted this bird flocking simulation to look like. In theory, this final lab is the 2D particle simulation, just with another dimension and some more particle-specific code like cohesion, alignment, and separation, the parallelisation architecture remained similar. The assignment brief specified how we were meant to parallelise the CPU solution, by splitting up birds into concurrent groups.

The three core features that govern bird behaviour are separation, alignment, and cohesion. These forces are very self-explanatory in how they control bird behaviour, and each force is capped at the bird's maximum force capability and weighted differently to achieve a natural looking flocking behaviour. Separation dictates that birds should not be too close to one another, alignment forces the birds to cluster towards each other, and cohesion is what enables the birds to move together in flocks. After calculating all three forces, their magnitudes are compared to determine the dominant force, which is used for giving the birds unique colour.

For this final lab, I used a spatial grid to divide the bounds into smaller cubes, allowing more efficient lookup of neighbouring birds. To reduce calculation time further, birds also compare their distance on 1 axis before continuing to measure the full distance. Birds are calculated as being in a cell of a spatial grid by their perception distance, if a bird can see a cell, it is in it. Therefore, birds can be in multiple cells, despite technically not actually residing in that particular cell. This is to ensure that two birds in neighbouring cells that are physically close can be correctly calculated. This is an optimisation I was trying out based on feedback from Warren Viant in the past lab where we had to create a 2D particle simulation. Previously I wrote code to search for the current cell in the spatial grid, along with its 8 neighbours. If I implemented that here, the program would have to search through 27 neighbours due to the extra dimension, still more efficient than looking at every other bird in the simulation, but still much less efficient in 3D.

Again, taking inspiration from my previous lab work, I employed a double-buffering approach where i maintain two separate collections of birds, `current_birds` (wrapped in an Arc) and `next_birds` (standard Vec). This technique is essential because it allows the simulation to compute the next state of all birds without modifying their current state during force calculations. When all birds are updated, the buffers are swapped using `mem::replace` and `mem::take`, avoiding expensive copying operations. Without this double buffer, I would either need to clone the entire flock each frame, which is extremely ineficient, or use complex locking mechanisms that would severely impact parallelism. The double buffer approach also simplifies the force calculation phase as threads can safely read the current state of all birds without worrying about race conditions or data inconsistencies that would occur if birds were updated in place.

For thread management, the simulation utilises two separate scoped threadpools from the `scoped_threadpool` crate, one is for force calculations and the other is for position updates. These scoped threadpools are particularly useful because they guarantee that all spawned threads will complete before the scope ends, elimating potential data races and dangling references. Due to this scoped threads can safely borrow data from their parent scope because they are guaranteed to terminate. In my implementation, the force pool performs the complex N-body calculations where the birds interact with their neighbours, while ht eupdate pool handles the simpler position and rotation updates. This division allows me to optimise thread counts separately for computation-heavy force calculations versus the lighter update operations. For example, assigning 12 force threads and 4 update threads. Each threadpool creates chunks of birds based on the available thread count, ensuring even distribution of work across CPU cores.

To dissuade excessive cloning throughout the simulation, I extensively use Rust's `Arc` for shared immutable access to bird data. When multiple threads need to read the same positions and velocities during force calculations, Arc allows them to share this data without duplicating it. This is accomplished by wrapping `current_birds` in an Arc and then having each worker thread create a lightweight clone of the Arc pointer (not the underlying data). This approach is great, because previously I was deep copying the entire flock for each thread, which is much less efficient. The Arc wrapper ensures that data remains valid until all threads finish using it, at which point I attempt to reclaim the buffer with `Arc::try_unwrap()`. In the rare case when some other part of the program still holds a reference, the code falls back to allocating a new buffer, ensuring the simulation continues to run efficiently without memory leaks or excessive allocations.

For boundary handling, I implemented a toroidal world where birds that exit one side of the simulation reappear on the opposite side. In the `apply_boundaries` method, each bird's position is checked against the minimum and maximum bounds on all three axes. If a bird exceeds any boundary, its position is wrapped to the opposite side of the simulation space. I implemented a wrapped distance calculation function which computes the shortest distance between two birds in this toroidal space. This is crucial for accurate force calculations, as without it, birds near opposite boundaries would incorrectly perceive a large distance between them. Originally, birds would just have a check to see if they are outside of bounds, of which it would just teleport them to the opposite side, but of course this messes up the distance related calculations.

To add some visual distinction to the birds, colour coding was employed. Birds are coloured based on their most dominant force, green is cohesion, blue is alignment, and red is separation. Besides being more visually appealing to look at, this feature aided in development and facilitated a much greater understanding on an individual bird's behaviour at a given moment. A decent portion of the code located in my `main.rs` was already provided to us, this made rendering the birds as triangles in OpenGL much easier than if we were not given this. When running the visual simulation, periodically it will print to the console, showcasing the current FPS, number of birds, simulation time (in ms), and rendering time (in ms). When the visual simulation is closed, it will give a performance report:

```
PS D:\Files\Documents\AProjects\Rust\bird-flocking> cargo run --release
    Finished `release` profile [optimized] target(s) in 1.04s
     Running `target\release\bird-flocking.exe`
FPS: 574.5 | Birds: 1000 | Sim time: 0.73ms | Render time: 0.58ms
FPS: 682.6 | Birds: 1000 | Sim time: 0.71ms | Render time: 0.57ms
FPS: 672.2 | Birds: 1000 | Sim time: 0.72ms | Render time: 0.57ms
FPS: 657.1 | Birds: 1000 | Sim time: 0.74ms | Render time: 0.58ms
FPS: 606.2 | Birds: 1000 | Sim time: 0.78ms | Render time: 0.65ms
FPS: 495.1 | Birds: 1000 | Sim time: 0.91ms | Render time: 0.88ms
FPS: 497.6 | Birds: 1000 | Sim time: 0.91ms | Render time: 0.88ms
FPS: 490.9 | Birds: 1000 | Sim time: 0.93ms | Render time: 0.88ms
FPS: 541.1 | Birds: 1000 | Sim time: 0.87ms | Render time: 0.74ms
FPS: 633.0 | Birds: 1000 | Sim time: 0.78ms | Render time: 0.58ms


=== Final Performance Report ===
Bird count: 1000
Average FPS: 585.0
Min FPS: 490.9
Max FPS: 682.6
Time breakdown per frame:
  - Simulation: 0.80ms (0.7%)
  - Rendering: 0.58ms (0.5%)
Component breakdown in simulation:
=== Performance Report ===
Total steps: 29679
Total time: 23.648 seconds
Steps per second: 1255.0
Time breakdown:
  - Spatial grid updates: 2.992s (12.7%)
  - Force calculations: 19.517s (82.5%)
  - Position updates: 1.124s (4.8%)
  - Other/overhead: 0.015s (0.1%)
=========================
```

To control whether or not the visualisation should run, the bird count, and the thread count, the program accepts parameters before running. The default `cargo run --release` runs the visualisation, with the default birdcount (1000), with the default number of threads (4 update threads, 4 force calculation threads). To run without visualisation, the command is `cargo run --release -- benchmark` which will run a benchmark test, giving a performance report at the end. To change the bird count or steps with this benchmark the command is `cargo run --release -- benchmark [birds] [steps]`. To alter the thread count, you can add it to the command with `--threads [threads]`. Here is an example of a benchmark performance report for 500 birds at 500 steps with a thread count of 8:

```
PS D:\Files\Documents\AProjects\Rust\bird-flocking> cargo run --release -- benchmark 500 500 --threads 8
    Finished `release` profile [optimized] target(s) in 1.39s
     Running `target\release\bird-flocking.exe benchmark 500 500 --threads 8`
Running performance benchmark with 500 birds for 500 steps (force threads: 8, update threads: 8)
Running benchmark for 500 steps...
Completed 100 steps (20.0%)
Completed 200 steps (40.0%)
Completed 300 steps (60.0%)
Completed 400 steps (80.0%)
Completed 500 steps (100.0%)
=== Performance Report ===
Total steps: 500
Total time: 0.163 seconds
Steps per second: 3064.2
Time breakdown:
  - Spatial grid updates: 0.023s (14.1%)
  - Force calculations: 0.110s (67.5%)
  - Position updates: 0.030s (18.2%)
  - Other/overhead: 0.000s (0.1%)
=========================
PS D:\Files\Documents\AProjects\Rust\bird-flocking>
```

### Potential Improvements & Other Parallelization Architecture

Despite the optimisations, CPU based parallelisation still will always have inherent limitations for large-scale simulations. As the number of birds increases, memory bandwidth becomes a significant bottleneck, as threads compete for access to the shared memory bus.

To further optimise the simulation, you could use a more advanced spatial grid structure than the one I currently have. My implementation uses uniform cell sizes across the simulation space, which is not always optimal, especially for this use-case as birds tend to flock is fairly close clusters. An adaptive grid structure would dynamically allocate smaller cells in dense regions and larger cells in sparse areas, potentially reducing both memory usage and computation time significantly. The challenge with this is the added computational overhead for maintenance, especially when the birds rapidly form and dissolve clusters.

Similarly, work stealing is another technique that could prove beneficial. My current approach divides birds into equal-sized chunks and statically assigns these to threads, which works well when the distribution of the birds in relatively uniform. However, this becomes inefficient when some birds require more computation than others due to the number of neighbours it has. A work stealing scheduler would enable threads that finish early to 'steal' pending tasks from threads with heavier workloads, creating a more balanced distribution of computation. From my understanding, the Rust ecosystem offers some very useful libraries like `Rayon` that provide work stealing pools with minimal code changes required. I did experiment with different static thread allocations, but dynamic work balancing would likely yield better results.

```
PS D:\Files\Documents\AProjects\Rust\bird-flocking> cargo run --release
    Finished `release` profile [optimized] target(s) in 1.04s
     Running `target\release\bird-flocking.exe`
FPS: 606.1 | Birds: 1000 | Sim time: 0.73ms | Render time: 0.60ms
FPS: 606.0 | Birds: 1000 | Sim time: 0.74ms | Render time: 0.59ms
FPS: 626.0 | Birds: 1000 | Sim time: 0.74ms | Render time: 0.60ms
FPS: 595.2 | Birds: 1000 | Sim time: 0.80ms | Render time: 0.65ms
FPS: 486.0 | Birds: 1000 | Sim time: 0.95ms | Render time: 0.82ms
FPS: 496.8 | Birds: 1000 | Sim time: 0.96ms | Render time: 0.80ms
FPS: 489.4 | Birds: 1000 | Sim time: 0.99ms | Render time: 0.80ms
FPS: 488.0 | Birds: 1000 | Sim time: 0.96ms | Render time: 0.75ms
```

As you can see, from running the simulation for a minute or so, the simulation time already increases due to the clustering of birds. If I were to expand this code further, this would be my immediate priority, but for the sake of keeping it relatively simple for the lab, I decided against it.

One technique I did not explore (because of the specificity of the lab brief) was SIMD (Single Instruction Multiple Data) vectorisation, which could significantly accelerate the vector math operations that dominate these kind of force calculations. Modern CPUs support AVX and SSE instructions that can process 4-8 floating-point operations simultaneously, therefore potentially offering 4-8x speedup for the heavy calculation portion of the simulation. From my understanding, Rust supports SIMD capabilities quite well through libraries like `packed_simd`. However, implementing this would require careful restructuring of the force calculation logic to take advantage of vectorised operations, and then integrating it within the existing non-SIMD parallelised code. Something like this seems like it would be the primary objective when designing a program, rather than slapping it on something pre-existing, and I would need to do some more research to fully understand SIMD enough to properly implement it.
