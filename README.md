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

```Rust
struct SpatialGrid {
    cells: Vec<Vec<usize>>,
    dim_x: usize,
    dim_y: usize,
    dim_z: usize,
    min_bounds: Vector3<f32>,
    cell_size: f32,
}

impl SpatialGrid {
    fn new(min_bounds: &Vector3<f32>, max_bounds: &Vector3<f32>) -> Self {
        let dim_x = ((max_bounds.x - min_bounds.x) / CELL_SIZE).ceil() as usize + 1;
        let dim_y = ((max_bounds.y - min_bounds.y) / CELL_SIZE).ceil() as usize + 1;
        let dim_z = ((max_bounds.z - min_bounds.z) / CELL_SIZE).ceil() as usize + 1;
        let total_cells = dim_x * dim_y * dim_z;

        let mut cells = Vec::with_capacity(total_cells);
        for _ in 0..total_cells {
            cells.push(Vec::new());
        }

        SpatialGrid {
            cells,
            dim_x,
            dim_y,
            dim_z,
            min_bounds: *min_bounds,
            cell_size: CELL_SIZE,
        }
    }
```

Again, taking inspiration from my previous lab work, I employed a double-buffering approach where I maintain two separate collections of birds, `current_birds` (wrapped in an Arc) and `next_birds` (standard Vec). This technique is essential because it allows the simulation to compute the next state of all birds without modifying their current state during force calculations. When all birds are updated, the buffers are swapped using `mem::replace` and `mem::take`, avoiding expensive copying operations. Without this double buffer, I would either need to clone the entire flock each frame, which is extremely inefficient, or use complex locking mechanisms that would severely impact parallelism. The double buffer approach also simplifies the force calculation phase as threads can safely read the current state of all birds without worrying about race conditions or data inconsistencies that would occur if birds were updated in place.

```Rust
let current_birds = Arc::new(std::mem::take(&mut next_birds));
```

For thread management, the simulation utilises two separate scoped thread pools from the `scoped_threadpool` crate, one is for force calculations and the other is for position updates. These scoped thread pools are particularly useful because they guarantee that all spawned threads will complete before the scope ends, eliminating potential data races and dangling references. Due to this scoped threads can safely borrow data from their parent scope because they are guaranteed to terminate. In my implementation, the force pool performs the complex N-body calculations where the birds interact with their neighbours, while the update pool handles the simpler position and rotation updates. This division allows me to optimise thread counts separately for computation-heavy force calculations versus the lighter update operations. For example, assigning 12 force threads and 4 update threads. Each thread pool creates chunks of birds based on the available thread count, ensuring even distribution of work across CPU cores.

To dissuade excessive cloning throughout the simulation, I extensively use Rust's `Arc` for shared immutable access to bird data. When multiple threads need to read the same positions and velocities during force calculations, Arc allows them to share this data without duplicating it. This is accomplished by wrapping `current_birds` in an Arc and then having each worker thread create a lightweight clone of the Arc pointer (not the underlying data). This approach is great, because previously I was deep copying the entire flock for each thread, which is much less efficient. The Arc wrapper ensures that data remains valid until all threads finish using it, at which point I attempt to reclaim the buffer with `Arc::try_unwrap()`. In the rare case when some other part of the program still holds a reference, the code falls back to allocating a new buffer, ensuring the simulation continues to run efficiently without memory leaks or excessive allocations.

```Rust
 self.performance.position_update_time += update_start.elapsed();

        // Swap without cloning using mem::replace
        let old_birds = std::mem::replace(
            &mut self.current_birds,
            Arc::new(std::mem::take(&mut self.next_birds)),
        );

        // Try to reclaim the old buffer
        if let Ok(birds) = Arc::try_unwrap(old_birds) {
            self.next_birds = birds;
        } else {
            // If we can't unwrap (other references exist), create a new buffer
            // This should be rare in practice
            self.next_birds = Vec::with_capacity(birds_count);
        }

        self.performance.steps_completed += 1;
        self.performance.total_time += start_time.elapsed();
```

For boundary handling, I implemented a toroidal world where birds that exit one side of the simulation reappear on the opposite side. In the `apply_boundaries` method, each bird's position is checked against the minimum and maximum bounds on all three axes. If a bird exceeds any boundary, its position is wrapped to the opposite side of the simulation space. I implemented a wrapped distance calculation function which computes the shortest distance between two birds in this toroidal space. This is crucial for accurate force calculations, as without it, birds near opposite boundaries would incorrectly perceive a large distance between them. Originally, birds would just have a check to see if they are outside of bounds, of which it would just teleport them to the opposite side, but of course this messes up the distance related calculations.

```Rust
    // Calculates the wrapped distance between two positions in a toroidal space
    fn calculate_wrapped_distance(
        pos1: &Vector3<f32>,
        pos2: &Vector3<f32>,
        min_bounds: &Vector3<f32>,
        max_bounds: &Vector3<f32>,
    ) -> Vector3<f32> {
        let size_x = max_bounds.x - min_bounds.x;
        let size_y = max_bounds.y - min_bounds.y;
        let size_z = max_bounds.z - min_bounds.z;

        let dx = pos1.x - pos2.x;
        let dy = pos1.y - pos2.y;
        let dz = pos1.z - pos2.z;

        let dx_wrap = dx - size_x * (dx / (size_x * 0.5)).round();
        let dy_wrap = dy - size_y * (dy / (size_y * 0.5)).round();
        let dz_wrap = dz - size_z * (dz / (size_z * 0.5)).round();

        Vector3::new(dx_wrap, dy_wrap, dz_wrap)
    }
```

To add some visual distinction to the birds, colour coding was used. Birds are coloured based on their most dominant force, green is cohesion, blue is alignment, and red is separation. Besides being more visually appealing to look at, this feature aided in development and facilitated a much greater understanding of an individual bird's behaviour at a given moment. A decent portion of the code located in my `main.rs` was already provided to us, this made rendering the birds as triangles in OpenGL much easier than if we were not given this. When running the visual simulation, periodically it will print to the console, showcasing the current FPS, number of birds, simulation time (in ms), and rendering time (in ms). When the visual simulation is closed, it will give a performance report:

```Rust
let dominant_force = flock.get_dominant_force(i);
let bird_color = BIRD_COLORS[dominant_force.min(3)];
```

![Bird Simulation with Colours](image-2.png)

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
- # Other/overhead: 0.015s (0.1%)

```

To control whether or not the visualisation should run, the bird count, and the thread count, the program accepts parameters before running. The default `cargo run --release` runs the visualisation, with the default bird count (1000), with the default number of threads (4 update threads, 4 force calculation threads). To run without visualisation, the command is `cargo run --release -- benchmark` which will run a benchmark test, giving a performance report at the end. To change the bird count or steps with this benchmark the command is `cargo run --release -- benchmark [birds] [steps]`. To alter the thread count, you can add it to the command with `--threads [threads]`. Here is an example of a benchmark performance report for 500 birds at 500 steps with a thread count of 8:

```Rust
if args.len() > 1 {
        match args[1].as_str() {
            "benchmark" => {
                let mut num_birds = DEFAULT_NUM_BIRDS;
                let mut benchmark_steps = BENCHMARK_STEPS;

                // Check if birds count is specified
                if args.len() > 2 && !args[2].starts_with("--") {
                    if let Ok(birds) = args[2].parse() {
                        num_birds = birds;
                    }

                    // Check if steps are also specified
                    if args.len() > 3 && !args[3].starts_with("--") {
                        if let Ok(steps) = args[3].parse() {
                            benchmark_steps = steps;
                        }
                    }
                }

                println!("Running performance benchmark with {} birds for {} steps (force threads: {}, update threads: {})",
                    num_birds, benchmark_steps, force_threads, update_threads);
                let mut flock = FlockManager::new(
                    num_birds,
                    min_bounds,
                    max_bounds,
                    force_threads,
                    update_threads,
                );
                flock.run_benchmark(benchmark_steps, SIMULATION_TIMESTEP);
                return;
            }
            "scaling" => {
                // Scaling test across different flock sizes
                println!("Running scaling test with various flock sizes (force threads: {}, update threads: {})",
                    force_threads, update_threads);
                for &size in &FLOCK_SIZES {
                    println!("\n=== Testing with {} birds ===", size);
                    let mut flock = FlockManager::new(
                        size,
                        min_bounds,
                        max_bounds,
                        force_threads,
                        update_threads,
                    );
                    flock.run_benchmark(SCALING_TEST_STEPS, SIMULATION_TIMESTEP);
                }
                return;
            }
            _ => {}
        }
    }
}
```

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
- # Other/overhead: 0.000s (0.1%)
  PS D:\Files\Documents\AProjects\Rust\bird-flocking>

```

### Potential Improvements & Other Parallelization Architecture

Despite the optimisations, CPU based parallelisation still will always have inherent limitations for large-scale simulations. As the number of birds increases, memory bandwidth becomes a significant bottleneck, as threads compete for access to the shared memory bus.

To further optimise the simulation, you could use a more advanced spatial grid structure than the one I currently have. My implementation uses uniform cell sizes across the simulation space, which is not always optimal, especially for this use case as birds tend to flock in fairly close clusters. An adaptive grid structure would dynamically allocate smaller cells in dense regions and larger cells in sparse areas, potentially reducing both memory usage and computation time significantly. The challenge with this is the added computational overhead for maintenance, especially when the birds rapidly form and dissolve clusters.

Similarly, work stealing is another technique that could prove beneficial. My current approach divides birds into equal-sized chunks and statically assigns these to threads, which works well when the distribution of the birds is relatively uniform. However, this becomes inefficient when some birds require more computation than others due to the number of neighbours it has. A work stealing scheduler would enable threads that finish early to 'steal' pending tasks from threads with heavier workloads, creating a more balanced distribution of computation. From my understanding, the Rust ecosystem offers some very useful libraries like `Rayon` that provide work stealing pools with minimal code changes required. I did experiment with different static thread allocations, but dynamic work balancing would likely yield better results.

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

![Bird Flocking Simulation Clustering](image-3.png)

As you can see, from running the simulation for a minute or so, the simulation time already increases due to the clustering of birds. If I were to expand this code further, this would be my immediate priority, but for the sake of keeping it relatively simple for the lab, I decided against it.

One technique I did not explore (because of the specificity of the lab brief) was SIMD (Single Instruction Multiple Data) vectorisation, which could significantly accelerate the vector math operations that dominate these kind of force calculations. Modern CPUs support AVX and SSE instructions that can process 4-8 floating-point operations simultaneously, therefore potentially offering 4-8x speedup for the heavy calculation portion of the simulation. From my understanding, Rust supports SIMD capabilities quite well through libraries like `packed_simd`. However, implementing this would require careful restructuring of the force calculation logic to take advantage of vectorised operations, and then integrating it within the existing non-SIMD parallelised code. Something like this seems like it would be the primary objective when designing a program, rather than slapping it on something pre-existing, and I would need to do some more research to fully understand SIMD enough to properly implement it.

## GPU Implementation

The code is split across two main files, `main.cu`, which contains the CUDA specific code like rendering birds, calculating values, and assigning each bird a thread, and `main.cpp` which contains the code that sets up the program, displays the results, and contains the main entry point and loop. This division keeps the CUDA kernels separate from the OpenGL and application logic, allowing me to better focus on the optimisation and implementation of both of these parts separately without worrying too much about affecting the other. This explanation will be more brief as I will omit some non-parallelisation-related specifics which closely resemble my Rust implementation.

![CUDA Simulation](image-4.png)

### Explanation

I found the GPU solution much more hands-on than the Rust programming, largely because I am not as confident in my CUDA abilities compared to Rust. Following the brief, each bird is assigned its own thread to work with. On this thread, it will calculate its separation, alignment, and cohesion. Since CUDA's architecture is designed to handle massive parallelism across thousands of threads, writing parallelised code was a lot more straightforward. The bird-to-thread mapping is implemented directly in the kernel launch configuration.

```cpp
dim3 blockSize(256);
dim3 gridSize((birdCount + blockSize.x - 1) / blockSize.x);
calculateForces<<<gridSize, blockSize>>>(separationWeight, alignmentWeight, cohesionWeight);
```

This ensures we have exactly one thread per bird while optimising for CUDA's execution model, where threads are grouped into blocks of 256 for efficient scheduling on the GPU. Unlike the CPU implementation where I need to manually divide birds among a smaller number of threads, the GPU naturally scales to thousands of concurrent operations.

Similar to the CPU version, birds determine their dominant force (separation, alignment, cohesion) by comparing the magnitudes of these forces, which affects their colour in the visualisation. Birds are colour coded the same way as in the CPU implementation with red for separation, blue for alignment, and green for cohesion. For rendering, I implemented a simple ray-casting approach where each CUDA thread handles a pixel in the output image. This creates a 3D visualisation where birds are rendered as small coloured spheres, with a lot of this code being taken from previous labs. The GPU implementation also uses a toroidal world with wrap-around boundary handling. I implemented a wrapped distance function that considers the toroidal space, ensuring that birds near the boundaries interact correctly with birds on the opposite side.

```cpp
__device__ float3 calculateWrappedDistance(float3 pos1, float3 pos2) {
    float3 diff = make_float3(pos1.x - pos2.x, pos1.y - pos2.y, pos1.z - pos2.z);

    float size_x = maxBounds.x - minBounds.x;
    float size_y = maxBounds.y - minBounds.y;
    float size_z = maxBounds.z - minBounds.z;

    if (abs(diff.x) > size_x * 0.5f) {
        diff.x = diff.x - copysignf(size_x, diff.x);
    }
    if (abs(diff.y) > size_y * 0.5f) {
        diff.y = diff.y - copysignf(size_y, diff.y);
    }
    if (abs(diff.z) > size_z * 0.5f) {
        diff.z = diff.z - copysignf(size_z, diff.z);
    }

    return diff;
}
```

For memory management, I used `__device__ __managed__` which automatically handles memory transfers between host and device. Using this means I did not need to write explicit `cudaMemcpy` calls to transfer data between the CPU and GPU. The CUDA runtime automatically handles these transfers as needed, which greatly simplified the code structure. This was useful for the performance metrics, which are updated on the GPU but need to be read on the CPU for reporting. This also means that the number of birds needs to be fixed at compile time.

```cpp
__device__ __managed__ Bird birds[MAX_BIRDS];
__device__ __managed__ int numBirds;
__device__ __managed__ float3 minBounds;
__device__ __managed__ float3 maxBounds;
__device__ __managed__ PerformanceMetrics metrics;
```

For performance measurement, I used CUDA events to precisely time different parts of the simulation. These events allow me to track the time spent in the force calculations versus position updates, similar to my CPU implementation. The benchmarking code provides detailed performance reports that break down where computation time is spent.

```cpp
extern "C" void updateSimulation(float dt, float separationWeight, float alignmentWeight, float cohesionWeight) {
    int birdCount;
    cudaMemcpyFromSymbol(&birdCount, numBirds, sizeof(int));

    dim3 blockSize(256);
    dim3 gridSize((birdCount + blockSize.x - 1) / blockSize.x);

    float elapsedTime;

    cudaEventRecord(startEvent);

    cudaEventRecord(forceStartEvent);
    calculateForces << <gridSize, blockSize, blockSize.x * sizeof(Bird) >> > (separationWeight, alignmentWeight, cohesionWeight);
    cudaEventRecord(forceStopEvent);
    cudaDeviceSynchronize();

    cudaEventRecord(posStartEvent);
    updatePositions << <gridSize, blockSize >> > (dt);
    cudaEventRecord(posStopEvent);
    cudaDeviceSynchronize();

    cudaEventRecord(stopEvent);
    cudaEventSynchronize(stopEvent);

    cudaEventElapsedTime(&elapsedTime, forceStartEvent, forceStopEvent);
    PerformanceMetrics h_metrics;
    cudaMemcpyFromSymbol(&h_metrics, metrics, sizeof(PerformanceMetrics));
    h_metrics.forceCalculationTime += elapsedTime / 1000.0f;

    cudaEventElapsedTime(&elapsedTime, posStartEvent, posStopEvent);
    h_metrics.positionUpdateTime += elapsedTime / 1000.0f;

    cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
    h_metrics.totalTime += elapsedTime / 1000.0f;

    h_metrics.stepsCompleted++;

    cudaMemcpyToSymbol(metrics, &h_metrics, sizeof(PerformanceMetrics));
}
```

```cpp
extern "C" void runBenchmark(int birdCount, int steps, float dt, float separationWeight, float alignmentWeight, float cohesionWeight) {
    resetMetrics << <1, 1 >> > ();
    cudaDeviceSynchronize();

    printf("Running benchmark for %d birds with %d steps...\n", birdCount, steps);

    for (int i = 1; i <= steps; i++) {
        updateSimulation(dt, separationWeight, alignmentWeight, cohesionWeight);

        if (i % 100 == 0 || i == steps) {
            printf("Completed %d steps (%.1f%%)\n", i, (i * 100.0f) / steps);
        }
    }

    PerformanceMetrics h_metrics;
    cudaMemcpyFromSymbol(&h_metrics, metrics, sizeof(PerformanceMetrics));

    printf("\n=== Performance Report ===\n");
    printf("Total steps: %d\n", h_metrics.stepsCompleted);
    printf("Total time: %.3f seconds\n", h_metrics.totalTime);
    printf("Steps per second: %.1f\n", h_metrics.stepsCompleted / h_metrics.totalTime);
    printf("Time breakdown:\n");
    printf("  - Force calculations: %.3fs (%.1f%%)\n",
        h_metrics.forceCalculationTime,
        (h_metrics.forceCalculationTime * 100.0f) / h_metrics.totalTime);
    printf("  - Position updates: %.3fs (%.1f%%)\n",
        h_metrics.positionUpdateTime,
        (h_metrics.positionUpdateTime * 100.0f) / h_metrics.totalTime);

    float overhead = h_metrics.totalTime -
        (h_metrics.forceCalculationTime +
            h_metrics.positionUpdateTime);

    printf("  - Other/overhead: %.3fs (%.1f%%)\n",
        overhead,
        (overhead * 100.0f) / h_metrics.totalTime);
    printf("=========================\n");
}
```

**Benchmark with 1000 Birds**

```
=== Testing with 1000 birds ===
Simulation initialized with 1000 birds
Running benchmark for 1000 birds with 500 steps...
Completed 100 steps (20.0%)
Completed 200 steps (40.0%)
Completed 300 steps (60.0%)
Completed 400 steps (80.0%)
Completed 500 steps (100.0%)

=== Performance Report ===
Total steps: 500
Total time: 0.107 seconds
Steps per second: 4674.4
Time breakdown:
  - Force calculations: 0.074s (69.2%)
  - Position updates: 0.011s (10.4%)
  - Other/overhead: 0.022s (20.5%)
=========================
```

One of the key optimisations in this implementation is the use of shared memory to improve performance. In the `calculateForces` kernel, I load bird data into shared memory to reduce global memory accesses. This approach significantly reduces the number of global memory accesses, as each thread in a block can access the bird data from the faster shared memory rather than repeatedly fetching it from global memory.

```cpp
__global__ void calculateForces(float separationWeight, float alignmentWeight, float cohesionWeight) {
    extern __shared__ Bird sharedBirds[];

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < numBirds) {
        Bird& bird = birds[tid];

        float3 separation = make_float3(0.0f, 0.0f, 0.0f);
        float3 alignment = make_float3(0.0f, 0.0f, 0.0f);
        float3 cohesion = make_float3(0.0f, 0.0f, 0.0f);
        int separationCount = 0;
        int alignmentCount = 0;
        int cohesionCount = 0;

        for (int chunkStart = 0; chunkStart < numBirds; chunkStart += blockDim.x) {
            int chunkIdx = chunkStart + threadIdx.x;
            if (chunkIdx < numBirds) {
                sharedBirds[threadIdx.x] = birds[chunkIdx];
            }

            __syncthreads();

            int chunkSize = min(blockDim.x, numBirds - chunkStart);
            for (int j = 0; j < chunkSize; j++) {
                int otherIdx = chunkStart + j;
                if (tid == otherIdx) continue;

                Bird& other = sharedBirds[j];

                float dx = abs(bird.position.x - other.position.x);
                float size_x = maxBounds.x - minBounds.x;
                if (dx > size_x * 0.5f) {
                    dx = size_x - dx;
                }
                if (dx > PERCEPTION_RADIUS) continue;

                float3 diff = calculateWrappedDistance(bird.position, other.position);
                float dist_sq = diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;

                if (dist_sq > EPSILON && dist_sq < (PERCEPTION_RADIUS * SEPARATION_RADIUS_FACTOR) * (PERCEPTION_RADIUS * SEPARATION_RADIUS_FACTOR)) {
                    float3 repulse = make_float3(diff.x / dist_sq, diff.y / dist_sq, diff.z / dist_sq);
                    separation.x += repulse.x;
                    separation.y += repulse.y;
                    separation.z += repulse.z;
                    separationCount++;
                }

                if (dist_sq < PERCEPTION_RADIUS * PERCEPTION_RADIUS) {
                    alignment.x += other.velocity.x;
                    alignment.y += other.velocity.y;
                    alignment.z += other.velocity.z;
                    alignmentCount++;

                    float3 otherPos = make_float3(
                        bird.position.x - diff.x,
                        bird.position.y - diff.y,
                        bird.position.z - diff.z
                    );
                    cohesion.x += otherPos.x;
                    cohesion.y += otherPos.y;
                    cohesion.z += otherPos.z;
                    cohesionCount++;
                }
            }

            __syncthreads();
        }
        ...
```

Instead of implementing a full spatial grid like in the Rust version, I opted for a simpler early rejection based on one axis comparison. Before performing the full distance calculation between birds, I first check if they are too far apart on the x-axis to even be considered as neighbouring. This allows for quick better performance for bird lookups, yet still not nearly as much as implementing a proper spatial partitioned grid. I made this decision partly because of my lack of experience in CUDA to implement one as efficiently as my Rust solution.

### Potential Improvements & Other Parallelization Architecture

Similarly to my Rust implementation, a spatial grid that the birds live in would prove very beneficial. Currently, the lookup for birds is n^2, with every bird in the simulation checking every other bird. While the GPU can handle this better than the CPU for moderate flock sizes, it still becomes inefficient for large flocks. A spatial grid would reduce the number of bird-to-bird comparisons needed, potentially improving performance dramatically for larger flocks.

Like the Rust implementation, this simulation also suffers from clustering after the birds find their footing and begin to actually flock together. Due to each bird having its own thread instead of a thread managing a group of birds, the solution of work-stealing is not appropriate here. Instead, a dynamic load balancing approach that reassigns threads based on the spatial distribution of birds could be implemented, dividing the simulation space into regions and assigning thread blocks to regions based on bird density.

![CUDA Birds Clustering](image-5.png)

From my understanding, I believe the ideal solution would be a hybrid CPU-GPU architecture. The CPU could handle high-level coordination and spatial partitioning, while the GPU focuses on the computationally intensive force calculations. This approach would leverage the strengths of both platforms, with the CPU handling tasks that require more complex decision making and the GPU handling the highly parallel mathematical operations.

## Performance Comparison

Understandably, even with the 3D rendering of the birds, the GPU solution reigned supreme in terms of performance. This was expected as GPUs are specifically designed for running many actions concurrently, while the architecture of a CPU at its core relies on a sequential step-by-step execution cycle. Looking at the scaling test for each implementation, even despite the CUDA code being less optimised than the Rust code, it is very clear that CUDA is more suitable for a task like this.

### CUDA Scaling Test

This scaling test uses the default block size of 256 for the birds.

```
PS C:\Users\Willy\Documents\Jayden\GitHub\cuda-bird-flocking\BirdSim\x64\Release> .\BirdSim.exe scaling
GPU Device 0: "Ada" with compute capability 8.9

CUDA device [NVIDIA GeForce RTX 4070 Ti SUPER] has 66 Multi-Processors
Running scaling test with various flock sizes
Running scaling test with various flock sizes...

=== Testing with 100 birds ===
Simulation initialized with 100 birds
Running benchmark for 100 birds with 500 steps...
Completed 100 steps (20.0%)
Completed 200 steps (40.0%)
Completed 300 steps (60.0%)
Completed 400 steps (80.0%)
Completed 500 steps (100.0%)

=== Performance Report ===
Total steps: 500
Total time: 0.049 seconds
Steps per second: 10142.5
Time breakdown:
  - Force calculations: 0.018s (36.9%)
  - Position updates: 0.011s (21.9%)
  - Other/overhead: 0.020s (41.2%)
=========================

=== Testing with 500 birds ===
Simulation initialized with 500 birds
Running benchmark for 500 birds with 500 steps...
Completed 100 steps (20.0%)
Completed 200 steps (40.0%)
Completed 300 steps (60.0%)
Completed 400 steps (80.0%)
Completed 500 steps (100.0%)

=== Performance Report ===
Total steps: 500
Total time: 0.074 seconds
Steps per second: 6730.5
Time breakdown:
  - Force calculations: 0.042s (57.1%)
  - Position updates: 0.011s (14.6%)
  - Other/overhead: 0.021s (28.3%)
=========================

=== Testing with 1000 birds ===
Simulation initialized with 1000 birds
Running benchmark for 1000 birds with 500 steps...
Completed 100 steps (20.0%)
Completed 200 steps (40.0%)
Completed 300 steps (60.0%)
Completed 400 steps (80.0%)
Completed 500 steps (100.0%)

=== Performance Report ===
Total steps: 500
Total time: 0.107 seconds
Steps per second: 4674.4
Time breakdown:
  - Force calculations: 0.074s (69.2%)
  - Position updates: 0.011s (10.4%)
  - Other/overhead: 0.022s (20.5%)
=========================

=== Testing with 5000 birds ===
Simulation initialized with 5000 birds
Running benchmark for 5000 birds with 500 steps...
Completed 100 steps (20.0%)
Completed 200 steps (40.0%)
Completed 300 steps (60.0%)
Completed 400 steps (80.0%)
Completed 500 steps (100.0%)

=== Performance Report ===
Total steps: 500
Total time: 0.345 seconds
Steps per second: 1447.4
Time breakdown:
  - Force calculations: 0.306s (88.5%)
  - Position updates: 0.014s (4.0%)
  - Other/overhead: 0.026s (7.5%)
=========================
```

### Rust Scaling Test

This Rust scaling test uses 4 threads for position updates, and another 4 threads for force calculations.

```
D:\Files\Documents\AProjects\Rust\bird-flocking>cargo run --release -- scaling
    Finished `release` profile [optimized] target(s) in 0.10s
     Running `target\release\bird-flocking.exe scaling`
Running scaling test with various flock sizes (force threads: 4, update threads: 4)

=== Testing with 100 birds ===
Running benchmark for 500 steps...
Completed 100 steps (20.0%)
Completed 200 steps (40.0%)
Completed 300 steps (60.0%)
Completed 400 steps (80.0%)
Completed 500 steps (100.0%)
=== Performance Report ===
Total steps: 500
Total time: 0.040 seconds
Steps per second: 12428.7
Time breakdown:
  - Spatial grid updates: 0.005s (11.7%)
  - Force calculations: 0.020s (49.9%)
  - Position updates: 0.015s (38.0%)
  - Other/overhead: 0.000s (0.3%)
=========================

=== Testing with 500 birds ===
Running benchmark for 500 steps...
Completed 100 steps (20.0%)
Completed 200 steps (40.0%)
Completed 300 steps (60.0%)
Completed 400 steps (80.0%)
Completed 500 steps (100.0%)
=== Performance Report ===
Total steps: 500
Total time: 0.147 seconds
Steps per second: 3401.5
Time breakdown:
  - Spatial grid updates: 0.021s (14.6%)
  - Force calculations: 0.109s (74.2%)
  - Position updates: 0.016s (11.0%)
  - Other/overhead: 0.000s (0.1%)
=========================

=== Testing with 1000 birds ===
Running benchmark for 500 steps...
Completed 100 steps (20.0%)
Completed 200 steps (40.0%)
Completed 300 steps (60.0%)
Completed 400 steps (80.0%)
Completed 500 steps (100.0%)
=== Performance Report ===
Total steps: 500
Total time: 0.395 seconds
Steps per second: 1265.7
Time breakdown:
  - Spatial grid updates: 0.043s (10.9%)
  - Force calculations: 0.333s (84.4%)
  - Position updates: 0.018s (4.6%)
  - Other/overhead: 0.000s (0.0%)
=========================

=== Testing with 5000 birds ===
Running benchmark for 500 steps...
Completed 100 steps (20.0%)
Completed 200 steps (40.0%)
Completed 300 steps (60.0%)
Completed 400 steps (80.0%)
Completed 500 steps (100.0%)
=== Performance Report ===
Total steps: 500
Total time: 6.719 seconds
Steps per second: 74.4
Time breakdown:
  - Spatial grid updates: 0.219s (3.3%)
  - Force calculations: 6.479s (96.4%)
  - Position updates: 0.021s (0.3%)
  - Other/overhead: 0.000s (0.0%)
=========================
```

The CPU Rust implementation shines best with smaller bird sizes, yet it still pales in comparison to the CUDA GPU implementation. At its best with only 100 birds, the CPU actually outperforms the GPU, with the Rust implementation achieving 12,428.7 steps per second while the CUDA implementation reaches 10,142.5 steps per second. I believe this is due to the spatial grid which is absent in the CUDA code. However, at the next benchmark of 500 birds, the CUDA implementation is almost twice as fast as Rust, with 6,730.5 steps per second versus 3,401.5 steps per second. At the highest benchmark of 5000 birds, the GPU really highlights its strengths in these types of simulations, with a steps per second of 1,447.4 compared to the Rust implementation's 74.4. With more optimisation to the CUDA code, I believe it would outperform the CPU solution at 100 birds.

It is also worth noting that despite the GPU performing better in the benchmarks, in the visualisation, the Rust implementation outperforms CUDA in frames per second. When running the visualisation, the CUDA implementation hovers around 90-100fps, while the Rust implementation tends to hover around 400-600fps. However, it seems like this discrepancy comes from the 3D rendering of the CUDA code with lit spheres to represent birds, while the Rust code uses flat 2D triangles to represent the birds.

## Reflection

The 2D implementation of a simple particle simulation in Rust proved to be invaluable during the programming for this final lab. I have learnt a lot from this final lab, and I am pleased with not only my implementation, but also from seeing how knowledge gathered from these labs culminate into a slightly larger project like this one. I think the final lab did a great job of demonstrating how parallelisation and concurrency can dramatically impact a program, and how different techniques on different hardware can yield very distinct results.
