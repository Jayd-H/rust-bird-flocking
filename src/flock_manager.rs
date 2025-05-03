use crate::bird::Bird;
use nalgebra::Vector3;
use scoped_threadpool::Pool;
use std::sync::Arc;
use std::time::{Duration, Instant};

const CELL_SIZE: f32 = 30.0;
const EPSILON: f32 = 1e-6;
const SEPARATION_RADIUS_FACTOR: f32 = 0.3;

pub struct PerformanceMetrics {
    pub grid_update_time: Duration,
    pub force_calculation_time: Duration,
    pub position_update_time: Duration,
    pub steps_completed: usize,
    pub total_time: Duration,
}

impl PerformanceMetrics {
    pub fn new() -> Self {
        PerformanceMetrics {
            grid_update_time: Duration::new(0, 0),
            force_calculation_time: Duration::new(0, 0),
            position_update_time: Duration::new(0, 0),
            steps_completed: 0,
            total_time: Duration::new(0, 0),
        }
    }

    pub fn reset(&mut self) {
        *self = PerformanceMetrics::new();
    }

    pub fn report(&self) {
        println!("=== Performance Report ===");
        println!("Total steps: {}", self.steps_completed);
        println!("Total time: {:.3} seconds", self.total_time.as_secs_f32());
        println!(
            "Steps per second: {:.1}",
            self.steps_completed as f32 / self.total_time.as_secs_f32()
        );
        println!("Time breakdown:");
        println!(
            "  - Spatial grid updates: {:.3}s ({:.1}%)",
            self.grid_update_time.as_secs_f32(),
            self.grid_update_time.as_secs_f32() * 100.0 / self.total_time.as_secs_f32()
        );
        println!(
            "  - Force calculations: {:.3}s ({:.1}%)",
            self.force_calculation_time.as_secs_f32(),
            self.force_calculation_time.as_secs_f32() * 100.0 / self.total_time.as_secs_f32()
        );
        println!(
            "  - Position updates: {:.3}s ({:.1}%)",
            self.position_update_time.as_secs_f32(),
            self.position_update_time.as_secs_f32() * 100.0 / self.total_time.as_secs_f32()
        );
        let overhead = self.total_time.as_secs_f32()
            - (self.grid_update_time.as_secs_f32()
                + self.force_calculation_time.as_secs_f32()
                + self.position_update_time.as_secs_f32());
        println!(
            "  - Other/overhead: {:.3}s ({:.1}%)",
            overhead,
            overhead * 100.0 / self.total_time.as_secs_f32()
        );
        println!("=========================");
    }
}

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

    fn clear(&mut self) {
        for cell in &mut self.cells {
            cell.clear();
        }
    }

    fn get_cell_coords(&self, position: &Vector3<f32>) -> (usize, usize, usize) {
        let x_cell = ((position.x - self.min_bounds.x) / self.cell_size)
            .floor()
            .max(0.0)
            .min((self.dim_x - 1) as f32) as usize;
        let y_cell = ((position.y - self.min_bounds.y) / self.cell_size)
            .floor()
            .max(0.0)
            .min((self.dim_y - 1) as f32) as usize;
        let z_cell = ((position.z - self.min_bounds.z) / self.cell_size)
            .floor()
            .max(0.0)
            .min((self.dim_z - 1) as f32) as usize;
        (x_cell, y_cell, z_cell)
    }

    fn get_cell_index(&self, x: usize, y: usize, z: usize) -> usize {
        x + y * self.dim_x + z * (self.dim_x * self.dim_y)
    }

    fn get_overlapping_cells(&self, bird: &Bird) -> Vec<(usize, usize, usize)> {
        let perception_radius = bird.perception_radius;

        let min_pos = Vector3::new(
            bird.position.x - perception_radius,
            bird.position.y - perception_radius,
            bird.position.z - perception_radius,
        );

        let max_pos = Vector3::new(
            bird.position.x + perception_radius,
            bird.position.y + perception_radius,
            bird.position.z + perception_radius,
        );

        let (min_x, min_y, min_z) = self.get_cell_coords(&min_pos);
        let (max_x, max_y, max_z) = self.get_cell_coords(&max_pos);

        let mut overlapping_cells = Vec::new();
        for x in min_x..=max_x {
            for y in min_y..=max_y {
                for z in min_z..=max_z {
                    if x < self.dim_x && y < self.dim_y && z < self.dim_z {
                        overlapping_cells.push((x, y, z));
                    }
                }
            }
        }

        overlapping_cells
    }

    fn add_bird(&mut self, bird_idx: usize, bird: &Bird) {
        for (x_cell, y_cell, z_cell) in self.get_overlapping_cells(bird) {
            let cell_idx = self.get_cell_index(x_cell, y_cell, z_cell);
            if cell_idx < self.cells.len() {
                self.cells[cell_idx].push(bird_idx);
            }
        }
    }

    fn get_neighbor_indices(&self, bird: &Bird) -> Vec<usize> {
        let (x_cell, y_cell, z_cell) = self.get_cell_coords(&bird.position);
        let cell_idx = self.get_cell_index(x_cell, y_cell, z_cell);

        if cell_idx < self.cells.len() {
            self.cells[cell_idx].clone()
        } else {
            Vec::new()
        }
    }
}

pub struct FlockManager {
    pub current_birds: Arc<Vec<Bird>>,
    pub next_birds: Vec<Bird>,
    pub min_bounds: Vector3<f32>,
    pub max_bounds: Vector3<f32>,
    pub separation_weight: f32,
    pub alignment_weight: f32,
    pub cohesion_weight: f32,
    pub dominant_forces: Vec<usize>,
    spatial_grid: SpatialGrid,
    force_pool: Pool,
    update_pool: Pool,
    force_thread_count: usize,
    update_thread_count: usize,
    pub performance: PerformanceMetrics,
}

impl FlockManager {
    // Creates a new flock manager with the specified number of birds and thread counts
    pub fn new(
        num_birds: usize,
        min_bounds: Vector3<f32>,
        max_bounds: Vector3<f32>,
        force_threads: usize,
        update_threads: usize,
    ) -> Self {
        // Create birds in next_birds first
        let mut next_birds = Vec::with_capacity(num_birds);
        for _ in 0..num_birds {
            next_birds.push(Bird::new(&min_bounds, &max_bounds));
        }

        // Move birds into Arc without cloning
        let current_birds = Arc::new(std::mem::take(&mut next_birds));

        // Reserve capacity for next_birds for future use
        next_birds = Vec::with_capacity(num_birds);

        FlockManager {
            current_birds,
            next_birds,
            min_bounds,
            max_bounds,
            separation_weight: 3.0,
            alignment_weight: 1.0,
            cohesion_weight: 0.5,
            dominant_forces: vec![0; num_birds],
            spatial_grid: SpatialGrid::new(&min_bounds, &max_bounds),
            force_pool: Pool::new(force_threads as u32),
            update_pool: Pool::new(update_threads as u32),
            force_thread_count: force_threads,
            update_thread_count: update_threads,
            performance: PerformanceMetrics::new(),
        }
    }

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

    // Calculates separation, alignment and cohesion forces for a bird
    fn calculate_forces(
        bird_index: usize,
        birds: &[Bird],
        neighbor_indices: &[usize],
        min_bounds: &Vector3<f32>,
        max_bounds: &Vector3<f32>,
        separation_weight: f32,
        alignment_weight: f32,
        cohesion_weight: f32,
    ) -> (Vector3<f32>, Vector3<f32>, Vector3<f32>, usize) {
        let bird = &birds[bird_index];

        let mut separation_force = Vector3::zeros();
        let mut average_velocity = Vector3::zeros();
        let mut cohesion_center = Vector3::zeros();
        let mut neighboring_birds_count = 0;

        for &j in neighbor_indices {
            if bird_index == j || j >= birds.len() {
                continue;
            }

            let other_bird = &birds[j];
            let wrapped_diff = Self::calculate_wrapped_distance(
                &bird.position,
                &other_bird.position,
                min_bounds,
                max_bounds,
            );
            let dist_sq = wrapped_diff.magnitude_squared();

            if dist_sq > EPSILON
                && dist_sq
                    < (bird.perception_radius * SEPARATION_RADIUS_FACTOR)
                        * (bird.perception_radius * SEPARATION_RADIUS_FACTOR)
            {
                separation_force += wrapped_diff.normalize() * (1.0 / dist_sq);
            }

            if dist_sq < bird.perception_radius * bird.perception_radius {
                average_velocity += other_bird.velocity;

                let other_pos = bird.position - wrapped_diff;
                cohesion_center += other_pos;

                neighboring_birds_count += 1;
            }
        }

        let alignment_force = if neighboring_birds_count > 0 {
            average_velocity /= neighboring_birds_count as f32;
            let mut force = average_velocity - bird.velocity;

            if force.magnitude() > bird.max_force {
                force = force.normalize() * bird.max_force;
            }
            force
        } else {
            Vector3::zeros()
        };

        let cohesion_force = if neighboring_birds_count > 0 {
            cohesion_center /= neighboring_birds_count as f32;
            let desired_velocity = cohesion_center - bird.position;
            let mut force = desired_velocity;

            if force.magnitude() > bird.max_force {
                force = force.normalize() * bird.max_force;
            }
            force
        } else {
            Vector3::zeros()
        };

        let sep_force = separation_force * separation_weight;
        let align_force = alignment_force * alignment_weight;
        let coh_force = cohesion_force * cohesion_weight;

        let sep_mag = sep_force.magnitude();
        let align_mag = align_force.magnitude();
        let coh_mag = coh_force.magnitude();

        let dominant_force = if sep_mag > align_mag && sep_mag > coh_mag {
            0
        } else if align_mag > sep_mag && align_mag > coh_mag {
            1
        } else {
            2
        };

        (sep_force, align_force, coh_force, dominant_force)
    }

    // Gets a reference to the current birds for rendering
    pub fn get_birds(&self) -> &[Bird] {
        &self.current_birds
    }

    // Gets the total number of birds in the simulation
    pub fn get_bird_count(&self) -> usize {
        self.current_birds.len()
    }

    // Gets the dominant force for a specific bird
    pub fn get_dominant_force(&self, bird_index: usize) -> usize {
        self.dominant_forces[bird_index]
    }

    // Updates the flock simulation by one timestep
    pub fn update(&mut self, dt: f32) {
        let start_time = Instant::now();

        // Update spatial grid using current birds
        let grid_start = Instant::now();
        self.spatial_grid.clear();
        for (idx, bird) in self.current_birds.iter().enumerate() {
            self.spatial_grid.add_bird(idx, bird);
        }
        self.performance.grid_update_time += grid_start.elapsed();

        // Force calculation
        let force_start = Instant::now();

        // Pre-collect all neighbor indices
        let birds_with_neighbors: Vec<Vec<usize>> = self
            .current_birds
            .iter()
            .map(|bird| self.spatial_grid.get_neighbor_indices(bird))
            .collect();

        // Prepare thread local storage for results
        let birds_count = self.current_birds.len();
        let mut thread_results = vec![Vec::new(); self.force_thread_count];

        // Ensure next_birds is the right size
        if self.next_birds.len() != birds_count {
            self.next_birds = Vec::with_capacity(birds_count);
            self.next_birds.resize_with(birds_count, || {
                Bird::new(&self.min_bounds, &self.max_bounds)
            });
        }

        // Scope all Arc references to ensure they're dropped before buffer swap
        {
            // Copy configuration data that will be shared with threads
            let min_bounds = self.min_bounds;
            let max_bounds = self.max_bounds;
            let separation_weight = self.separation_weight;
            let alignment_weight = self.alignment_weight;
            let cohesion_weight = self.cohesion_weight;
            let thread_count = self.force_thread_count;
            let birds_arc = Arc::clone(&self.current_birds);
            let birds_with_neighbors_arc = Arc::new(birds_with_neighbors);
            let chunk_size = (birds_count + thread_count - 1) / thread_count;

            // Process in parallel
            self.force_pool.scoped(|scope| {
                for (thread_idx, thread_result) in thread_results.iter_mut().enumerate() {
                    let start_idx = thread_idx * chunk_size;
                    let end_idx = (start_idx + chunk_size).min(birds_count);

                    let birds_ref = Arc::clone(&birds_arc);
                    let neighbors_ref = Arc::clone(&birds_with_neighbors_arc);

                    let mut local_results = Vec::with_capacity(end_idx - start_idx);

                    scope.execute(move || {
                        for bird_idx in start_idx..end_idx {
                            // Calculate forces for this bird
                            let (sep, align, coh, dominant) = Self::calculate_forces(
                                bird_idx,
                                &birds_ref,
                                &neighbors_ref[bird_idx],
                                &min_bounds,
                                &max_bounds,
                                separation_weight,
                                alignment_weight,
                                cohesion_weight,
                            );

                            // Store result for this bird
                            local_results.push((bird_idx, sep, align, coh, dominant));
                        }

                        *thread_result = local_results;
                    });
                }
            });
            // All Arc clones are dropped here when this scope ends
        }

        // Apply forces to next_birds (sequential operation)
        //TODO : Realistically, this should be parallelized too but for now, it's not worth the effort
        for thread_result in &thread_results {
            for &(bird_idx, sep, align, coh, dominant) in thread_result {
                if bird_idx < self.next_birds.len() {
                    self.next_birds[bird_idx] = self.current_birds[bird_idx];
                    self.next_birds[bird_idx].apply_force(sep);
                    self.next_birds[bird_idx].apply_force(align);
                    self.next_birds[bird_idx].apply_force(coh);
                    self.dominant_forces[bird_idx] = dominant;
                }
            }
        }

        self.performance.force_calculation_time += force_start.elapsed();

        // Update positions in parallel
        let update_start = Instant::now();

        // Prepare data for position updates
        let update_thread_count = self.update_thread_count;
        let next_birds_len = self.next_birds.len();
        let min_bounds = self.min_bounds;
        let max_bounds = self.max_bounds;
        let chunk_size = (next_birds_len + update_thread_count - 1) / update_thread_count;

        // Split next_birds into chunks and process in parallel
        self.update_pool.scoped(|scope| {
            for next_birds_chunk in self.next_birds.chunks_mut(chunk_size) {
                let min_bounds_copy = min_bounds;
                let max_bounds_copy = max_bounds;

                scope.execute(move || {
                    for bird in next_birds_chunk {
                        bird.update(dt);
                        bird.apply_boundaries(&min_bounds_copy, &max_bounds_copy);
                    }
                });
            }
        });

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
    }

    // Runs a benchmark for the specified number of steps
    pub fn run_benchmark(&mut self, steps: usize, dt: f32) {
        self.performance.reset();
        println!("Running benchmark for {} steps...", steps);

        for i in 1..=steps {
            self.update(dt);

            if i % 100 == 0 || i == steps {
                println!(
                    "Completed {} steps ({:.1}%)",
                    i,
                    (i as f32 / steps as f32) * 100.0
                );
            }
        }

        self.performance.report();
    }
}
