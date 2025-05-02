use crate::bird::Bird;
use nalgebra::Vector3;
use scoped_threadpool::Pool;
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
    grid_dim: usize,
    min_bounds: Vector3<f32>,
    cell_size: f32,
}

impl SpatialGrid {
    fn new(min_bounds: &Vector3<f32>, max_bounds: &Vector3<f32>) -> Self {
        let size_x = max_bounds.x - min_bounds.x;
        let size_y = max_bounds.y - min_bounds.y;
        let size_z = max_bounds.z - min_bounds.z;
        let max_size = size_x.max(size_y).max(size_z);
        let grid_dim = (max_size / CELL_SIZE).ceil() as usize + 1;

        let mut cells = Vec::with_capacity(grid_dim * grid_dim * grid_dim);
        for _ in 0..(grid_dim * grid_dim * grid_dim) {
            cells.push(Vec::new());
        }

        SpatialGrid {
            cells,
            grid_dim,
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
            .min((self.grid_dim - 1) as f32) as usize;
        let y_cell = ((position.y - self.min_bounds.y) / self.cell_size)
            .floor()
            .max(0.0)
            .min((self.grid_dim - 1) as f32) as usize;
        let z_cell = ((position.z - self.min_bounds.z) / self.cell_size)
            .floor()
            .max(0.0)
            .min((self.grid_dim - 1) as f32) as usize;
        (x_cell, y_cell, z_cell)
    }

    fn get_overlapping_cells(&self, bird: &Bird) -> Vec<(usize, usize, usize)> {
        let perception_radius = bird.perception_radius;

        // Create a position at the minimum corner of the bird's perception radius
        let min_pos = Vector3::new(
            bird.position.x - perception_radius,
            bird.position.y - perception_radius,
            bird.position.z - perception_radius,
        );

        // Create a position at the maximum corner of the bird's perception radius
        let max_pos = Vector3::new(
            bird.position.x + perception_radius,
            bird.position.y + perception_radius,
            bird.position.z + perception_radius,
        );

        // Get the cell coordinates for the min and max positions
        let (min_x, min_y, min_z) = self.get_cell_coords(&min_pos);
        let (max_x, max_y, max_z) = self.get_cell_coords(&max_pos);

        // Generate all cell coordinates within this range
        let mut overlapping_cells = Vec::new();
        for x in min_x..=max_x {
            for y in min_y..=max_y {
                for z in min_z..=max_z {
                    if x < self.grid_dim && y < self.grid_dim && z < self.grid_dim {
                        overlapping_cells.push((x, y, z));
                    }
                }
            }
        }

        overlapping_cells
    }

    fn add_bird(&mut self, bird_idx: usize, bird: &Bird) {
        // Add bird to all cells its perception radius overlaps
        for (x_cell, y_cell, z_cell) in self.get_overlapping_cells(bird) {
            let cell_idx = z_cell * self.grid_dim * self.grid_dim + y_cell * self.grid_dim + x_cell;
            if cell_idx < self.cells.len() {
                self.cells[cell_idx].push(bird_idx);
            }
        }
    }

    fn get_neighbor_indices(&self, bird: &Bird) -> Vec<usize> {
        let (x_cell, y_cell, z_cell) = self.get_cell_coords(&bird.position);
        let cell_idx = z_cell * self.grid_dim * self.grid_dim + y_cell * self.grid_dim + x_cell;

        if cell_idx < self.cells.len() {
            self.cells[cell_idx].clone()
        } else {
            Vec::new()
        }
    }
}

pub struct FlockManager {
    pub current_birds: Vec<Bird>,
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
    pub fn new(
        num_birds: usize,
        min_bounds: Vector3<f32>,
        max_bounds: Vector3<f32>,
        force_threads: usize,
        update_threads: usize,
    ) -> Self {
        let mut current_birds = Vec::with_capacity(num_birds);
        for _ in 0..num_birds {
            current_birds.push(Bird::new(&min_bounds, &max_bounds));
        }
        let next_birds = current_birds.clone();
        let dominant_forces = vec![0; num_birds];

        FlockManager {
            current_birds,
            next_birds,
            min_bounds,
            max_bounds,
            separation_weight: 3.0,
            alignment_weight: 1.0,
            cohesion_weight: 0.5,
            dominant_forces,
            spatial_grid: SpatialGrid::new(&min_bounds, &max_bounds),
            force_pool: Pool::new(force_threads as u32),
            update_pool: Pool::new(update_threads as u32),
            force_thread_count: force_threads,
            update_thread_count: update_threads,
            performance: PerformanceMetrics::new(),
        }
    }

    fn update_spatial_grid(&mut self) {
        self.spatial_grid.clear();
        for (idx, bird) in self.current_birds.iter().enumerate() {
            self.spatial_grid.add_bird(idx, bird);
        }
    }

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

        // Calculate separation force
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
                // For alignment
                average_velocity += other_bird.velocity;

                // For cohesion - calculate the wrapped position of the other bird
                let other_pos = bird.position - wrapped_diff;
                cohesion_center += other_pos;

                neighboring_birds_count += 1;
            }
        }

        // Finalize alignment force
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

        // Finalize cohesion force
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

        // Apply weights
        let sep_force = separation_force * separation_weight;
        let align_force = alignment_force * alignment_weight;
        let coh_force = cohesion_force * cohesion_weight;

        // Determine dominant force
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

    pub fn update(&mut self, dt: f32) {
        let start_time = Instant::now();

        let grid_start = Instant::now();
        self.update_spatial_grid();
        self.performance.grid_update_time += grid_start.elapsed();

        let force_start = Instant::now();

        // Pre-collect all neighbor indices
        let birds_with_neighbors: Vec<Vec<usize>> = self
            .current_birds
            .iter()
            .map(|bird| self.spatial_grid.get_neighbor_indices(bird))
            .collect();

        // Create thread-safe copies of shared data
        let birds = self.current_birds.clone();
        let min_bounds = self.min_bounds;
        let max_bounds = self.max_bounds;
        let separation_weight = self.separation_weight;
        let alignment_weight = self.alignment_weight;
        let cohesion_weight = self.cohesion_weight;

        // Use force_thread_count instead of NUM_FORCE_THREADS
        let mut thread_forces: Vec<Vec<(usize, Vector3<f32>, Vector3<f32>, Vector3<f32>, usize)>> =
            vec![Vec::new(); self.force_thread_count];

        // Use force_thread_count instead of NUM_FORCE_THREADS
        let chunk_size =
            (self.current_birds.len() + self.force_thread_count - 1) / self.force_thread_count;

        self.force_pool.scoped(|scope| {
            for (chunk_idx, thread_result) in thread_forces.iter_mut().enumerate() {
                let start_idx = chunk_idx * chunk_size;
                let end_idx = (start_idx + chunk_size).min(birds.len());

                // Create thread-local copies
                let birds_copy = birds.clone();
                let birds_with_neighbors_copy = birds_with_neighbors.clone();
                let min_bounds_copy = min_bounds;
                let max_bounds_copy = max_bounds;
                let sep_weight_copy = separation_weight;
                let align_weight_copy = alignment_weight;
                let cohesion_weight_copy = cohesion_weight;

                scope.execute(move || {
                    let mut local_results = Vec::new();

                    for bird_idx in start_idx..end_idx {
                        let (sep, align, coh, dominant) = Self::calculate_forces(
                            bird_idx,
                            &birds_copy,
                            &birds_with_neighbors_copy[bird_idx],
                            &min_bounds_copy,
                            &max_bounds_copy,
                            sep_weight_copy,
                            align_weight_copy,
                            cohesion_weight_copy,
                        );

                        local_results.push((bird_idx, sep, align, coh, dominant));
                    }

                    *thread_result = local_results;
                });
            }
        });

        // Apply the forces to next_birds
        for thread_result in &thread_forces {
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

        let update_start = Instant::now();

        let next_birds_len = self.next_birds.len();
        let min_bounds = self.min_bounds;
        let max_bounds = self.max_bounds;

        // Use update_thread_count instead of NUM_UPDATE_THREADS
        let chunk_size = (next_birds_len + self.update_thread_count - 1) / self.update_thread_count;

        // Update positions in parallel
        {
            let next_birds_slice = self.next_birds.as_mut_slice();

            self.update_pool.scoped(|scope| {
                for next_birds_chunk in next_birds_slice.chunks_mut(chunk_size) {
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
        }

        self.performance.position_update_time += update_start.elapsed();

        std::mem::swap(&mut self.current_birds, &mut self.next_birds);

        self.performance.steps_completed += 1;
        self.performance.total_time += start_time.elapsed();
    }

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

    pub fn get_dominant_force(&self, bird_index: usize) -> usize {
        self.dominant_forces[bird_index]
    }

    pub fn get_birds(&self) -> &Vec<Bird> {
        &self.current_birds
    }

    pub fn get_bird_count(&self) -> usize {
        self.current_birds.len()
    }
}
