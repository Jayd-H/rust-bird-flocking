use nalgebra::Vector3;
use crate::bird::Bird;
use scoped_threadpool::Pool;
use std::time::{Duration, Instant};

const CELL_SIZE: f32 = 15.0;
const NUM_FORCE_THREADS: usize = 4;
const NUM_UPDATE_THREADS: usize = 4;

//* Performance metrics structure to track time spent in different parts of the simulation */
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
        println!("Steps per second: {:.1}", 
                 self.steps_completed as f32 / self.total_time.as_secs_f32());
        println!("Time breakdown:");
        println!("  - Spatial grid updates: {:.3}s ({:.1}%)", 
                 self.grid_update_time.as_secs_f32(),
                 self.grid_update_time.as_secs_f32() * 100.0 / self.total_time.as_secs_f32());
        println!("  - Force calculations: {:.3}s ({:.1}%)", 
                 self.force_calculation_time.as_secs_f32(),
                 self.force_calculation_time.as_secs_f32() * 100.0 / self.total_time.as_secs_f32());
        println!("  - Position updates: {:.3}s ({:.1}%)", 
                 self.position_update_time.as_secs_f32(),
                 self.position_update_time.as_secs_f32() * 100.0 / self.total_time.as_secs_f32());
        let overhead = self.total_time.as_secs_f32() - 
                       (self.grid_update_time.as_secs_f32() + 
                        self.force_calculation_time.as_secs_f32() + 
                        self.position_update_time.as_secs_f32());
        println!("  - Other/overhead: {:.3}s ({:.1}%)", 
                 overhead,
                 overhead * 100.0 / self.total_time.as_secs_f32());
        println!("=========================");
    }
}

//*  SpatialGrid partitions the 3D space into cells for efficient neighbor lookup */
struct SpatialGrid {
    cells: Vec<Vec<usize>>,
    grid_dim: usize,
}

impl SpatialGrid {
    // Creates a new SpatialGrid covering the simulation boundaries
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
        
        SpatialGrid { cells, grid_dim }
    }
    
    // Clears all cells to prepare for a new update
    fn clear(&mut self) {
        for cell in &mut self.cells {
            cell.clear();
        }
    }
    
    // Adds a bird (by its index) to the correct cell based on its position
    fn add_bird(&mut self, bird_idx: usize, bird: &Bird) {
        let (x_cell, y_cell, z_cell) = self.get_cell_coords(bird);
        let cell_idx = z_cell * self.grid_dim * self.grid_dim + y_cell * self.grid_dim + x_cell;
        if cell_idx < self.cells.len() {
            self.cells[cell_idx].push(bird_idx);
        }
    }
    
    // Computes the cell coordinates for a given bird
    fn get_cell_coords(&self, bird: &Bird) -> (usize, usize, usize) {
        let x_cell = (bird.position.x / CELL_SIZE).floor().max(0.0).min((self.grid_dim - 1) as f32) as usize;
        let y_cell = (bird.position.y / CELL_SIZE).floor().max(0.0).min((self.grid_dim - 1) as f32) as usize;
        let z_cell = (bird.position.z / CELL_SIZE).floor().max(0.0).min((self.grid_dim - 1) as f32) as usize;
        (x_cell, y_cell, z_cell)
    }
    
    // Retrieves the indices of birds located in neighboring cells
    fn get_neighbor_indices(&self, bird: &Bird) -> Vec<usize> {
        let (x_cell, y_cell, z_cell) = self.get_cell_coords(bird);
        let mut neighbors = Vec::new();
        for dz in -1..=1 {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let nx = x_cell as isize + dx;
                    let ny = y_cell as isize + dy;
                    let nz = z_cell as isize + dz;
                    if nx < 0 || ny < 0 || nz < 0 || 
                       nx >= self.grid_dim as isize || 
                       ny >= self.grid_dim as isize || 
                       nz >= self.grid_dim as isize {
                        continue;
                    }
                    let cell_idx = (nz as usize) * self.grid_dim * self.grid_dim +
                                   (ny as usize) * self.grid_dim +
                                   (nx as usize);
                    if cell_idx < self.cells.len() {
                        neighbors.extend(&self.cells[cell_idx]);
                    }
                }
            }
        }
        neighbors
    }
}

//* FlockManager handles the birds, flocking behavior, and parallel updates, could be called bird manager realistically */
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
    pub performance: PerformanceMetrics,
}

impl FlockManager {
    // Creates and initializes a new FlockManager with birds and thread pools
    pub fn new(num_birds: usize, min_bounds: Vector3<f32>, max_bounds: Vector3<f32>) -> Self {
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
            separation_weight: 1.5,
            alignment_weight: 1.0,
            cohesion_weight: 1.0,
            dominant_forces,
            spatial_grid: SpatialGrid::new(&min_bounds, &max_bounds),
            force_pool: Pool::new(NUM_FORCE_THREADS as u32),
            update_pool: Pool::new(NUM_UPDATE_THREADS as u32),
            performance: PerformanceMetrics::new(),
        }
    }
    
    // Updates the spatial grid by placing each bird into its corresponding cell
    fn update_spatial_grid(&mut self) {
        self.spatial_grid.clear();
        for (idx, bird) in self.current_birds.iter().enumerate() {
            self.spatial_grid.add_bird(idx, bird);
        }
    }
    
    // Performs a simulation update with performance tracking
    pub fn update(&mut self, dt: f32) {
        let start_time = Instant::now();
        
        // Update spatial grid
        let grid_start = Instant::now();
        self.update_spatial_grid();
        self.performance.grid_update_time += grid_start.elapsed();
        
        let mut thread_dominant_forces: Vec<Vec<(usize, usize)>> = vec![Vec::new(); NUM_FORCE_THREADS];
        
        // Calculate forces
        let force_start = Instant::now();
        {
            let current_birds = &self.current_birds;
            let spatial_grid = &self.spatial_grid;
            let separation_weight = self.separation_weight;
            let alignment_weight = self.alignment_weight;
            let cohesion_weight = self.cohesion_weight;
            let next_birds_slice = self.next_birds.as_mut_slice();
            let chunk_size = (current_birds.len() + NUM_FORCE_THREADS - 1) / NUM_FORCE_THREADS;
            
            self.force_pool.scoped(|scope| {
                for ((chunk_idx, next_birds_chunk), thread_forces) in 
                    next_birds_slice.chunks_mut(chunk_size).enumerate().zip(thread_dominant_forces.iter_mut())
                {
                    let start_idx = chunk_idx * chunk_size;
                    scope.execute(move || {
                        for (i, next_bird) in next_birds_chunk.iter_mut().enumerate() {
                            let global_idx = start_idx + i;
                            if global_idx >= current_birds.len() {
                                continue;
                            }
                            
                            let bird = &current_birds[global_idx];
                            *next_bird = *bird;
                            let neighbors = spatial_grid.get_neighbor_indices(bird);
                            let mut separation_force = Vector3::zeros();
                            let mut alignment_force = Vector3::zeros();
                            let mut cohesion_force = Vector3::zeros();
                            let mut neighboring_birds_count = 0;
                            let mut center_of_mass = Vector3::zeros();
                            let mut average_velocity = Vector3::zeros();
                            
                            for &j in &neighbors {
                                if global_idx == j {
                                    continue;
                                }
                                let other_bird = &current_birds[j];
                                let distance = (bird.position - other_bird.position).magnitude();
                                if distance < bird.perception_radius {
                                    separation_force += (bird.position - other_bird.position).normalize();
                                    average_velocity += other_bird.velocity;
                                    center_of_mass += other_bird.position;
                                    neighboring_birds_count += 1;
                                }
                            }
                            
                            if neighboring_birds_count > 0 {
                                average_velocity /= neighboring_birds_count as f32;
                                alignment_force = average_velocity - bird.velocity;
                                if alignment_force.magnitude() > bird.max_force {
                                    alignment_force = alignment_force.normalize() * bird.max_force;
                                }
                                
                                center_of_mass /= neighboring_birds_count as f32;
                                let desired_velocity = center_of_mass - bird.position;
                                cohesion_force = desired_velocity - bird.velocity;
                                if cohesion_force.magnitude() > bird.max_force {
                                    cohesion_force = cohesion_force.normalize() * bird.max_force;
                                }
                            }
                            
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
                            
                            thread_forces.push((global_idx, dominant_force));
                            next_bird.apply_force(sep_force);
                            next_bird.apply_force(align_force);
                            next_bird.apply_force(coh_force);
                        }
                    });
                }
            });
        }
        self.performance.force_calculation_time += force_start.elapsed();
        
        for thread_forces in thread_dominant_forces {
            for (idx, force) in thread_forces {
                if idx < self.dominant_forces.len() {
                    self.dominant_forces[idx] = force;
                }
            }
        }
        
        // Update positions
        let update_start = Instant::now();
        {
            let next_birds_len = self.next_birds.len();
            let chunk_size = (next_birds_len + NUM_UPDATE_THREADS - 1) / NUM_UPDATE_THREADS;
            
            let next_birds_slice = self.next_birds.as_mut_slice();
            let min_bounds = self.min_bounds;
            let max_bounds = self.max_bounds;
            
            self.update_pool.scoped(|scope| {
                for next_birds_chunk in next_birds_slice.chunks_mut(chunk_size) {
                    scope.execute(move || {
                        for bird in next_birds_chunk.iter_mut() {
                            bird.update(dt);
                            bird.apply_boundaries(&min_bounds, &max_bounds);
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
    
    // Runs the simulation for a fixed number of steps without rendering
    pub fn run_benchmark(&mut self, steps: usize, dt: f32) {
        self.performance.reset();
        println!("Running benchmark for {} steps...", steps);
        
        for i in 1..=steps {
            self.update(dt);
            
            if i % 100 == 0 || i == steps {
                println!("Completed {} steps ({:.1}%)", i, (i as f32 / steps as f32) * 100.0);
            }
        }
        
        self.performance.report();
    }
    
    // Returns the dominant force index for the bird at the given index
    pub fn get_dominant_force(&self, bird_index: usize) -> usize {
        self.dominant_forces[bird_index]
    }
    
    // Returns a reference to the current bird vector
    pub fn get_birds(&self) -> &Vec<Bird> {
        &self.current_birds
    }
    
    // Returns the number of birds in the simulation
    pub fn get_bird_count(&self) -> usize {
        self.current_birds.len()
    }
}