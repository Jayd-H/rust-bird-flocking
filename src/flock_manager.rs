use nalgebra::Vector3;
use crate::bird::Bird;

pub struct FlockManager {
    pub birds: Vec<Bird>,
    pub min_bounds: Vector3<f32>,
    pub max_bounds: Vector3<f32>,
    pub separation_weight: f32,
    pub alignment_weight: f32,
    pub cohesion_weight: f32,
    pub dominant_forces: Vec<usize>,
}

impl FlockManager {
    pub fn new(
        num_birds: usize,
        min_bounds: Vector3<f32>,
        max_bounds: Vector3<f32>,
    ) -> Self {
        let mut birds = Vec::with_capacity(num_birds);
        
        for _ in 0..num_birds {
            birds.push(Bird::new(&min_bounds, &max_bounds));
        }
        
        Self {
            birds,
            min_bounds,
            max_bounds,
            separation_weight: 1.5,
            alignment_weight: 1.0,
            cohesion_weight: 1.0,
            dominant_forces: vec![0; num_birds],
        }
    }
    
    pub fn update(&mut self, dt: f32) {
        for i in 0..self.birds.len() {
            let (separation, alignment, cohesion) = self.calculate_flocking_forces(i);
            
            let bird = &mut self.birds[i];
            
            // Apply weighted forces (Algorithm Step 3)
            let sep_force = separation * self.separation_weight;
            let align_force = alignment * self.alignment_weight;
            let coh_force = cohesion * self.cohesion_weight;
            
            // Track dominant force for visualization
            let sep_mag = sep_force.magnitude();
            let align_mag = align_force.magnitude();
            let coh_mag = coh_force.magnitude();
            
            if sep_mag > align_mag && sep_mag > coh_mag {
                self.dominant_forces[i] = 0;
            } else if align_mag > sep_mag && align_mag > coh_mag {
                self.dominant_forces[i] = 1;
            } else {
                self.dominant_forces[i] = 2;
            }
            
            // Apply all forces (Algorithm Step 4)
            bird.apply_force(sep_force);
            bird.apply_force(align_force);
            bird.apply_force(coh_force);
            
            // Update velocity and position (Algorithm Steps 5-6)
            bird.update(dt);
            
            // Apply boundary conditions (Algorithm Step 6)
            bird.apply_boundaries(&self.min_bounds, &self.max_bounds);
        }
    }
    
    fn calculate_flocking_forces(&self, bird_index: usize) -> (Vector3<f32>, Vector3<f32>, Vector3<f32>) {
        let bird = &self.birds[bird_index];
        
        let mut separation_force = Vector3::zeros();
        let mut alignment_force = Vector3::zeros();
        let mut cohesion_force = Vector3::zeros();
        
        let mut neighboring_birds_count = 0;
        let mut center_of_mass = Vector3::zeros();
        let mut average_velocity = Vector3::zeros();
        
        // Calculate forces from nearby birds (Algorithm Step 2)
        for (j, other_bird) in self.birds.iter().enumerate() {
            if j == bird_index {
                continue;
            }
            
            let distance = (bird.position - other_bird.position).magnitude();
            
            if distance < bird.perception_radius {
                // Separation: vector pointing away from neighbor (Algorithm Step 2.1)
                let diff = (bird.position - other_bird.position).normalize();
                separation_force += diff;
                
                // Alignment and Cohesion data collection (Algorithm Steps 2.2-2.3)
                average_velocity += other_bird.velocity;
                center_of_mass += other_bird.position;
                
                neighboring_birds_count += 1;
            }
        }
        
        if neighboring_birds_count > 0 {
            // Alignment: steer towards average velocity (Algorithm Step 2.2)
            average_velocity /= neighboring_birds_count as f32;
            alignment_force = average_velocity - bird.velocity;
            if alignment_force.magnitude() > bird.max_force {
                alignment_force = alignment_force.normalize() * bird.max_force;
            }
            
            // Cohesion: steer towards center of mass (Algorithm Step 2.3)
            center_of_mass /= neighboring_birds_count as f32;
            let desired_velocity = center_of_mass - bird.position;
            cohesion_force = desired_velocity - bird.velocity;
            if cohesion_force.magnitude() > bird.max_force {
                cohesion_force = cohesion_force.normalize() * bird.max_force;
            }
        }
        
        (separation_force, alignment_force, cohesion_force)
    }
    
    pub fn get_dominant_force(&self, bird_index: usize) -> usize {
        self.dominant_forces[bird_index]
    }
    
    pub fn get_birds(&self) -> &Vec<Bird> {
        &self.birds
    }
}