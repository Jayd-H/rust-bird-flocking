use nalgebra::Vector3;
use rand::Rng;

pub struct Bird {
    pub position: Vector3<f32>,
    pub velocity: Vector3<f32>,
    pub acceleration: Vector3<f32>,
    pub max_speed: f32,
    pub max_force: f32,
    pub perception_radius: f32,
}

impl Bird {
    pub fn new(min_bounds: &Vector3<f32>, max_bounds: &Vector3<f32>) -> Self {
        let mut rng = rand::thread_rng();
        
        // Initialize position randomly (Algorithm Step 1)
        let position = Vector3::new(
            rng.gen::<f32>() * (max_bounds.x - min_bounds.x) + min_bounds.x,
            rng.gen::<f32>() * (max_bounds.y - min_bounds.y) + min_bounds.y,
            rng.gen::<f32>() * (max_bounds.z - min_bounds.z) + min_bounds.z,
        );
        
        // Initialize velocity randomly (Algorithm Step 1)
        let velocity = Vector3::new(
            rng.gen::<f32>() * 2.0 - 1.0,
            rng.gen::<f32>() * 2.0 - 1.0,
            rng.gen::<f32>() * 2.0 - 1.0,
        ).normalize() * (rng.gen::<f32>() * 1.5 + 0.5);
        
        Self {
            position,
            velocity,
            acceleration: Vector3::zeros(),
            max_speed: 3.0,
            max_force: 0.1,
            perception_radius: 15.0,
        }
    }
    
    pub fn update(&mut self, dt: f32) {
        // Update velocity based on acceleration (Algorithm Step 5)
        self.velocity += self.acceleration * dt;
        
        // Limit velocity to max_speed (Algorithm Step 5)
        if self.velocity.magnitude() > self.max_speed {
            self.velocity = self.velocity.normalize() * self.max_speed;
        }
        
        // Update position based on velocity (Algorithm Step 6)
        self.position += self.velocity * dt;
        
        // Reset acceleration for next frame
        self.acceleration = Vector3::zeros();
    }
    
    pub fn apply_force(&mut self, force: Vector3<f32>) {
        // Add force to acceleration (Algorithm Step 4)
        self.acceleration += force;
    }
    
    pub fn apply_boundaries(&mut self, min_bounds: &Vector3<f32>, max_bounds: &Vector3<f32>) {
        // Wrap around at boundaries (Algorithm Step 6)
        if self.position.x < min_bounds.x { self.position.x = max_bounds.x; }
        if self.position.y < min_bounds.y { self.position.y = max_bounds.y; }
        if self.position.z < min_bounds.z { self.position.z = max_bounds.z; }
        if self.position.x > max_bounds.x { self.position.x = min_bounds.x; }
        if self.position.y > max_bounds.y { self.position.y = min_bounds.y; }
        if self.position.z > max_bounds.z { self.position.z = min_bounds.z; }
    }
}