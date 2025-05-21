use nalgebra::Vector3;
use rand::Rng;

//* BIRD CONFIGS!! */
pub const MAX_SPEED: f32 = 10.0;
pub const MAX_FORCE: f32 = 2.0;
pub const PERCEPTION_RADIUS: f32 = 15.0;

pub const INITIAL_VELOCITY_RANGE: f32 = 10.0;
pub const INITIAL_VELOCITY_OFFSET: f32 = 1.0;

#[derive(Debug, Copy, Clone)]
pub struct Bird {
    pub position: Vector3<f32>,
    pub velocity: Vector3<f32>,
    pub acceleration: Vector3<f32>,
    pub max_speed: f32,
    pub max_force: f32,
    pub perception_radius: f32,
}

impl Bird {
    // Creates a new bird with random position and velocity
    pub fn new(min_bounds: &Vector3<f32>, max_bounds: &Vector3<f32>) -> Self {
        let mut rng = rand::thread_rng();

        let position = Vector3::new(
            rng.gen::<f32>() * (max_bounds.x - min_bounds.x) + min_bounds.x,
            rng.gen::<f32>() * (max_bounds.y - min_bounds.y) + min_bounds.y,
            rng.gen::<f32>() * (max_bounds.z - min_bounds.z) + min_bounds.z,
        );

        let velocity = Vector3::new(
            rng.gen::<f32>() * 2.0 - 1.0,
            rng.gen::<f32>() * 2.0 - 1.0,
            rng.gen::<f32>() * 2.0 - 1.0,
        )
        .normalize()
            * (rng.gen::<f32>() * INITIAL_VELOCITY_RANGE + INITIAL_VELOCITY_OFFSET);

        Self {
            position,
            velocity,
            acceleration: Vector3::zeros(),
            max_speed: MAX_SPEED,
            max_force: MAX_FORCE,
            perception_radius: PERCEPTION_RADIUS,
        }
    }
    // Updates the birds velocity and position based on its acceleration
    pub fn update(&mut self, dt: f32) {
        self.velocity += self.acceleration * dt;

        if self.velocity.magnitude() > self.max_speed {
            self.velocity = self.velocity.normalize() * self.max_speed;
        }

        self.position += self.velocity * dt;
        self.acceleration = Vector3::zeros();
    }

    // Applies a force to the bird, affecting its acceleration
    pub fn apply_force(&mut self, force: Vector3<f32>) {
        self.acceleration += force;
    }

    // Ensures the bird stays within the simulation boundaries
    pub fn apply_boundaries(&mut self, min_bounds: &Vector3<f32>, max_bounds: &Vector3<f32>) {
        if self.position.x < min_bounds.x {
            self.position.x = max_bounds.x;
        }
        if self.position.y < min_bounds.y {
            self.position.y = max_bounds.y;
        }
        if self.position.z < min_bounds.z {
            self.position.z = max_bounds.z;
        }
        if self.position.x > max_bounds.x {
            self.position.x = min_bounds.x;
        }
        if self.position.y > max_bounds.y {
            self.position.y = min_bounds.y;
        }
        if self.position.z > max_bounds.z {
            self.position.z = min_bounds.z;
        }
    }
}
