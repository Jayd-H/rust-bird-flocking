#[macro_use]
extern crate glium;
extern crate winit;

use nalgebra::{Matrix4, Perspective3, Point3, Vector3};
use std::time::Instant;

mod bird;
mod flock_manager;
use flock_manager::FlockManager;

fn main() {
    use glium::{glutin, Surface};
    let event_loop = winit::event_loop::EventLoop::new();
    let wb = winit::window::WindowBuilder::new()
        .with_inner_size(winit::dpi::LogicalSize::new(1024.0, 768.0))
        .with_title("Bird Flocking Simulation");
    let cb = glutin::ContextBuilder::new().with_depth_buffer(24);
    let display = glium::Display::new(wb, cb, &event_loop).unwrap();

    #[derive(Copy, Clone)]
    struct Vertex {
        position: [f32; 3],
    }

    implement_vertex!(Vertex, position);

    let vertex1 = Vertex { position: [-0.5, -0.3, 0.0] };
    let vertex2 = Vertex { position: [0.0, 0.6, 0.0] };
    let vertex3 = Vertex { position: [0.5, -0.3, 0.0] };
    let shape = vec![vertex1, vertex2, vertex3];
    let vertex_buffer = glium::VertexBuffer::new(&display, &shape).unwrap();
    let indices = glium::index::NoIndices(glium::index::PrimitiveType::TrianglesList);

    let vertex_shader_src = r#"
        #version 140
        in vec3 position;
        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;
        out vec3 frag_pos;
        
        void main() {
            frag_pos = vec3(model * vec4(position, 1.0));
            gl_Position = projection * view * model * vec4(position, 1.0);
        }
    "#;

    let fragment_shader_src = r#"
        #version 140
        uniform vec3 bird_color;
        uniform vec3 camera_pos;
        in vec3 frag_pos;
        out vec4 color;
        
        void main() {
            float dist = distance(frag_pos, camera_pos);
            float attenuation = 1.0 / (1.0 + 0.001 * dist + 0.00001 * dist * dist);
            attenuation = clamp(attenuation, 0.3, 1.0);
            color = vec4(bird_color * attenuation, 1.0);
        }
    "#;

    let program = glium::Program::from_source(&display, vertex_shader_src, fragment_shader_src, None).unwrap();

    let min_bounds = Vector3::new(-50.0, -50.0, -50.0);
    let max_bounds = Vector3::new(50.0, 50.0, 50.0);
    
    let num_birds = 200;
    let mut flock = FlockManager::new(num_birds, min_bounds, max_bounds);
    
    let mut last_update = Instant::now();
    let mut simulation_time = 0.0;

    let cam_x = 0.0;
    let cam_y = 0.0;
    let cam_z = 120.0;

    event_loop.run(move |event, _, control_flow| {
        match event {
            winit::event::Event::WindowEvent { event, .. } => match event {
                winit::event::WindowEvent::CloseRequested => {
                    *control_flow = winit::event_loop::ControlFlow::Exit;
                    return;
                },
                _ => return,
            },
            winit::event::Event::RedrawRequested(_) => {
                let now = Instant::now();
                let dt = now.duration_since(last_update).as_secs_f32();
                last_update = now;
                
                simulation_time += dt;
                flock.update(dt);
                
                let mut target = display.draw();
                target.clear_color_and_depth((0.1, 0.1, 0.15, 1.0), 1.0);

                let eye = Point3::new(cam_x, cam_y, cam_z);
                let target_point = Point3::new(0.0, 0.0, 0.0);
                let up = Vector3::new(0.0, 1.0, 0.0);
                let view_matrix: [[f32; 4]; 4] = *Matrix4::look_at_rh(&eye, &target_point, &up).as_ref();
                
                let perspective = Perspective3::new(
                    display.get_framebuffer_dimensions().0 as f32 / display.get_framebuffer_dimensions().1 as f32,
                    std::f32::consts::FRAC_PI_4,
                    0.1,
                    1000.0
                );
                let projection_matrix: [[f32; 4]; 4] = *perspective.as_matrix().as_ref();
                
                for (i, bird) in flock.get_birds().iter().enumerate() {
                    let pos_x = bird.position.x;
                    let pos_y = bird.position.y;
                    let pos_z = bird.position.z;
                    
                    let look_dir = Vector3::new(cam_x - pos_x, cam_y - pos_y, cam_z - pos_z).normalize();
                    let up = Vector3::new(0.0, 1.0, 0.0);
                    let right = up.cross(&look_dir).normalize();
                    let adjusted_up = look_dir.cross(&right).normalize();
                    
                    let model_matrix = [
                        [right.x, adjusted_up.x, look_dir.x, 0.0],
                        [right.y, adjusted_up.y, look_dir.y, 0.0],
                        [right.z, adjusted_up.z, look_dir.z, 0.0],
                        [pos_x, pos_y, pos_z, 1.0f32],
                    ];
                    
                    let dominant_force = flock.get_dominant_force(i);
                    let bird_color = match dominant_force {
                        0 => [0.9f32, 0.2f32, 0.2f32],
                        1 => [0.2f32, 0.9f32, 0.2f32],
                        2 => [0.2f32, 0.2f32, 0.9f32],
                        _ => [0.7f32, 0.7f32, 0.7f32],
                    };
                    
                    let uniforms = uniform! {
                        model: model_matrix,
                        view: view_matrix,
                        projection: projection_matrix,
                        bird_color: bird_color,
                        camera_pos: [cam_x, cam_y, cam_z],
                    };
                    
                    target.draw(&vertex_buffer, &indices, &program, &uniforms, 
                                &glium::DrawParameters {
                                    depth: glium::Depth {
                                        test: glium::draw_parameters::DepthTest::IfLess,
                                        write: true,
                                        ..Default::default()
                                    },
                                    ..Default::default()
                                }).unwrap();
                }
                
                target.finish().unwrap();
                
                if simulation_time % 5.0 < dt {
                    println!("FPS: {:.1}", 1.0 / dt);
                }
            },
            winit::event::Event::MainEventsCleared => {
                display.gl_window().window().request_redraw();
            },
            _ => return,
        };
    });
}