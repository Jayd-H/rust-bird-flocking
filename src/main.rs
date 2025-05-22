//* Bird Flocking Simulation */
//* APRIL/MAY 2025 - Jayden Holdsworth */
#[macro_use]
extern crate glium;
extern crate nalgebra;
extern crate rand;
extern crate scoped_threadpool;
extern crate winit;
use nalgebra::{Matrix4, Perspective3, Point3, Vector3};
use std::env;
use std::time::Instant;
mod bird;
mod flock_manager;
use flock_manager::FlockManager;

//* SIMULATION CONFIG */
const FLOCK_SIZES: [usize; 4] = [100, 500, 1000, 5000];
const BENCHMARK_STEPS: usize = 1000;
const SCALING_TEST_STEPS: usize = 500;
const SIMULATION_TIMESTEP: f32 = 0.016; // 60 FPS equivalent

// Default values
// This could be a struct with ::Default imlementatiom
const DEFAULT_NUM_BIRDS: usize = 1000;
const DEFAULT_FORCE_THREADS: usize = 4;
const DEFAULT_UPDATE_THREADS: usize = 4;
const DEFAULT_MIN_BOUNDS: Vector3<f32> = Vector3::new(-50.0, -50.0, -50.0);
const DEFAULT_MAX_BOUNDS: Vector3<f32> = Vector3::new(50.0, 50.0, 50.0);

//* VISUALIZATION CONFIG */
const WINDOW_WIDTH: f32 = 1024.0;
const WINDOW_HEIGHT: f32 = 768.0;
const WINDOW_TITLE: &str = "Bird Flocking Simulation";
const BIRD_COLORS: [[f32; 3]; 4] = [
    [0.9, 0.2, 0.2], // Separation (red)
    [0.2, 0.9, 0.2], // Alignment (green)
    [0.2, 0.2, 0.9], // Cohesion (blue)
    [0.7, 0.7, 0.7], // Default (grey)
];
const BACKGROUND_COLOR: (f32, f32, f32, f32) = (0.1, 0.1, 0.15, 1.0);

//* CAMERA CONFIG */
const CAMERA_POSITION: Vector3<f32> = Vector3::new(0.0, 0.0, 120.0);
const CAMERA_TARGET: Vector3<f32> = Vector3::new(0.0, 0.0, 0.0);
const CAMERA_UP: Vector3<f32> = Vector3::new(0.0, 1.0, 0.0);
const CAMERA_FOV: f32 = std::f32::consts::FRAC_PI_4;
const CAMERA_NEAR: f32 = 0.1;
const CAMERA_FAR: f32 = 1000.0;

//* PERFORMANCE MONITORING */
const REPORT_INTERVAL_SECS: u64 = 5;

fn main() {
    let args: Vec<String> = env::args().collect();

    // Default configuration
    let min_bounds = DEFAULT_MIN_BOUNDS;
    let max_bounds = DEFAULT_MAX_BOUNDS;
    let mut force_threads = DEFAULT_FORCE_THREADS;
    let mut update_threads = DEFAULT_UPDATE_THREADS;

    // Parse thread counts from arguments
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--threads" && i + 1 < args.len() {
            if let Ok(count) = args[i + 1].parse() {
                force_threads = count;
                update_threads = count;
                i += 2;
                continue;
            }
        } else if args[i] == "--force-threads" && i + 1 < args.len() {
            if let Ok(count) = args[i + 1].parse() {
                force_threads = count;
                i += 2;
                continue;
            }
        } else if args[i] == "--update-threads" && i + 1 < args.len() {
            if let Ok(count) = args[i + 1].parse() {
                update_threads = count;
                i += 2;
                continue;
            }
        }
        i += 1;
    }

    // Parse command line arguments to determine mode
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

    run_interactive_mode(min_bounds, max_bounds, force_threads, update_threads);
}

fn run_interactive_mode(
    min_bounds: Vector3<f32>,
    max_bounds: Vector3<f32>,
    force_threads: usize,
    update_threads: usize,
) {
    use glium::{glutin, Surface};

    let event_loop = winit::event_loop::EventLoop::new();
    let wb = winit::window::WindowBuilder::new()
        .with_inner_size(winit::dpi::LogicalSize::new(WINDOW_WIDTH, WINDOW_HEIGHT))
        .with_title(WINDOW_TITLE);
    let cb = glutin::ContextBuilder::new().with_depth_buffer(24);
    let display = glium::Display::new(wb, cb, &event_loop).unwrap();

    #[derive(Copy, Clone)]
    struct Vertex {
        position: [f32; 3],
    }

    implement_vertex!(Vertex, position);

    let vertex1 = Vertex {
        position: [-0.5, -0.3, 0.0],
    };
    let vertex2 = Vertex {
        position: [0.0, 0.6, 0.0],
    };
    let vertex3 = Vertex {
        position: [0.5, -0.3, 0.0],
    };
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

    let program =
        glium::Program::from_source(&display, vertex_shader_src, fragment_shader_src, None)
            .unwrap();

    let mut flock = FlockManager::new(
        DEFAULT_NUM_BIRDS,
        min_bounds,
        max_bounds,
        force_threads,
        update_threads,
    );

    //* Performance tracking variables */
    let mut last_update = Instant::now();
    let mut fps_counter = 0;
    let mut fps_timer = Instant::now();
    let mut fps_history = Vec::new();
    let mut total_frame_time = 0.0;
    let mut simulation_time = 0.0;
    let mut render_time = 0.0;

    let report_interval = std::time::Duration::from_secs(REPORT_INTERVAL_SECS);
    let mut last_report = Instant::now();

    event_loop.run(move |event, _, control_flow| {
        match event {
            winit::event::Event::WindowEvent { event, .. } => match event {
                winit::event::WindowEvent::CloseRequested => {
                    *control_flow = winit::event_loop::ControlFlow::Exit;

                    // Final performance report
                    if !fps_history.is_empty() {
                        let avg_fps = fps_history.iter().sum::<f32>() / fps_history.len() as f32;
                        let min_fps = fps_history.iter().fold(f32::INFINITY, |a, &b| a.min(b));
                        let max_fps = fps_history.iter().fold(0.0, |a: f32, &b| a.max(b));

                        println!();
                        println!("\n=== Final Performance Report ===");
                        println!("Bird count: {}", flock.get_bird_count());
                        println!("Average FPS: {:.1}", avg_fps);
                        println!("Min FPS: {:.1}", min_fps);
                        println!("Max FPS: {:.1}", max_fps);
                        println!("Time breakdown per frame:");
                        println!(
                            "  - Simulation: {:.2}ms ({:.1}%)",
                            simulation_time * 1000.0 / fps_counter as f32,
                            simulation_time * 100.0 / total_frame_time
                        );
                        println!(
                            "  - Rendering: {:.2}ms ({:.1}%)",
                            render_time * 1000.0 / fps_counter as f32,
                            render_time * 100.0 / total_frame_time
                        );
                        println!("Component breakdown in simulation:");
                        flock.performance.report();
                    }

                    return;
                }
                _ => return,
            },
            winit::event::Event::RedrawRequested(_) => {
                let now = Instant::now();
                let dt = now.duration_since(last_update).as_secs_f32();
                last_update = now;
                total_frame_time += dt;

                // Measure simulation time
                let sim_start = Instant::now();
                flock.update(dt);
                let sim_duration = sim_start.elapsed().as_secs_f32();
                simulation_time += sim_duration;

                // Measure rendering time
                let render_start = Instant::now();

                // Render scene
                let mut target = display.draw();
                target.clear_color_and_depth(BACKGROUND_COLOR, 1.0);

                let eye = Point3::new(CAMERA_POSITION.x, CAMERA_POSITION.y, CAMERA_POSITION.z);
                let target_point = Point3::new(CAMERA_TARGET.x, CAMERA_TARGET.y, CAMERA_TARGET.z);
                let up = CAMERA_UP;
                let view_matrix: [[f32; 4]; 4] =
                    *Matrix4::look_at_rh(&eye, &target_point, &up).as_ref();

                let perspective = Perspective3::new(
                    display.get_framebuffer_dimensions().0 as f32
                        / display.get_framebuffer_dimensions().1 as f32,
                    CAMERA_FOV,
                    CAMERA_NEAR,
                    CAMERA_FAR,
                );
                let projection_matrix: [[f32; 4]; 4] = *perspective.as_matrix().as_ref();

                for (i, bird) in flock.get_birds().iter().enumerate() {
                    let pos_x = bird.position.x;
                    let pos_y = bird.position.y;
                    let pos_z = bird.position.z;

                    let look_dir = Vector3::new(
                        CAMERA_POSITION.x - pos_x,
                        CAMERA_POSITION.y - pos_y,
                        CAMERA_POSITION.z - pos_z,
                    )
                    .normalize();
                    let up = CAMERA_UP;
                    let right = up.cross(&look_dir).normalize();
                    let adjusted_up = look_dir.cross(&right).normalize();

                    let model_matrix = [
                        [right.x, adjusted_up.x, look_dir.x, 0.0],
                        [right.y, adjusted_up.y, look_dir.y, 0.0],
                        [right.z, adjusted_up.z, look_dir.z, 0.0],
                        [pos_x, pos_y, pos_z, 1.0f32],
                    ];

                    let dominant_force = flock.get_dominant_force(i);
                    let bird_color = BIRD_COLORS[dominant_force.min(3)];

                    let uniforms = uniform! {
                        model: model_matrix,
                        view: view_matrix,
                        projection: projection_matrix,
                        bird_color: bird_color,
                        camera_pos: [CAMERA_POSITION.x, CAMERA_POSITION.y, CAMERA_POSITION.z],
                    };

                    target
                        .draw(
                            &vertex_buffer,
                            &indices,
                            &program,
                            &uniforms,
                            &glium::DrawParameters {
                                depth: glium::Depth {
                                    test: glium::draw_parameters::DepthTest::IfLess,
                                    write: true,
                                    ..Default::default()
                                },
                                ..Default::default()
                            },
                        )
                        .unwrap();
                }

                target.finish().unwrap();

                let render_duration = render_start.elapsed().as_secs_f32();
                render_time += render_duration;

                fps_counter += 1;

                //* Report FPS every interval */
                if last_report.elapsed() >= report_interval {
                    let current_fps = fps_counter as f32 / fps_timer.elapsed().as_secs_f32();
                    fps_history.push(current_fps);

                    println!(
                        "FPS: {:.1} | Birds: {} | Sim time: {:.2}ms | Render time: {:.2}ms",
                        current_fps,
                        flock.get_bird_count(),
                        (simulation_time / fps_counter as f32) * 1000.0,
                        (render_time / fps_counter as f32) * 1000.0
                    );

                    fps_counter = 0;
                    fps_timer = Instant::now();
                    last_report = Instant::now();
                    simulation_time = 0.0;
                    render_time = 0.0;
                }
            }
            winit::event::Event::MainEventsCleared => {
                display.gl_window().window().request_redraw();
            }
            _ => return,
        };
    });
}
