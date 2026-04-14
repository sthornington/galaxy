#[cfg(target_arch = "wasm32")]
mod wasm {
    use std::{cell::RefCell, rc::Rc, thread_local};

    use sim_core::{PreviewFrame, PreviewParticle, decode_preview_packet};
    use wasm_bindgen::{JsCast, closure::Closure, prelude::*};
    use web_sys::{
        BinaryType, CanvasRenderingContext2d, Event, HtmlCanvasElement, MessageEvent, MouseEvent,
        WebSocket, WheelEvent, Window,
    };

    #[derive(Clone, Copy)]
    enum DragMode {
        Orbit,
        Pan,
    }

    #[derive(Clone)]
    struct CameraState {
        yaw: f64,
        pitch: f64,
        distance_scale: f64,
        base_distance: f64,
        auto_frame: bool,
        dragging: bool,
        drag_mode: DragMode,
        last_x: f64,
        last_y: f64,
        focus: [f64; 3],
        scene_radius: f64,
    }

    impl Default for CameraState {
        fn default() -> Self {
            Self {
                yaw: 0.4,
                pitch: 0.9,
                distance_scale: 1.2,
                base_distance: 120.0,
                auto_frame: true,
                dragging: false,
                drag_mode: DragMode::Orbit,
                last_x: 0.0,
                last_y: 0.0,
                focus: [0.0, 0.0, 0.0],
                scene_radius: 120.0,
            }
        }
    }

    struct ViewerState {
        websocket: WebSocket,
        canvas: HtmlCanvasElement,
        _frame: Rc<RefCell<Option<PreviewFrame>>>,
        _previous_frame: Rc<RefCell<Option<PreviewFrame>>>,
        _camera: Rc<RefCell<CameraState>>,
        _blend_started_ms: Rc<RefCell<f64>>,
        _blend_scheduled: Rc<RefCell<bool>>,
        _onmessage: Closure<dyn FnMut(MessageEvent)>,
        _animation: Rc<RefCell<Option<Closure<dyn FnMut(f64)>>>>,
        oncontextmenu: Closure<dyn FnMut(MouseEvent)>,
        onmousedown: Closure<dyn FnMut(MouseEvent)>,
        onmousemove: Closure<dyn FnMut(MouseEvent)>,
        onmouseup: Closure<dyn FnMut(MouseEvent)>,
        onmouseleave: Closure<dyn FnMut(MouseEvent)>,
        onwheel: Closure<dyn FnMut(WheelEvent)>,
        ondblclick: Closure<dyn FnMut(MouseEvent)>,
    }

    impl ViewerState {
        fn dispose(self) {
            self.websocket.set_onmessage(None);
            let _ = self.websocket.close();
            let _ = self.canvas.remove_event_listener_with_callback(
                "contextmenu",
                self.oncontextmenu.as_ref().unchecked_ref(),
            );
            let _ = self.canvas.remove_event_listener_with_callback(
                "mousedown",
                self.onmousedown.as_ref().unchecked_ref(),
            );
            let _ = self.canvas.remove_event_listener_with_callback(
                "mousemove",
                self.onmousemove.as_ref().unchecked_ref(),
            );
            let _ = self.canvas.remove_event_listener_with_callback(
                "mouseup",
                self.onmouseup.as_ref().unchecked_ref(),
            );
            let _ = self.canvas.remove_event_listener_with_callback(
                "mouseleave",
                self.onmouseleave.as_ref().unchecked_ref(),
            );
            let _ = self.canvas.remove_event_listener_with_callback(
                "wheel",
                self.onwheel.as_ref().unchecked_ref(),
            );
            let _ = self.canvas.remove_event_listener_with_callback(
                "dblclick",
                self.ondblclick.as_ref().unchecked_ref(),
            );
        }
    }

    struct ProjectedParticle {
        x: f64,
        y: f64,
        depth: f64,
        perspective: f64,
        radial_velocity_kms: f64,
        particle: PreviewParticle,
    }

    struct RenderParticle {
        x: f64,
        y: f64,
        depth: f64,
        glow_radius: f64,
        core_radius: f64,
        glow_alpha: f64,
        core_alpha: f64,
        color: [f64; 3],
    }

    struct HaloField {
        density: Vec<f64>,
        grid_w: usize,
        grid_h: usize,
        cell_w: f64,
        cell_h: f64,
        max_density: f64,
    }

    struct CameraBasis {
        distance: f64,
        camera_position: [f64; 3],
        forward: [f64; 3],
        right: [f64; 3],
        up: [f64; 3],
    }

    thread_local! {
        static VIEWER_STATE: RefCell<Option<ViewerState>> = const { RefCell::new(None) };
    }

    #[wasm_bindgen]
    pub fn shutdown() {
        VIEWER_STATE.with(|state| {
            if let Some(previous) = state.borrow_mut().take() {
                previous.dispose();
            }
        });
    }

    fn dispatch_window_event(window: &Window, event_name: &str) {
        if let Ok(event) = Event::new(event_name) {
            let _ = window.dispatch_event(&event);
        }
    }

    #[wasm_bindgen]
    pub fn boot(canvas_id: &str, session_id: &str) -> Result<(), JsValue> {
        shutdown();

        let window = web_sys::window().ok_or_else(|| JsValue::from_str("missing window"))?;
        let document = window
            .document()
            .ok_or_else(|| JsValue::from_str("missing document"))?;
        let canvas = document
            .get_element_by_id(canvas_id)
            .ok_or_else(|| JsValue::from_str("missing canvas"))?
            .dyn_into::<HtmlCanvasElement>()?;
        let context = canvas
            .get_context("2d")?
            .ok_or_else(|| JsValue::from_str("missing 2d context"))?
            .dyn_into::<CanvasRenderingContext2d>()?;

        let frame = Rc::new(RefCell::new(None));
        let previous_frame = Rc::new(RefCell::new(None));
        let camera = Rc::new(RefCell::new(CameraState::default()));
        let saw_first_frame = Rc::new(RefCell::new(false));
        let blend_started_ms = Rc::new(RefCell::new(0.0));
        let blend_scheduled = Rc::new(RefCell::new(false));
        let animation: Rc<RefCell<Option<Closure<dyn FnMut(f64)>>>> =
            Rc::new(RefCell::new(None));

        let location = window.location();
        let scheme = if location.protocol()?.starts_with("https") {
            "wss"
        } else {
            "ws"
        };
        let host = location.host()?;
        let ws = WebSocket::new(&format!("{scheme}://{host}/ws/frames/{session_id}"))?;
        ws.set_binary_type(BinaryType::Arraybuffer);

        let schedule_blend = {
            let window = window.clone();
            let context = context.clone();
            let canvas = canvas.clone();
            let frame = frame.clone();
            let previous_frame = previous_frame.clone();
            let camera = camera.clone();
            let blend_started_ms = blend_started_ms.clone();
            let blend_scheduled = blend_scheduled.clone();
            let animation = animation.clone();

            move || {
                if *blend_scheduled.borrow() {
                    return;
                }
                *blend_scheduled.borrow_mut() = true;
                if animation.borrow().is_none() {
                    let window_for_anim = window.clone();
                    let context_for_anim = context.clone();
                    let canvas_for_anim = canvas.clone();
                    let frame_for_anim = frame.clone();
                    let previous_for_anim = previous_frame.clone();
                    let camera_for_anim = camera.clone();
                    let blend_started_for_anim = blend_started_ms.clone();
                    let blend_scheduled_for_anim = blend_scheduled.clone();
                    let animation_for_anim = animation.clone();
                    *animation.borrow_mut() = Some(Closure::<dyn FnMut(f64)>::new(move |timestamp: f64| {
                        let maybe_current = frame_for_anim.borrow().clone();
                        let maybe_previous = previous_for_anim.borrow().clone();
                        let Some(current) = maybe_current else {
                            *blend_scheduled_for_anim.borrow_mut() = false;
                            return;
                        };
                        let alpha = clamp((timestamp - *blend_started_for_anim.borrow()) / 90.0, 0.0, 1.0);
                        {
                            let camera = camera_for_anim.borrow();
                            if let Some(previous) = maybe_previous.as_ref() {
                                draw_frame_blended(
                                    &context_for_anim,
                                    &canvas_for_anim,
                                    previous,
                                    &current,
                                    alpha,
                                    &camera,
                                );
                            } else {
                                draw_frame(&context_for_anim, &canvas_for_anim, &current, &camera);
                            }
                        }
                        if alpha < 1.0 {
                            if let Some(callback) = animation_for_anim.borrow().as_ref() {
                                let _ = window_for_anim.request_animation_frame(
                                    callback.as_ref().unchecked_ref(),
                                );
                            }
                        } else {
                            *blend_scheduled_for_anim.borrow_mut() = false;
                        }
                    }));
                }
                if let Some(callback) = animation.borrow().as_ref() {
                    let _ = window.request_animation_frame(callback.as_ref().unchecked_ref());
                }
            }
        };

        let onmessage = {
            let context = context.clone();
            let canvas = canvas.clone();
            let frame = frame.clone();
            let previous_frame = previous_frame.clone();
            let camera = camera.clone();
            let window = window.clone();
            let saw_first_frame = saw_first_frame.clone();
            let blend_started_ms = blend_started_ms.clone();
            let schedule_blend = schedule_blend.clone();
            Closure::<dyn FnMut(MessageEvent)>::new(move |event: MessageEvent| {
                match event.data().dyn_into::<js_sys::ArrayBuffer>() {
                    Ok(buffer) => {
                        let bytes = js_sys::Uint8Array::new(&buffer).to_vec();
                        match decode_preview_packet(&bytes) {
                            Ok(decoded) => {
                                if let Some(current) = frame.borrow().as_ref() {
                                    if decoded.sim_time_myr <= current.sim_time_myr + 1.0e-9 {
                                        return;
                                    }
                                }
                                {
                                    let mut camera = camera.borrow_mut();
                                    update_scene_bounds(&mut camera, &decoded, false);
                                }
                                if let Some(current) = frame.borrow().as_ref() {
                                    *previous_frame.borrow_mut() = Some(current.clone());
                                } else {
                                    *previous_frame.borrow_mut() = Some(decoded.clone());
                                }
                                *frame.borrow_mut() = Some(decoded);
                                *blend_started_ms.borrow_mut() = js_sys::Date::now();
                                schedule_blend();
                                if !*saw_first_frame.borrow() {
                                    *saw_first_frame.borrow_mut() = true;
                                    dispatch_window_event(&window, "galaxy-viewer-frame");
                                }
                            }
                            Err(_) => {
                                dispatch_window_event(&window, "galaxy-viewer-error");
                            }
                        }
                    }
                    Err(_) => {
                        dispatch_window_event(&window, "galaxy-viewer-error");
                    }
                }
            })
        };
        ws.set_onmessage(Some(onmessage.as_ref().unchecked_ref()));

        let oncontextmenu = Closure::<dyn FnMut(MouseEvent)>::new(move |event: MouseEvent| {
            event.prevent_default();
        });
        canvas.add_event_listener_with_callback(
            "contextmenu",
            oncontextmenu.as_ref().unchecked_ref(),
        )?;

        let onmousedown = {
            let camera = camera.clone();
            Closure::<dyn FnMut(MouseEvent)>::new(move |event: MouseEvent| {
                event.prevent_default();
                let mut camera = camera.borrow_mut();
                camera.auto_frame = false;
                camera.dragging = true;
                camera.drag_mode = if event.button() == 2 || event.shift_key() {
                    DragMode::Pan
                } else {
                    DragMode::Orbit
                };
                camera.last_x = f64::from(event.client_x());
                camera.last_y = f64::from(event.client_y());
            })
        };
        canvas
            .add_event_listener_with_callback("mousedown", onmousedown.as_ref().unchecked_ref())?;

        let onmousemove = {
            let context = context.clone();
            let canvas = canvas.clone();
            let frame = frame.clone();
            let previous_frame = previous_frame.clone();
            let camera = camera.clone();
            let blend_started_ms = blend_started_ms.clone();
            Closure::<dyn FnMut(MouseEvent)>::new(move |event: MouseEvent| {
                let mut camera_state = camera.borrow_mut();
                if !camera_state.dragging {
                    return;
                }
                let next_x = f64::from(event.client_x());
                let next_y = f64::from(event.client_y());
                let dx = next_x - camera_state.last_x;
                let dy = next_y - camera_state.last_y;
                match camera_state.drag_mode {
                    DragMode::Pan => {
                        let basis = camera_basis(&camera_state);
                        let fov = std::f64::consts::FRAC_PI_4;
                        let pan_scale = (basis.distance * (fov * 0.5).tan())
                            / (f64::from(canvas.width().min(canvas.height())) * 0.5);
                        camera_state.focus[0] +=
                            (-dx * pan_scale) * basis.right[0] + (dy * pan_scale) * basis.up[0];
                        camera_state.focus[1] +=
                            (-dx * pan_scale) * basis.right[1] + (dy * pan_scale) * basis.up[1];
                        camera_state.focus[2] +=
                            (-dx * pan_scale) * basis.right[2] + (dy * pan_scale) * basis.up[2];
                    }
                    DragMode::Orbit => {
                        camera_state.yaw -= dx * 0.006;
                        camera_state.pitch =
                            clamp(camera_state.pitch - dy * 0.006, -1.45, 1.45);
                    }
                }
                camera_state.last_x = next_x;
                camera_state.last_y = next_y;
                if let Some(frame) = frame.borrow().as_ref() {
                    let alpha =
                        clamp((js_sys::Date::now() - *blend_started_ms.borrow()) / 90.0, 0.0, 1.0);
                    if let Some(previous) = previous_frame.borrow().as_ref() {
                        draw_frame_blended(&context, &canvas, previous, frame, alpha, &camera_state);
                    } else {
                        draw_frame(&context, &canvas, frame, &camera_state);
                    }
                }
            })
        };
        canvas
            .add_event_listener_with_callback("mousemove", onmousemove.as_ref().unchecked_ref())?;

        let onmouseup = {
            let camera = camera.clone();
            Closure::<dyn FnMut(MouseEvent)>::new(move |_event: MouseEvent| {
                let mut camera = camera.borrow_mut();
                camera.dragging = false;
                camera.drag_mode = DragMode::Orbit;
            })
        };
        canvas.add_event_listener_with_callback("mouseup", onmouseup.as_ref().unchecked_ref())?;

        let onmouseleave = {
            let camera = camera.clone();
            Closure::<dyn FnMut(MouseEvent)>::new(move |_event: MouseEvent| {
                let mut camera = camera.borrow_mut();
                camera.dragging = false;
                camera.drag_mode = DragMode::Orbit;
            })
        };
        canvas.add_event_listener_with_callback(
            "mouseleave",
            onmouseleave.as_ref().unchecked_ref(),
        )?;

        let onwheel = {
            let context = context.clone();
            let canvas = canvas.clone();
            let frame = frame.clone();
            let previous_frame = previous_frame.clone();
            let camera = camera.clone();
            let blend_started_ms = blend_started_ms.clone();
            Closure::<dyn FnMut(WheelEvent)>::new(move |event: WheelEvent| {
                event.prevent_default();
                let mut camera_state = camera.borrow_mut();
                camera_state.auto_frame = false;
                let factor = if event.delta_y() < 0.0 { 0.78 } else { 1.0 / 0.78 };
                camera_state.distance_scale =
                    clamp(camera_state.distance_scale * factor, 0.003, 20.0);
                if let Some(frame) = frame.borrow().as_ref() {
                    let alpha =
                        clamp((js_sys::Date::now() - *blend_started_ms.borrow()) / 90.0, 0.0, 1.0);
                    if let Some(previous) = previous_frame.borrow().as_ref() {
                        draw_frame_blended(&context, &canvas, previous, frame, alpha, &camera_state);
                    } else {
                        draw_frame(&context, &canvas, frame, &camera_state);
                    }
                }
            })
        };
        canvas.add_event_listener_with_callback("wheel", onwheel.as_ref().unchecked_ref())?;

        let ondblclick = {
            let context = context.clone();
            let canvas = canvas.clone();
            let frame = frame.clone();
            let camera = camera.clone();
            Closure::<dyn FnMut(MouseEvent)>::new(move |_event: MouseEvent| {
                let mut camera_state = camera.borrow_mut();
                *camera_state = CameraState::default();
                if let Some(frame) = frame.borrow().as_ref() {
                    update_scene_bounds(&mut camera_state, frame, true);
                    draw_frame(&context, &canvas, frame, &camera_state);
                }
            })
        };
        canvas.add_event_listener_with_callback("dblclick", ondblclick.as_ref().unchecked_ref())?;

        VIEWER_STATE.with(|state| {
            *state.borrow_mut() = Some(ViewerState {
                websocket: ws,
                canvas,
                _frame: frame,
                _previous_frame: previous_frame,
                _camera: camera,
                _blend_started_ms: blend_started_ms,
                _blend_scheduled: blend_scheduled,
                _onmessage: onmessage,
                _animation: animation,
                oncontextmenu,
                onmousedown,
                onmousemove,
                onmouseup,
                onmouseleave,
                onwheel,
                ondblclick,
            });
        });

        Ok(())
    }

    fn clamp(value: f64, min: f64, max: f64) -> f64 {
        value.max(min).min(max)
    }

    fn normalize3(vector: [f64; 3]) -> [f64; 3] {
        let length =
            (vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]).sqrt();
        if length <= f64::EPSILON {
            [0.0, 0.0, 0.0]
        } else {
            [vector[0] / length, vector[1] / length, vector[2] / length]
        }
    }

    fn dot3(left: [f64; 3], right: [f64; 3]) -> f64 {
        left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
    }

    fn cross3(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
        [
            left[1] * right[2] - left[2] * right[1],
            left[2] * right[0] - left[0] * right[2],
            left[0] * right[1] - left[1] * right[0],
        ]
    }

    fn mix_color(a: [f64; 3], b: [f64; 3], t: f64) -> [f64; 3] {
        [
            a[0] + (b[0] - a[0]) * t,
            a[1] + (b[1] - a[1]) * t,
            a[2] + (b[2] - a[2]) * t,
        ]
    }

    fn clamp_color(color: [f64; 3]) -> [f64; 3] {
        [
            clamp(color[0], 0.0, 1.0),
            clamp(color[1], 0.0, 1.0),
            clamp(color[2], 0.0, 1.0),
        ]
    }

    fn update_scene_bounds(camera: &mut CameraState, frame: &PreviewFrame, force: bool) {
        if !camera.auto_frame && !force {
            return;
        }

        let luminous: Vec<_> = frame
            .particles
            .iter()
            .filter(|particle| particle.component != 0 && particle.component != 3)
            .collect();
        let particles: Vec<_> = if luminous.is_empty() {
            frame.particles.iter().collect()
        } else {
            luminous
        };

        if particles.is_empty() {
            camera.focus = [0.0, 0.0, 0.0];
            camera.scene_radius = 1.0;
            return;
        }

        let mut sum = [0.0, 0.0, 0.0];
        for particle in &particles {
            sum[0] += f64::from(particle.position_kpc[0]);
            sum[1] += f64::from(particle.position_kpc[1]);
            sum[2] += f64::from(particle.position_kpc[2]);
        }

        camera.focus = [
            sum[0] / particles.len() as f64,
            sum[1] / particles.len() as f64,
            sum[2] / particles.len() as f64,
        ];

        let mut max_radius = 1.0_f64;
        for particle in &particles {
            let dx = f64::from(particle.position_kpc[0]) - camera.focus[0];
            let dy = f64::from(particle.position_kpc[1]) - camera.focus[1];
            let dz = f64::from(particle.position_kpc[2]) - camera.focus[2];
            max_radius = max_radius.max((dx * dx + dy * dy + dz * dz).sqrt());
        }
        camera.scene_radius = max_radius;
        camera.base_distance = (camera.scene_radius * 0.9) / std::f64::consts::FRAC_PI_8.tan();
        if !force && particles.len() >= 32 && max_radius > 0.5 {
            camera.auto_frame = false;
        }
    }

    fn camera_basis(camera: &CameraState) -> CameraBasis {
        let distance = (camera.base_distance * camera.distance_scale).max(0.08);
        let camera_position = [
            camera.focus[0] + distance * camera.pitch.cos() * camera.yaw.cos(),
            camera.focus[1] + distance * camera.pitch.cos() * camera.yaw.sin(),
            camera.focus[2] + distance * camera.pitch.sin(),
        ];
        let forward = normalize3([
            camera.focus[0] - camera_position[0],
            camera.focus[1] - camera_position[1],
            camera.focus[2] - camera_position[2],
        ]);
        let mut right = cross3(forward, [0.0, 0.0, 1.0]);
        if dot3(right, right).sqrt() <= 1.0e-6 {
            right = [1.0, 0.0, 0.0];
        } else {
            right = normalize3(right);
        }
        let up = normalize3(cross3(right, forward));
        CameraBasis {
            distance,
            camera_position,
            forward,
            right,
            up,
        }
    }

    fn project_point(
        position: [f64; 3],
        width: f64,
        height: f64,
        camera: &CameraState,
    ) -> Option<(f64, f64, f64, f64, [f64; 3])> {
        let fov = std::f64::consts::FRAC_PI_4;
        let basis = camera_basis(camera);
        let relative = [
            position[0] - basis.camera_position[0],
            position[1] - basis.camera_position[1],
            position[2] - basis.camera_position[2],
        ];
        let depth = dot3(relative, basis.forward);
        if depth <= 0.1 {
            return None;
        }

        let focal_length = (width.min(height) * 0.5) / (fov * 0.5).tan();
        let x = width * 0.5 + dot3(relative, basis.right) * focal_length / depth;
        let y = height * 0.5 - dot3(relative, basis.up) * focal_length / depth;
        let perspective = clamp((focal_length / depth) * 0.18, 0.08, 3.5);
        Some((x, y, depth, perspective, basis.forward))
    }

    fn project_particle(
        particle: &PreviewParticle,
        width: f64,
        height: f64,
        camera: &CameraState,
    ) -> Option<ProjectedParticle> {
        let (x, y, depth, perspective, forward) = project_point(
            [
                f64::from(particle.position_kpc[0]),
                f64::from(particle.position_kpc[1]),
                f64::from(particle.position_kpc[2]),
            ],
            width,
            height,
            camera,
        )?;

        Some(ProjectedParticle {
            x,
            y,
            depth,
            perspective,
            radial_velocity_kms: dot3(
                [
                    f64::from(particle.velocity_kms[0]),
                    f64::from(particle.velocity_kms[1]),
                    f64::from(particle.velocity_kms[2]),
                ],
                forward,
            ),
            particle: particle.clone(),
        })
    }

    fn stellar_base_color(particle: &PreviewParticle) -> Option<[f64; 3]> {
        let component = particle.component;
        let mass_msun = f64::from(particle.mass_msun).max(1.0);
        let mass_bias = clamp((mass_msun.log10() - 4.2) / 1.6, 0.0, 1.0);
        match component {
            0 | 3 => None,
            2 => Some(mix_color(
                [1.0, 0.82, 0.68],
                [1.0, 0.93, 0.82],
                0.35 + 0.2 * mass_bias,
            )),
            _ => Some(mix_color(
                [1.0, 0.92, 0.8],
                [0.79, 0.88, 1.0],
                0.45 + 0.35 * mass_bias,
            )),
        }
    }

    fn apply_doppler_shift(color: [f64; 3], radial_velocity_kms: f64) -> [f64; 3] {
        let shift = clamp(radial_velocity_kms / 700.0, -0.28, 0.28);
        if shift >= 0.0 {
            clamp_color([
                color[0] * (1.0 + 0.9 * shift),
                color[1] * (1.0 + 0.2 * shift),
                color[2] * (1.0 - 0.75 * shift),
            ])
        } else {
            let blue = -shift;
            clamp_color([
                color[0] * (1.0 - 0.75 * blue),
                color[1] * (1.0 + 0.1 * blue),
                color[2] * (1.0 + 0.95 * blue),
            ])
        }
    }

    fn render_style(projected: ProjectedParticle) -> Option<RenderParticle> {
        let base_color = stellar_base_color(&projected.particle)?;
        let mass_msun = f64::from(projected.particle.mass_msun).max(1.0);
        let luminosity = clamp((mass_msun.log10() - 3.7) / 2.2, 0.25, 1.8);
        let render_luminosity = luminosity.powf(0.58);
        let color = apply_doppler_shift(base_color, projected.radial_velocity_kms);
        let size_scale = projected.perspective.powf(1.18);
        let alpha_scale = projected.perspective.powf(0.92);

        Some(RenderParticle {
            x: projected.x,
            y: projected.y,
            depth: projected.depth,
            glow_radius: (0.52 * render_luminosity * size_scale).max(0.08),
            core_radius: (0.12 * render_luminosity * size_scale).max(0.03),
            glow_alpha: clamp(0.0032 * render_luminosity * alpha_scale, 0.0012, 0.012),
            core_alpha: clamp(0.072 * render_luminosity * alpha_scale, 0.012, 0.18),
            color,
        })
    }

    fn project_interpolated_particle(
        previous: &PreviewParticle,
        current: &PreviewParticle,
        alpha: f64,
        width: f64,
        height: f64,
        camera: &CameraState,
    ) -> Option<ProjectedParticle> {
        if previous.component != current.component {
            return project_particle(current, width, height, camera);
        }

        let position = [
            f64::from(previous.position_kpc[0])
                + (f64::from(current.position_kpc[0]) - f64::from(previous.position_kpc[0])) * alpha,
            f64::from(previous.position_kpc[1])
                + (f64::from(current.position_kpc[1]) - f64::from(previous.position_kpc[1])) * alpha,
            f64::from(previous.position_kpc[2])
                + (f64::from(current.position_kpc[2]) - f64::from(previous.position_kpc[2])) * alpha,
        ];
        let (x, y, depth, perspective, forward) =
            project_point(position, width, height, camera)?;
        let velocity = [
            f64::from(previous.velocity_kms[0])
                + (f64::from(current.velocity_kms[0]) - f64::from(previous.velocity_kms[0])) * alpha,
            f64::from(previous.velocity_kms[1])
                + (f64::from(current.velocity_kms[1]) - f64::from(previous.velocity_kms[1])) * alpha,
            f64::from(previous.velocity_kms[2])
                + (f64::from(current.velocity_kms[2]) - f64::from(previous.velocity_kms[2])) * alpha,
        ];

        Some(ProjectedParticle {
            x,
            y,
            depth,
            perspective,
            radial_velocity_kms: dot3(velocity, forward),
            particle: current.clone(),
        })
    }

    fn build_halo_field(points: &[ProjectedParticle], width: f64, height: f64) -> HaloField {
        let grid_w = ((width / 10.0).floor() as usize).max(48);
        let grid_h = ((height / 10.0).floor() as usize).max(27);
        let mut density = vec![0.0; grid_w * grid_h];
        let mut max_density: f64 = 0.0;

        let mut accumulate = |ix: isize, iy: isize, weight: f64| {
            if ix < 0 || iy < 0 || weight <= 0.0 {
                return;
            }
            let ix = ix as usize;
            let iy = iy as usize;
            if ix >= grid_w || iy >= grid_h {
                return;
            }
            let slot = iy * grid_w + ix;
            density[slot] += weight;
            max_density = max_density.max(density[slot]);
        };

        for point in points {
            let gx = clamp(point.x / width * grid_w as f64, 0.0, grid_w as f64 - 1.0001);
            let gy = clamp(point.y / height * grid_h as f64, 0.0, grid_h as f64 - 1.0001);
            let ix = gx.floor() as isize;
            let iy = gy.floor() as isize;
            let tx = gx - ix as f64;
            let ty = gy - iy as f64;
            let mass_weight = clamp(
                f64::from(point.particle.mass_msun).max(1.0).log10() / 6.5,
                0.18,
                1.0,
            );
            let perspective_weight = point.perspective.max(0.08).powf(0.85);
            let weight = mass_weight * perspective_weight;
            accumulate(ix, iy, weight * (1.0 - tx) * (1.0 - ty));
            accumulate(ix + 1, iy, weight * tx * (1.0 - ty));
            accumulate(ix, iy + 1, weight * (1.0 - tx) * ty);
            accumulate(ix + 1, iy + 1, weight * tx * ty);
        }

        HaloField {
            density,
            grid_w,
            grid_h,
            cell_w: width / grid_w as f64,
            cell_h: height / grid_h as f64,
            max_density,
        }
    }

    fn edge_point(
        x0: f64,
        y0: f64,
        v0: f64,
        x1: f64,
        y1: f64,
        v1: f64,
        threshold: f64,
    ) -> Option<(f64, f64)> {
        let dv = v1 - v0;
        if dv.abs() <= 1.0e-8 {
            return None;
        }
        let t = (threshold - v0) / dv;
        if !(0.0..=1.0).contains(&t) {
            return None;
        }
        Some((x0 + (x1 - x0) * t, y0 + (y1 - y0) * t))
    }

    fn draw_halo_fog_and_contours(
        context: &CanvasRenderingContext2d,
        width: f64,
        height: f64,
        halo_points: &[ProjectedParticle],
    ) {
        if halo_points.is_empty() {
            return;
        }
        let field = build_halo_field(halo_points, width, height);
        if !(field.max_density > 0.0) {
            return;
        }

        context.save();
        for y in 0..field.grid_h {
            for x in 0..field.grid_w {
                let density = field.density[y * field.grid_w + x] / field.max_density;
                if density < 0.03 {
                    continue;
                }
                let fog = density.powf(0.62);
                let alpha = clamp(0.015 + 0.11 * fog, 0.0, 0.12);
                let red = (36.0 + 28.0 * fog) as u32;
                let green = (88.0 + 72.0 * fog) as u32;
                let blue = (132.0 + 108.0 * fog) as u32;
                context.set_fill_style_str(&format!(
                    "rgba({}, {}, {}, {})",
                    red, green, blue, alpha
                ));
                context.fill_rect(
                    x as f64 * field.cell_w,
                    y as f64 * field.cell_h,
                    field.cell_w + 0.7,
                    field.cell_h + 0.7,
                );
            }
        }
        context.restore();

        let thresholds = [
            (0.16, "rgba(110, 170, 255, 0.12)", 0.8),
            (0.32, "rgba(130, 205, 255, 0.18)", 0.95),
            (0.54, "rgba(185, 235, 255, 0.26)", 1.05),
        ];
        for (level, color, line_width) in thresholds {
            context.save();
            context.set_stroke_style_str(color);
            context.set_line_width(line_width);
            for y in 0..field.grid_h.saturating_sub(1) {
                for x in 0..field.grid_w.saturating_sub(1) {
                    let v00 = field.density[y * field.grid_w + x] / field.max_density;
                    let v10 = field.density[y * field.grid_w + x + 1] / field.max_density;
                    let v01 = field.density[(y + 1) * field.grid_w + x] / field.max_density;
                    let v11 = field.density[(y + 1) * field.grid_w + x + 1] / field.max_density;
                    let min_value = v00.min(v10).min(v01).min(v11);
                    let max_value = v00.max(v10).max(v01).max(v11);
                    if min_value > level || max_value < level {
                        continue;
                    }

                    let x0 = x as f64 * field.cell_w;
                    let x1 = (x + 1) as f64 * field.cell_w;
                    let y0 = y as f64 * field.cell_h;
                    let y1 = (y + 1) as f64 * field.cell_h;
                    let mut points = Vec::with_capacity(4);
                    if let Some(point) = edge_point(x0, y0, v00, x1, y0, v10, level) {
                        points.push(point);
                    }
                    if let Some(point) = edge_point(x1, y0, v10, x1, y1, v11, level) {
                        points.push(point);
                    }
                    if let Some(point) = edge_point(x1, y1, v11, x0, y1, v01, level) {
                        points.push(point);
                    }
                    if let Some(point) = edge_point(x0, y1, v01, x0, y0, v00, level) {
                        points.push(point);
                    }
                    if points.len() < 2 {
                        continue;
                    }
                    context.begin_path();
                    context.move_to(points[0].0, points[0].1);
                    context.line_to(points[1].0, points[1].1);
                    context.stroke();
                    if points.len() == 4 {
                        context.begin_path();
                        context.move_to(points[2].0, points[2].1);
                        context.line_to(points[3].0, points[3].1);
                        context.stroke();
                    }
                }
            }
            context.restore();
        }
    }

    fn draw_frame(
        context: &CanvasRenderingContext2d,
        canvas: &HtmlCanvasElement,
        frame: &PreviewFrame,
        camera: &CameraState,
    ) {
        let width = canvas.width() as f64;
        let height = canvas.height() as f64;
        context.clear_rect(0.0, 0.0, width, height);

        context.set_fill_style_str("#020810");
        context.fill_rect(0.0, 0.0, width, height);

        let mut projected = Vec::new();
        for particle in &frame.particles {
            let Some(projected_particle) = project_particle(particle, width, height, camera) else {
                continue;
            };
            if particle.component == 0 {
                continue;
            }
            if let Some(rendered) = render_style(projected_particle) {
                projected.push(rendered);
            }
        }
        projected.sort_by(|left, right| right.depth.total_cmp(&left.depth));

        let _ = context.set_global_composite_operation("lighter");
        for point in projected {
            let [r, g, b] = point.color;
            context.begin_path();
            context.set_fill_style_str(&format!(
                "rgba({}, {}, {}, {})",
                (r * 255.0) as u32,
                (g * 255.0) as u32,
                (b * 255.0) as u32,
                point.glow_alpha
            ));
            let _ = context.arc(point.x, point.y, point.glow_radius, 0.0, std::f64::consts::TAU);
            context.fill();

            context.begin_path();
            context.set_fill_style_str(&format!(
                "rgba({}, {}, {}, {})",
                (r * 255.0) as u32,
                (g * 255.0) as u32,
                (b * 255.0) as u32,
                point.core_alpha
            ));
            let _ = context.arc(point.x, point.y, point.core_radius, 0.0, std::f64::consts::TAU);
            context.fill();
        }
        let _ = context.set_global_composite_operation("source-over");
        if camera.dragging {
            draw_xy_plane_grid(context, width, height, camera);
        }
        draw_origin_axes(context, width, height, camera);
    }

    fn draw_frame_blended(
        context: &CanvasRenderingContext2d,
        canvas: &HtmlCanvasElement,
        previous: &PreviewFrame,
        current: &PreviewFrame,
        alpha: f64,
        camera: &CameraState,
    ) {
        let width = canvas.width() as f64;
        let height = canvas.height() as f64;
        context.clear_rect(0.0, 0.0, width, height);

        context.set_fill_style_str("#020810");
        context.fill_rect(0.0, 0.0, width, height);

        let mut projected = Vec::new();
        let count = previous.particles.len().min(current.particles.len());
        for index in 0..count {
            let previous_particle = &previous.particles[index];
            let current_particle = &current.particles[index];
            let Some(projected_particle) = project_interpolated_particle(
                previous_particle,
                current_particle,
                alpha,
                width,
                height,
                camera,
            ) else {
                continue;
            };
            if current_particle.component == 0 {
                continue;
            }
            if let Some(rendered) = render_style(projected_particle) {
                projected.push(rendered);
            }
        }
        projected.sort_by(|left, right| right.depth.total_cmp(&left.depth));

        let _ = context.set_global_composite_operation("lighter");
        for point in projected {
            let [r, g, b] = point.color;
            context.begin_path();
            context.set_fill_style_str(&format!(
                "rgba({}, {}, {}, {})",
                (r * 255.0) as u32,
                (g * 255.0) as u32,
                (b * 255.0) as u32,
                point.glow_alpha
            ));
            let _ = context.arc(point.x, point.y, point.glow_radius, 0.0, std::f64::consts::TAU);
            context.fill();

            context.begin_path();
            context.set_fill_style_str(&format!(
                "rgba({}, {}, {}, {})",
                (r * 255.0) as u32,
                (g * 255.0) as u32,
                (b * 255.0) as u32,
                point.core_alpha
            ));
            let _ = context.arc(point.x, point.y, point.core_radius, 0.0, std::f64::consts::TAU);
            context.fill();
        }
        let _ = context.set_global_composite_operation("source-over");
        if camera.dragging {
            draw_xy_plane_grid(context, width, height, camera);
        }
        draw_origin_axes(context, width, height, camera);
    }

    fn nice_grid_spacing(target: f64) -> f64 {
        let normalized = target.max(0.25);
        let exponent = normalized.log10().floor();
        let base = 10_f64.powf(exponent);
        let scaled = normalized / base;
        if scaled <= 1.0 {
            base
        } else if scaled <= 2.0 {
            2.0 * base
        } else if scaled <= 5.0 {
            5.0 * base
        } else {
            10.0 * base
        }
    }

    fn draw_projected_segment(
        context: &CanvasRenderingContext2d,
        start: Option<(f64, f64, f64, f64, [f64; 3])>,
        end: Option<(f64, f64, f64, f64, [f64; 3])>,
        color: &str,
        width: f64,
    ) {
        let (Some((x0, y0, _depth0, _perspective0, _forward0)), Some((x1, y1, _depth1, _perspective1, _forward1))) =
            (start, end)
        else {
            return;
        };

        context.begin_path();
        context.set_stroke_style_str(color);
        context.set_line_width(width);
        context.move_to(x0, y0);
        context.line_to(x1, y1);
        context.stroke();
    }

    fn draw_xy_plane_grid(
        context: &CanvasRenderingContext2d,
        width: f64,
        height: f64,
        camera: &CameraState,
    ) {
        let extent = camera.scene_radius.max(10.0) * 0.9;
        let spacing = nice_grid_spacing(extent / 6.0);
        let line_count = ((extent / spacing).ceil() as i32).clamp(2, 10);

        context.save();
        for i in -line_count..=line_count {
            let axis_offset = f64::from(i) * spacing;
            let major = i % 5 == 0;
            let color = if major {
                "rgba(210, 224, 245, 0.11)"
            } else {
                "rgba(210, 224, 245, 0.055)"
            };
            let stroke_width = if major { 0.9 } else { 0.65 };

            draw_projected_segment(
                context,
                project_point([axis_offset, -extent, 0.0], width, height, camera),
                project_point([axis_offset, extent, 0.0], width, height, camera),
                color,
                stroke_width,
            );
            draw_projected_segment(
                context,
                project_point([-extent, axis_offset, 0.0], width, height, camera),
                project_point([extent, axis_offset, 0.0], width, height, camera),
                color,
                stroke_width,
            );
        }
        context.restore();
    }

    fn draw_origin_axes(
        context: &CanvasRenderingContext2d,
        width: f64,
        height: f64,
        camera: &CameraState,
    ) {
        let axis_length = camera.scene_radius.max(1.4) * 0.09;
        let Some((origin_x, origin_y, _origin_depth, origin_perspective, _forward)) =
            project_point([0.0, 0.0, 0.0], width, height, camera)
        else {
            return;
        };

        let axes = [
            ("X", "rgba(255, 110, 110, 0.56)", [axis_length, 0.0, 0.0]),
            ("Y", "rgba(120, 255, 170, 0.56)", [0.0, axis_length, 0.0]),
            ("Z", "rgba(110, 170, 255, 0.56)", [0.0, 0.0, axis_length]),
        ];

        context.save();
        context.set_line_width(clamp(origin_perspective * 0.75, 0.8, 1.4));
        context.set_font("500 10px \"IBM Plex Sans\", sans-serif");
        context.set_text_align("center");
        context.set_text_baseline("middle");

        for (label, color, endpoint) in axes {
            let Some((x, y, _depth, _perspective, _forward)) =
                project_point(endpoint, width, height, camera)
            else {
                continue;
            };
            context.set_stroke_style_str(color);
            context.begin_path();
            context.move_to(origin_x, origin_y);
            context.line_to(x, y);
            context.stroke();
            context.set_fill_style_str(color);
            let _ = context.fill_text(label, x, y);
        }

        context.begin_path();
        context.set_fill_style_str("rgba(245, 248, 255, 0.6)");
        let _ = context.arc(
            origin_x,
            origin_y,
            clamp(origin_perspective * 0.9, 1.1, 2.2),
            0.0,
            std::f64::consts::TAU,
        );
        context.fill();
        context.set_fill_style_str("rgba(245, 248, 255, 0.45)");
        let _ = context.fill_text("O", origin_x, origin_y - 9.0);
        context.restore();
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub fn boot(_: &str, _: &str) {}
