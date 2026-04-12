#[cfg(target_arch = "wasm32")]
mod wasm {
    use std::{cell::RefCell, thread_local};

    use wasm_bindgen::{JsCast, closure::Closure, prelude::*};
    use web_sys::{
        BinaryType, CanvasRenderingContext2d, HtmlCanvasElement, MessageEvent, WebSocket,
    };

    use sim_core::PreviewFrame;

    struct ViewerState {
        websocket: WebSocket,
        _onmessage: Closure<dyn FnMut(MessageEvent)>,
    }

    thread_local! {
        static VIEWER_STATE: RefCell<Option<ViewerState>> = const { RefCell::new(None) };
    }

    #[wasm_bindgen]
    pub fn boot(canvas_id: &str, session_id: &str) -> Result<(), JsValue> {
        VIEWER_STATE.with(|state| {
            if let Some(previous) = state.borrow_mut().take() {
                previous.websocket.set_onmessage(None);
                let _ = previous.websocket.close();
            }
        });

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

        let location = window.location();
        let scheme = if location.protocol()?.starts_with("https") {
            "wss"
        } else {
            "ws"
        };
        let host = location.host()?;
        let ws = WebSocket::new(&format!("{scheme}://{host}/ws/frames/{session_id}"))?;
        ws.set_binary_type(BinaryType::Arraybuffer);

        let onmessage = Closure::<dyn FnMut(MessageEvent)>::new(move |event: MessageEvent| {
            if let Ok(buffer) = event.data().dyn_into::<js_sys::ArrayBuffer>() {
                let bytes = js_sys::Uint8Array::new(&buffer).to_vec();
                if let Ok(frame) = bincode::deserialize::<PreviewFrame>(&bytes) {
                    draw_frame(&context, &canvas, &frame);
                }
            }
        });
        ws.set_onmessage(Some(onmessage.as_ref().unchecked_ref()));

        VIEWER_STATE.with(|state| {
            *state.borrow_mut() = Some(ViewerState {
                websocket: ws,
                _onmessage: onmessage,
            });
        });

        Ok(())
    }

    fn draw_frame(
        context: &CanvasRenderingContext2d,
        canvas: &HtmlCanvasElement,
        frame: &PreviewFrame,
    ) {
        let width = canvas.width() as f64;
        let height = canvas.height() as f64;
        context.clear_rect(0.0, 0.0, width, height);

        context.set_fill_style_str("#020810");
        context.fill_rect(0.0, 0.0, width, height);

        let scale = 2.6_f64;
        for particle in &frame.particles {
            let x = width * 0.5 + f64::from(particle.position_kpc[0]) * scale;
            let y = height * 0.5 + f64::from(particle.position_kpc[1]) * scale;
            let radius = (f64::from(particle.intensity) * 1.6).max(0.5);
            let [r, g, b, a] = particle.color_rgba;
            context.begin_path();
            context.set_fill_style_str(&format!(
                "rgba({}, {}, {}, {})",
                (r * 255.0) as u32,
                (g * 255.0) as u32,
                (b * 255.0) as u32,
                a.max(particle.intensity)
            ));
            let _ = context.arc(x, y, radius, 0.0, std::f64::consts::TAU);
            context.fill();
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub fn boot(_: &str, _: &str) {}
