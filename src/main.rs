use std::path::PathBuf;
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use std::time::Instant;

#[cfg(unix)]
use std::os::unix::process::ExitStatusExt;
#[cfg(windows)]
use std::os::windows::process::ExitStatusExt;

use eframe::egui;
use egui::{ColorImage, TextureHandle};
use rfd::FileDialog;
use serde::{Deserialize, Serialize};

/// Mitsuba Studio: a pragmatic front-end for Mitsuba scene setup + rendering.
///
/// Design goals:
/// - Provide a friendlier workflow than hand-editing XML
/// - Cover the most frequently tweaked knobs first: camera, film resolution, sampler count,
///   light intensity, and a sphere's transform/material
/// - Output standard Mitsuba XML (loadable via `mi.load_file(...)` in Python)
///
/// Future extensions:
/// - Multiple shapes (Vec<Shape>) with add/remove
/// - Mesh import (ply/obj), textures, more BSDFs, multiple emitters
/// - A "Render via Python" button that calls your script
fn main() -> eframe::Result<()> {
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1520.0, 1108.0])
            .with_title("Mitsuba Studio"),
        ..Default::default()
    };

    eframe::run_native(
        "Mitsuba Studio",
        native_options,
        Box::new(|cc| Ok(Box::new(MitsubaStudioApp::new(cc)))),
    )
}

#[derive(Clone, Serialize, Deserialize)]
struct CameraConfig {
    fov_deg: f32,
    origin: [f32; 3],
    target: [f32; 3],
    up: [f32; 3],
}

#[derive(Clone, Serialize, Deserialize)]
struct FilmConfig {
    width: u32,
    height: u32,
}

#[derive(Clone, Serialize, Deserialize)]
struct SamplerConfig {
    sample_count: u32,
}

#[derive(Clone, Serialize, Deserialize)]
struct AreaLightConfig {
    enabled: bool,
    radiance_rgb: [f32; 3],
    scale_xy: f32,
    y: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
enum ObjectKind {
    Sphere,
    Cube,
    PlyMesh,
    ObjMesh,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
enum SceneEnvironment {
    Empty,
    CornellBox,
    ObjFile,
}

#[derive(Clone, Serialize, Deserialize)]
enum ParamValue {
    Float(f32),
    Integer(i32),
    Boolean(bool),
    Rgb([f32; 3]),
    Spectrum(String),
    String(String),
    Bitmap {
        filename: String,
        // If true, treat the bitmap as sRGB (raw=false). If false, treat it as linear (raw=true).
        srgb: bool,
    },
}

#[derive(Clone, Serialize, Deserialize)]
struct BsdfParam {
    name: String,
    value: ParamValue,
}

/// A generic, nestable BSDF node.
///
/// This intentionally models Mitsuba XML in a structured way:
/// - `plugin_type` maps to `<bsdf type="...">`
/// - `params` maps to `<float/integer/boolean/rgb/string/texture ...>` children
/// - `inner` maps to nested `<bsdf ...>` (useful for wrapper BSDFs like twosided/normalmap/bumpmap/mask)
#[derive(Clone, Serialize, Deserialize)]
struct BsdfNode {
    plugin_type: String,
    params: Vec<BsdfParam>,
    inner: Option<Box<BsdfNode>>,
}

impl Default for BsdfNode {
    fn default() -> Self {
        Self {
            plugin_type: "diffuse".to_string(),
            params: vec![BsdfParam {
                name: "reflectance".to_string(),
                value: ParamValue::Rgb([0.8, 0.8, 0.8]),
            }],
            inner: None,
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
struct SceneEnvironmentConfig {
    kind: SceneEnvironment,
    obj_filename: String,
    obj_scale: f32,
}

impl Default for SceneEnvironmentConfig {
    fn default() -> Self {
        Self {
            kind: SceneEnvironment::CornellBox,
            obj_filename: "meshes/room.obj".to_string(),
            obj_scale: 1.0,
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
struct ObjectConfig {
    enabled: bool,
    kind: ObjectKind,

    // Shared
    translate: [f32; 3],
    #[serde(default)]
    bsdf: BsdfNode,

    // Sphere
    sphere_radius: f32,

    // Cube
    cube_scale: [f32; 3],

    // Mesh
    ply_filename: String,
    obj_filename: String,
    mesh_scale: f32,
}

#[derive(Clone, Serialize, Deserialize)]
struct CornellBoxConfig {
    camera: CameraConfig,
    film: FilmConfig,
    sampler: SamplerConfig,
    light: AreaLightConfig,
    #[serde(default)]
    scene_environment: SceneEnvironmentConfig,
    object: ObjectConfig,
}

impl Default for CornellBoxConfig {
    fn default() -> Self {
        let default_object_bsdf = BsdfNode {
            plugin_type: "diffuse".to_string(),
            params: vec![BsdfParam {
                name: "reflectance".to_string(),
                value: ParamValue::Rgb([0.25, 0.35, 0.8]),
            }],
            inner: None,
        };

        Self {
            camera: CameraConfig {
                fov_deg: 39.0,
                origin: [0.0, 1.0, 3.2],
                target: [0.0, 1.0, 0.0],
                up: [0.0, 1.0, 0.0],
            },
            film: FilmConfig {
                width: 512,
                height: 512,
            },
            sampler: SamplerConfig { sample_count: 16 },
            light: AreaLightConfig {
                enabled: true,
                radiance_rgb: [18.0, 18.0, 18.0],
                scale_xy: 0.35,
                y: 1.99,
            },
            scene_environment: SceneEnvironmentConfig::default(),
            object: ObjectConfig {
                enabled: true,
                translate: [0.0, 0.35, 0.0],
                bsdf: default_object_bsdf,
                kind: ObjectKind::Sphere,
                sphere_radius: 0.35,
                cube_scale: [0.5, 0.8, 0.5],
                ply_filename: "meshes/teapot.ply".to_string(),
                obj_filename: "meshes/teapot.obj".to_string(),
                mesh_scale: 1.0,
            },
        }
    }
}

struct MitsubaStudioApp {
    config: CornellBoxConfig,
    last_status: String,
    xml_preview: String,

    // --- UI ---
    sidebar_tab: SidebarTab,
    main_tab: MainTab,
    style_applied: bool,

    // --- Rendering integration (spawns Python subprocesses) ---
    python_exe: String,
    render_scene_path: String,
    render_variant: String,
    render_spp: u32,
    render_out_path: String,
    drjit_libllvm_path: String,

    // --- Fitting (nvdiffrast) ---
    fit_steps: u32,
    fit_lr: f32,
    fit_out_dir: String,

    job: RenderJobState,
    training_progress: Option<TrainingProgress>,

    // --- Embedded preview image (Training 2x2 grid) ---
    preview_path: String,
    preview_texture: Option<TextureHandle>,
    preview_error: String,
    preview_zoom: f32,
    preview_auto_refresh: bool,
    preview_refresh_interval_secs: f32,
    preview_last_refresh: Option<Instant>,

    // --- Path tracing render output ---
    pathtracing_path: String,
    pathtracing_texture: Option<TextureHandle>,
    pathtracing_error: String,
    pathtracing_zoom: f32,

    // --- Persistence ---
    state_dirty: bool,
    state_last_save: Option<Instant>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SidebarTab {
    Scene,
    PathTracing,
    Training,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MainTab {
    Training,
    PathTracing,
    Log,
    Xml,
}

#[derive(Clone, Serialize, Deserialize)]
struct PersistedState {
    config: CornellBoxConfig,

    python_exe: String,
    render_scene_path: String,
    render_variant: String,
    render_spp: u32,
    render_out_path: String,
    drjit_libllvm_path: String,

    fit_steps: u32,
    fit_lr: f32,
    fit_out_dir: String,

    // Training preview (2x2 grid: GT|Current / Diff|Params)
    preview_path: String,
    preview_zoom: f32,
    preview_auto_refresh: bool,
    preview_refresh_interval_secs: f32,

    // Path tracing render output
    #[serde(default = "default_pathtracing_path")]
    pathtracing_path: String,
    #[serde(default = "default_pathtracing_zoom")]
    pathtracing_zoom: f32,
}

fn default_pathtracing_path() -> String {
    "renders/preview.png".to_string()
}

fn default_pathtracing_zoom() -> f32 {
    1.0
}

enum RenderJobState {
    Idle,
    Running {
        started_at: Instant,
        rx: mpsc::Receiver<RenderJobResult>,
        live_log: Arc<Mutex<Vec<String>>>,
    },
    Finished(RenderJobResult),
}

struct RenderJobResult {
    ok: bool,
    exit_code: Option<i32>,
    stdout: String,
    stderr: String,
}

#[derive(Debug, Clone)]
struct TrainingProgress {
    step: u32,
    total_steps: u32,
    loss: f32,
    // Simple diffuse fitting
    albedo: Option<[f32; 3]>,
    // Disney BRDF parameters
    base_color: Option<[f32; 3]>,
    roughness: Option<f32>,
    metallic: Option<f32>,
    specular: Option<f32>,
}

impl MitsubaStudioApp {
    fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        let mut app = Self::new_default();
        if let Some(st) = app.load_state_from_disk() {
            app.apply_persisted_state(st);
            app.last_status = "Loaded previous GUI state".to_string();
        }

        // Auto-fix Python path for cross-platform compatibility
        app.python_exe = Self::normalize_python_path(&app.python_exe);

        app.regenerate_preview();
        app
    }

    /// Normalize and validate Python path for the current platform
    fn normalize_python_path(path: &str) -> String {
        let trimmed = path.trim();

        // If it's just "python" or "python3", keep it as is (system Python)
        if trimmed == "python" || trimmed == "python3" || trimmed == "py" {
            return trimmed.to_string();
        }

        // Check if the provided path exists and is valid
        let path_buf = PathBuf::from(trimmed);
        if path_buf.is_file() {
            return trimmed.to_string();
        }

        // Auto-detect platform-specific virtual environment path
        let venv_paths = if cfg!(windows) {
            vec![
                ".venv/Scripts/python.exe",
                ".venv\\Scripts\\python.exe",
                "venv/Scripts/python.exe",
                "venv\\Scripts\\python.exe",
            ]
        } else {
            vec![
                ".venv/bin/python",
                ".venv/bin/python3",
                "venv/bin/python",
                "venv/bin/python3",
            ]
        };

        // Try to find an existing Python executable
        for venv_path in &venv_paths {
            let pb = PathBuf::from(venv_path);
            if pb.is_file() {
                return venv_path.to_string();
            }
        }

        // If nothing found, return platform-appropriate default
        Self::get_default_python_path()
    }

    /// Get platform-specific default Python path
    fn get_default_python_path() -> String {
        if cfg!(windows) {
            ".venv/Scripts/python.exe".to_string()
        } else if cfg!(target_os = "macos") {
            ".venv/bin/python3".to_string()
        } else {
            ".venv/bin/python".to_string()
        }
    }

    fn parse_training_progress(line: &str, total_steps: u32) -> Option<TrainingProgress> {
        // Parse training log supporting both simple and Disney BRDF formats:
        // Simple: "step=0025 loss=0.123456 albedo=[0.1 0.2 0.3]"
        // Disney: "step=0025 loss=0.123456 baseColor=[0.1 0.2 0.3] roughness=0.5 metallic=0.0 specular=0.5"

        if !line.contains("step=") || !line.contains("loss=") {
            return None;
        }

        let step = line
            .split("step=")
            .nth(1)?
            .split_whitespace()
            .next()?
            .parse::<u32>()
            .ok()?;

        let loss = line
            .split("loss=")
            .nth(1)?
            .split_whitespace()
            .next()?
            .parse::<f32>()
            .ok()?;

        // Try parsing Disney BRDF parameters first
        let base_color = Self::parse_vec3_param(line, "baseColor");
        let roughness = Self::parse_float_param(line, "roughness");
        let metallic = Self::parse_float_param(line, "metallic");
        let specular = Self::parse_float_param(line, "specular");

        // If Disney params found, use them
        if base_color.is_some() || roughness.is_some() {
            return Some(TrainingProgress {
                step,
                total_steps,
                loss,
                albedo: None,
                base_color,
                roughness,
                metallic,
                specular,
            });
        }

        // Otherwise try parsing simple albedo
        let albedo = Self::parse_vec3_param(line, "albedo");

        Some(TrainingProgress {
            step,
            total_steps,
            loss,
            albedo,
            base_color: None,
            roughness: None,
            metallic: None,
            specular: None,
        })
    }

    fn parse_vec3_param(line: &str, param_name: &str) -> Option<[f32; 3]> {
        let param_str = line.split(&format!("{}=", param_name)).nth(1)?;
        let values: Vec<f32> = param_str
            .trim_start_matches('[')
            .split(']')
            .next()?
            .split_whitespace()
            .filter_map(|s| s.parse::<f32>().ok())
            .collect();

        if values.len() == 3 {
            Some([values[0], values[1], values[2]])
        } else {
            None
        }
    }

    fn parse_float_param(line: &str, param_name: &str) -> Option<f32> {
        line.split(&format!("{}=", param_name))
            .nth(1)?
            .split_whitespace()
            .next()?
            .parse::<f32>()
            .ok()
    }

    fn new_default() -> Self {
        let config = CornellBoxConfig::default();
        let xml_preview = generate_cbox_xml(&config);
        let drjit_libllvm_path = default_drjit_libllvm_path();

        Self {
            config,
            last_status: "".to_string(),
            xml_preview,

            sidebar_tab: SidebarTab::Scene,
            main_tab: MainTab::Training,
            style_applied: false,

            // Default paths assume `current_dir` will be the workspace root (we set it when spawning).
            python_exe: Self::get_default_python_path(),
            render_scene_path: "scenes/cbox.xml".to_string(),
            render_variant: "scalar_rgb".to_string(),
            render_spp: 64,
            render_out_path: "renders/preview.png".to_string(),
            drjit_libllvm_path,

            fit_steps: 400,
            fit_lr: 0.05,
            fit_out_dir: "renders/fit".to_string(),

            job: RenderJobState::Idle,
            training_progress: None,

            preview_path: "renders/fit/progress.png".to_string(),
            preview_texture: None,
            preview_error: "".to_string(),
            preview_zoom: 1.0,
            preview_auto_refresh: true,
            preview_refresh_interval_secs: 2.0,
            preview_last_refresh: None,

            pathtracing_path: "renders/preview.png".to_string(),
            pathtracing_texture: None,
            pathtracing_error: "".to_string(),
            pathtracing_zoom: 1.0,

            state_dirty: false,
            state_last_save: None,
        }
    }

    fn apply_style(ctx: &egui::Context) {
        let mut style = (*ctx.style()).clone();
        style.visuals = egui::Visuals::dark();
        style.spacing.item_spacing = egui::vec2(10.0, 8.0);
        style.spacing.button_padding = egui::vec2(10.0, 6.0);
        style.spacing.window_margin = egui::Margin::same(12.0);
        style.visuals.window_rounding = egui::Rounding::same(10.0);
        style.visuals.widgets.inactive.rounding = egui::Rounding::same(8.0);
        style.visuals.widgets.hovered.rounding = egui::Rounding::same(8.0);
        style.visuals.widgets.active.rounding = egui::Rounding::same(8.0);
        ctx.set_style(style);
    }

    fn regenerate_preview(&mut self) {
        self.xml_preview = generate_cbox_xml(&self.config);
    }

    fn state_file_path(&self) -> PathBuf {
        self.workspace_root().join(".mitsuba_studio_state.json")
    }

    fn load_state_from_disk(&self) -> Option<PersistedState> {
        let path = self.state_file_path();
        let bytes = std::fs::read(path).ok()?;
        serde_json::from_slice::<PersistedState>(&bytes).ok()
    }

    fn apply_persisted_state(&mut self, st: PersistedState) {
        self.config = st.config;
        self.python_exe = st.python_exe;
        self.render_scene_path = st.render_scene_path;
        self.render_variant = st.render_variant;
        self.render_spp = st.render_spp;
        self.render_out_path = st.render_out_path;
        self.drjit_libllvm_path = if st.drjit_libllvm_path.trim().is_empty() {
            default_drjit_libllvm_path()
        } else {
            st.drjit_libllvm_path
        };

        self.preview_path = st.preview_path;
        self.preview_zoom = st.preview_zoom;
        self.preview_auto_refresh = st.preview_auto_refresh;
        self.preview_refresh_interval_secs = st.preview_refresh_interval_secs;

        self.pathtracing_path = st.pathtracing_path;
        self.pathtracing_zoom = st.pathtracing_zoom;

        self.fit_steps = st.fit_steps;
        self.fit_lr = st.fit_lr;
        self.fit_out_dir = st.fit_out_dir;
    }

    fn mark_state_dirty(&mut self) {
        self.state_dirty = true;
    }

    fn persist_state_if_needed(&mut self) {
        if !self.state_dirty {
            return;
        }

        let now = Instant::now();
        if let Some(t) = self.state_last_save {
            if now.duration_since(t) < Duration::from_secs_f32(1.0) {
                return;
            }
        }

        let st = PersistedState {
            config: self.config.clone(),

            python_exe: self.python_exe.clone(),
            render_scene_path: self.render_scene_path.clone(),
            render_variant: self.render_variant.clone(),
            render_spp: self.render_spp,
            render_out_path: self.render_out_path.clone(),
            drjit_libllvm_path: self.drjit_libllvm_path.clone(),

            fit_steps: self.fit_steps,
            fit_lr: self.fit_lr,
            fit_out_dir: self.fit_out_dir.clone(),

            preview_path: self.preview_path.clone(),
            preview_zoom: self.preview_zoom,
            preview_auto_refresh: self.preview_auto_refresh,
            preview_refresh_interval_secs: self.preview_refresh_interval_secs,

            pathtracing_path: self.pathtracing_path.clone(),
            pathtracing_zoom: self.pathtracing_zoom,
        };

        let path = self.state_file_path();
        let tmp = path.with_extension("json.tmp");
        if let Ok(bytes) = serde_json::to_vec_pretty(&st) {
            if std::fs::write(&tmp, bytes).is_ok() {
                let _ = std::fs::rename(&tmp, &path);
            }
        }

        self.state_dirty = false;
        self.state_last_save = Some(now);
    }

    fn write_xml_to_scene_path(&mut self) -> bool {
        // Write XML to the currently selected scene path in the Render panel.
        let rel = self.render_scene_path.trim();
        if rel.is_empty() {
            self.last_status = "Scene path is empty".to_string();
            return false;
        }

        let path = self.workspace_root().join(rel);
        let xml = generate_cbox_xml(&self.config);
        match std::fs::write(&path, xml.as_bytes()) {
            Ok(()) => {
                self.last_status = format!("Overwrote: {}", path.display());
                true
            }
            Err(err) => {
                self.last_status = format!("Overwrite failed: {}", err);
                false
            }
        }
    }

    fn load_preview_texture(&mut self, ctx: &egui::Context, set_status: bool) {
        let rel = self.preview_path.trim();
        if rel.is_empty() {
            self.preview_error = "Preview path is empty".to_string();
            self.preview_texture = None;
            return;
        }

        let path = self.workspace_root().join(rel);
        let bytes = match std::fs::read(&path) {
            Ok(b) => b,
            Err(err) => {
                self.preview_error = format!("Failed to read {}: {}", path.display(), err);
                self.preview_texture = None;
                return;
            }
        };

        let dyn_img = match image::load_from_memory(&bytes) {
            Ok(img) => img,
            Err(err) => {
                self.preview_error = format!("Failed to decode image: {}", err);
                self.preview_texture = None;
                return;
            }
        };

        let rgba = dyn_img.to_rgba8();
        let size = [rgba.width() as usize, rgba.height() as usize];
        let color_image = ColorImage::from_rgba_unmultiplied(size, rgba.as_raw());

        self.preview_texture =
            Some(ctx.load_texture("preview_png", color_image, egui::TextureOptions::LINEAR));
        self.preview_error.clear();
        if set_status {
            self.last_status = format!("Loaded preview: {}", path.display());
        }
        self.preview_last_refresh = Some(Instant::now());
    }

    fn load_pathtracing_texture(&mut self, ctx: &egui::Context, set_status: bool) {
        let rel = self.pathtracing_path.trim();
        if rel.is_empty() {
            self.pathtracing_error = "Path tracing path is empty".to_string();
            self.pathtracing_texture = None;
            return;
        }

        let path = self.workspace_root().join(rel);
        let bytes = match std::fs::read(&path) {
            Ok(b) => b,
            Err(err) => {
                self.pathtracing_error = format!("Failed to read {}: {}", path.display(), err);
                self.pathtracing_texture = None;
                return;
            }
        };

        let dyn_img = match image::load_from_memory(&bytes) {
            Ok(img) => img,
            Err(err) => {
                self.pathtracing_error = format!("Failed to decode image: {}", err);
                self.pathtracing_texture = None;
                return;
            }
        };

        let rgba = dyn_img.to_rgba8();
        let size = [rgba.width() as usize, rgba.height() as usize];
        let color_image = ColorImage::from_rgba_unmultiplied(size, rgba.as_raw());

        self.pathtracing_texture =
            Some(ctx.load_texture("pathtracing_png", color_image, egui::TextureOptions::LINEAR));
        self.pathtracing_error.clear();
        if set_status {
            self.last_status = format!("Loaded path tracing: {}", path.display());
        }
    }

    fn maybe_refresh_preview(&mut self, ctx: &egui::Context) {
        if !self.preview_auto_refresh {
            return;
        }

        // Ensure the UI keeps repainting even without user interaction.
        let interval = self.preview_refresh_interval_secs.max(0.2);
        ctx.request_repaint_after(Duration::from_secs_f32(interval.min(1.0)));

        let now = Instant::now();
        let should = match self.preview_last_refresh {
            None => true,
            Some(t) => now.duration_since(t).as_secs_f32() >= interval,
        };

        if should {
            self.load_preview_texture(ctx, false);
        }
    }

    fn workspace_root(&self) -> PathBuf {
        // Anchor all relative paths to the repository root.
        // When this crate lives at the repo root, CARGO_MANIFEST_DIR is exactly what we want.
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
    }

    fn can_start_job(&self) -> bool {
        matches!(self.job, RenderJobState::Idle | RenderJobState::Finished(_))
    }

    fn start_python_job(&mut self, args: Vec<String>) {
        if !self.can_start_job() {
            self.last_status = "A job is already running".to_string();
            return;
        }

        // Normalize Python path before starting job
        self.python_exe = Self::normalize_python_path(&self.python_exe);

        let (tx, rx) = mpsc::channel::<RenderJobResult>();
        let live_log = Arc::new(Mutex::new(Vec::<String>::new()));
        let live_log_clone = Arc::clone(&live_log);

        let cwd = self.workspace_root();
        let python_exe = self.python_exe.trim().to_string();
        let drjit_libllvm_path = self.drjit_libllvm_path.trim().to_string();

        std::thread::spawn(move || {
            use std::io::{BufRead, BufReader};
            use std::process::Stdio;

            let mut cmd = std::process::Command::new(python_exe);
            cmd.current_dir(cwd);
            for a in &args {
                cmd.arg(a);
            }
            if !drjit_libllvm_path.is_empty() {
                cmd.env("DRJIT_LIBLLVM_PATH", drjit_libllvm_path);
            }

            // Capture stdout and stderr separately for real-time streaming
            cmd.stdout(Stdio::piped());
            cmd.stderr(Stdio::piped());

            let spawn_result = cmd.spawn();
            let result = match spawn_result {
                Ok(mut child) => {
                    let mut stdout_lines = Vec::new();
                    let mut stderr_lines = Vec::new();

                    // Read stdout in real-time
                    if let Some(stdout) = child.stdout.take() {
                        let reader = BufReader::new(stdout);
                        for line in reader.lines() {
                            if let Ok(line) = line {
                                // Append to live log
                                if let Ok(mut log) = live_log_clone.lock() {
                                    log.push(line.clone());
                                }
                                stdout_lines.push(line);
                            }
                        }
                    }

                    // Read stderr
                    if let Some(stderr) = child.stderr.take() {
                        let reader = BufReader::new(stderr);
                        for line in reader.lines() {
                            if let Ok(line) = line {
                                if let Ok(mut log) = live_log_clone.lock() {
                                    log.push(format!("[stderr] {}", line));
                                }
                                stderr_lines.push(line);
                            }
                        }
                    }

                    let status = child.wait().unwrap_or_else(|_| {
                        std::process::ExitStatus::from_raw(1)
                    });

                    RenderJobResult {
                        ok: status.success(),
                        exit_code: status.code(),
                        stdout: stdout_lines.join("\n"),
                        stderr: stderr_lines.join("\n"),
                    }
                }
                Err(err) => RenderJobResult {
                    ok: false,
                    exit_code: None,
                    stdout: "".to_string(),
                    stderr: format!("Failed to spawn process: {err}"),
                },
            };

            let _ = tx.send(result);
        });

        self.last_status = "Job started...".to_string();
        self.job = RenderJobState::Running {
            started_at: Instant::now(),
            rx,
            live_log,
        };
    }

    fn maybe_pick_obj_file(&mut self) {
        let picked = FileDialog::new()
            .add_filter("Wavefront OBJ", &["obj"])
            .pick_file();

        let Some(path) = picked else {
            return;
        };

        // Prefer workspace-relative paths (so moving the project keeps working).
        let ws = self.workspace_root();
        let s = path
            .strip_prefix(&ws)
            .ok()
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_else(|| path.to_string_lossy().to_string());

        self.config.object.obj_filename = s;
        self.mark_state_dirty();
    }

    fn maybe_pick_ply_file(&mut self) {
        let picked = FileDialog::new()
            .add_filter("Stanford PLY", &["ply"])
            .pick_file();

        let Some(path) = picked else {
            return;
        };

        let ws = self.workspace_root();
        let s = path
            .strip_prefix(&ws)
            .ok()
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_else(|| path.to_string_lossy().to_string());

        self.config.object.ply_filename = s;
        self.mark_state_dirty();
    }
}

fn default_drjit_libllvm_path() -> String {
    #[cfg(target_os = "macos")]
    {
        let candidates = [
            "/opt/homebrew/opt/llvm/lib/libLLVM.dylib",
            "/usr/local/opt/llvm/lib/libLLVM.dylib",
        ];

        for p in candidates {
            if std::path::Path::new(p).exists() {
                return p.to_string();
            }
        }
    }

    String::new()
}

impl eframe::App for MitsubaStudioApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if !self.style_applied {
            Self::apply_style(ctx);
            self.style_applied = true;
        }

        // Periodic preview refresh (every N seconds)
        self.maybe_refresh_preview(ctx);

        // Poll subprocess results and parse training progress
        if let RenderJobState::Running { started_at: _, rx, live_log } = &self.job {
            // Parse latest training progress from live log
            if let Ok(log) = live_log.lock() {
                if !log.is_empty() {
                    // Parse from the last few lines
                    for line in log.iter().rev().take(10) {
                        if let Some(progress) = Self::parse_training_progress(line, self.fit_steps) {
                            self.training_progress = Some(progress);
                            break;
                        }
                    }
                }
            }

            match rx.try_recv() {
                Ok(result) => {
                    let job_succeeded = result.ok;
                    self.last_status = if result.ok {
                        "Job finished successfully".to_string()
                    } else {
                        format!("Job failed (exit={:?})", result.exit_code)
                    };
                    self.job = RenderJobState::Finished(result);

                    // Refresh appropriate image based on current tab
                    if job_succeeded {
                        match self.main_tab {
                            MainTab::Training => self.load_preview_texture(ctx, false),
                            MainTab::PathTracing => self.load_pathtracing_texture(ctx, false),
                            _ => {}
                        }
                    }
                }
                Err(mpsc::TryRecvError::Empty) => {
                    // Request repaint for live log updates
                    ctx.request_repaint();
                }
                Err(mpsc::TryRecvError::Disconnected) => {
                    self.last_status = "Job channel disconnected".to_string();
                    self.job = RenderJobState::Finished(RenderJobResult {
                        ok: false,
                        exit_code: None,
                        stdout: "".to_string(),
                        stderr: "Job channel disconnected".to_string(),
                    });
                }
            }
        }

        let running = matches!(self.job, RenderJobState::Running { .. });

        egui::TopBottomPanel::top("top_bar")
            .resizable(false)
            .exact_height(44.0)
            .show(ctx, |ui| {
                ui.add_space(6.0);
                ui.columns(3, |cols| {
                    cols[0].horizontal(|ui| {
                        ui.label(egui::RichText::new("Mitsuba Studio").strong().size(18.0));
                    });

                    cols[1].horizontal(|ui| {
                        ui.add_enabled_ui(!running, |ui| {
                            if ui.button("Render").clicked() {
                                self.mark_state_dirty();
                                if self.write_xml_to_scene_path() {
                                    // Set pathtracing path to render output and switch to PathTracing tab
                                    self.pathtracing_path = self.render_out_path.clone();
                                    self.main_tab = MainTab::PathTracing;

                                    self.start_python_job(vec![
                                        "tools/mitsuba_render.py".to_string(),
                                        "--scene".to_string(),
                                        self.render_scene_path.clone(),
                                        "--variant".to_string(),
                                        self.render_variant.clone(),
                                        "--spp".to_string(),
                                        self.render_spp.to_string(),
                                        "--out".to_string(),
                                        self.render_out_path.clone(),
                                    ]);
                                }
                            }

                            if ui.button("Diff Render").clicked() {
                                self.mark_state_dirty();
                                if self.write_xml_to_scene_path() {
                                    // Set pathtracing path to render output and switch to PathTracing tab
                                    self.pathtracing_path = self.render_out_path.clone();
                                    self.main_tab = MainTab::PathTracing;

                                    self.start_python_job(vec![
                                        "tools/mitsuba_diff_smoketest.py".to_string(),
                                        "--scene".to_string(),
                                        self.render_scene_path.clone(),
                                        "--variant".to_string(),
                                        "cuda_ad_rgb".to_string(),
                                        "--spp".to_string(),
                                        "4".to_string(),
                                    ]);
                                }
                            }
                        });

                        ui.separator();
                        ui.label("Variant");
                        ui.add(
                            egui::TextEdit::singleline(&mut self.render_variant)
                                .desired_width(120.0),
                        );
                        ui.label("SPP");
                        ui.add(
                            egui::DragValue::new(&mut self.render_spp)
                                .speed(1)
                                .range(1..=4096),
                        );
                    });

                    cols[2].with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if running {
                            ui.label(
                                egui::RichText::new("Running…").color(egui::Color32::LIGHT_YELLOW),
                            );
                        }
                        if !self.last_status.is_empty() {
                            ui.label(&self.last_status);
                        }
                    });
                });
            });

        egui::SidePanel::left("left_panel")
            .resizable(true)
            .default_width(360.0)
            .show(ctx, |ui| {
                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut self.sidebar_tab, SidebarTab::Scene, "Scene");
                    ui.selectable_value(&mut self.sidebar_tab, SidebarTab::PathTracing, "Path Tracing");
                    ui.selectable_value(&mut self.sidebar_tab, SidebarTab::Training, "Training");
                });
                ui.separator();

                egui::ScrollArea::vertical().show(ui, |ui| {
                    let mut changed = false;
                    let mut state_changed = false;

                    match self.sidebar_tab {
                        SidebarTab::Scene => {
                            egui::CollapsingHeader::new("Camera")
                                .default_open(true)
                                .show(ui, |ui| {
                                    changed |= ui
                                        .add(
                                            egui::Slider::new(
                                                &mut self.config.camera.fov_deg,
                                                10.0..=120.0,
                                            )
                                            .text("FOV (deg)"),
                                        )
                                        .changed();
                                    changed |= vec3_ui(
                                        ui,
                                        "origin",
                                        &mut self.config.camera.origin,
                                        -50.0..=50.0,
                                    );
                                    changed |= vec3_ui(
                                        ui,
                                        "target",
                                        &mut self.config.camera.target,
                                        -50.0..=50.0,
                                    );
                                    changed |=
                                        vec3_ui(ui, "up", &mut self.config.camera.up, -1.0..=1.0);
                                });

                            egui::CollapsingHeader::new("Light")
                                .default_open(true)
                                .show(ui, |ui| {
                                    changed |= ui
                                        .checkbox(
                                            &mut self.config.light.enabled,
                                            "Enable ceiling area light",
                                        )
                                        .changed();
                                    changed |= color3_ui(
                                        ui,
                                        "radiance",
                                        &mut self.config.light.radiance_rgb,
                                        0.0..=200.0,
                                    );
                                    changed |= ui
                                        .add(
                                            egui::Slider::new(
                                                &mut self.config.light.scale_xy,
                                                0.05..=1.0,
                                            )
                                            .text("scale_xy"),
                                        )
                                        .changed();
                                    changed |= ui
                                        .add(
                                            egui::Slider::new(&mut self.config.light.y, 0.5..=3.0)
                                                .text("y"),
                                        )
                                        .changed();
                                });

                            egui::CollapsingHeader::new("Scene Environment")
                                .default_open(true)
                                .show(ui, |ui| {
                                    ui.horizontal(|ui| {
                                        ui.label("Type");
                                        egui::ComboBox::from_id_salt("scene_environment_kind")
                                            .selected_text(match self.config.scene_environment.kind {
                                                SceneEnvironment::Empty => "Empty",
                                                SceneEnvironment::CornellBox => "Cornell Box",
                                                SceneEnvironment::ObjFile => "OBJ File",
                                            })
                                            .show_ui(ui, |ui| {
                                                changed |= ui
                                                    .selectable_value(
                                                        &mut self.config.scene_environment.kind,
                                                        SceneEnvironment::Empty,
                                                        "Empty",
                                                    )
                                                    .changed();
                                                changed |= ui
                                                    .selectable_value(
                                                        &mut self.config.scene_environment.kind,
                                                        SceneEnvironment::CornellBox,
                                                        "Cornell Box",
                                                    )
                                                    .changed();
                                                changed |= ui
                                                    .selectable_value(
                                                        &mut self.config.scene_environment.kind,
                                                        SceneEnvironment::ObjFile,
                                                        "OBJ File",
                                                    )
                                                    .changed();
                                            });
                                    });

                                    if self.config.scene_environment.kind == SceneEnvironment::ObjFile {
                                        ui.label("OBJ file (relative to scenes/)");
                                        ui.horizontal(|ui| {
                                            changed |= ui
                                                .text_edit_singleline(&mut self.config.scene_environment.obj_filename)
                                                .changed();
                                            if ui.button("Browse").clicked() {
                                                let picked = rfd::FileDialog::new()
                                                    .add_filter("Wavefront OBJ", &["obj"])
                                                    .pick_file();
                                                if let Some(path) = picked {
                                                    let ws = self.workspace_root();
                                                    let s = path
                                                        .strip_prefix(&ws)
                                                        .ok()
                                                        .map(|p| p.to_string_lossy().to_string())
                                                        .unwrap_or_else(|| path.to_string_lossy().to_string());
                                                    self.config.scene_environment.obj_filename = s;
                                                    changed = true;
                                                }
                                            }
                                        });
                                        changed |= ui
                                            .add(
                                                egui::Slider::new(&mut self.config.scene_environment.obj_scale, 0.01..=10.0)
                                                    .text("scale"),
                                            )
                                            .changed();
                                    }
                                });

                            egui::CollapsingHeader::new("Object")
                                .default_open(true)
                                .show(ui, |ui| {
                                    changed |= ui
                                        .checkbox(&mut self.config.object.enabled, "Enable object")
                                        .changed();

                                    ui.horizontal(|ui| {
                                        ui.label("Type");
                                        egui::ComboBox::from_id_salt("object_kind")
                                            .selected_text(match self.config.object.kind {
                                                ObjectKind::Sphere => "sphere",
                                                ObjectKind::Cube => "cube",
                                                ObjectKind::PlyMesh => "ply mesh",
                                                ObjectKind::ObjMesh => "obj mesh",
                                            })
                                            .show_ui(ui, |ui| {
                                                changed |= ui
                                                    .selectable_value(
                                                        &mut self.config.object.kind,
                                                        ObjectKind::Sphere,
                                                        "sphere",
                                                    )
                                                    .changed();
                                                changed |= ui
                                                    .selectable_value(
                                                        &mut self.config.object.kind,
                                                        ObjectKind::Cube,
                                                        "cube",
                                                    )
                                                    .changed();
                                                changed |= ui
                                                    .selectable_value(
                                                        &mut self.config.object.kind,
                                                        ObjectKind::PlyMesh,
                                                        "ply mesh",
                                                    )
                                                    .changed();
                                                changed |= ui
                                                    .selectable_value(
                                                        &mut self.config.object.kind,
                                                        ObjectKind::ObjMesh,
                                                        "obj mesh",
                                                    )
                                                    .changed();
                                            });
                                    });

                                    changed |= vec3_ui(
                                        ui,
                                        "translate",
                                        &mut self.config.object.translate,
                                        -2.0..=2.0,
                                    );

                                    match self.config.object.kind {
                                        ObjectKind::Sphere => {
                                            changed |= ui
                                                .add(
                                                    egui::Slider::new(
                                                        &mut self.config.object.sphere_radius,
                                                        0.05..=1.0,
                                                    )
                                                    .text("radius"),
                                                )
                                                .changed();
                                        }
                                        ObjectKind::Cube => {
                                            changed |= vec3_ui(
                                                ui,
                                                "scale",
                                                &mut self.config.object.cube_scale,
                                                0.05..=2.0,
                                            );
                                        }
                                        ObjectKind::PlyMesh => {
                                            ui.label("PLY (relative to scenes/)");
                                            ui.horizontal(|ui| {
                                                changed |= ui
                                                    .text_edit_singleline(
                                                        &mut self.config.object.ply_filename,
                                                    )
                                                    .changed();
                                                if ui.button("Browse").clicked() {
                                                    self.maybe_pick_ply_file();
                                                    changed = true;
                                                }
                                            });
                                            changed |= ui
                                                .add(
                                                    egui::Slider::new(
                                                        &mut self.config.object.mesh_scale,
                                                        0.01..=10.0,
                                                    )
                                                    .text("mesh_scale"),
                                                )
                                                .changed();
                                        }
                                        ObjectKind::ObjMesh => {
                                            ui.label("OBJ (relative to scenes/)");
                                            ui.horizontal(|ui| {
                                                changed |= ui
                                                    .text_edit_singleline(
                                                        &mut self.config.object.obj_filename,
                                                    )
                                                    .changed();
                                                if ui.button("Browse").clicked() {
                                                    self.maybe_pick_obj_file();
                                                    changed = true;
                                                }
                                            });
                                            changed |= ui
                                                .add(
                                                    egui::Slider::new(
                                                        &mut self.config.object.mesh_scale,
                                                        0.01..=10.0,
                                                    )
                                                    .text("mesh_scale"),
                                                )
                                                .changed();
                                        }
                                    }

                                    ui.add_space(10.0);
                                    egui::CollapsingHeader::new("Material (BSDF)")
                                        .default_open(true)
                                        .show(ui, |ui| {
                                            changed |= bsdf_node_ui(
                                                ui,
                                                &self.workspace_root(),
                                                &mut self.config.object.bsdf,
                                                0,
                                            );
                                        });
                                });
                        }
                        SidebarTab::PathTracing => {
                            ui.label("Path tracing renderer settings (Mitsuba 3)");
                            ui.add_space(6.0);

                            egui::CollapsingHeader::new("Film")
                                .default_open(true)
                                .show(ui, |ui| {
                                    egui::Grid::new("film_grid")
                                        .num_columns(2)
                                        .spacing(egui::vec2(10.0, 6.0))
                                        .show(ui, |ui| {
                                            ui.label("Width");
                                            changed |= ui
                                                .add(
                                                    egui::DragValue::new(
                                                        &mut self.config.film.width,
                                                    )
                                                    .speed(1)
                                                    .range(16..=8192),
                                                )
                                                .changed();
                                            ui.end_row();

                                            ui.label("Height");
                                            changed |= ui
                                                .add(
                                                    egui::DragValue::new(
                                                        &mut self.config.film.height,
                                                    )
                                                    .speed(1)
                                                    .range(16..=8192),
                                                )
                                                .changed();
                                            ui.end_row();
                                        });
                                });

                            egui::CollapsingHeader::new("Sampler")
                                .default_open(true)
                                .show(ui, |ui| {
                                    egui::Grid::new("sampler_grid")
                                        .num_columns(2)
                                        .spacing(egui::vec2(10.0, 6.0))
                                        .show(ui, |ui| {
                                            ui.label("Sample count (SPP)");
                                            changed |= ui
                                                .add(
                                                    egui::DragValue::new(
                                                        &mut self.config.sampler.sample_count,
                                                    )
                                                    .speed(1)
                                                    .range(1..=4096),
                                                )
                                                .changed();
                                            ui.end_row();
                                        });
                                });

                            egui::CollapsingHeader::new("Render Settings")
                                .default_open(true)
                                .show(ui, |ui| {
                                    egui::Grid::new("render_settings_grid")
                                        .num_columns(2)
                                        .spacing(egui::vec2(10.0, 8.0))
                                        .show(ui, |ui| {
                                            ui.label("Variant");
                                            state_changed |=
                                                ui.text_edit_singleline(&mut self.render_variant).changed();
                                            ui.end_row();

                                            ui.label("SPP (override)");
                                            state_changed |= ui
                                                .add(
                                                    egui::DragValue::new(&mut self.render_spp)
                                                        .speed(1)
                                                        .range(1..=4096),
                                                )
                                                .changed();
                                            ui.end_row();

                                            ui.label("Output path");
                                            state_changed |= ui
                                                .text_edit_singleline(&mut self.render_out_path)
                                                .changed();
                                            ui.end_row();

                                            ui.label("Scene path");
                                            state_changed |= ui
                                                .text_edit_singleline(&mut self.render_scene_path)
                                                .changed();
                                            ui.end_row();
                                        });
                                });

                            egui::CollapsingHeader::new("Advanced")
                                .default_open(false)
                                .show(ui, |ui| {
                                    ui.label("DRJIT_LIBLLVM_PATH for llvm_ad_* variants");
                                    state_changed |= ui
                                        .text_edit_singleline(&mut self.drjit_libllvm_path)
                                        .changed();
                                    ui.label("Windows: C:\\Program Files\\LLVM\\bin\\LLVM-C.dll");
                                    ui.label("macOS: /opt/homebrew/opt/llvm/lib/libLLVM.dylib");
                                });
                        }
                        SidebarTab::Training => {
                            ui.label("Material fitting with nvdiffrast (requires CUDA GPU)");
                            ui.add_space(6.0);

                            egui::Grid::new("python_grid")
                                .num_columns(2)
                                .spacing(egui::vec2(10.0, 8.0))
                                .show(ui, |ui| {
                                    ui.label("Python exe");
                                    state_changed |=
                                        ui.text_edit_singleline(&mut self.python_exe).changed();
                                    ui.end_row();
                                });

                            egui::CollapsingHeader::new("Training Parameters")
                                .default_open(true)
                                .show(ui, |ui| {

                                    egui::Grid::new("fit_grid")
                                        .num_columns(2)
                                        .spacing(egui::vec2(10.0, 8.0))
                                        .show(ui, |ui| {
                                            ui.label("Steps");
                                            state_changed |= ui
                                                .add(
                                                    egui::DragValue::new(&mut self.fit_steps)
                                                        .speed(10)
                                                        .range(10..=20000),
                                                )
                                                .changed();
                                            ui.end_row();

                                            ui.label("LR");
                                            state_changed |= ui
                                                .add(
                                                    egui::DragValue::new(&mut self.fit_lr)
                                                        .speed(0.005)
                                                        .range(1e-5..=1.0),
                                                )
                                                .changed();
                                            ui.end_row();

                                            ui.label("Out dir");
                                            state_changed |= ui.text_edit_singleline(&mut self.fit_out_dir).changed();
                                            ui.end_row();
                                        });

                                    ui.add_space(8.0);
                                    ui.add_enabled_ui(!running, |ui| {
                                        ui.horizontal(|ui| {
                                            if ui.button("Fit diffuse albedo").clicked() {
                                                self.mark_state_dirty();
                                                if self.write_xml_to_scene_path() {
                                                    // Set preview path to show training progress
                                                    self.preview_path = format!("{}/progress.png", self.fit_out_dir);
                                                    self.preview_auto_refresh = true;
                                                    self.preview_refresh_interval_secs = 1.0; // Refresh every second

                                                    self.start_python_job(vec![
                                                        "tools/mitsuba_raster_fit_nvdiffrast.py".to_string(),
                                                        "--scene".to_string(),
                                                        self.render_scene_path.clone(),
                                                        "--gt-image".to_string(),
                                                        self.pathtracing_path.clone(),
                                                        "--gt-variant".to_string(),
                                                        self.render_variant.clone(),
                                                        "--gt-spp".to_string(),
                                                        self.render_spp.to_string(),
                                                        "--steps".to_string(),
                                                        self.fit_steps.to_string(),
                                                        "--lr".to_string(),
                                                        self.fit_lr.to_string(),
                                                        "--out-dir".to_string(),
                                                        self.fit_out_dir.clone(),
                                                    ]);
                                                }
                                            }

                                            if ui.button("Fit Disney BRDF").clicked() {
                                                self.mark_state_dirty();
                                                if self.write_xml_to_scene_path() {
                                                    // Set preview path to show training progress
                                                    self.preview_path = format!("{}_disney/progress.png", self.fit_out_dir);
                                                    self.preview_auto_refresh = true;
                                                    self.preview_refresh_interval_secs = 1.0; // Refresh every second

                                                    self.start_python_job(vec![
                                                        "tools/mitsuba_raster_fit_disney.py".to_string(),
                                                        "--scene".to_string(),
                                                        self.render_scene_path.clone(),
                                                        "--gt-image".to_string(),
                                                        self.pathtracing_path.clone(),
                                                        "--gt-variant".to_string(),
                                                        self.render_variant.clone(),
                                                        "--gt-spp".to_string(),
                                                        self.render_spp.to_string(),
                                                        "--steps".to_string(),
                                                        self.fit_steps.to_string(),
                                                        "--lr".to_string(),
                                                        self.fit_lr.to_string(),
                                                        "--out-dir".to_string(),
                                                        format!("{}_disney", self.fit_out_dir),
                                                    ]);
                                                }
                                            }
                                        });
                                    });
                                });

                            ui.add_space(10.0);
                            ui.add_enabled_ui(!running, |ui| {
                                if ui.button("Write XML to scene path").clicked() {
                                    self.mark_state_dirty();
                                    let _ = self.write_xml_to_scene_path();
                                }
                            });
                        }
                    }

                    if changed {
                        self.regenerate_preview();
                        self.mark_state_dirty();
                    }
                    if state_changed {
                        self.mark_state_dirty();
                    }
                });
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.add_space(6.0);
            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.main_tab, MainTab::Training, "Training");
                ui.selectable_value(&mut self.main_tab, MainTab::PathTracing, "Path Tracing");
                ui.selectable_value(&mut self.main_tab, MainTab::Log, "Log");
                ui.selectable_value(&mut self.main_tab, MainTab::Xml, "XML");
            });
            ui.separator();

            match self.main_tab {
                MainTab::Training => {
                    ui.horizontal(|ui| {
                        ui.label("Training Progress (2x2 grid)");
                        ui.add_space(8.0);
                        ui.label("Path");
                        if ui
                            .add(
                                egui::TextEdit::singleline(&mut self.preview_path)
                                    .desired_width(260.0),
                            )
                            .changed()
                        {
                            self.mark_state_dirty();
                        }
                        if ui.button("Use fit output").clicked() {
                            self.preview_path = format!("{}/progress.png", self.fit_out_dir);
                            self.mark_state_dirty();
                        }
                        if ui.button("Reload").clicked() {
                            self.load_preview_texture(ctx, true);
                        }
                        if ui
                            .checkbox(&mut self.preview_auto_refresh, "Auto")
                            .changed()
                        {
                            self.mark_state_dirty();
                        }
                        ui.add_enabled_ui(self.preview_auto_refresh, |ui| {
                            ui.label("Every (s)");
                            if ui
                                .add(
                                    egui::DragValue::new(&mut self.preview_refresh_interval_secs)
                                        .speed(0.1)
                                        .range(0.2..=60.0),
                                )
                                .changed()
                            {
                                self.mark_state_dirty();
                            }
                        });
                        ui.separator();
                        ui.label("Zoom");
                        if ui
                            .add(
                                egui::Slider::new(&mut self.preview_zoom, 0.25..=2.0)
                                    .show_value(false),
                            )
                            .changed()
                        {
                            self.mark_state_dirty();
                        }
                        ui.label(format!("{:.2}×", self.preview_zoom));
                    });
                    ui.add_space(8.0);

                    // First-time auto load
                    if self.preview_texture.is_none() && self.preview_error.is_empty() {
                        self.load_preview_texture(ctx, false);
                    }

                    let available = ui.available_size();
                    let (rect, _) = ui.allocate_exact_size(available, egui::Sense::hover());
                    let frame = egui::Frame::canvas(ui.style())
                        .rounding(egui::Rounding::same(12.0))
                        .inner_margin(egui::Margin::same(12.0));
                    frame.show(ui, |ui| {
                        ui.allocate_new_ui(egui::UiBuilder::new().max_rect(rect), |ui| {
                            ui.set_min_size(rect.size());

                            if let Some(tex) = &self.preview_texture {
                                // Fit using configured film aspect ratio, not the decoded PNG dimensions.
                                let film_aspect = if self.config.film.width > 0 {
                                    self.config.film.height as f32 / self.config.film.width as f32
                                } else {
                                    1.0
                                };

                                let max_w = ui.available_width().max(1.0);
                                let max_h = ui.available_height().max(1.0);
                                let mut w = max_w;
                                let mut h = w * film_aspect;
                                if h > max_h {
                                    h = max_h;
                                    w = (h / film_aspect).max(1.0);
                                }

                                ui.centered_and_justified(|ui| {
                                    ui.image((
                                        tex.id(),
                                        egui::vec2(w * self.preview_zoom, h * self.preview_zoom),
                                    ));
                                });
                            } else if !self.preview_error.is_empty() {
                                ui.centered_and_justified(|ui| {
                                    ui.colored_label(egui::Color32::LIGHT_RED, &self.preview_error);
                                });
                            } else {
                                ui.centered_and_justified(|ui| {
                                    ui.label(egui::RichText::new("No preview loaded").weak());
                                });
                            }
                        });
                    });
                }
                MainTab::PathTracing => {
                    ui.horizontal(|ui| {
                        ui.label("Path Tracing Render");
                        ui.add_space(8.0);
                        ui.label("Path");
                        if ui
                            .add(
                                egui::TextEdit::singleline(&mut self.pathtracing_path)
                                    .desired_width(260.0),
                            )
                            .changed()
                        {
                            self.mark_state_dirty();
                        }
                        if ui.button("Use render output").clicked() {
                            self.pathtracing_path = self.render_out_path.clone();
                            self.mark_state_dirty();
                        }
                        if ui.button("Reload").clicked() {
                            self.load_pathtracing_texture(ctx, true);
                        }
                        ui.separator();
                        ui.label("Zoom");
                        if ui
                            .add(
                                egui::Slider::new(&mut self.pathtracing_zoom, 0.25..=2.0)
                                    .show_value(false),
                            )
                            .changed()
                        {
                            self.mark_state_dirty();
                        }
                        ui.label(format!("{:.2}×", self.pathtracing_zoom));
                    });
                    ui.add_space(8.0);

                    // First-time auto load
                    if self.pathtracing_texture.is_none() && self.pathtracing_error.is_empty() {
                        self.load_pathtracing_texture(ctx, false);
                    }

                    let available = ui.available_size();
                    let (rect, _) = ui.allocate_exact_size(available, egui::Sense::hover());
                    let frame = egui::Frame::canvas(ui.style())
                        .rounding(egui::Rounding::same(12.0))
                        .inner_margin(egui::Margin::same(12.0));
                    frame.show(ui, |ui| {
                        ui.allocate_new_ui(egui::UiBuilder::new().max_rect(rect), |ui| {
                            ui.set_min_size(rect.size());

                            if let Some(tex) = &self.pathtracing_texture {
                                // Fit using configured film aspect ratio
                                let film_aspect = if self.config.film.width > 0 {
                                    self.config.film.height as f32 / self.config.film.width as f32
                                } else {
                                    1.0
                                };

                                let max_w = ui.available_width().max(1.0);
                                let max_h = ui.available_height().max(1.0);
                                let mut w = max_w;
                                let mut h = w * film_aspect;
                                if h > max_h {
                                    h = max_h;
                                    w = (h / film_aspect).max(1.0);
                                }

                                ui.centered_and_justified(|ui| {
                                    ui.image((
                                        tex.id(),
                                        egui::vec2(w * self.pathtracing_zoom, h * self.pathtracing_zoom),
                                    ));
                                });
                            } else if !self.pathtracing_error.is_empty() {
                                ui.centered_and_justified(|ui| {
                                    ui.colored_label(egui::Color32::LIGHT_RED, &self.pathtracing_error);
                                });
                            } else {
                                ui.centered_and_justified(|ui| {
                                    ui.label(egui::RichText::new("No render loaded").weak());
                                });
                            }
                        });
                    });
                }
                MainTab::Log => {
                    // Show training progress if available
                    if let Some(progress) = &self.training_progress {
                        ui.group(|ui| {
                            ui.heading("Training Progress");
                            ui.add_space(4.0);

                            // Progress bar
                            let progress_ratio = progress.step as f32 / progress.total_steps.max(1) as f32;
                            ui.add(
                                egui::ProgressBar::new(progress_ratio)
                                    .text(format!("Step {}/{}", progress.step, progress.total_steps))
                                    .animate(true),
                            );

                            ui.add_space(6.0);
                            egui::Grid::new("progress_grid")
                                .num_columns(2)
                                .spacing(egui::vec2(10.0, 6.0))
                                .show(ui, |ui| {
                                    ui.label("Loss:");
                                    ui.label(format!("{:.6}", progress.loss));
                                    ui.end_row();

                                    // Display simple albedo if available
                                    if let Some(albedo) = progress.albedo {
                                        ui.label("Albedo:");
                                        ui.horizontal(|ui| {
                                            let rgb = [
                                                (albedo[0] * 255.0) as u8,
                                                (albedo[1] * 255.0) as u8,
                                                (albedo[2] * 255.0) as u8,
                                            ];
                                            ui.label(format!("[{:.3}, {:.3}, {:.3}]", albedo[0], albedo[1], albedo[2]));
                                            let mut color = egui::Color32::from_rgb(rgb[0], rgb[1], rgb[2]);
                                            ui.color_edit_button_srgba(&mut color);
                                        });
                                        ui.end_row();
                                    }

                                    // Display Disney BRDF parameters if available
                                    if let Some(base_color) = progress.base_color {
                                        ui.label("Base Color:");
                                        ui.horizontal(|ui| {
                                            let rgb = [
                                                (base_color[0] * 255.0) as u8,
                                                (base_color[1] * 255.0) as u8,
                                                (base_color[2] * 255.0) as u8,
                                            ];
                                            ui.label(format!("[{:.3}, {:.3}, {:.3}]", base_color[0], base_color[1], base_color[2]));
                                            let mut color = egui::Color32::from_rgb(rgb[0], rgb[1], rgb[2]);
                                            ui.color_edit_button_srgba(&mut color);
                                        });
                                        ui.end_row();
                                    }

                                    if let Some(roughness) = progress.roughness {
                                        ui.label("Roughness:");
                                        ui.add(egui::ProgressBar::new(roughness).text(format!("{:.3}", roughness)));
                                        ui.end_row();
                                    }

                                    if let Some(metallic) = progress.metallic {
                                        ui.label("Metallic:");
                                        ui.add(egui::ProgressBar::new(metallic).text(format!("{:.3}", metallic)));
                                        ui.end_row();
                                    }

                                    if let Some(specular) = progress.specular {
                                        ui.label("Specular:");
                                        ui.add(egui::ProgressBar::new(specular).text(format!("{:.3}", specular)));
                                        ui.end_row();
                                    }
                                });
                        });
                        ui.add_space(8.0);
                    }

                    let mut log_text = String::new();
                    match &self.job {
                        RenderJobState::Idle => log_text.push_str("(idle)\n"),
                        RenderJobState::Running { started_at, live_log, .. } => {
                            log_text.push_str(&format!(
                                "Running for {:.1}s...\n\n",
                                started_at.elapsed().as_secs_f32()
                            ));

                            // Show live log
                            if let Ok(log) = live_log.lock() {
                                for line in log.iter() {
                                    log_text.push_str(line);
                                    log_text.push('\n');
                                }
                            }
                        }
                        RenderJobState::Finished(r) => {
                            log_text.push_str(&format!("ok={} exit={:?}\n", r.ok, r.exit_code));
                            if !r.stdout.is_empty() {
                                log_text.push_str("\n--- stdout ---\n");
                                log_text.push_str(&r.stdout);
                            }
                            if !r.stderr.is_empty() {
                                log_text.push_str("\n--- stderr ---\n");
                                log_text.push_str(&r.stderr);
                            }
                        }
                    }

                    ui.horizontal(|ui| {
                        if ui.button("Copy").clicked() {
                            ui.output_mut(|o| o.copied_text = log_text.clone());
                        }
                        if matches!(self.job, RenderJobState::Running { .. }) {
                            ui.label(egui::RichText::new("● Live").color(egui::Color32::GREEN));
                        }
                    });
                    ui.add_space(6.0);

                    egui::ScrollArea::vertical()
                        .auto_shrink([false; 2])
                        .stick_to_bottom(true)
                        .show(ui, |ui| {
                            ui.add(
                                egui::TextEdit::multiline(&mut log_text)
                                    .font(egui::TextStyle::Monospace)
                                    .desired_width(f32::INFINITY)
                                    .desired_rows(24),
                            );
                        });
                }
                MainTab::Xml => {
                    ui.horizontal(|ui| {
                        if ui.button("Copy").clicked() {
                            ui.output_mut(|o| o.copied_text = self.xml_preview.clone());
                        }
                    });
                    ui.add_space(6.0);

                    ui.add(
                        egui::TextEdit::multiline(&mut self.xml_preview)
                            .font(egui::TextStyle::Monospace)
                            .desired_width(f32::INFINITY)
                            .desired_rows(40),
                    );
                }
            }
        });

        // Persist GUI state (throttled)
        self.persist_state_if_needed();
    }
}

fn vec3_ui(
    ui: &mut egui::Ui,
    label: &str,
    v: &mut [f32; 3],
    range: std::ops::RangeInclusive<f32>,
) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.label(label);
        changed |= ui
            .add(
                egui::DragValue::new(&mut v[0])
                    .range(range.clone())
                    .speed(0.01),
            )
            .changed();
        changed |= ui
            .add(
                egui::DragValue::new(&mut v[1])
                    .range(range.clone())
                    .speed(0.01),
            )
            .changed();
        changed |= ui
            .add(egui::DragValue::new(&mut v[2]).range(range).speed(0.01))
            .changed();
    });
    changed
}

fn color3_ui(
    ui: &mut egui::Ui,
    label: &str,
    c: &mut [f32; 3],
    range: std::ops::RangeInclusive<f32>,
) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.label(label);
        changed |= ui
            .add(
                egui::DragValue::new(&mut c[0])
                    .range(range.clone())
                    .speed(0.01),
            )
            .changed();
        changed |= ui
            .add(
                egui::DragValue::new(&mut c[1])
                    .range(range.clone())
                    .speed(0.01),
            )
            .changed();
        changed |= ui
            .add(egui::DragValue::new(&mut c[2]).range(range).speed(0.01))
            .changed();
    });
    changed
}

fn bsdf_node_ui(
    ui: &mut egui::Ui,
    workspace_root: &PathBuf,
    node: &mut BsdfNode,
    depth: usize,
) -> bool {
    let mut changed = false;
    if depth >= 3 {
        ui.colored_label(egui::Color32::LIGHT_RED, "Max BSDF nesting depth reached");
        return false;
    }

    ui.horizontal(|ui| {
        ui.label("type");

        // Common BSDF plugin presets (still editable via the text field).
        egui::ComboBox::from_id_salt(format!("bsdf_type_{depth}"))
            .selected_text(node.plugin_type.as_str())
            .show_ui(ui, |ui| {
                for t in [
                    "diffuse",
                    "plastic",
                    "roughplastic",
                    "conductor",
                    "roughconductor",
                    "dielectric",
                    "principled",
                    "twosided",
                    "mask",
                    "normalmap",
                    "bumpmap",
                    "blendbsdf",
                ] {
                    if ui.selectable_label(node.plugin_type == t, t).clicked() {
                        node.plugin_type = t.to_string();
                        changed = true;
                    }
                }
            });

        ui.add_space(6.0);
        ui.label("custom");
        changed |= ui.text_edit_singleline(&mut node.plugin_type).changed();
    });

    ui.add_space(4.0);
    ui.label("params");

    let mut remove_idx: Option<usize> = None;
    for (i, p) in node.params.iter_mut().enumerate() {
        ui.group(|ui| {
            ui.horizontal(|ui| {
                ui.label("name");
                changed |= ui.text_edit_singleline(&mut p.name).changed();

                ui.add_space(6.0);
                ui.label("kind");
                let mut kind = match &p.value {
                    ParamValue::Float(_) => 0,
                    ParamValue::Integer(_) => 1,
                    ParamValue::Boolean(_) => 2,
                    ParamValue::Rgb(_) => 3,
                    ParamValue::Spectrum(_) => 4,
                    ParamValue::String(_) => 5,
                    ParamValue::Bitmap { .. } => 6,
                };

                egui::ComboBox::from_id_salt(format!("bsdf_param_kind_{depth}_{i}"))
                    .selected_text(match kind {
                        0 => "float",
                        1 => "integer",
                        2 => "boolean",
                        3 => "rgb",
                        4 => "spectrum",
                        5 => "string",
                        _ => "bitmap",
                    })
                    .show_ui(ui, |ui| {
                        changed |= ui.selectable_value(&mut kind, 0, "float").changed();
                        changed |= ui.selectable_value(&mut kind, 1, "integer").changed();
                        changed |= ui.selectable_value(&mut kind, 2, "boolean").changed();
                        changed |= ui.selectable_value(&mut kind, 3, "rgb").changed();
                        changed |= ui.selectable_value(&mut kind, 4, "spectrum").changed();
                        changed |= ui.selectable_value(&mut kind, 5, "string").changed();
                        changed |= ui.selectable_value(&mut kind, 6, "bitmap").changed();
                    });

                // If user changed kind, reset value.
                let reset_to = match kind {
                    0 => ParamValue::Float(0.0),
                    1 => ParamValue::Integer(0),
                    2 => ParamValue::Boolean(false),
                    3 => ParamValue::Rgb([0.8, 0.8, 0.8]),
                    4 => ParamValue::Spectrum("0.8,0.8,0.8".to_string()),
                    5 => ParamValue::String("".to_string()),
                    _ => ParamValue::Bitmap {
                        filename: "".to_string(),
                        srgb: true,
                    },
                };

                // Apply reset if the variant doesn't match.
                let mismatch = match (&p.value, &reset_to) {
                    (ParamValue::Float(_), ParamValue::Float(_)) => false,
                    (ParamValue::Integer(_), ParamValue::Integer(_)) => false,
                    (ParamValue::Boolean(_), ParamValue::Boolean(_)) => false,
                    (ParamValue::Rgb(_), ParamValue::Rgb(_)) => false,
                    (ParamValue::Spectrum(_), ParamValue::Spectrum(_)) => false,
                    (ParamValue::String(_), ParamValue::String(_)) => false,
                    (ParamValue::Bitmap { .. }, ParamValue::Bitmap { .. }) => false,
                    _ => true,
                };
                if mismatch {
                    p.value = reset_to;
                    changed = true;
                }

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.button("Remove").clicked() {
                        remove_idx = Some(i);
                    }
                });
            });

            changed |= param_value_ui(ui, workspace_root, &mut p.value, depth, i);
        });
    }

    if let Some(idx) = remove_idx {
        if idx < node.params.len() {
            node.params.remove(idx);
            changed = true;
        }
    }

    if ui.button("Add parameter").clicked() {
        node.params.push(BsdfParam {
            name: "param".to_string(),
            value: ParamValue::Float(0.0),
        });
        changed = true;
    }

    ui.add_space(6.0);
    let mut has_inner = node.inner.is_some();
    if ui.checkbox(&mut has_inner, "Nested BSDF (inner)").changed() {
        if has_inner {
            node.inner = Some(Box::new(BsdfNode {
                plugin_type: "diffuse".to_string(),
                params: vec![BsdfParam {
                    name: "reflectance".to_string(),
                    value: ParamValue::Rgb([0.8, 0.8, 0.8]),
                }],
                inner: None,
            }));
        } else {
            node.inner = None;
        }
        changed = true;
    }

    if let Some(inner) = node.inner.as_deref_mut() {
        ui.add_space(6.0);
        ui.group(|ui| {
            ui.label("inner");
            changed |= bsdf_node_ui(ui, workspace_root, inner, depth + 1);
        });
    }

    changed
}

fn param_value_ui(
    ui: &mut egui::Ui,
    workspace_root: &PathBuf,
    v: &mut ParamValue,
    depth: usize,
    idx: usize,
) -> bool {
    let mut changed = false;

    match v {
        ParamValue::Float(x) => {
            ui.horizontal(|ui| {
                ui.label("value");
                changed |= ui.add(egui::DragValue::new(x).speed(0.01)).changed();
            });
        }
        ParamValue::Integer(x) => {
            ui.horizontal(|ui| {
                ui.label("value");
                changed |= ui.add(egui::DragValue::new(x).speed(1)).changed();
            });
        }
        ParamValue::Boolean(b) => {
            ui.horizontal(|ui| {
                changed |= ui.checkbox(b, "true").changed();
            });
        }
        ParamValue::Rgb(c) => {
            changed |= color3_ui(ui, "rgb", c, 0.0..=10.0);
        }
        ParamValue::Spectrum(s) => {
            ui.horizontal(|ui| {
                ui.label("value");
                changed |= ui.text_edit_singleline(s).changed();
            });
            ui.label("Example: 0.8,0.8,0.8 or a spectrum string supported by Mitsuba");
        }
        ParamValue::String(s) => {
            ui.horizontal(|ui| {
                ui.label("value");
                changed |= ui.text_edit_singleline(s).changed();
            });
        }
        ParamValue::Bitmap { filename, srgb } => {
            ui.horizontal(|ui| {
                ui.label("file");
                changed |= ui.text_edit_singleline(filename).changed();
                if ui.button("Browse...").clicked() {
                    let picked = FileDialog::new()
                        .add_filter(
                            "Images",
                            &[
                                "png", "jpg", "jpeg", "tga", "bmp", "exr", "hdr", "pfm", "tif",
                                "tiff",
                            ],
                        )
                        .pick_file();

                    if let Some(path) = picked {
                        let s = path
                            .strip_prefix(workspace_root)
                            .ok()
                            .map(|p| p.to_string_lossy().to_string())
                            .unwrap_or_else(|| path.to_string_lossy().to_string());
                        *filename = s;
                        changed = true;
                    }
                }
            });
            ui.horizontal(|ui| {
                ui.label("color space");
                changed |= ui.checkbox(srgb, "sRGB").changed();
                ui.label("(unchecked = linear/raw)");
            });
        }
    }

    let _ = (depth, idx);
    changed
}

fn emit_bsdf_xml(out: &mut String, node: &BsdfNode, indent: usize) {
    let pad = " ".repeat(indent);
    out.push_str(&format!(
        "{pad}<bsdf type=\"{}\">\n",
        escape_xml_attr(&node.plugin_type)
    ));

    for p in &node.params {
        emit_param_xml(out, p, indent + 4);
    }

    if let Some(inner) = &node.inner {
        emit_bsdf_xml(out, inner, indent + 4);
    }

    out.push_str(&format!("{pad}</bsdf>\n"));
}

fn emit_param_xml(out: &mut String, p: &BsdfParam, indent: usize) {
    let pad = " ".repeat(indent);
    let name = escape_xml_attr(p.name.trim());

    match &p.value {
        ParamValue::Float(x) => {
            out.push_str(&format!("{pad}<float name=\"{name}\" value=\"{x}\"/>\n"));
        }
        ParamValue::Integer(x) => {
            out.push_str(&format!("{pad}<integer name=\"{name}\" value=\"{x}\"/>\n"));
        }
        ParamValue::Boolean(b) => {
            out.push_str(&format!(
                "{pad}<boolean name=\"{name}\" value=\"{}\"/>\n",
                if *b { "true" } else { "false" }
            ));
        }
        ParamValue::Rgb(c) => {
            out.push_str(&format!(
                "{pad}<rgb name=\"{name}\" value=\"{}, {}, {}\"/>\n",
                c[0], c[1], c[2]
            ));
        }
        ParamValue::Spectrum(s) => {
            out.push_str(&format!(
                "{pad}<spectrum name=\"{name}\" value=\"{}\"/>\n",
                escape_xml_attr(s)
            ));
        }
        ParamValue::String(s) => {
            out.push_str(&format!(
                "{pad}<string name=\"{name}\" value=\"{}\"/>\n",
                escape_xml_attr(s)
            ));
        }
        ParamValue::Bitmap { filename, srgb } => {
            // Mitsuba bitmap texture: `raw=true` means no sRGB->linear conversion.
            // Here we invert it via `srgb` for a friendlier UI label.
            let raw = if *srgb { "false" } else { "true" };
            out.push_str(&format!("{pad}<texture type=\"bitmap\" name=\"{name}\">\n"));
            out.push_str(&format!(
                "{pad}    <string name=\"filename\" value=\"{}\"/>\n",
                escape_xml_attr(filename)
            ));
            out.push_str(&format!(
                "{pad}    <boolean name=\"raw\" value=\"{raw}\"/>\n"
            ));
            out.push_str(&format!("{pad}</texture>\n"));
        }
    }
}

fn escape_xml_attr(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('"', "&quot;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

/// Generates a "Cornell-box-like + ceiling area light + object" Mitsuba XML scene.
///
/// Notes:
/// - This intentionally uses template-like string generation instead of an XML library.
///   It's easier to learn and debug by directly comparing the emitted XML.
fn generate_cbox_xml(cfg: &CornellBoxConfig) -> String {
    let mut s = String::new();

    s.push_str("<scene version=\"3.0.0\">\n");
    s.push_str("    <integrator type=\"path\"/>\n\n");

    s.push_str("    <sensor type=\"perspective\">\n");
    s.push_str(&format!(
        "        <float name=\"fov\" value=\"{}\"/>\n",
        cfg.camera.fov_deg
    ));
    s.push_str("        <transform name=\"to_world\">\n");
    s.push_str(&format!(
        "            <lookat origin=\"{}, {}, {}\" target=\"{}, {}, {}\" up=\"{}, {}, {}\"/>\n",
        cfg.camera.origin[0],
        cfg.camera.origin[1],
        cfg.camera.origin[2],
        cfg.camera.target[0],
        cfg.camera.target[1],
        cfg.camera.target[2],
        cfg.camera.up[0],
        cfg.camera.up[1],
        cfg.camera.up[2]
    ));
    s.push_str("        </transform>\n\n");

    s.push_str("        <sampler type=\"independent\">\n");
    s.push_str(&format!(
        "            <integer name=\"sample_count\" value=\"{}\"/>\n",
        cfg.sampler.sample_count
    ));
    s.push_str("        </sampler>\n\n");

    s.push_str("        <film type=\"hdrfilm\">\n");
    s.push_str(&format!(
        "            <integer name=\"width\" value=\"{}\"/>\n",
        cfg.film.width
    ));
    s.push_str(&format!(
        "            <integer name=\"height\" value=\"{}\"/>\n",
        cfg.film.height
    ));
    s.push_str("            <rfilter type=\"tent\"/>\n");
    s.push_str("            <string name=\"pixel_format\" value=\"rgb\"/>\n");
    s.push_str("        </film>\n");
    s.push_str("    </sensor>\n\n");

    // --- Scene Environment ---
    match cfg.scene_environment.kind {
        SceneEnvironment::Empty => {
            // No environment geometry
        }
        SceneEnvironment::CornellBox => {
            // --- Cornell box: 5 walls (floor/ceiling/back/left/right) ---
            // Floor
            s.push_str("    <shape type=\"rectangle\">\n");
            s.push_str("        <transform name=\"to_world\">\n");
            s.push_str("            <rotate x=\"1\" y=\"0\" z=\"0\" angle=\"-90\"/>\n");
            s.push_str("            <translate x=\"0\" y=\"0\" z=\"0\"/>\n");
            s.push_str("        </transform>\n");
            s.push_str("        <bsdf type=\"diffuse\">\n");
            s.push_str("            <rgb name=\"reflectance\" value=\"0.8, 0.8, 0.8\"/>\n");
            s.push_str("        </bsdf>\n");
            s.push_str("    </shape>\n\n");

            // Ceiling
            s.push_str("    <shape type=\"rectangle\">\n");
            s.push_str("        <transform name=\"to_world\">\n");
            s.push_str("            <rotate x=\"1\" y=\"0\" z=\"0\" angle=\"90\"/>\n");
            s.push_str("            <translate x=\"0\" y=\"2\" z=\"0\"/>\n");
            s.push_str("        </transform>\n");
            s.push_str("        <bsdf type=\"diffuse\">\n");
            s.push_str("            <rgb name=\"reflectance\" value=\"0.8, 0.8, 0.8\"/>\n");
            s.push_str("        </bsdf>\n");
            s.push_str("    </shape>\n\n");

            // Back wall
            s.push_str("    <shape type=\"rectangle\">\n");
            s.push_str("        <transform name=\"to_world\">\n");
            s.push_str("            <translate x=\"0\" y=\"1\" z=\"-1\"/>\n");
            s.push_str("        </transform>\n");
            s.push_str("        <bsdf type=\"diffuse\">\n");
            s.push_str("            <rgb name=\"reflectance\" value=\"0.8, 0.8, 0.8\"/>\n");
            s.push_str("        </bsdf>\n");
            s.push_str("    </shape>\n\n");

            // Left wall (red)
            s.push_str("    <shape type=\"rectangle\">\n");
            s.push_str("        <transform name=\"to_world\">\n");
            s.push_str("            <rotate x=\"0\" y=\"1\" z=\"0\" angle=\"90\"/>\n");
            s.push_str("            <translate x=\"-1\" y=\"1\" z=\"0\"/>\n");
            s.push_str("        </transform>\n");
            s.push_str("        <bsdf type=\"diffuse\">\n");
            s.push_str("            <rgb name=\"reflectance\" value=\"0.75, 0.15, 0.15\"/>\n");
            s.push_str("        </bsdf>\n");
            s.push_str("    </shape>\n\n");

            // Right wall (green)
            s.push_str("    <shape type=\"rectangle\">\n");
            s.push_str("        <transform name=\"to_world\">\n");
            s.push_str("            <rotate x=\"0\" y=\"1\" z=\"0\" angle=\"-90\"/>\n");
            s.push_str("            <translate x=\"1\" y=\"1\" z=\"0\"/>\n");
            s.push_str("        </transform>\n");
            s.push_str("        <bsdf type=\"diffuse\">\n");
            s.push_str("            <rgb name=\"reflectance\" value=\"0.15, 0.75, 0.15\"/>\n");
            s.push_str("        </bsdf>\n");
            s.push_str("    </shape>\n\n");
        }
        SceneEnvironment::ObjFile => {
            // Load OBJ file as environment
            s.push_str("    <shape type=\"obj\">\n");
            s.push_str(&format!(
                "        <string name=\"filename\" value=\"{}\"/>\n",
                escape_xml_attr(&cfg.scene_environment.obj_filename)
            ));
            s.push_str("        <transform name=\"to_world\">\n");
            s.push_str(&format!(
                "            <scale x=\"{}\" y=\"{}\" z=\"{}\"/>\n",
                cfg.scene_environment.obj_scale,
                cfg.scene_environment.obj_scale,
                cfg.scene_environment.obj_scale
            ));
            s.push_str("        </transform>\n");
            s.push_str("        <bsdf type=\"diffuse\">\n");
            s.push_str("            <rgb name=\"reflectance\" value=\"0.8, 0.8, 0.8\"/>\n");
            s.push_str("        </bsdf>\n");
            s.push_str("    </shape>\n\n");
        }
    }

    // Light
    if cfg.light.enabled {
        s.push_str("    <shape type=\"rectangle\">\n");
        s.push_str("        <transform name=\"to_world\">\n");
        s.push_str(&format!(
            "            <scale x=\"{}\" y=\"{}\" z=\"{}\"/>\n",
            cfg.light.scale_xy, cfg.light.scale_xy, cfg.light.scale_xy
        ));
        s.push_str("            <rotate x=\"1\" y=\"0\" z=\"0\" angle=\"90\"/>\n");
        s.push_str(&format!(
            "            <translate x=\"0\" y=\"{}\" z=\"0\"/>\n",
            cfg.light.y
        ));
        s.push_str("        </transform>\n");
        s.push_str("        <emitter type=\"area\">\n");
        s.push_str(&format!(
            "            <rgb name=\"radiance\" value=\"{}, {}, {}\"/>\n",
            cfg.light.radiance_rgb[0], cfg.light.radiance_rgb[1], cfg.light.radiance_rgb[2]
        ));
        s.push_str("        </emitter>\n");
        s.push_str("    </shape>\n\n");
    }

    // Object
    if cfg.object.enabled {
        match cfg.object.kind {
            ObjectKind::Sphere => {
                s.push_str("    <shape type=\"sphere\">\n");
                s.push_str(&format!(
                    "        <float name=\"radius\" value=\"{}\"/>\n",
                    cfg.object.sphere_radius
                ));
                s.push_str("        <transform name=\"to_world\">\n");
                s.push_str(&format!(
                    "            <translate x=\"{}\" y=\"{}\" z=\"{}\"/>\n",
                    cfg.object.translate[0], cfg.object.translate[1], cfg.object.translate[2]
                ));
                s.push_str("        </transform>\n");
                emit_bsdf_xml(&mut s, &cfg.object.bsdf, 8);
                s.push_str("    </shape>\n");
            }
            ObjectKind::Cube => {
                s.push_str("    <shape type=\"cube\">\n");
                s.push_str("        <transform name=\"to_world\">\n");
                s.push_str(&format!(
                    "            <scale x=\"{}\" y=\"{}\" z=\"{}\"/>\n",
                    cfg.object.cube_scale[0], cfg.object.cube_scale[1], cfg.object.cube_scale[2]
                ));
                s.push_str(&format!(
                    "            <translate x=\"{}\" y=\"{}\" z=\"{}\"/>\n",
                    cfg.object.translate[0], cfg.object.translate[1], cfg.object.translate[2]
                ));
                s.push_str("        </transform>\n");
                emit_bsdf_xml(&mut s, &cfg.object.bsdf, 8);
                s.push_str("    </shape>\n");
            }
            ObjectKind::PlyMesh => {
                s.push_str("    <shape type=\"ply\">\n");
                s.push_str(&format!(
                    "        <string name=\"filename\" value=\"{}\"/>\n",
                    escape_xml_attr(&cfg.object.ply_filename)
                ));
                s.push_str("        <transform name=\"to_world\">\n");
                s.push_str(&format!(
                    "            <scale x=\"{}\" y=\"{}\" z=\"{}\"/>\n",
                    cfg.object.mesh_scale, cfg.object.mesh_scale, cfg.object.mesh_scale
                ));
                s.push_str(&format!(
                    "            <translate x=\"{}\" y=\"{}\" z=\"{}\"/>\n",
                    cfg.object.translate[0], cfg.object.translate[1], cfg.object.translate[2]
                ));
                s.push_str("        </transform>\n");
                emit_bsdf_xml(&mut s, &cfg.object.bsdf, 8);
                s.push_str("    </shape>\n");
            }
            ObjectKind::ObjMesh => {
                s.push_str("    <shape type=\"obj\">\n");
                s.push_str(&format!(
                    "        <string name=\"filename\" value=\"{}\"/>\n",
                    escape_xml_attr(&cfg.object.obj_filename)
                ));
                s.push_str("        <transform name=\"to_world\">\n");
                s.push_str(&format!(
                    "            <scale x=\"{}\" y=\"{}\" z=\"{}\"/>\n",
                    cfg.object.mesh_scale, cfg.object.mesh_scale, cfg.object.mesh_scale
                ));
                s.push_str(&format!(
                    "            <translate x=\"{}\" y=\"{}\" z=\"{}\"/>\n",
                    cfg.object.translate[0], cfg.object.translate[1], cfg.object.translate[2]
                ));
                s.push_str("        </transform>\n");
                emit_bsdf_xml(&mut s, &cfg.object.bsdf, 8);
                s.push_str("    </shape>\n");
            }
        }
    }

    s.push_str("</scene>\n");
    s
}
