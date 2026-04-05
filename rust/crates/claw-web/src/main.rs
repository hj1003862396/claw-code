//! Local web UI + JSON API: system prompt preview and multi-turn chat via the same runtime as agents.
//!
//! Run from the `rust/` directory:
//! `cargo run -p claw-web`
//!
//! Provider auth uses the same environment / config as `tools` background agents (`ProviderClient`).
//! Bind address: `CLAW_WEB_ADDR` (default `127.0.0.1:3000`).

use std::collections::HashMap;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use tokio::sync::Mutex;

use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use runtime::{load_system_prompt, ContentBlock, Session, TurnSummary};
use serde::{Deserialize, Serialize};
use tools::{
    resume_standalone_conversation_runtime, standalone_allowed_tools,
    summarize_turn_assistant_text, DEFAULT_STANDALONE_AGENT_MAX_ITERATIONS,
};
use tower_http::cors::CorsLayer;
use tower_http::services::ServeDir;
use tower_http::trace::{DefaultMakeSpan, DefaultOnResponse, TraceLayer};
use tracing::Level;
use tracing_subscriber::EnvFilter;

#[derive(Clone)]
struct ChatSessionState {
    session: Session,
    system_prompt: Vec<String>,
    model: String,
    allowed_tools: std::collections::BTreeSet<String>,
}

#[derive(Clone)]
struct AppState {
    sessions: Arc<Mutex<HashMap<String, ChatSessionState>>>,
}

#[derive(Debug, Deserialize)]
struct PromptQuery {
    cwd: Option<String>,
    date: Option<String>,
}

#[derive(Debug, Serialize)]
struct HealthBody {
    status: &'static str,
    service: &'static str,
}

#[derive(Debug, Serialize)]
struct SystemPromptBody {
    section_count: usize,
    preview_chars: usize,
    preview: String,
}

#[derive(Debug, Serialize)]
struct ErrorBody {
    error: String,
}

#[derive(Debug, Deserialize)]
struct EchoRequest {
    message: String,
}

#[derive(Debug, Serialize)]
struct EchoResponse {
    echo: String,
}

#[derive(Debug, Deserialize)]
struct CreateSessionBody {
    cwd: Option<String>,
    date: Option<String>,
    /// Model id passed to the provider (optional; default matches tools agent default).
    model: Option<String>,
    /// Sub-agent profile for tool allowlist: `General` (default, shell + 写文件), `Explore` (只读), `Plan`, …
    subagent_type: Option<String>,
}

#[derive(Debug, Serialize)]
struct CreateSessionResponse {
    session_id: String,
    model: String,
    subagent_type: String,
}

#[derive(Debug, Deserialize)]
struct ChatMessageBody {
    session_id: String,
    message: String,
}

#[derive(Debug, Serialize)]
struct ToolRunDto {
    tool_name: String,
    is_error: bool,
    output_preview: String,
}

#[derive(Debug, Serialize)]
struct UsageDto {
    input_tokens: u32,
    output_tokens: u32,
    cache_creation_input_tokens: u32,
    cache_read_input_tokens: u32,
}

#[derive(Debug, Serialize)]
struct ChatMessageResponse {
    reply: String,
    iterations: usize,
    tool_runs: Vec<ToolRunDto>,
    usage: UsageDto,
}

#[derive(Debug, Deserialize)]
struct SessionIdBody {
    session_id: String,
}

fn err(status: StatusCode, msg: impl Into<String>) -> (StatusCode, Json<ErrorBody>) {
    (status, Json(ErrorBody { error: msg.into() }))
}

fn resolve_cwd(explicit: Option<String>) -> Result<PathBuf, String> {
    explicit
        .map(PathBuf::from)
        .map_or_else(|| std::env::current_dir().map_err(|e| e.to_string()), Ok)
}

fn build_web_system_prompt(cwd: PathBuf, date: String) -> Result<Vec<String>, String> {
    let mut sections = load_system_prompt(cwd, date, std::env::consts::OS, "unknown")
        .map_err(|e| e.to_string())?;
    sections.push(
        "You are assisting the user through a local Claw web UI. Answer clearly; use tools when they help."
            .to_string(),
    );
    Ok(sections)
}

fn tool_runs_from_summary(summary: &TurnSummary) -> Vec<ToolRunDto> {
    let mut out = Vec::new();
    for message in &summary.tool_results {
        for block in &message.blocks {
            if let ContentBlock::ToolResult {
                tool_name,
                output,
                is_error,
                ..
            } = block
            {
                out.push(ToolRunDto {
                    tool_name: tool_name.clone(),
                    is_error: *is_error,
                    output_preview: output.chars().take(600).collect(),
                });
            }
        }
    }
    out
}

async fn health() -> Json<HealthBody> {
    Json(HealthBody {
        status: "ok",
        service: "claw-web",
    })
}

async fn system_prompt(Query(q): Query<PromptQuery>) -> Response {
    let cwd = match resolve_cwd(q.cwd) {
        Ok(p) => p,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorBody { error: e }),
            )
                .into_response();
        }
    };
    let date = q.date.unwrap_or_else(|| "unknown".to_string());
    match load_system_prompt(cwd, date, std::env::consts::OS, "unknown") {
        Ok(sections) => {
            let rendered = sections.join("\n\n");
            let preview = rendered.chars().take(8_000).collect::<String>();
            let preview_chars = preview.chars().count();
            Json(SystemPromptBody {
                section_count: sections.len(),
                preview_chars,
                preview,
            })
            .into_response()
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorBody {
                error: e.to_string(),
            }),
        )
            .into_response(),
    }
}

async fn echo(Json(body): Json<EchoRequest>) -> Json<EchoResponse> {
    Json(EchoResponse { echo: body.message })
}

async fn chat_create_session(
    State(state): State<AppState>,
    Json(body): Json<CreateSessionBody>,
) -> Result<Json<CreateSessionResponse>, (StatusCode, Json<ErrorBody>)> {
    let t0 = Instant::now();
    let cwd = resolve_cwd(body.cwd).map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, e))?;
    let date = body.date.clone().unwrap_or_else(|| "unknown".to_string());
    let subagent_type = body
        .subagent_type
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .unwrap_or("General")
        .to_string();
    let model_opt = body
        .model
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty());

    let system_prompt = build_web_system_prompt(cwd, date)
        .map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, e))?;
    let allowed = standalone_allowed_tools(Some(subagent_type.as_str()));

    let session_id = uuid::Uuid::new_v4().to_string();
    let model_for_response = model_opt
        .map(|m| m.to_string())
        .or_else(|| std::env::var("CLAW_WEB_MODEL").ok())
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| "claude-opus-4-6".to_string());

    let chat = ChatSessionState {
        session: Session::new(),
        system_prompt,
        model: model_for_response.clone(),
        allowed_tools: allowed,
    };
    let tool_count = chat.allowed_tools.len();

    state.sessions.lock().await.insert(session_id.clone(), chat);

    tracing::info!(
        target: "claw_web",
        session_id = %session_id,
        subagent_type = %subagent_type,
        tool_count,
        elapsed_ms = t0.elapsed().as_millis() as u64,
        "chat session created"
    );

    Ok(Json(CreateSessionResponse {
        session_id,
        model: model_for_response,
        subagent_type,
    }))
}

async fn chat_send(
    State(state): State<AppState>,
    Json(body): Json<ChatMessageBody>,
) -> Result<Json<ChatMessageResponse>, (StatusCode, Json<ErrorBody>)> {
    if body.message.trim().is_empty() {
        return Err(err(StatusCode::BAD_REQUEST, "message must not be empty"));
    }
    if body.session_id.trim().is_empty() {
        return Err(err(StatusCode::BAD_REQUEST, "session_id must not be empty"));
    }

    let t0 = Instant::now();
    tracing::info!(
        target: "claw_web",
        session_id = %body.session_id,
        message_chars = body.message.chars().count(),
        "chat message: run_turn start"
    );

    let mut map = state.sessions.lock().await;
    let entry = map.get_mut(&body.session_id).ok_or_else(|| {
        err(
            StatusCode::NOT_FOUND,
            "unknown session_id; create a session first",
        )
    })?;

    let mut runtime = resume_standalone_conversation_runtime(
        entry.session.clone(),
        entry.system_prompt.clone(),
        Some(entry.model.as_str()),
        entry.allowed_tools.clone(),
    )
    .map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, e))?;
    runtime = runtime.with_max_iterations(DEFAULT_STANDALONE_AGENT_MAX_ITERATIONS);

    let summary = runtime
        .run_turn(body.message.clone(), None)
        .map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    entry.session = runtime.into_session();

    tracing::info!(
        target: "claw_web",
        session_id = %body.session_id,
        elapsed_ms = t0.elapsed().as_millis() as u64,
        iterations = summary.iterations,
        tool_results = summary.tool_results.len(),
        "chat message: run_turn done"
    );

    let u = summary.usage;
    Ok(Json(ChatMessageResponse {
        reply: summarize_turn_assistant_text(&summary),
        iterations: summary.iterations,
        tool_runs: tool_runs_from_summary(&summary),
        usage: UsageDto {
            input_tokens: u.input_tokens,
            output_tokens: u.output_tokens,
            cache_creation_input_tokens: u.cache_creation_input_tokens,
            cache_read_input_tokens: u.cache_read_input_tokens,
        },
    }))
}

async fn chat_close_session(
    State(state): State<AppState>,
    Json(body): Json<SessionIdBody>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ErrorBody>)> {
    let removed = state
        .sessions
        .lock()
        .await
        .remove(&body.session_id)
        .is_some();
    if !removed {
        return Err(err(StatusCode::NOT_FOUND, "unknown session_id"));
    }
    Ok(Json(serde_json::json!({ "ok": true })))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let static_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("static");

    let state = AppState {
        sessions: Arc::new(Mutex::new(HashMap::new())),
    };

    let app = Router::new()
        .route("/api/health", get(health))
        .route("/api/system-prompt", get(system_prompt))
        .route("/api/echo", post(echo))
        .route("/api/chat/session", post(chat_create_session))
        .route("/api/chat/message", post(chat_send))
        .route("/api/chat/session/close", post(chat_close_session))
        .with_state(state)
        .fallback_service(ServeDir::new(static_root))
        .layer(CorsLayer::permissive())
        .layer(
            TraceLayer::new_for_http()
                .make_span_with(DefaultMakeSpan::new().level(Level::INFO))
                .on_response(DefaultOnResponse::new().level(Level::INFO)),
        );

    let addr: SocketAddr = std::env::var("CLAW_WEB_ADDR")
        .unwrap_or_else(|_| "127.0.0.1:3000".to_string())
        .parse()?;

    tracing::info!("claw-web listening on http://{addr}/");
    tracing::info!(
        "logs go to stderr (this terminal). default filter is RUST_LOG=info; try RUST_LOG=debug,tower_http=info,tokio=warn,hyper=warn,reqwest=warn for more"
    );
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}
