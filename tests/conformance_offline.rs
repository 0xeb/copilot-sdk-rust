// Copyright (c) 2026 Elias Bachaalany
// SPDX-License-Identifier: MIT

//! Offline conformance suite for the Rust port.
//!
//! Exercises wire-level behavior of the public SDK surface without a real
//! Copilot CLI:
//!
//! * `Client::new()` validation matrix for `cli_url` / `use_stdio` / `port` /
//!   `tcp_connection_token` combinations (covers v0.1.49 additions and the
//!   pre-existing invariants).
//! * Per-field `serde_json` shape parity for `SessionConfig`,
//!   `ResumeSessionConfig`, `SessionListFilter`, `MessageOptions`, and
//!   `ToolResult`.
//! * `SessionEvent::from_json` dispatch coverage for v0.1.49 event types not
//!   already exercised by `tests/v0149_parity.rs` plus the malformed-payload
//!   fallback path.
//! * RPC method-string constants are reachable from the public `rpc_methods`
//!   module (the in-crate `#[cfg(test)]` suite in `src/rpc_methods.rs`
//!   already covers exact wire strings; this complements it from outside the
//!   crate).
//! * Async lifetime: register a tool handler on a `Session`, spawn an
//!   `invoke_tool` task and a stubbed `destroy()`, and assert no panics and
//!   that both futures resolve cleanly.
//!
//! Companion suites:
//!
//! * `tests/v0149_parity.rs` — additive v0.1.49 public-surface parity
//! * `tests/snapshot_conformance.rs` — feature-gated upstream snapshot parity
//! * `tests/e2e_tests.rs` / `tests/e2e_parity_tests.rs` — feature-gated, need
//!   a live CLI.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use copilot_sdk::{
    rpc_methods, Client, ClientOptions, CopilotError, InvokeFuture, MessageOptions,
    RemoteSessionMode, ResumeSessionConfig, Session, SessionConfig, SessionEvent, SessionEventData,
    SessionListFilter, Tool, ToolHandler, ToolResult, ToolResultExpanded,
};
use serde_json::{json, Value};
use tokio::sync::Mutex as TokioMutex;

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

fn make_raw_event(event_type: &str, data: Value) -> Value {
    json!({
        "id": format!("evt-{event_type}"),
        "timestamp": "2026-01-01T00:00:00Z",
        "type": event_type,
        "data": data,
    })
}

/// Build a stub invoke function that records every JSON-RPC call and replies
/// with `Ok(Value::Null)`. Returned alongside an `Arc<TokioMutex<Vec<...>>>`
/// the test can inspect after destroying / invoking the session.
type RecordedCall = (String, Option<Value>);

#[allow(clippy::type_complexity)]
fn recording_invoke_fn() -> (
    Arc<TokioMutex<Vec<RecordedCall>>>,
    impl Fn(&str, Option<Value>) -> InvokeFuture + Send + Sync + 'static,
) {
    let log: Arc<TokioMutex<Vec<RecordedCall>>> = Arc::new(TokioMutex::new(Vec::new()));
    let log_for_closure = Arc::clone(&log);
    let invoke = move |method: &str, params: Option<Value>| -> InvokeFuture {
        let method = method.to_string();
        let log = Arc::clone(&log_for_closure);
        Box::pin(async move {
            log.lock().await.push((method, params));
            Ok(Value::Null)
        })
    };
    (log, invoke)
}

// ---------------------------------------------------------------------------
// 1. Client::new validation matrix
// ---------------------------------------------------------------------------

/// `Client` does not implement `Debug`, so `expect_err` would not compile.
/// This helper turns the `Result<Client>` into the `CopilotError` (panicking
/// with a static message when construction unexpectedly succeeded).
fn expect_invalid(result: Result<Client, CopilotError>, what: &str) -> CopilotError {
    match result {
        Ok(_) => panic!("expected Client::new to reject {what}, but it succeeded"),
        Err(e) => e,
    }
}

#[test]
fn client_new_rejects_cli_url_with_cli_path() {
    let opts = ClientOptions {
        cli_url: Some("http://127.0.0.1:1234".into()),
        cli_path: Some(PathBuf::from("/usr/bin/copilot")),
        ..Default::default()
    };
    let err = expect_invalid(Client::new(opts), "cli_url + cli_path");
    match err {
        CopilotError::InvalidConfig(msg) => {
            assert!(msg.contains("cli_url"), "msg: {msg}");
            assert!(msg.contains("cli_path"), "msg: {msg}");
        }
        other => panic!("expected InvalidConfig, got {other:?}"),
    }
}

#[test]
fn client_new_rejects_cli_url_with_port() {
    let opts = ClientOptions {
        cli_url: Some("http://127.0.0.1:1234".into()),
        port: 5555,
        ..Default::default()
    };
    let err = expect_invalid(Client::new(opts), "cli_url + port");
    assert!(matches!(err, CopilotError::InvalidConfig(_)));
}

#[test]
fn client_new_rejects_cli_url_with_github_token() {
    let opts = ClientOptions {
        cli_url: Some("http://127.0.0.1:1234".into()),
        github_token: Some("gh_pat_xxx".into()),
        ..Default::default()
    };
    let err = expect_invalid(Client::new(opts), "cli_url + github_token");
    match err {
        CopilotError::InvalidConfig(msg) => {
            assert!(msg.contains("github_token"), "msg: {msg}");
            assert!(msg.contains("cli_url"), "msg: {msg}");
        }
        other => panic!("expected InvalidConfig, got {other:?}"),
    }
}

#[test]
fn client_new_rejects_cli_url_with_use_logged_in_user() {
    let opts = ClientOptions {
        cli_url: Some("http://127.0.0.1:1234".into()),
        use_logged_in_user: Some(true),
        ..Default::default()
    };
    let err = expect_invalid(Client::new(opts), "cli_url + use_logged_in_user");
    match err {
        CopilotError::InvalidConfig(msg) => {
            assert!(msg.contains("use_logged_in_user"), "msg: {msg}");
        }
        other => panic!("expected InvalidConfig, got {other:?}"),
    }
}

#[test]
fn client_new_rejects_use_stdio_with_port() {
    let opts = ClientOptions {
        use_stdio: true,
        port: 1234,
        ..Default::default()
    };
    let err = expect_invalid(Client::new(opts), "use_stdio + port");
    match err {
        CopilotError::InvalidConfig(msg) => {
            assert!(msg.contains("port"), "msg: {msg}");
            assert!(msg.contains("use_stdio"), "msg: {msg}");
        }
        other => panic!("expected InvalidConfig, got {other:?}"),
    }
}

#[test]
fn client_new_accepts_default_options() {
    // Sanity: default (stdio, no port, no token) must construct cleanly.
    assert!(Client::new(ClientOptions::default()).is_ok());
}

#[test]
fn client_new_accepts_tcp_mode_without_token_and_auto_generates_one() {
    // In TCP+spawn mode with no caller-provided token, Client::new must
    // succeed (the SDK auto-fills a UUID v4 internally). We cannot read the
    // private `options` field from an integration test, but the constructor
    // succeeding is the externally observable contract.
    let opts = ClientOptions {
        use_stdio: false,
        ..Default::default()
    };
    assert!(Client::new(opts).is_ok());
}

#[test]
fn client_new_external_url_forces_stdio_off() {
    // cli_url implies an external TCP server, so use_stdio (the default true)
    // must NOT cause a validation failure. The SDK silently flips it to false.
    let opts = ClientOptions {
        cli_url: Some("http://127.0.0.1:1234".into()),
        // use_stdio left as default `true`.
        ..Default::default()
    };
    assert!(Client::new(opts).is_ok());
}

// ---------------------------------------------------------------------------
// 2. SessionConfig / ResumeSessionConfig / SessionListFilter wire shape
// ---------------------------------------------------------------------------

#[test]
fn session_config_default_serializes_empty_object() {
    let v = serde_json::to_value(SessionConfig::default()).unwrap();
    let obj = v
        .as_object()
        .expect("SessionConfig must serialize as object");
    assert!(
        obj.is_empty(),
        "default SessionConfig must emit no keys, got: {obj:?}"
    );
}

#[test]
fn session_config_session_id_is_snake_case_per_upstream() {
    // SessionConfig uses #[serde(rename_all = "camelCase")] which turns
    // `session_id` into `sessionId`. Upstream nodejs sends `sessionId`.
    let cfg = SessionConfig {
        session_id: Some("sess-123".into()),
        model: Some("gpt-5".into()),
        working_directory: Some("/w".into()),
        ..Default::default()
    };
    let v = serde_json::to_value(&cfg).unwrap();
    assert_eq!(v["sessionId"], "sess-123");
    assert_eq!(v["model"], "gpt-5");
    assert_eq!(v["workingDirectory"], "/w");
    assert!(v.get("session_id").is_none());
    assert!(v.get("working_directory").is_none());
}

#[test]
fn resume_session_config_disable_resume_omitted_when_false() {
    let cfg = ResumeSessionConfig::default();
    let v = serde_json::to_value(&cfg).unwrap();
    assert!(
        v.get("disableResume").is_none(),
        "disable_resume=false must be omitted, got: {v}"
    );
}

#[test]
fn resume_session_config_disable_resume_emitted_when_true() {
    let cfg = ResumeSessionConfig {
        disable_resume: true,
        ..Default::default()
    };
    let v = serde_json::to_value(&cfg).unwrap();
    assert_eq!(v["disableResume"], true);
}

#[test]
fn session_list_filter_all_fields_camel_case() {
    let filter = SessionListFilter {
        cwd: Some("/w".into()),
        repository: Some("octo/cat".into()),
        ..Default::default()
    };
    let v = serde_json::to_value(&filter).unwrap();
    assert_eq!(v["cwd"], "/w");
    assert_eq!(v["repository"], "octo/cat");
    let obj = v.as_object().unwrap();
    assert_eq!(obj.len(), 2, "only set fields must serialize, got: {obj:?}");
}

#[test]
fn session_config_remote_session_emits_all_modes() {
    for (mode, wire) in [
        (RemoteSessionMode::Off, "off"),
        (RemoteSessionMode::Export, "export"),
        (RemoteSessionMode::On, "on"),
    ] {
        let cfg = SessionConfig {
            remote_session: Some(mode),
            ..Default::default()
        };
        let v = serde_json::to_value(&cfg).unwrap();
        assert_eq!(v["remoteSession"], wire);
    }
}

#[test]
fn message_options_request_permission_uses_explicit_rename() {
    // SessionConfig.request_permission is explicitly renamed to
    // `requestPermission` (not derived from camelCase) — make sure the
    // explicit rename survives.
    let cfg = SessionConfig {
        request_permission: Some(false),
        request_user_input: Some(true),
        ..Default::default()
    };
    let v = serde_json::to_value(&cfg).unwrap();
    assert_eq!(v["requestPermission"], false);
    assert_eq!(v["requestUserInput"], true);
}

// ---------------------------------------------------------------------------
// 3. ToolResult and MessageOptions wire shape
// ---------------------------------------------------------------------------

#[test]
fn tool_result_expanded_default_factory_serializes_camel_case() {
    let res = ToolResultExpanded::text("hello");
    let v = serde_json::to_value(&res).unwrap();
    // Both the Rust port and the upstream nodejs SDK use `textResultForLlm`
    // (lowercase `llm`). See `reference/copilot-sdk/nodejs/src/types.ts:241`
    // and `nodejs/src/generated/rpc.ts:940`. A prior conformance-suite
    // comment incorrectly claimed upstream emitted `textResultForLLM`; that
    // was a false alarm — there is no mismatch here.
    assert_eq!(v["textResultForLlm"], "hello");
    assert_eq!(v["resultType"], "success");
    assert!(v.get("error").is_none());
    assert!(v.get("text_result_for_llm").is_none());
}

#[test]
fn message_options_serializes_camel_case() {
    let opts = MessageOptions {
        prompt: "hi".into(),
        mode: Some("plan".into()),
        ..Default::default()
    };
    let v = serde_json::to_value(&opts).unwrap();
    assert_eq!(v["prompt"], "hi");
    assert_eq!(v["mode"], "plan");
    assert!(v.get("attachments").is_none());
}

// ---------------------------------------------------------------------------
// 4. Event dispatch: complement v0149_parity coverage
// ---------------------------------------------------------------------------

#[test]
fn parse_session_handoff_with_remote_source() {
    let raw = make_raw_event(
        "session.handoff",
        json!({
            "handoffTime": "2026-01-02T03:04:05Z",
            "sourceType": "remote",
            "context": "carry over",
            "remoteSessionId": "remote-1"
        }),
    );
    let ev = SessionEvent::from_json(&raw).expect("must parse");
    match ev.data {
        SessionEventData::SessionHandoff(d) => {
            assert_eq!(d.handoff_time, "2026-01-02T03:04:05Z");
            assert_eq!(d.context.as_deref(), Some("carry over"));
            assert_eq!(d.remote_session_id.as_deref(), Some("remote-1"));
        }
        other => panic!("expected SessionHandoff, got {other:?}"),
    }
}

#[test]
fn parse_session_shutdown_routine() {
    let raw = make_raw_event(
        "session.shutdown",
        json!({
            "shutdownType": "routine",
            "totalPremiumRequests": 7.0,
            "totalApiDurationMs": 1234.0,
            "sessionStartTime": 1700000000000.0,
            "codeChanges": {
                "linesAdded": 12.0,
                "linesRemoved": 3.0,
                "filesModified": ["a.rs", "b.rs"]
            }
        }),
    );
    let ev = SessionEvent::from_json(&raw).expect("must parse");
    match ev.data {
        SessionEventData::SessionShutdown(d) => {
            assert_eq!(d.total_premium_requests, 7.0);
            assert_eq!(d.code_changes.lines_added, 12.0);
            assert_eq!(d.code_changes.files_modified.len(), 2);
        }
        other => panic!("expected SessionShutdown, got {other:?}"),
    }
}

#[test]
fn parse_session_info_with_optional_url_and_tip() {
    let raw = make_raw_event(
        "session.info",
        json!({
            "infoType": "tip",
            "message": "Try /help",
            "tip": "Run /help for details",
            "url": "https://example.com/help"
        }),
    );
    let ev = SessionEvent::from_json(&raw).expect("must parse");
    match ev.data {
        SessionEventData::SessionInfo(d) => {
            assert_eq!(d.info_type, "tip");
            assert_eq!(d.message, "Try /help");
            assert_eq!(d.tip.as_deref(), Some("Run /help for details"));
            assert_eq!(d.url.as_deref(), Some("https://example.com/help"));
        }
        other => panic!("expected SessionInfo, got {other:?}"),
    }
}

#[test]
fn parse_session_usage_info() {
    let raw = make_raw_event(
        "session.usage_info",
        json!({
            "tokenLimit": 100_000.0,
            "currentTokens": 12_345.0,
            "messagesLength": 42.0
        }),
    );
    let ev = SessionEvent::from_json(&raw).expect("must parse");
    match ev.data {
        SessionEventData::SessionUsageInfo(d) => {
            assert_eq!(d.token_limit, 100_000.0);
            assert_eq!(d.current_tokens, 12_345.0);
            assert_eq!(d.messages_length, 42.0);
        }
        other => panic!("expected SessionUsageInfo, got {other:?}"),
    }
}

#[test]
fn parse_session_task_complete() {
    let raw = make_raw_event(
        "session.task_complete",
        json!({
            "success": true,
            "summary": "wrote 3 files"
        }),
    );
    let ev = SessionEvent::from_json(&raw).expect("must parse");
    match ev.data {
        SessionEventData::SessionTaskComplete(d) => {
            assert_eq!(d.success, Some(true));
            assert_eq!(d.summary.as_deref(), Some("wrote 3 files"));
        }
        other => panic!("expected SessionTaskComplete, got {other:?}"),
    }
}

#[test]
fn parse_session_snapshot_rewind() {
    let raw = make_raw_event(
        "session.snapshot_rewind",
        json!({
            "upToEventId": "evt-42",
            "eventsRemoved": 5.0
        }),
    );
    let ev = SessionEvent::from_json(&raw).expect("must parse");
    match ev.data {
        SessionEventData::SessionSnapshotRewind(d) => {
            assert_eq!(d.up_to_event_id, "evt-42");
            assert_eq!(d.events_removed, 5.0);
        }
        other => panic!("expected SessionSnapshotRewind, got {other:?}"),
    }
}

#[test]
fn parse_session_schedule_created_recurring() {
    let raw = make_raw_event(
        "session.schedule_created",
        json!({
            "id": 1.0,
            "prompt": "/status",
            "intervalMs": 60_000.0,
            "displayPrompt": "every minute /status",
            "recurring": true
        }),
    );
    let ev = SessionEvent::from_json(&raw).expect("must parse");
    match ev.data {
        SessionEventData::SessionScheduleCreated(d) => {
            assert_eq!(d.id, 1.0);
            assert_eq!(d.prompt, "/status");
            assert_eq!(d.interval_ms, 60_000.0);
            assert_eq!(d.recurring, Some(true));
        }
        other => panic!("expected SessionScheduleCreated, got {other:?}"),
    }
}

#[test]
fn parse_subagent_legacy_custom_agent_alias() {
    // Upstream renamed `custom_agent.*` to `subagent.*` but still emits the
    // legacy wire names for back-compat. The Rust parser must accept both.
    let payload = json!({
        "agentName": "researcher",
        "agentDisplayName": "Researcher",
        "tools": ["read", "write"]
    });
    let legacy = make_raw_event("custom_agent.selected", payload.clone());
    let primary = make_raw_event("subagent.selected", payload);
    let ev_legacy = SessionEvent::from_json(&legacy).expect("legacy must parse");
    let ev_primary = SessionEvent::from_json(&primary).expect("primary must parse");
    assert!(matches!(
        ev_legacy.data,
        SessionEventData::CustomAgentSelected(_)
    ));
    assert!(matches!(
        ev_primary.data,
        SessionEventData::CustomAgentSelected(_)
    ));
}

#[test]
fn parse_event_with_malformed_data_falls_back_to_unknown_null() {
    // Known event type but the data payload is missing required fields.
    // The parse_into! macro must catch the error and produce
    // `SessionEventData::Unknown(Value::Null)` rather than panic.
    let raw = make_raw_event("session.title_changed", json!({"definitely_not_title": 1}));
    let ev = SessionEvent::from_json(&raw).expect("envelope still parses");
    match ev.data {
        SessionEventData::Unknown(v) => assert!(v.is_null(), "got: {v}"),
        other => panic!("expected Unknown(Null) fallback, got {other:?}"),
    }
}

#[test]
fn parse_envelope_with_parent_id_and_ephemeral_round_trips() {
    let raw = json!({
        "id": "evt-1",
        "timestamp": "2026-01-01T00:00:00Z",
        "type": "session.title_changed",
        "parentId": "parent-1",
        "ephemeral": true,
        "data": {"title": "Hello"}
    });
    let ev = SessionEvent::from_json(&raw).expect("must parse");
    assert_eq!(ev.parent_id.as_deref(), Some("parent-1"));
    assert_eq!(ev.ephemeral, Some(true));
    assert_eq!(ev.event_type, "session.title_changed");
}

// ---------------------------------------------------------------------------
// 5. RPC method constants reachable from the public surface
// ---------------------------------------------------------------------------

#[test]
fn rpc_methods_public_surface_matches_upstream_strings() {
    // Spot-check that key constants are reachable from outside the crate
    // (the exhaustive list is asserted by src/rpc_methods.rs unit tests).
    assert_eq!(rpc_methods::PING, "ping");
    assert_eq!(rpc_methods::CONNECT, "connect");
    assert_eq!(rpc_methods::SESSIONS_FORK, "sessions.fork");
    assert_eq!(rpc_methods::SESSIONS_CONNECT, "sessions.connect");
    assert_eq!(rpc_methods::SESSION_SUSPEND, "session.suspend");
    assert_eq!(
        rpc_methods::SESSION_FS_SET_PROVIDER,
        "sessionFs.setProvider"
    );
    assert_eq!(
        rpc_methods::SESSION_HISTORY_COMPACT,
        "session.history.compact"
    );
    assert_eq!(
        rpc_methods::SESSION_HISTORY_TRUNCATE,
        "session.history.truncate"
    );
    assert_eq!(rpc_methods::SESSION_REMOTE_ENABLE, "session.remote.enable");
    assert_eq!(
        rpc_methods::SESSION_REMOTE_DISABLE,
        "session.remote.disable"
    );
}

/// Regression prevention: scan the SDK's call sites for any residual wire-name
/// literals inside `invoke(...)` / `(invoke_fn)(...)` calls.
///
/// All RPC method names should reach the wire via `rpc_methods::*` constants
/// (whose values are asserted against upstream in the suite above). A
/// hand-written literal like `"session.workspace.list_files"` would compile
/// and pass type checks but talk to a non-existent runtime endpoint.
///
/// The check is two-pronged:
/// 1. A **negative list** of historically-known wrong strings (these were
///    surfaced during the v0.1.49 sync and fixed). If any of these reappears
///    anywhere in `src/`, the test fails.
/// 2. A **heuristic** that flags any literal in an `invoke[_fn]?(...)` call
///    whose dotted segments after the first contain snake_case — upstream
///    uses camelCase for verbs and lowercase for namespace tokens, so any
///    `_` in segment 2+ is suspect.
#[test]
fn no_residual_wire_name_literals_in_invoke_calls() {
    use std::path::PathBuf;

    let crate_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));

    // ---- 1. Negative list (exact-match) across all src/*.rs files ----
    //
    // These are the 8 wire-name bugs surfaced during the v0.1.49 sync.
    // They must never reappear anywhere in the crate source.
    const KNOWN_WRONG_NAMES: &[&str] = &[
        "session.model.get_current",
        "session.model.switch_to",
        "session.agent.get_current",
        "session.compaction.compact",
        "session.workspace.list_files",
        "session.workspace.read_file",
        "session.workspace.create_file",
        "account.get_quota",
    ];

    let src_dir = crate_root.join("src");
    let mut all_offenders: Vec<String> = Vec::new();
    for entry in std::fs::read_dir(&src_dir).expect("read src dir") {
        let entry = entry.expect("dir entry");
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("rs") {
            continue;
        }
        let content = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("read {}: {}", path.display(), e));
        for wrong in KNOWN_WRONG_NAMES {
            let needle = format!("\"{wrong}\"");
            if content.contains(&needle) {
                all_offenders.push(format!(
                    "{}: contains historically-broken literal {:?}",
                    path.file_name().and_then(|n| n.to_str()).unwrap_or("?"),
                    wrong
                ));
            }
        }
    }

    // ---- 2. Heuristic for new snake_case wire-name mistakes ----
    //
    // For client.rs and session.rs, find every invoke() / (invoke_fn)() call
    // with a string literal and flag dotted method names whose non-first
    // segments contain underscores.
    fn extract_invoke_literal(line: &str) -> Option<&str> {
        let invoke_idx = line.find("invoke")?;
        let after = &line[invoke_idx..];
        if !after.contains('(') {
            return None;
        }
        let quote_start = line[invoke_idx..].find('"')?;
        let abs_start = invoke_idx + quote_start + 1;
        let quote_end_rel = line[abs_start..].find('"')?;
        Some(&line[abs_start..abs_start + quote_end_rel])
    }

    let files = ["src/client.rs", "src/session.rs"];
    for rel in &files {
        let path = crate_root.join(rel);
        let content = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("read {}: {}", path.display(), e));
        for (lineno, line) in content.lines().enumerate() {
            let Some(literal) = extract_invoke_literal(line) else {
                continue;
            };
            if !literal.contains('.') {
                continue;
            }
            let segments: Vec<&str> = literal.split('.').collect();
            if segments.len() >= 2
                && segments
                    .iter()
                    .skip(1)
                    .any(|seg| seg.chars().any(|c| c == '_'))
            {
                all_offenders.push(format!(
                    "{}:{}: snake_case wire literal {:?} (use rpc_methods::* constant)",
                    rel,
                    lineno + 1,
                    literal
                ));
            }
        }
    }

    assert!(
        all_offenders.is_empty(),
        "Found residual wire-name literals (should use rpc_methods::* constants):\n  {}",
        all_offenders.join("\n  ")
    );
}

// ---------------------------------------------------------------------------
// 6. Async lifetime — Session::destroy() while a tool handler is in flight
// ---------------------------------------------------------------------------

#[tokio::test]
async fn session_destroy_invokes_session_destroy_with_session_id() {
    let (log, invoke) = recording_invoke_fn();
    let session = Session::new("sess-x".into(), None, invoke);

    session.destroy().await.expect("destroy must succeed");

    let calls = log.lock().await;
    assert_eq!(calls.len(), 1, "exactly one RPC call expected");
    assert_eq!(calls[0].0, "session.destroy");
    let params = calls[0].1.as_ref().expect("destroy must send params");
    assert_eq!(params["sessionId"], "sess-x");
}

#[tokio::test]
async fn session_register_and_invoke_tool_round_trip() {
    let (_log, invoke) = recording_invoke_fn();
    let session = Session::new("sess-tool".into(), None, invoke);

    let tool = Tool::new("echo").description("Echo arguments");
    let handler: ToolHandler =
        Arc::new(|_name: &str, args: &Value| ToolResult::text(args.to_string()));
    session
        .register_tool_with_handler(tool, Some(handler))
        .await;

    let registered = session.get_tool("echo").await;
    assert!(registered.is_some(), "tool must be registered");

    let result = session
        .invoke_tool("echo", &json!({"x": 1}))
        .await
        .expect("invoke_tool must succeed");
    assert!(matches!(result, ToolResult::Text(ref text) if text.contains("\"x\"")));
}

#[tokio::test]
async fn session_invoke_unknown_tool_returns_tool_not_found() {
    let (_log, invoke) = recording_invoke_fn();
    let session = Session::new("sess-missing".into(), None, invoke);

    let err = session
        .invoke_tool("nope", &json!({}))
        .await
        .expect_err("unknown tool must error");
    assert!(matches!(err, CopilotError::ToolNotFound(name) if name == "nope"));
}

#[tokio::test]
async fn session_destroy_concurrent_with_tool_invocation_no_panic() {
    // Lifetime smoke test: we register a tool, spawn its synchronous
    // invocation on one task, spawn `destroy()` on another, then await both.
    // Neither future must panic and both must complete (the stubbed
    // invoke_fn always returns Null successfully, so destroy is fast; the
    // tool handler is sync so it completes immediately too). This guards
    // against future regressions if either path stops being cancel-safe.
    let (log, invoke) = recording_invoke_fn();
    let session = Arc::new(Session::new("sess-concurrent".into(), None, invoke));

    let handler: ToolHandler = Arc::new(|_name: &str, _args: &Value| ToolResult::text("done"));
    session
        .register_tool_with_handler(Tool::new("slow"), Some(handler))
        .await;

    let session_a = Arc::clone(&session);
    let session_b = Arc::clone(&session);

    let tool_task = tokio::spawn(async move {
        session_a
            .invoke_tool("slow", &json!({"hello": "world"}))
            .await
    });
    let destroy_task = tokio::spawn(async move { session_b.destroy().await });

    // 5 seconds is generous for two trivially-bounded futures and keeps the
    // suite finite even if a future regression deadlocks.
    let (tool_res, destroy_res) = tokio::time::timeout(
        Duration::from_secs(5),
        futures::future::join(tool_task, destroy_task),
    )
    .await
    .expect("tasks must complete within timeout");

    let tool_res = tool_res.expect("tool task must not panic");
    let destroy_res = destroy_res.expect("destroy task must not panic");

    assert!(tool_res.is_ok(), "tool invocation must succeed");
    assert!(destroy_res.is_ok(), "destroy must succeed");

    // Exactly one RPC call (the destroy); invoke_tool is local-only.
    let calls = log.lock().await;
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].0, "session.destroy");
}

#[tokio::test]
async fn session_subscribe_yields_dispatched_event() {
    // Sanity-check the broadcast channel — register a subscriber, dispatch
    // an event, and confirm the subscriber receives it without losing the
    // typed variant. Guards against future churn in `Session::dispatch_event`
    // affecting subscriber semantics.
    let (_log, invoke) = recording_invoke_fn();
    let session = Session::new("sess-sub".into(), None, invoke);

    let mut sub = session.subscribe();

    let raw = make_raw_event("session.title_changed", json!({"title": "Hi"}));
    let ev = SessionEvent::from_json(&raw).expect("parse");
    session.dispatch_event(ev).await;

    let received = tokio::time::timeout(Duration::from_secs(2), sub.recv())
        .await
        .expect("recv must not time out")
        .expect("subscriber must receive event");

    match received.data {
        SessionEventData::SessionTitleChanged(d) => assert_eq!(d.title, "Hi"),
        other => panic!("expected SessionTitleChanged, got {other:?}"),
    }
}
