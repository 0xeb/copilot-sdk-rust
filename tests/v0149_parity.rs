// Copyright (c) 2026 Elias Bachaalany
// SPDX-License-Identifier: MIT

//! Integration tests covering upstream nodejs v0.1.49 additive features.
//!
//! These tests build the SDK's public types end-to-end (through the
//! re-exports in `lib.rs`) and assert wire-level shape parity:
//!
//! * `SessionConfig` / `ResumeSessionConfig` v0.1.49 fields serialize as
//!   `camelCase` and are omitted entirely when `None`, so v0.1.23 servers
//!   see no change.
//! * `RemoteSessionMode` round-trips for every variant (`off`/`export`/`on`).
//! * `ClientOptions` and `ClientBuilder` expose the new fields
//!   (`tcp_connection_token`, `copilot_home`, `session_idle_timeout_seconds`,
//!   `remote`) and reject invalid combinations.
//! * `SessionListFilter` / `SessionContext` shape match upstream
//!   `listSessions(filter)` and `getSessionMetadata()`.
//! * Event parser dispatches the v0.1.49 event types through `SessionEvent`'s
//!   public `from_json` entry point (Unknown-fallback round-trip).

use copilot_sdk::{
    Client, ClientOptions, RemoteSessionMode, ResumeSessionConfig, SessionConfig, SessionContext,
    SessionEvent, SessionEventData, SessionListFilter, SessionMetadata,
};
use serde_json::json;

// ---------------------------------------------------------------------------
// SessionConfig v0.1.49 wire format
// ---------------------------------------------------------------------------

#[test]
fn session_config_v0149_all_fields_camel_case() {
    let cfg = SessionConfig {
        client_name: Some("ports-rust".into()),
        agent: Some("default".into()),
        working_directory: Some("/work".into()),
        enable_session_telemetry: Some(true),
        include_sub_agent_streaming_events: Some(true),
        enable_config_discovery: Some(false),
        instruction_directories: Some(vec!["/etc/inst".into(), "./local-inst".into()]),
        remote_session: Some(RemoteSessionMode::Export),
        ..Default::default()
    };
    let v = serde_json::to_value(&cfg).unwrap();
    // v0.1.49 fields all use the upstream camelCase wire names.
    assert_eq!(v["enableSessionTelemetry"], true);
    assert_eq!(v["includeSubAgentStreamingEvents"], true);
    assert_eq!(v["enableConfigDiscovery"], false);
    assert_eq!(v["instructionDirectories"][0], "/etc/inst");
    assert_eq!(v["instructionDirectories"][1], "./local-inst");
    assert_eq!(v["remoteSession"], "export");
    // Existing fields still serialize alongside the new ones.
    assert_eq!(v["clientName"], "ports-rust");
    assert_eq!(v["agent"], "default");
    assert_eq!(v["workingDirectory"], "/work");
    // Confirm the snake_case wire names are NOT emitted.
    assert!(v.get("enable_session_telemetry").is_none());
    assert!(v.get("instruction_directories").is_none());
    assert!(v.get("remote_session").is_none());
}

#[test]
fn session_config_empty_instruction_directories_serializes_as_empty_array() {
    let cfg = SessionConfig {
        instruction_directories: Some(Vec::new()),
        ..Default::default()
    };
    let v = serde_json::to_value(&cfg).unwrap();
    assert_eq!(v["instructionDirectories"], json!([]));
}

#[test]
fn resume_session_config_v0149_all_fields_camel_case() {
    let cfg = ResumeSessionConfig {
        client_name: Some("resumer".into()),
        agent: Some("alt".into()),
        enable_session_telemetry: Some(false),
        include_sub_agent_streaming_events: Some(true),
        enable_config_discovery: Some(true),
        instruction_directories: Some(vec!["/r/inst".into()]),
        remote_session: Some(RemoteSessionMode::On),
        ..Default::default()
    };
    let v = serde_json::to_value(&cfg).unwrap();
    assert_eq!(v["enableSessionTelemetry"], false);
    assert_eq!(v["includeSubAgentStreamingEvents"], true);
    assert_eq!(v["enableConfigDiscovery"], true);
    assert_eq!(v["instructionDirectories"], json!(["/r/inst"]));
    assert_eq!(v["remoteSession"], "on");
    assert_eq!(v["clientName"], "resumer");
    assert_eq!(v["agent"], "alt");
}

#[test]
fn resume_session_config_default_omits_v0149_fields() {
    let cfg = ResumeSessionConfig::default();
    let v = serde_json::to_value(&cfg).unwrap();
    for key in [
        "enableSessionTelemetry",
        "includeSubAgentStreamingEvents",
        "enableConfigDiscovery",
        "instructionDirectories",
        "remoteSession",
    ] {
        assert!(
            v.get(key).is_none(),
            "ResumeSessionConfig default unexpectedly serialized `{key}`"
        );
    }
}

// ---------------------------------------------------------------------------
// RemoteSessionMode round-trip
// ---------------------------------------------------------------------------

#[test]
fn remote_session_mode_round_trip_all_variants() {
    for (mode, wire) in [
        (RemoteSessionMode::Off, "off"),
        (RemoteSessionMode::Export, "export"),
        (RemoteSessionMode::On, "on"),
    ] {
        let v = serde_json::to_value(mode).unwrap();
        assert_eq!(v, json!(wire), "serialize {mode:?}");
        let back: RemoteSessionMode = serde_json::from_value(json!(wire)).unwrap();
        assert_eq!(back, mode, "deserialize {wire}");
    }
}

#[test]
fn remote_session_mode_rejects_unknown_value() {
    let err = serde_json::from_value::<RemoteSessionMode>(json!("partial")).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("partial") || msg.to_lowercase().contains("unknown variant"),
        "unexpected error: {msg}"
    );
}

// ---------------------------------------------------------------------------
// ClientOptions / ClientBuilder v0.1.49 wiring
// ---------------------------------------------------------------------------

#[test]
fn client_options_default_has_v0149_fields_unset() {
    let opts = ClientOptions::default();
    assert!(opts.tcp_connection_token.is_none());
    assert!(opts.copilot_home.is_none());
    assert!(opts.session_idle_timeout_seconds.is_none());
    assert!(!opts.remote);
}

#[test]
fn client_builder_propagates_all_v0149_options() {
    // `Client::options` is private; we can only assert the build succeeds
    // with all v0.1.49 setters chained. Per-field propagation is exercised
    // by the in-crate `#[cfg(test)]` unit tests in `src/client.rs`.
    let result = Client::builder()
        .use_stdio(false)
        .tcp_connection_token("abc-123")
        .copilot_home("/var/lib/copilot")
        .session_idle_timeout_seconds(900)
        .remote(true)
        .build();
    assert!(
        result.is_ok(),
        "builder must accept v0.1.49 setters: {:?}",
        result.err()
    );
}

#[test]
fn client_options_struct_exposes_v0149_fields() {
    // Build the options struct manually (public fields) to confirm the
    // v0.1.49 surface is reachable from downstream crates.
    let opts = ClientOptions {
        use_stdio: false,
        tcp_connection_token: Some("token-xyz".into()),
        copilot_home: Some(std::path::PathBuf::from("/opt/copilot")),
        session_idle_timeout_seconds: Some(120),
        remote: true,
        ..Default::default()
    };
    assert_eq!(opts.tcp_connection_token.as_deref(), Some("token-xyz"));
    assert_eq!(
        opts.copilot_home.as_ref().and_then(|p| p.to_str()),
        Some("/opt/copilot")
    );
    assert_eq!(opts.session_idle_timeout_seconds, Some(120));
    assert!(opts.remote);
    // And the constructed options are accepted by Client::new.
    assert!(Client::new(opts).is_ok());
}

#[test]
fn client_builder_rejects_empty_tcp_token() {
    let result = Client::builder()
        .use_stdio(false)
        .tcp_connection_token("")
        .build();
    assert!(
        result.is_err(),
        "empty tcp_connection_token must be rejected"
    );
}

#[test]
fn client_builder_rejects_tcp_token_with_stdio() {
    let result = Client::builder()
        .use_stdio(true)
        .tcp_connection_token("any-non-empty")
        .build();
    assert!(
        result.is_err(),
        "tcp_connection_token must be rejected when use_stdio is true"
    );
}

// ---------------------------------------------------------------------------
// Lifecycle types: SessionListFilter / SessionContext / SessionMetadata
// ---------------------------------------------------------------------------

#[test]
fn session_list_filter_serializes_only_set_fields() {
    let filter = SessionListFilter {
        repository: Some("octo/cat".into()),
        ..Default::default()
    };
    let v = serde_json::to_value(&filter).unwrap();
    let obj = v.as_object().expect("filter must serialize as object");
    assert_eq!(obj.len(), 1, "only set fields should be emitted: {obj:?}");
    assert_eq!(obj["repository"], "octo/cat");
}

#[test]
fn session_context_round_trip() {
    let ctx = SessionContext {
        cwd: "/w".into(),
        git_root: Some("/w".into()),
        repository: Some("o/r".into()),
        branch: Some("main".into()),
    };
    let v = serde_json::to_value(&ctx).unwrap();
    assert_eq!(v["cwd"], "/w");
    assert_eq!(v["gitRoot"], "/w");
    assert_eq!(v["repository"], "o/r");
    assert_eq!(v["branch"], "main");
    let back: SessionContext = serde_json::from_value(v).unwrap();
    assert_eq!(back.cwd, "/w");
    assert_eq!(back.git_root.as_deref(), Some("/w"));
    assert_eq!(back.repository.as_deref(), Some("o/r"));
    assert_eq!(back.branch.as_deref(), Some("main"));
}

#[test]
fn session_metadata_parses_v0149_context_field() {
    let raw = json!({
        "sessionId": "sess-1",
        "isRemote": false,
        "context": {
            "cwd": "/x",
            "gitRoot": null,
            "repository": "o/r",
            "branch": "dev"
        }
    });
    let md: SessionMetadata = serde_json::from_value(raw).unwrap();
    let ctx = md.context.expect("context must be parsed");
    assert_eq!(ctx.cwd, "/x");
    assert!(ctx.git_root.is_none());
    assert_eq!(ctx.repository.as_deref(), Some("o/r"));
    assert_eq!(ctx.branch.as_deref(), Some("dev"));
}

// ---------------------------------------------------------------------------
// Event dispatch round-trip (public surface via SessionEvent::from_json)
// ---------------------------------------------------------------------------

fn make_raw(event_type: &str, data: serde_json::Value) -> serde_json::Value {
    json!({
        "id": format!("evt-{event_type}"),
        "timestamp": "2026-01-01T00:00:00Z",
        "type": event_type,
        "data": data,
    })
}

#[test]
fn parse_session_remote_steerable_changed_via_public_api() {
    let raw = make_raw(
        "session.remote_steerable_changed",
        json!({ "remoteSteerable": true }),
    );
    let ev = SessionEvent::from_json(&raw).expect("must parse");
    match ev.data {
        SessionEventData::SessionRemoteSteerableChanged(d) => {
            assert!(d.remote_steerable);
        }
        other => panic!("expected SessionRemoteSteerableChanged, got {other:?}"),
    }
}

#[test]
fn parse_session_title_changed_via_public_api() {
    let raw = make_raw("session.title_changed", json!({ "title": "Hi" }));
    let ev = SessionEvent::from_json(&raw).unwrap();
    match ev.data {
        SessionEventData::SessionTitleChanged(d) => assert_eq!(d.title, "Hi"),
        other => panic!("expected SessionTitleChanged, got {other:?}"),
    }
}

#[test]
fn parse_model_call_failure_via_public_api() {
    let raw = make_raw(
        "model.call_failure",
        json!({
            "source": "top_level",
            "model": "gpt-4",
            "statusCode": 429,
            "errorMessage": "rate limited"
        }),
    );
    let ev = SessionEvent::from_json(&raw).unwrap();
    assert!(matches!(ev.data, SessionEventData::ModelCallFailure(_)));
}

#[test]
fn parse_capabilities_changed_via_public_api() {
    let raw = make_raw(
        "capabilities.changed",
        json!({ "capabilities": { "tools": ["read", "write"] } }),
    );
    let ev = SessionEvent::from_json(&raw).unwrap();
    assert!(matches!(ev.data, SessionEventData::CapabilitiesChanged(_)));
}

#[test]
fn parse_commands_changed_via_public_api() {
    let raw = make_raw("commands.changed", json!({ "commands": [] }));
    let ev = SessionEvent::from_json(&raw).unwrap();
    assert!(matches!(ev.data, SessionEventData::CommandsChanged(_)));
}

#[test]
fn parse_subagent_deselected_via_public_api() {
    let raw = make_raw("subagent.deselected", json!({}));
    let ev = SessionEvent::from_json(&raw).unwrap();
    assert!(matches!(ev.data, SessionEventData::SubagentDeselected(_)));
}

#[test]
fn unknown_event_type_falls_back_to_unknown() {
    let raw = make_raw("completely.fictional.event.from.future", json!({ "x": 1 }));
    let ev = SessionEvent::from_json(&raw).unwrap();
    assert!(matches!(ev.data, SessionEventData::Unknown(_)));
    assert_eq!(ev.event_type, "completely.fictional.event.from.future");
}
