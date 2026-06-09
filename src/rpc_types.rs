// Copyright (c) 2026 Elias Bachaalany
// SPDX-License-Identifier: MIT

//! Typed request/response shapes for the v3 generated RPC families that are
//! not yet covered by [`crate::types`].
//!
//! The struct definitions mirror the TypeScript interfaces exported by
//! `reference/copilot-sdk/nodejs/src/generated/rpc.ts`. The intent is to give
//! Rust callers strongly-typed payloads they can serialize/deserialize when
//! exchanging JSON-RPC messages with the Copilot CLI, even when the higher
//! level [`crate::Client`] / [`crate::Session`] surface doesn't yet expose a
//! dedicated method.
//!
//! Coverage in this module:
//! * `sessions.fork` (request / result)
//! * `session.commands.*` (list / invoke / handle-pending / respond-to-queued)
//! * `session.ui.elicitation` (response + handle-pending request + result)
//! * `session.history.compact` / `session.history.truncate`
//! * `sessionFs.*` server-handled callbacks
//!
//! These types intentionally avoid duplicating shapes already defined in
//! [`crate::types`] (e.g. `PlanData`, `LogOptions`, `ShellExecOptions`).

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

// =============================================================================
// sessions.fork
// =============================================================================

/// `SessionsForkRequest` — fork an existing session up to an optional event
/// boundary.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct SessionsForkRequest {
    /// Source session ID to fork from.
    pub session_id: String,

    /// Optional event ID boundary. When provided, the fork includes only events
    /// before this ID (exclusive). When omitted, all events are included.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub to_event_id: Option<String>,

    /// Optional friendly name to assign to the forked session.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

/// `SessionsForkResult` — identifier (and optional name) of the newly forked
/// session.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct SessionsForkResult {
    /// The new forked session's ID.
    pub session_id: String,

    /// Friendly name assigned to the forked session, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

// =============================================================================
// session.commands.*
// =============================================================================

/// `SlashCommandKind` — provenance of a slash command.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum SlashCommandKind {
    Builtin,
    Skill,
    Client,
}

/// `SlashCommandInputCompletion` — completion hint for command input.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum SlashCommandInputCompletion {
    Directory,
}

/// `SlashCommandInput` — optional unstructured input hint for a command.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct SlashCommandInput {
    pub hint: String,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub required: Option<bool>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completion: Option<SlashCommandInputCompletion>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preserve_multiline_input: Option<bool>,
}

/// `SlashCommandInfo` — metadata describing a registered slash command.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct SlashCommandInfo {
    pub name: String,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub aliases: Option<Vec<String>>,

    pub description: String,
    pub kind: SlashCommandKind,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input: Option<SlashCommandInput>,

    pub allow_during_agent_execution: bool,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub experimental: Option<bool>,
}

/// `CommandsListRequest` — filters controlling which command sources are
/// returned.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct CommandsListRequest {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub include_builtins: Option<bool>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub include_skills: Option<bool>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub include_client_commands: Option<bool>,
}

/// `CommandsListResult` — convenience wrapper over the upstream array result.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct CommandsListResult {
    #[serde(default)]
    pub commands: Vec<SlashCommandInfo>,
}

/// `CommandsInvokeRequest` — invoke a slash command by name.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct CommandsInvokeRequest {
    pub name: String,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input: Option<String>,
}

/// `CommandsHandlePendingCommandRequest` — finalize a pending client-handled
/// command, optionally reporting an error.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct CommandsHandlePendingCommandRequest {
    pub request_id: String,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// `CommandsHandlePendingCommandResult` — outcome of a handle-pending call.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct CommandsHandlePendingCommandResult {
    pub success: bool,
}

/// `QueuedCommandResult` — outcome reported by the client for a queued command.
///
/// Upstream models this as a discriminated union on the `handled` boolean field
/// (`QueuedCommandHandled` vs `QueuedCommandNotHandled`). We flatten the union
/// into a single struct here so the JSON shape stays unambiguous on the wire
/// (untagged serde enums can't distinguish two structs differing only in a
/// boolean literal).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct QueuedCommandResult {
    /// `true` when the client handled the queued command, `false` otherwise.
    pub handled: bool,

    /// Only meaningful when `handled = true`: if `Some(true)`, the runtime
    /// should stop processing remaining queued items.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_processing_queue: Option<bool>,
}

impl QueuedCommandResult {
    /// Build a "handled" response, optionally requesting the runtime to stop
    /// draining the queue.
    pub fn handled(stop_processing_queue: Option<bool>) -> Self {
        Self {
            handled: true,
            stop_processing_queue,
        }
    }

    /// Build a "not handled" response.
    pub fn not_handled() -> Self {
        Self {
            handled: false,
            stop_processing_queue: None,
        }
    }
}

/// `QueuedCommandHandled` — type alias kept for upstream-shape parity.
///
/// In practice [`QueuedCommandResult`] covers both upstream variants; this
/// alias is retained so callers searching for the upstream interface name can
/// still find it.
pub type QueuedCommandHandled = QueuedCommandResult;

/// `QueuedCommandNotHandled` — type alias kept for upstream-shape parity.
pub type QueuedCommandNotHandled = QueuedCommandResult;

/// `CommandsRespondToQueuedCommandRequest` — respond to a queued command.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct CommandsRespondToQueuedCommandRequest {
    pub request_id: String,
    pub result: QueuedCommandResult,
}

/// `CommandsRespondToQueuedCommandResult` — whether the response was accepted.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct CommandsRespondToQueuedCommandResult {
    pub success: bool,
}

// =============================================================================
// session.ui.elicitation
// =============================================================================

/// `UIElicitationResponseAction` — accept / decline / cancel.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum UIElicitationResponseAction {
    Accept,
    Decline,
    Cancel,
}

/// `UIElicitationResponse` — the user's submitted form values plus action.
///
/// `content` is left as a free-form JSON object because upstream models field
/// values as a polymorphic union (`string | number | boolean | string[]`).
/// Callers that want stronger typing can convert from [`serde_json::Value`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct UIElicitationResponse {
    pub action: UIElicitationResponseAction,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<HashMap<String, JsonValue>>,
}

/// `UIHandlePendingElicitationRequest` — finalize a pending elicitation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct UIHandlePendingElicitationRequest {
    pub request_id: String,
    pub result: UIElicitationResponse,
}

/// `UIElicitationResult` — whether the elicitation response was accepted.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct UIElicitationResult {
    pub success: bool,
}

// =============================================================================
// session.history.compact / session.history.truncate
// =============================================================================

/// `HistoryCompactContextWindow` — token/message accounting for the model
/// context window at the time of a compaction.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct HistoryCompactContextWindow {
    pub token_limit: u64,
    pub current_tokens: u64,
    pub messages_length: u64,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system_tokens: Option<u64>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub conversation_tokens: Option<u64>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_definitions_tokens: Option<u64>,
}

/// `HistoryCompactResult` — outcome of `session.history.compact`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct HistoryCompactResult {
    pub success: bool,
    pub tokens_removed: u64,
    pub messages_removed: u64,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_window: Option<HistoryCompactContextWindow>,
}

/// `HistoryTruncateRequest` — truncate the session history at an event ID.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct HistoryTruncateRequest {
    pub event_id: String,
}

/// `HistoryTruncateResult` — number of events removed by the truncation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct HistoryTruncateResult {
    pub events_removed: u64,
}

// =============================================================================
// sessionFs.* — SDK-implemented filesystem callbacks
// =============================================================================

/// `SessionFsErrorCode` — coarse error classification for filesystem callbacks.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum SessionFsErrorCode {
    #[serde(rename = "ENOENT")]
    NoEntry,
    #[serde(rename = "UNKNOWN")]
    Unknown,
}

/// `SessionFsError` — error payload returned by filesystem callbacks.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct SessionFsError {
    pub code: SessionFsErrorCode,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
}

/// `SessionFsReaddirWithTypesEntryType` — file / directory discriminator.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum SessionFsReaddirWithTypesEntryType {
    File,
    Directory,
}

/// `SessionFsSetProviderConventions` — path-convention selector.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum SessionFsSetProviderConventions {
    Windows,
    Posix,
}

/// `SessionFsSetProviderRequest` — register the SDK as the filesystem provider.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct SessionFsSetProviderRequest {
    pub initial_cwd: String,
    pub session_state_path: String,
    pub conventions: SessionFsSetProviderConventions,
}

/// `SessionFsSetProviderResult` — whether provider registration succeeded.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct SessionFsSetProviderResult {
    pub success: bool,
}

/// `SessionFsReadFileRequest`
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct SessionFsReadFileRequest {
    pub session_id: String,
    pub path: String,
}

/// `SessionFsReadFileResult`
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct SessionFsReadFileResult {
    pub content: String,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<SessionFsError>,
}

/// `SessionFsWriteFileRequest`
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct SessionFsWriteFileRequest {
    pub session_id: String,
    pub path: String,
    pub content: String,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mode: Option<u32>,
}

/// `SessionFsAppendFileRequest`
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct SessionFsAppendFileRequest {
    pub session_id: String,
    pub path: String,
    pub content: String,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mode: Option<u32>,
}

/// `SessionFsExistsRequest`
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct SessionFsExistsRequest {
    pub session_id: String,
    pub path: String,
}

/// `SessionFsExistsResult`
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct SessionFsExistsResult {
    pub exists: bool,
}

/// `SessionFsStatRequest`
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct SessionFsStatRequest {
    pub session_id: String,
    pub path: String,
}

/// `SessionFsStatResult`
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct SessionFsStatResult {
    pub is_file: bool,
    pub is_directory: bool,
    pub size: u64,
    pub mtime: String,
    pub birthtime: String,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<SessionFsError>,
}

/// `SessionFsMkdirRequest`
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct SessionFsMkdirRequest {
    pub session_id: String,
    pub path: String,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recursive: Option<bool>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mode: Option<u32>,
}

/// `SessionFsReaddirRequest`
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct SessionFsReaddirRequest {
    pub session_id: String,
    pub path: String,
}

/// `SessionFsReaddirResult`
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct SessionFsReaddirResult {
    #[serde(default)]
    pub entries: Vec<String>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<SessionFsError>,
}

/// `SessionFsReaddirWithTypesEntry`
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct SessionFsReaddirWithTypesEntry {
    pub name: String,
    #[serde(rename = "type")]
    pub entry_type: SessionFsReaddirWithTypesEntryType,
}

/// `SessionFsReaddirWithTypesRequest`
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct SessionFsReaddirWithTypesRequest {
    pub session_id: String,
    pub path: String,
}

/// `SessionFsReaddirWithTypesResult`
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct SessionFsReaddirWithTypesResult {
    #[serde(default)]
    pub entries: Vec<SessionFsReaddirWithTypesEntry>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<SessionFsError>,
}

/// `SessionFsRmRequest`
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct SessionFsRmRequest {
    pub session_id: String,
    pub path: String,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recursive: Option<bool>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub force: Option<bool>,
}

/// `SessionFsRenameRequest`
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct SessionFsRenameRequest {
    pub session_id: String,
    pub src: String,
    pub dest: String,
}

// =============================================================================
// Unit tests — round-trip a handful of representative shapes through JSON
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn roundtrip<T>(value: &T) -> T
    where
        T: Serialize + for<'de> Deserialize<'de>,
    {
        let s = serde_json::to_string(value).expect("serialize");
        serde_json::from_str(&s).expect("deserialize")
    }

    #[test]
    fn sessions_fork_request_camel_case_round_trip() {
        let req = SessionsForkRequest {
            session_id: "s1".into(),
            to_event_id: Some("evt-42".into()),
            name: Some("branch-a".into()),
        };

        let raw = serde_json::to_value(&req).unwrap();
        assert_eq!(
            raw,
            json!({
                "sessionId": "s1",
                "toEventId": "evt-42",
                "name": "branch-a"
            })
        );

        let back: SessionsForkRequest = roundtrip(&req);
        assert_eq!(back, req);
    }

    #[test]
    fn sessions_fork_request_skips_none_fields() {
        let req = SessionsForkRequest {
            session_id: "s1".into(),
            to_event_id: None,
            name: None,
        };
        let raw = serde_json::to_value(&req).unwrap();
        assert_eq!(raw, json!({ "sessionId": "s1" }));
    }

    #[test]
    fn sessions_fork_result_round_trip() {
        let res = SessionsForkResult {
            session_id: "s2".into(),
            name: None,
        };
        let raw = serde_json::to_value(&res).unwrap();
        assert_eq!(raw, json!({ "sessionId": "s2" }));
        assert_eq!(roundtrip(&res), res);
    }

    #[test]
    fn commands_list_request_round_trip() {
        let req = CommandsListRequest {
            include_builtins: Some(true),
            include_skills: Some(false),
            include_client_commands: None,
        };
        let raw = serde_json::to_value(&req).unwrap();
        assert_eq!(
            raw,
            json!({
                "includeBuiltins": true,
                "includeSkills": false
            })
        );
        assert_eq!(roundtrip(&req), req);
    }

    #[test]
    fn commands_invoke_request_round_trip() {
        let req = CommandsInvokeRequest {
            name: "model".into(),
            input: Some("claude-opus".into()),
        };
        let raw = serde_json::to_value(&req).unwrap();
        assert_eq!(raw, json!({ "name": "model", "input": "claude-opus" }));
        assert_eq!(roundtrip(&req), req);
    }

    #[test]
    fn commands_handle_pending_round_trip() {
        let req = CommandsHandlePendingCommandRequest {
            request_id: "req-1".into(),
            error: None,
        };
        let raw = serde_json::to_value(&req).unwrap();
        assert_eq!(raw, json!({ "requestId": "req-1" }));
        assert_eq!(roundtrip(&req), req);

        let res = CommandsHandlePendingCommandResult { success: true };
        assert_eq!(roundtrip(&res), res);
    }

    #[test]
    fn queued_command_result_handled_round_trip() {
        let r = QueuedCommandResult::handled(Some(true));
        let raw = serde_json::to_value(&r).unwrap();
        assert_eq!(raw, json!({ "handled": true, "stopProcessingQueue": true }));
        assert_eq!(roundtrip(&r), r);
    }

    #[test]
    fn queued_command_result_not_handled_round_trip() {
        let r = QueuedCommandResult::not_handled();
        let raw = serde_json::to_value(&r).unwrap();
        assert_eq!(raw, json!({ "handled": false }));
        assert_eq!(roundtrip(&r), r);
    }

    #[test]
    fn ui_elicitation_response_round_trip() {
        let mut content = HashMap::new();
        content.insert("name".into(), json!("Alice"));
        content.insert("count".into(), json!(7));
        let resp = UIElicitationResponse {
            action: UIElicitationResponseAction::Accept,
            content: Some(content),
        };
        let back: UIElicitationResponse = roundtrip(&resp);
        assert_eq!(back, resp);

        let raw = serde_json::to_value(&resp).unwrap();
        assert_eq!(raw["action"], json!("accept"));
    }

    #[test]
    fn ui_handle_pending_elicitation_round_trip() {
        let req = UIHandlePendingElicitationRequest {
            request_id: "elic-1".into(),
            result: UIElicitationResponse {
                action: UIElicitationResponseAction::Cancel,
                content: None,
            },
        };
        let raw = serde_json::to_value(&req).unwrap();
        assert_eq!(
            raw,
            json!({
                "requestId": "elic-1",
                "result": { "action": "cancel" }
            })
        );
        assert_eq!(roundtrip(&req), req);
    }

    #[test]
    fn history_compact_result_round_trip() {
        let r = HistoryCompactResult {
            success: true,
            tokens_removed: 1024,
            messages_removed: 4,
            context_window: Some(HistoryCompactContextWindow {
                token_limit: 200_000,
                current_tokens: 12_345,
                messages_length: 12,
                system_tokens: Some(100),
                conversation_tokens: Some(12_000),
                tool_definitions_tokens: Some(245),
            }),
        };
        let raw = serde_json::to_value(&r).unwrap();
        assert_eq!(raw["tokensRemoved"], json!(1024));
        assert_eq!(raw["contextWindow"]["tokenLimit"], json!(200_000));
        assert_eq!(roundtrip(&r), r);
    }

    #[test]
    fn history_truncate_round_trip() {
        let req = HistoryTruncateRequest {
            event_id: "evt-99".into(),
        };
        assert_eq!(
            serde_json::to_value(&req).unwrap(),
            json!({ "eventId": "evt-99" })
        );
        assert_eq!(roundtrip(&req), req);

        let res = HistoryTruncateResult { events_removed: 3 };
        assert_eq!(
            serde_json::to_value(&res).unwrap(),
            json!({ "eventsRemoved": 3 })
        );
        assert_eq!(roundtrip(&res), res);
    }

    #[test]
    fn session_fs_set_provider_round_trip() {
        let req = SessionFsSetProviderRequest {
            initial_cwd: "/work".into(),
            session_state_path: "/.copilot/session-state".into(),
            conventions: SessionFsSetProviderConventions::Posix,
        };
        let raw = serde_json::to_value(&req).unwrap();
        assert_eq!(
            raw,
            json!({
                "initialCwd": "/work",
                "sessionStatePath": "/.copilot/session-state",
                "conventions": "posix"
            })
        );
        assert_eq!(roundtrip(&req), req);
    }

    #[test]
    fn session_fs_error_code_serializes_as_uppercase() {
        let err = SessionFsError {
            code: SessionFsErrorCode::NoEntry,
            message: Some("not found".into()),
        };
        let raw = serde_json::to_value(&err).unwrap();
        assert_eq!(raw["code"], json!("ENOENT"));
        assert_eq!(roundtrip(&err), err);
    }

    #[test]
    fn session_fs_readdir_with_types_round_trip() {
        let res = SessionFsReaddirWithTypesResult {
            entries: vec![
                SessionFsReaddirWithTypesEntry {
                    name: "src".into(),
                    entry_type: SessionFsReaddirWithTypesEntryType::Directory,
                },
                SessionFsReaddirWithTypesEntry {
                    name: "Cargo.toml".into(),
                    entry_type: SessionFsReaddirWithTypesEntryType::File,
                },
            ],
            error: None,
        };
        let raw = serde_json::to_value(&res).unwrap();
        assert_eq!(raw["entries"][0]["type"], json!("directory"));
        assert_eq!(raw["entries"][1]["type"], json!("file"));
        assert_eq!(roundtrip(&res), res);
    }

    #[test]
    fn session_fs_stat_round_trip() {
        let res = SessionFsStatResult {
            is_file: true,
            is_directory: false,
            size: 42,
            mtime: "2026-01-01T00:00:00Z".into(),
            birthtime: "2025-12-31T23:59:59Z".into(),
            error: None,
        };
        let raw = serde_json::to_value(&res).unwrap();
        assert_eq!(raw["isFile"], json!(true));
        assert_eq!(raw["isDirectory"], json!(false));
        assert_eq!(raw["mtime"], json!("2026-01-01T00:00:00Z"));
        assert_eq!(roundtrip(&res), res);
    }
}
