// Copyright (c) 2026 Elias Bachaalany
// SPDX-License-Identifier: MIT

//! Session event types for the Copilot SDK.
//!
//! Events are received from the Copilot CLI during a session. They include
//! assistant messages, tool executions, session lifecycle events, and more.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// =============================================================================
// Nested Types (used within event data)
// =============================================================================

/// Handoff source type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum HandoffSourceType {
    Remote,
    Local,
}

/// System message role.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SystemMessageRole {
    System,
    Developer,
}

/// Repository info for handoff events.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RepositoryInfo {
    pub owner: String,
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub branch: Option<String>,
}

/// Attachment in user message.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UserMessageAttachmentItem {
    #[serde(rename = "type")]
    pub attachment_type: super::AttachmentType,
    pub path: String,
    pub display_name: String,
}

/// Tool request in assistant message.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolRequestItem {
    pub tool_call_id: String,
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<serde_json::Value>,
}

/// Tool execution result content.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResultContent {
    pub content: String,
}

/// Tool execution error.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolExecutionError {
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
}

/// Hook error.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HookError {
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stack: Option<String>,
}

/// System message metadata.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SystemMessageMetadata {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_version: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub variables: Option<HashMap<String, serde_json::Value>>,
}

// =============================================================================
// Event Data Types
// =============================================================================

/// Data for session.start event.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionStartData {
    #[serde(default)]
    pub session_id: String,
    #[serde(default)]
    pub version: f64,
    #[serde(default)]
    pub producer: String,
    #[serde(default)]
    pub copilot_version: String,
    #[serde(default)]
    pub start_time: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub selected_model: Option<String>,
}

/// Data for session.resume event.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionResumeData {
    pub resume_time: String,
    pub event_count: f64,
}

/// Data for session.error event.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionErrorData {
    pub error_type: String,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stack: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_call_id: Option<String>,
}

/// Data for session.idle event.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionIdleData {
    /// True when the preceding agentic loop was cancelled via abort signal.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub aborted: Option<bool>,
}

/// Data for session.info event.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionInfoData {
    pub info_type: String,
    pub message: String,
    /// Optional actionable tip displayed with this message.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tip: Option<String>,
    /// Optional URL associated with this message that the user can open in a browser.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
}

/// Data for `session.warning` event.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionWarningData {
    /// Category of warning (e.g., "subscription", "policy", "mcp").
    pub warning_type: String,
    /// Human-readable warning message for display in the timeline.
    pub message: String,
    /// Optional URL associated with this warning that the user can open in a browser.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
}

/// Data for `session.remote_steerable_changed` event.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionRemoteSteerableChangedData {
    /// Whether this session now supports remote steering via GitHub.
    pub remote_steerable: bool,
}

/// Data for `session.title_changed` event.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionTitleChangedData {
    /// The new display title for the session.
    pub title: String,
}

/// Data for `session.schedule_created` event.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionScheduleCreatedData {
    /// Sequential id assigned to the scheduled prompt within the session.
    pub id: f64,
    /// Prompt text that gets enqueued on every tick.
    pub prompt: String,
    /// Interval between ticks in milliseconds.
    pub interval_ms: f64,
    /// Optional user-facing label shown in the timeline instead of the actual prompt.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub display_prompt: Option<String>,
    /// Whether the schedule re-arms after each tick (`/every`) or fires once (`/after`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub recurring: Option<bool>,
}

/// Data for `session.schedule_cancelled` event.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionScheduleCancelledData {
    /// Id of the scheduled prompt that was cancelled.
    pub id: f64,
}

/// Operation applied to the plan file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PlanChangedOperation {
    Create,
    Update,
    Delete,
}

/// Data for `session.plan_changed` event.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionPlanChangedData {
    pub operation: PlanChangedOperation,
}

/// Operation applied to a workspace file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum WorkspaceFileChangedOperation {
    Create,
    Update,
}

/// Data for `session.workspace_file_changed` event.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionWorkspaceFileChangedData {
    pub operation: WorkspaceFileChangedOperation,
    /// Relative path within the session workspace files directory.
    pub path: String,
}

/// Mode descriptor for `session.mode_changed` event.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionModeChangedData {
    /// Agent mode before the change (e.g., "interactive", "plan", "autopilot").
    pub previous_mode: String,
    /// Agent mode after the change.
    pub new_mode: String,
}

/// Hosting platform type of the working directory's repository.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum WorkingDirectoryHostType {
    Github,
    Ado,
}

/// Working-directory context shared by `session.context_changed` and `session.resume`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WorkingDirectoryContext {
    /// Current working directory path.
    #[serde(default)]
    pub cwd: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub branch: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub base_commit: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub head_commit: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub git_root: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub host_type: Option<WorkingDirectoryHostType>,
    /// Repository identifier derived from the git remote URL (e.g., "owner/name").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repository: Option<String>,
    /// Raw host string from the git remote URL.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repository_host: Option<String>,
}

/// Data for `session.task_complete` event.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionTaskCompleteData {
    /// Whether the tool call succeeded. False when validation failed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub success: Option<bool>,
    /// Summary of the completed task, provided by the agent.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary: Option<String>,
}

/// Data for session.model_change event.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionModelChangeData {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub previous_model: Option<String>,
    pub new_model: String,
}

/// Data for session.handoff event.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionHandoffData {
    pub handoff_time: String,
    pub source_type: HandoffSourceType,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repository: Option<RepositoryInfo>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub remote_session_id: Option<String>,
}

/// Data for session.truncation event.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionTruncationData {
    pub token_limit: f64,
    pub pre_truncation_tokens_in_messages: f64,
    pub pre_truncation_messages_length: f64,
    pub post_truncation_tokens_in_messages: f64,
    pub post_truncation_messages_length: f64,
    pub tokens_removed_during_truncation: f64,
    pub messages_removed_during_truncation: f64,
    pub performed_by: String,
}

/// Data for user.message event.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UserMessageData {
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub transformed_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub attachments: Option<Vec<UserMessageAttachmentItem>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
}

/// Data for pending_messages.modified event.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PendingMessagesModifiedData {}

/// Data for assistant.turn_start event.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AssistantTurnStartData {
    pub turn_id: String,
}

/// Data for assistant.intent event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssistantIntentData {
    pub intent: String,
}

/// Data for assistant.reasoning event.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AssistantReasoningData {
    pub reasoning_id: String,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chunk_content: Option<String>,
}

/// Data for assistant.reasoning_delta event.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AssistantReasoningDeltaData {
    pub reasoning_id: String,
    pub delta_content: String,
}

/// Data for assistant.message event.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AssistantMessageData {
    pub message_id: String,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chunk_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_response_size_bytes: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_requests: Option<Vec<ToolRequestItem>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_tool_call_id: Option<String>,
}

/// Data for assistant.message_delta event.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AssistantMessageDeltaData {
    pub message_id: String,
    pub delta_content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_response_size_bytes: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_tool_call_id: Option<String>,
}

/// Data for assistant.turn_end event.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AssistantTurnEndData {
    pub turn_id: String,
}

/// Data for assistant.usage event.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AssistantUsageData {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_tokens: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_tokens: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_read_tokens: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_write_tokens: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cost: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub initiator: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quota_snapshots: Option<HashMap<String, serde_json::Value>>,
}

/// Data for abort event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AbortData {
    pub reason: String,
}

/// Data for tool.user_requested event.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolUserRequestedData {
    pub tool_call_id: String,
    pub tool_name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<serde_json::Value>,
}

/// Data for tool.execution_start event.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolExecutionStartData {
    pub tool_call_id: String,
    pub tool_name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_tool_call_id: Option<String>,
}

/// Data for tool.execution_partial_result event.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolExecutionPartialResultData {
    pub tool_call_id: String,
    pub partial_output: String,
}

/// Data for tool.execution_complete event.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolExecutionCompleteData {
    pub tool_call_id: String,
    pub success: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub is_user_requested: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<ToolResultContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<ToolExecutionError>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_telemetry: Option<HashMap<String, serde_json::Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mcp_server_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mcp_tool_name: Option<String>,
}

/// Data for custom_agent.started event.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CustomAgentStartedData {
    pub tool_call_id: String,
    pub agent_name: String,
    pub agent_display_name: String,
    pub agent_description: String,
}

/// Data for custom_agent.completed event.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CustomAgentCompletedData {
    pub tool_call_id: String,
    pub agent_name: String,
}

/// Data for custom_agent.failed event.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CustomAgentFailedData {
    pub tool_call_id: String,
    pub agent_name: String,
    pub error: String,
}

/// Data for custom_agent.selected event.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CustomAgentSelectedData {
    pub agent_name: String,
    pub agent_display_name: String,
    pub tools: Vec<String>,
}

/// Data for hook.start event.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HookStartData {
    pub hook_invocation_id: String,
    pub hook_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input: Option<serde_json::Value>,
}

/// Data for hook.end event.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HookEndData {
    pub hook_invocation_id: String,
    pub hook_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output: Option<serde_json::Value>,
    pub success: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<HookError>,
}

/// Data for system.message event.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SystemMessageEventData {
    pub content: String,
    pub role: SystemMessageRole,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<SystemMessageMetadata>,
}

/// Data for session.compaction_start event.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SessionCompactionStartData {}

/// Tokens used during compaction.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CompactionTokensUsed {
    #[serde(default)]
    pub input: f64,
    #[serde(default)]
    pub output: f64,
    #[serde(default)]
    pub cached_input: f64,
}

/// Data for session.compaction_complete event.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionCompactionCompleteData {
    pub success: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pre_compaction_tokens: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub post_compaction_tokens: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pre_compaction_messages_length: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub post_compaction_messages_length: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub compaction_tokens_used: Option<CompactionTokensUsed>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub messages_removed: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tokens_removed: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub checkpoint_number: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub checkpoint_path: Option<String>,
}

/// Shutdown type for session.shutdown event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ShutdownType {
    Routine,
    Error,
}

/// Code changes reported in shutdown event.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ShutdownCodeChanges {
    #[serde(default)]
    pub lines_added: f64,
    #[serde(default)]
    pub lines_removed: f64,
    #[serde(default)]
    pub files_modified: Vec<String>,
}

/// Data for session.shutdown event.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionShutdownData {
    pub shutdown_type: ShutdownType,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_reason: Option<String>,
    #[serde(default)]
    pub total_premium_requests: f64,
    #[serde(default)]
    pub total_api_duration_ms: f64,
    #[serde(default)]
    pub session_start_time: f64,
    #[serde(default)]
    pub code_changes: ShutdownCodeChanges,
    #[serde(default)]
    pub model_metrics: HashMap<String, serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub current_model: Option<String>,
}

/// Data for session.snapshot_rewind event.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionSnapshotRewindData {
    pub up_to_event_id: String,
    #[serde(default)]
    pub events_removed: f64,
}

/// Data for session.usage_info event.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionUsageInfoData {
    #[serde(default)]
    pub token_limit: f64,
    #[serde(default)]
    pub current_tokens: f64,
    #[serde(default)]
    pub messages_length: f64,
}

/// Data for tool.execution_progress event.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolExecutionProgressData {
    pub tool_call_id: String,
    pub progress_message: String,
}

/// Data for skill.invoked event.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SkillInvokedData {
    pub name: String,
    pub path: String,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allowed_tools: Option<Vec<String>>,
}

// =============================================================================
// Session Event (Discriminated Union)
// =============================================================================

/// Data for `assistant.streaming_delta` event (cumulative byte count).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AssistantStreamingDeltaData {
    /// Cumulative total bytes received from the streaming response so far.
    pub total_response_size_bytes: f64,
}

/// Data for `assistant.message_start` event (streaming message announce).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AssistantMessageStartData {
    pub message_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub phase: Option<String>,
}

/// Source of a failed model call.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelCallFailureSource {
    TopLevel,
    Subagent,
    McpSampling,
}

/// Data for `model.call_failure` event (telemetry-only).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelCallFailureData {
    pub source: ModelCallFailureSource,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub initiator: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status_code: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_message: Option<String>,
}

/// Data for `subagent.deselected` event (empty payload).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SubagentDeselectedData {}

/// Data for `system.notification` event. `kind` is preserved as raw JSON
/// because its shape varies by notification subtype (agent_completed,
/// agent_idle, new_inbox_message, shell_completed, shell_detached_completed,
/// instruction_discovered).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SystemNotificationData {
    /// The notification text, typically wrapped in `<system_notification>` XML tags.
    pub content: String,
    /// Structured metadata identifying what triggered this notification.
    pub kind: serde_json::Value,
}

/// Data for `permission.completed` event. The `result` payload is preserved as
/// raw JSON because its shape varies across the `PermissionResult` union.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PermissionCompletedData {
    pub request_id: String,
    pub result: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

/// Data for `user_input.requested` event.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UserInputRequestedData {
    pub request_id: String,
    /// The question or prompt to present to the user.
    pub question: String,
    /// Predefined choices for the user to select from, if applicable.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub choices: Option<Vec<String>>,
    /// Whether the user can provide a free-form text response.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allow_freeform: Option<bool>,
    /// LLM-assigned tool call ID that triggered this request.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

/// Data for `user_input.completed` event.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UserInputCompletedData {
    pub request_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub answer: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub was_freeform: Option<bool>,
}

/// Elicitation mode (form-based or URL-based).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ElicitationRequestedMode {
    Form,
    Url,
}

/// Data for `elicitation.requested` event. The `requestedSchema` field is preserved
/// as raw JSON because its `properties` map is open-ended.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ElicitationRequestedData {
    pub request_id: String,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mode: Option<ElicitationRequestedMode>,
    /// URL to open in the user's browser (url mode only).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    /// JSON Schema describing the form fields to present (form mode only).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub requested_schema: Option<serde_json::Value>,
    /// MCP server name (or absent for agent-initiated).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub elicitation_source: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

/// User action returned with an elicitation completion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ElicitationCompletedAction {
    Accept,
    Decline,
    Cancel,
}

/// Data for `elicitation.completed` event. `content` values may be string, number,
/// boolean, or string array — preserved as raw JSON.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ElicitationCompletedData {
    pub request_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub action: Option<ElicitationCompletedAction>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<HashMap<String, serde_json::Value>>,
}

/// Data for `sampling.requested` event.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SamplingRequestedData {
    pub request_id: String,
    /// Name of the MCP server that initiated the sampling request.
    pub server_name: String,
    /// The JSON-RPC request ID from the MCP protocol (string or number).
    pub mcp_request_id: serde_json::Value,
}

/// Data for `sampling.completed` event.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SamplingCompletedData {
    pub request_id: String,
}

/// Optional static OAuth client configuration provided with `mcp.oauth_required`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct McpOauthStaticClientConfig {
    pub client_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub grant_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub public_client: Option<bool>,
}

/// Data for `mcp.oauth_required` event.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct McpOauthRequiredData {
    pub request_id: String,
    pub server_name: String,
    pub server_url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub static_client_config: Option<McpOauthStaticClientConfig>,
}

/// Data for `mcp.oauth_completed` event.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct McpOauthCompletedData {
    pub request_id: String,
}

/// Data for `session.custom_notification` event. `payload` and `subject` are
/// preserved as raw JSON since they are source-defined.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CustomNotificationData {
    /// Namespace for the custom notification producer.
    pub source: String,
    /// Source-defined custom notification name.
    pub name: String,
    pub payload: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub subject: Option<HashMap<String, String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<f64>,
}

/// Data for `external_tool.completed` event.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ExternalToolCompletedData {
    pub request_id: String,
}

/// Data for `command.queued` event (slash command dispatch).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CommandQueuedData {
    pub request_id: String,
    /// The slash command text to be executed (e.g., `/help`, `/clear`).
    pub command: String,
}

/// Data for `command.execute` event (registered command dispatch).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CommandExecuteData {
    pub request_id: String,
    /// Full command text (e.g., `/deploy production`).
    pub command: String,
    /// Command name without the leading `/`.
    pub command_name: String,
    /// Raw argument string after the command name.
    pub args: String,
}

/// Data for `command.completed` event.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CommandCompletedData {
    pub request_id: String,
}

/// Data for `auto_mode_switch.requested` event.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AutoModeSwitchRequestedData {
    pub request_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_code: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub retry_after_seconds: Option<f64>,
}

/// Data for `auto_mode_switch.completed` event.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AutoModeSwitchCompletedData {
    pub request_id: String,
    /// The user's choice: `yes`, `yes_always`, or `no`.
    pub response: String,
}

/// SDK-registered slash command descriptor.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CommandsChangedCommand {
    /// Slash command name without the leading slash.
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

/// Data for `commands.changed` event.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CommandsChangedData {
    pub commands: Vec<CommandsChangedCommand>,
}

/// UI capability flags carried by `capabilities.changed`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CapabilitiesChangedUi {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub elicitation: Option<bool>,
}

/// Data for `capabilities.changed` event.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CapabilitiesChangedData {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ui: Option<CapabilitiesChangedUi>,
}

/// Data for `exit_plan_mode.requested` event.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ExitPlanModeRequestedData {
    pub request_id: String,
    /// Summary of the plan that was created.
    pub summary: String,
    /// Full content of the plan file.
    pub plan_content: String,
    /// Available actions the user can take.
    pub actions: Vec<String>,
    /// The recommended action for the user to take.
    pub recommended_action: String,
}

/// Data for `exit_plan_mode.completed` event.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ExitPlanModeCompletedData {
    pub request_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub approved: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub feedback: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub selected_action: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub auto_approve_edits: Option<bool>,
}

/// Data for `session.tools_updated` event.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionToolsUpdatedData {
    /// Identifier of the model the resolved tools apply to.
    pub model: String,
}

/// Data for `session.background_tasks_changed` event (empty payload).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SessionBackgroundTasksChangedData {}

/// One skill entry in `session.skills_loaded`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SkillsLoadedSkill {
    pub name: String,
    pub description: String,
    pub source: String,
    pub enabled: bool,
    pub user_invocable: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
}

/// Data for `session.skills_loaded` event.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionSkillsLoadedData {
    pub skills: Vec<SkillsLoadedSkill>,
}

/// One custom agent entry in `session.custom_agents_updated`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CustomAgentsUpdatedAgent {
    pub id: String,
    pub name: String,
    pub display_name: String,
    pub description: String,
    pub source: String,
    pub user_invocable: bool,
    /// List of tool names available to this agent, or null when all tools are available.
    pub tools: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
}

/// Data for `session.custom_agents_updated` event.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionCustomAgentsUpdatedData {
    pub agents: Vec<CustomAgentsUpdatedAgent>,
    #[serde(default)]
    pub errors: Vec<String>,
    #[serde(default)]
    pub warnings: Vec<String>,
}

/// Connection status of an MCP server.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum McpServerStatus {
    #[serde(rename = "connected")]
    Connected,
    #[serde(rename = "failed")]
    Failed,
    #[serde(rename = "needs-auth")]
    NeedsAuth,
    #[serde(rename = "pending")]
    Pending,
    #[serde(rename = "disabled")]
    Disabled,
    #[serde(rename = "not_configured")]
    NotConfigured,
}

/// One MCP server entry in `session.mcp_servers_loaded`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct McpServersLoadedServer {
    pub name: String,
    pub status: McpServerStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Data for `session.mcp_servers_loaded` event.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionMcpServersLoadedData {
    pub servers: Vec<McpServersLoadedServer>,
}

/// Data for `session.mcp_server_status_changed` event.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionMcpServerStatusChangedData {
    pub server_name: String,
    pub status: McpServerStatus,
}

/// Discovery source of a loaded extension.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ExtensionSource {
    Project,
    User,
}

/// Current status of a loaded extension.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ExtensionStatus {
    Running,
    Disabled,
    Failed,
    Starting,
}

/// One extension entry in `session.extensions_loaded`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ExtensionsLoadedExtension {
    pub id: String,
    pub name: String,
    pub source: ExtensionSource,
    pub status: ExtensionStatus,
}

/// Data for `session.extensions_loaded` event.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionExtensionsLoadedData {
    pub extensions: Vec<ExtensionsLoadedExtension>,
}

/// Data for `external_tool.requested` event (protocol v3 broadcast model).
///
/// In protocol v3, tool calls are broadcast as session events instead of
/// RPC requests. The SDK handles these internally and responds via
/// `session.tools.handlePendingToolCall` RPC.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ExternalToolRequestedData {
    /// Unique request ID for correlating the response.
    pub request_id: Option<String>,
    /// Name of the tool being requested.
    pub tool_name: Option<String>,
    /// Tool call ID for tracking.
    pub tool_call_id: Option<String>,
    /// Arguments to pass to the tool handler.
    pub arguments: Option<serde_json::Value>,
}

/// Data for `permission.requested` event (protocol v3 broadcast model).
///
/// In protocol v3, permission requests are broadcast as session events.
/// The SDK handles these internally and responds via
/// `session.permissions.handlePendingPermissionRequest` RPC.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PermissionRequestedData {
    /// Unique request ID for correlating the response.
    pub request_id: Option<String>,
    /// The permission request details.
    pub permission_request: Option<serde_json::Value>,
}

/// Event data variants - the payload of each event type.
#[derive(Debug, Clone, Serialize)]
pub enum SessionEventData {
    SessionStart(SessionStartData),
    SessionResume(SessionResumeData),
    SessionRemoteSteerableChanged(SessionRemoteSteerableChangedData),
    SessionError(SessionErrorData),
    SessionIdle(SessionIdleData),
    SessionTitleChanged(SessionTitleChangedData),
    SessionScheduleCreated(SessionScheduleCreatedData),
    SessionScheduleCancelled(SessionScheduleCancelledData),
    SessionInfo(SessionInfoData),
    SessionWarning(SessionWarningData),
    SessionModelChange(SessionModelChangeData),
    SessionModeChanged(SessionModeChangedData),
    SessionPlanChanged(SessionPlanChangedData),
    SessionWorkspaceFileChanged(SessionWorkspaceFileChangedData),
    SessionHandoff(SessionHandoffData),
    SessionTruncation(SessionTruncationData),
    SessionContextChanged(WorkingDirectoryContext),
    SessionTaskComplete(SessionTaskCompleteData),
    UserMessage(UserMessageData),
    PendingMessagesModified(PendingMessagesModifiedData),
    AssistantTurnStart(AssistantTurnStartData),
    AssistantIntent(AssistantIntentData),
    AssistantReasoning(AssistantReasoningData),
    AssistantReasoningDelta(AssistantReasoningDeltaData),
    AssistantStreamingDelta(AssistantStreamingDeltaData),
    AssistantMessage(AssistantMessageData),
    AssistantMessageStart(AssistantMessageStartData),
    AssistantMessageDelta(AssistantMessageDeltaData),
    AssistantTurnEnd(AssistantTurnEndData),
    AssistantUsage(AssistantUsageData),
    ModelCallFailure(ModelCallFailureData),
    Abort(AbortData),
    ToolUserRequested(ToolUserRequestedData),
    ToolExecutionStart(ToolExecutionStartData),
    ToolExecutionPartialResult(ToolExecutionPartialResultData),
    ToolExecutionComplete(ToolExecutionCompleteData),
    ToolExecutionProgress(ToolExecutionProgressData),
    CustomAgentStarted(CustomAgentStartedData),
    CustomAgentCompleted(CustomAgentCompletedData),
    CustomAgentFailed(CustomAgentFailedData),
    CustomAgentSelected(CustomAgentSelectedData),
    SubagentDeselected(SubagentDeselectedData),
    HookStart(HookStartData),
    HookEnd(HookEndData),
    SystemMessage(SystemMessageEventData),
    SystemNotification(SystemNotificationData),
    SessionCompactionStart(SessionCompactionStartData),
    SessionCompactionComplete(SessionCompactionCompleteData),
    SessionShutdown(SessionShutdownData),
    SessionSnapshotRewind(SessionSnapshotRewindData),
    SessionUsageInfo(SessionUsageInfoData),
    SkillInvoked(SkillInvokedData),
    /// External tool requested (protocol v3 broadcast).
    ExternalToolRequested(ExternalToolRequestedData),
    ExternalToolCompleted(ExternalToolCompletedData),
    /// Permission requested (protocol v3 broadcast).
    PermissionRequested(PermissionRequestedData),
    PermissionCompleted(PermissionCompletedData),
    UserInputRequested(UserInputRequestedData),
    UserInputCompleted(UserInputCompletedData),
    ElicitationRequested(ElicitationRequestedData),
    ElicitationCompleted(ElicitationCompletedData),
    SamplingRequested(SamplingRequestedData),
    SamplingCompleted(SamplingCompletedData),
    McpOauthRequired(McpOauthRequiredData),
    McpOauthCompleted(McpOauthCompletedData),
    CustomNotification(CustomNotificationData),
    CommandQueued(CommandQueuedData),
    CommandExecute(CommandExecuteData),
    CommandCompleted(CommandCompletedData),
    AutoModeSwitchRequested(AutoModeSwitchRequestedData),
    AutoModeSwitchCompleted(AutoModeSwitchCompletedData),
    CommandsChanged(CommandsChangedData),
    CapabilitiesChanged(CapabilitiesChangedData),
    ExitPlanModeRequested(ExitPlanModeRequestedData),
    ExitPlanModeCompleted(ExitPlanModeCompletedData),
    SessionToolsUpdated(SessionToolsUpdatedData),
    SessionBackgroundTasksChanged(SessionBackgroundTasksChangedData),
    SessionSkillsLoaded(SessionSkillsLoadedData),
    SessionCustomAgentsUpdated(SessionCustomAgentsUpdatedData),
    SessionMcpServersLoaded(SessionMcpServersLoadedData),
    SessionMcpServerStatusChanged(SessionMcpServerStatusChangedData),
    SessionExtensionsLoaded(SessionExtensionsLoadedData),
    /// Unknown event - preserves raw JSON for forward compatibility.
    Unknown(serde_json::Value),
}

/// Raw session event as received from the CLI.
///
/// The event has common fields (id, timestamp, type) and a data payload
/// that varies based on the event type.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RawSessionEvent {
    pub id: String,
    pub timestamp: String,
    #[serde(rename = "type")]
    pub event_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ephemeral: Option<bool>,
    pub data: serde_json::Value,
}

/// A parsed session event with typed data.
#[derive(Debug, Clone)]
pub struct SessionEvent {
    /// Unique event ID.
    pub id: String,
    /// ISO 8601 timestamp.
    pub timestamp: String,
    /// Original type string (e.g., "assistant.message").
    pub event_type: String,
    /// Parent event ID, if any.
    pub parent_id: Option<String>,
    /// Whether this event is ephemeral.
    pub ephemeral: Option<bool>,
    /// Typed event data.
    pub data: SessionEventData,
}

impl SessionEvent {
    /// Parse a session event from JSON.
    pub fn from_json(json: &serde_json::Value) -> Result<Self, serde_json::Error> {
        let raw: RawSessionEvent = serde_json::from_value(json.clone())?;
        Ok(Self::from_raw(raw))
    }

    /// Convert a raw event to a typed event.
    pub fn from_raw(raw: RawSessionEvent) -> Self {
        let data = parse_event_data(&raw.event_type, raw.data);
        Self {
            id: raw.id,
            timestamp: raw.timestamp,
            event_type: raw.event_type,
            parent_id: raw.parent_id,
            ephemeral: raw.ephemeral,
            data,
        }
    }

    // =========================================================================
    // Type checking helpers
    // =========================================================================

    /// Check if this is an assistant message event.
    pub fn is_assistant_message(&self) -> bool {
        matches!(self.data, SessionEventData::AssistantMessage(_))
    }

    /// Check if this is an assistant message delta event.
    pub fn is_assistant_message_delta(&self) -> bool {
        matches!(self.data, SessionEventData::AssistantMessageDelta(_))
    }

    /// Check if this is a session idle event.
    pub fn is_session_idle(&self) -> bool {
        matches!(self.data, SessionEventData::SessionIdle(_))
    }

    /// Check if this is a session error event.
    pub fn is_session_error(&self) -> bool {
        matches!(self.data, SessionEventData::SessionError(_))
    }

    /// Check if this is a terminal event (session ended).
    pub fn is_terminal(&self) -> bool {
        matches!(
            self.data,
            SessionEventData::SessionIdle(_) | SessionEventData::SessionError(_)
        )
    }

    // =========================================================================
    // Data extraction helpers
    // =========================================================================

    /// Get assistant message data if this is an assistant.message event.
    pub fn as_assistant_message(&self) -> Option<&AssistantMessageData> {
        match &self.data {
            SessionEventData::AssistantMessage(data) => Some(data),
            _ => None,
        }
    }

    /// Get assistant message delta data if this is an assistant.message_delta event.
    pub fn as_assistant_message_delta(&self) -> Option<&AssistantMessageDeltaData> {
        match &self.data {
            SessionEventData::AssistantMessageDelta(data) => Some(data),
            _ => None,
        }
    }

    /// Get session error data if this is a session.error event.
    pub fn as_session_error(&self) -> Option<&SessionErrorData> {
        match &self.data {
            SessionEventData::SessionError(data) => Some(data),
            _ => None,
        }
    }

    /// Get tool execution complete data if this is a tool.execution_complete event.
    pub fn as_tool_execution_complete(&self) -> Option<&ToolExecutionCompleteData> {
        match &self.data {
            SessionEventData::ToolExecutionComplete(data) => Some(data),
            _ => None,
        }
    }

    /// Extract the content from an assistant message or delta.
    pub fn content(&self) -> Option<&str> {
        match &self.data {
            SessionEventData::AssistantMessage(data) => Some(&data.content),
            SessionEventData::AssistantMessageDelta(data) => Some(&data.delta_content),
            _ => None,
        }
    }
}

/// Parse event data based on event type string.
fn parse_event_data(event_type: &str, data: serde_json::Value) -> SessionEventData {
    // Helper macros keep the dispatcher compact and uniform.
    macro_rules! parse_into {
        ($variant:ident) => {
            serde_json::from_value(data)
                .map(SessionEventData::$variant)
                .unwrap_or_else(|_| SessionEventData::Unknown(serde_json::Value::Null))
        };
    }

    match event_type {
        // -- session lifecycle / metadata --
        "session.start" => parse_into!(SessionStart),
        "session.resume" => parse_into!(SessionResume),
        "session.remote_steerable_changed" => parse_into!(SessionRemoteSteerableChanged),
        "session.error" => parse_into!(SessionError),
        "session.idle" => serde_json::from_value(data)
            .map(SessionEventData::SessionIdle)
            .unwrap_or_else(|_| SessionEventData::SessionIdle(SessionIdleData::default())),
        "session.title_changed" => parse_into!(SessionTitleChanged),
        "session.schedule_created" => parse_into!(SessionScheduleCreated),
        "session.schedule_cancelled" => parse_into!(SessionScheduleCancelled),
        "session.info" => parse_into!(SessionInfo),
        "session.warning" => parse_into!(SessionWarning),
        "session.model_change" => parse_into!(SessionModelChange),
        "session.mode_changed" => parse_into!(SessionModeChanged),
        "session.plan_changed" => parse_into!(SessionPlanChanged),
        "session.workspace_file_changed" => parse_into!(SessionWorkspaceFileChanged),
        "session.handoff" => parse_into!(SessionHandoff),
        "session.truncation" => parse_into!(SessionTruncation),
        "session.context_changed" => parse_into!(SessionContextChanged),
        "session.task_complete" => parse_into!(SessionTaskComplete),
        "session.snapshot_rewind" => parse_into!(SessionSnapshotRewind),
        "session.shutdown" => parse_into!(SessionShutdown),
        "session.usage_info" => parse_into!(SessionUsageInfo),
        "session.compaction_start" => {
            SessionEventData::SessionCompactionStart(SessionCompactionStartData {})
        }
        "session.compaction_complete" => parse_into!(SessionCompactionComplete),
        "session.custom_notification" => parse_into!(CustomNotification),
        "session.tools_updated" => parse_into!(SessionToolsUpdated),
        "session.background_tasks_changed" => {
            SessionEventData::SessionBackgroundTasksChanged(SessionBackgroundTasksChangedData {})
        }
        "session.skills_loaded" => parse_into!(SessionSkillsLoaded),
        "session.custom_agents_updated" => parse_into!(SessionCustomAgentsUpdated),
        "session.mcp_servers_loaded" => parse_into!(SessionMcpServersLoaded),
        "session.mcp_server_status_changed" => parse_into!(SessionMcpServerStatusChanged),
        "session.extensions_loaded" => parse_into!(SessionExtensionsLoaded),

        // -- user/assistant turn flow --
        "user.message" => parse_into!(UserMessage),
        "pending_messages.modified" => {
            SessionEventData::PendingMessagesModified(PendingMessagesModifiedData {})
        }
        "assistant.turn_start" => parse_into!(AssistantTurnStart),
        "assistant.intent" => parse_into!(AssistantIntent),
        "assistant.reasoning" => parse_into!(AssistantReasoning),
        "assistant.reasoning_delta" => parse_into!(AssistantReasoningDelta),
        "assistant.streaming_delta" => parse_into!(AssistantStreamingDelta),
        "assistant.message" => parse_into!(AssistantMessage),
        "assistant.message_start" => parse_into!(AssistantMessageStart),
        "assistant.message_delta" => parse_into!(AssistantMessageDelta),
        "assistant.turn_end" => parse_into!(AssistantTurnEnd),
        "assistant.usage" => parse_into!(AssistantUsage),
        "model.call_failure" => parse_into!(ModelCallFailure),
        "abort" => parse_into!(Abort),

        // -- tool/skill execution --
        "tool.user_requested" => parse_into!(ToolUserRequested),
        "tool.execution_start" => parse_into!(ToolExecutionStart),
        "tool.execution_partial_result" => parse_into!(ToolExecutionPartialResult),
        "tool.execution_complete" => parse_into!(ToolExecutionComplete),
        "tool.execution_progress" => parse_into!(ToolExecutionProgress),
        "skill.invoked" => parse_into!(SkillInvoked),

        // Primary wire names (subagent.*) + legacy aliases (custom_agent.*).
        "subagent.started" | "custom_agent.started" => parse_into!(CustomAgentStarted),
        "subagent.completed" | "custom_agent.completed" => parse_into!(CustomAgentCompleted),
        "subagent.failed" | "custom_agent.failed" => parse_into!(CustomAgentFailed),
        "subagent.selected" | "custom_agent.selected" => parse_into!(CustomAgentSelected),
        "subagent.deselected" | "custom_agent.deselected" => {
            SessionEventData::SubagentDeselected(SubagentDeselectedData {})
        }

        // -- hooks / system messages --
        "hook.start" => parse_into!(HookStart),
        "hook.end" => parse_into!(HookEnd),
        "system.message" => parse_into!(SystemMessage),
        "system.notification" => parse_into!(SystemNotification),

        // -- protocol v3 broadcasts and request/response pairs --
        "external_tool.requested" => parse_into!(ExternalToolRequested),
        "external_tool.completed" => parse_into!(ExternalToolCompleted),
        "permission.requested" => parse_into!(PermissionRequested),
        "permission.completed" => parse_into!(PermissionCompleted),
        "user_input.requested" => parse_into!(UserInputRequested),
        "user_input.completed" => parse_into!(UserInputCompleted),
        "elicitation.requested" => parse_into!(ElicitationRequested),
        "elicitation.completed" => parse_into!(ElicitationCompleted),
        "sampling.requested" => parse_into!(SamplingRequested),
        "sampling.completed" => parse_into!(SamplingCompleted),
        "mcp.oauth_required" => parse_into!(McpOauthRequired),
        "mcp.oauth_completed" => parse_into!(McpOauthCompleted),
        "command.queued" => parse_into!(CommandQueued),
        "command.execute" => parse_into!(CommandExecute),
        "command.completed" => parse_into!(CommandCompleted),
        "auto_mode_switch.requested" => parse_into!(AutoModeSwitchRequested),
        "auto_mode_switch.completed" => parse_into!(AutoModeSwitchCompleted),
        "commands.changed" => parse_into!(CommandsChanged),
        "capabilities.changed" => parse_into!(CapabilitiesChanged),
        "exit_plan_mode.requested" => parse_into!(ExitPlanModeRequested),
        "exit_plan_mode.completed" => parse_into!(ExitPlanModeCompleted),

        // Unknown event type - preserve raw data.
        _ => SessionEventData::Unknown(data),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_parse_assistant_message() {
        let json = json!({
            "id": "evt_123",
            "timestamp": "2024-01-15T10:30:00Z",
            "type": "assistant.message",
            "data": {
                "messageId": "msg_456",
                "content": "Hello, world!"
            }
        });

        let event = SessionEvent::from_json(&json).unwrap();
        assert_eq!(event.id, "evt_123");
        assert_eq!(event.event_type, "assistant.message");
        assert!(event.is_assistant_message());

        let msg = event.as_assistant_message().unwrap();
        assert_eq!(msg.message_id, "msg_456");
        assert_eq!(msg.content, "Hello, world!");
    }

    #[test]
    fn test_parse_assistant_message_delta() {
        let json = json!({
            "id": "evt_124",
            "timestamp": "2024-01-15T10:30:01Z",
            "type": "assistant.message_delta",
            "data": {
                "messageId": "msg_456",
                "deltaContent": "Hello"
            }
        });

        let event = SessionEvent::from_json(&json).unwrap();
        assert!(event.is_assistant_message_delta());
        assert_eq!(event.content(), Some("Hello"));
    }

    #[test]
    fn test_parse_session_idle() {
        let json = json!({
            "id": "evt_125",
            "timestamp": "2024-01-15T10:30:02Z",
            "type": "session.idle",
            "data": {}
        });

        let event = SessionEvent::from_json(&json).unwrap();
        assert!(event.is_session_idle());
        assert!(event.is_terminal());
    }

    #[test]
    fn test_parse_external_tool_requested() {
        let json = json!({
            "id": "evt_125b",
            "timestamp": "2024-01-15T10:30:02Z",
            "type": "external_tool.requested",
            "data": {
                "requestId": "req_123",
                "toolName": "echo",
                "toolCallId": "call_456",
                "arguments": {
                    "text": "hello"
                }
            }
        });

        let event = SessionEvent::from_json(&json).unwrap();
        match &event.data {
            SessionEventData::ExternalToolRequested(data) => {
                assert_eq!(data.request_id.as_deref(), Some("req_123"));
                assert_eq!(data.tool_name.as_deref(), Some("echo"));
                assert_eq!(data.tool_call_id.as_deref(), Some("call_456"));
                assert_eq!(data.arguments.as_ref().unwrap()["text"], "hello");
            }
            other => panic!("Expected ExternalToolRequested, got {other:?}"),
        }
    }

    #[test]
    fn test_parse_permission_requested() {
        let json = json!({
            "id": "evt_125c",
            "timestamp": "2024-01-15T10:30:02Z",
            "type": "permission.requested",
            "data": {
                "requestId": "req_789",
                "permissionRequest": {
                    "kind": "tool_execution",
                    "toolCallId": "call_456",
                    "toolName": "shell"
                }
            }
        });

        let event = SessionEvent::from_json(&json).unwrap();
        match &event.data {
            SessionEventData::PermissionRequested(data) => {
                assert_eq!(data.request_id.as_deref(), Some("req_789"));
                assert_eq!(
                    data.permission_request.as_ref().unwrap()["kind"],
                    "tool_execution"
                );
                assert_eq!(
                    data.permission_request.as_ref().unwrap()["toolName"],
                    "shell"
                );
            }
            other => panic!("Expected PermissionRequested, got {other:?}"),
        }
    }

    #[test]
    fn test_parse_session_error() {
        let json = json!({
            "id": "evt_126",
            "timestamp": "2024-01-15T10:30:03Z",
            "type": "session.error",
            "data": {
                "errorType": "api_error",
                "message": "Rate limit exceeded"
            }
        });

        let event = SessionEvent::from_json(&json).unwrap();
        assert!(event.is_session_error());
        assert!(event.is_terminal());

        let err = event.as_session_error().unwrap();
        assert_eq!(err.error_type, "api_error");
        assert_eq!(err.message, "Rate limit exceeded");
    }

    #[test]
    fn test_parse_tool_execution_complete() {
        let json = json!({
            "id": "evt_127",
            "timestamp": "2024-01-15T10:30:04Z",
            "type": "tool.execution_complete",
            "data": {
                "toolCallId": "call_789",
                "success": true,
                "result": {
                    "content": "Tool output"
                }
            }
        });

        let event = SessionEvent::from_json(&json).unwrap();
        let tool = event.as_tool_execution_complete().unwrap();
        assert_eq!(tool.tool_call_id, "call_789");
        assert!(tool.success);
        assert_eq!(tool.result.as_ref().unwrap().content, "Tool output");
    }

    #[test]
    fn test_parse_unknown_event() {
        let json = json!({
            "id": "evt_128",
            "timestamp": "2024-01-15T10:30:05Z",
            "type": "future.unknown_event",
            "data": {
                "someField": "someValue"
            }
        });

        let event = SessionEvent::from_json(&json).unwrap();
        assert_eq!(event.event_type, "future.unknown_event");
        assert!(matches!(event.data, SessionEventData::Unknown(_)));
    }

    #[test]
    fn test_parse_session_start() {
        let json = json!({
            "id": "evt_001",
            "timestamp": "2024-01-15T10:30:00Z",
            "type": "session.start",
            "data": {
                "sessionId": "sess_123",
                "version": 1.0,
                "producer": "copilot-cli",
                "copilotVersion": "1.0.0",
                "startTime": "2024-01-15T10:30:00Z"
            }
        });

        let event = SessionEvent::from_json(&json).unwrap();
        if let SessionEventData::SessionStart(data) = &event.data {
            assert_eq!(data.session_id, "sess_123");
            assert_eq!(data.producer, "copilot-cli");
        } else {
            panic!("Expected SessionStart");
        }
    }

    #[test]
    fn test_event_with_parent_id() {
        let json = json!({
            "id": "evt_129",
            "timestamp": "2024-01-15T10:30:06Z",
            "type": "assistant.message",
            "parentId": "evt_128",
            "ephemeral": true,
            "data": {
                "messageId": "msg_789",
                "content": "Nested message"
            }
        });

        let event = SessionEvent::from_json(&json).unwrap();
        assert_eq!(event.parent_id, Some("evt_128".to_string()));
        assert_eq!(event.ephemeral, Some(true));
    }

    #[test]
    fn test_parse_subagent_started() {
        let json = json!({
            "id": "evt_200",
            "timestamp": "2024-01-15T10:30:00Z",
            "type": "subagent.started",
            "data": {
                "toolCallId": "call_1",
                "agentName": "test-agent",
                "agentDisplayName": "Test Agent",
                "agentDescription": "A test agent"
            }
        });
        let event = SessionEvent::from_json(&json).unwrap();
        assert!(matches!(
            event.data,
            SessionEventData::CustomAgentStarted(_)
        ));
        if let SessionEventData::CustomAgentStarted(data) = &event.data {
            assert_eq!(data.agent_name, "test-agent");
        }
    }

    #[test]
    fn test_parse_subagent_completed_legacy_alias() {
        // Verify legacy custom_agent.* wire names still work
        let json = json!({
            "id": "evt_201",
            "timestamp": "2024-01-15T10:30:00Z",
            "type": "custom_agent.completed",
            "data": {
                "toolCallId": "call_1",
                "agentName": "test-agent"
            }
        });
        let event = SessionEvent::from_json(&json).unwrap();
        assert!(matches!(
            event.data,
            SessionEventData::CustomAgentCompleted(_)
        ));
    }

    #[test]
    fn test_parse_subagent_all_wire_names() {
        for wire_name in &["subagent.failed", "custom_agent.failed"] {
            let json = json!({
                "id": "evt_202",
                "timestamp": "2024-01-15T10:30:00Z",
                "type": wire_name,
                "data": {
                    "toolCallId": "call_1",
                    "agentName": "agent",
                    "error": "boom"
                }
            });
            let event = SessionEvent::from_json(&json).unwrap();
            assert!(
                matches!(event.data, SessionEventData::CustomAgentFailed(_)),
                "Failed to parse {wire_name}"
            );
        }
    }

    #[test]
    fn test_parse_session_compaction_start() {
        let json = json!({
            "id": "evt_300",
            "timestamp": "2024-01-15T10:30:00Z",
            "type": "session.compaction_start",
            "data": {}
        });
        let event = SessionEvent::from_json(&json).unwrap();
        assert!(matches!(
            event.data,
            SessionEventData::SessionCompactionStart(_)
        ));
    }

    #[test]
    fn test_parse_session_compaction_complete() {
        let json = json!({
            "id": "evt_301",
            "timestamp": "2024-01-15T10:30:00Z",
            "type": "session.compaction_complete",
            "data": {
                "success": true,
                "preCompactionTokens": 50000.0,
                "postCompactionTokens": 10000.0,
                "compactionTokensUsed": {
                    "input": 100.0,
                    "output": 200.0,
                    "cachedInput": 50.0
                },
                "summaryContent": "Session was compacted"
            }
        });
        let event = SessionEvent::from_json(&json).unwrap();
        if let SessionEventData::SessionCompactionComplete(data) = &event.data {
            assert!(data.success);
            assert_eq!(data.pre_compaction_tokens, Some(50000.0));
            assert_eq!(data.compaction_tokens_used.as_ref().unwrap().input, 100.0);
        } else {
            panic!("Expected SessionCompactionComplete");
        }
    }

    #[test]
    fn test_parse_session_shutdown() {
        let json = json!({
            "id": "evt_302",
            "timestamp": "2024-01-15T10:30:00Z",
            "type": "session.shutdown",
            "data": {
                "shutdownType": "routine",
                "totalPremiumRequests": 5.0,
                "totalApiDurationMs": 1200.0,
                "sessionStartTime": 1700000000.0,
                "codeChanges": {
                    "linesAdded": 10.0,
                    "linesRemoved": 3.0,
                    "filesModified": ["src/main.rs"]
                },
                "modelMetrics": {},
                "currentModel": "gpt-4"
            }
        });
        let event = SessionEvent::from_json(&json).unwrap();
        if let SessionEventData::SessionShutdown(data) = &event.data {
            assert_eq!(data.shutdown_type, ShutdownType::Routine);
            assert_eq!(data.current_model, Some("gpt-4".to_string()));
            assert_eq!(data.code_changes.lines_added, 10.0);
        } else {
            panic!("Expected SessionShutdown");
        }
    }

    #[test]
    fn test_parse_session_snapshot_rewind() {
        let json = json!({
            "id": "evt_303",
            "timestamp": "2024-01-15T10:30:00Z",
            "type": "session.snapshot_rewind",
            "data": {
                "upToEventId": "evt_100",
                "eventsRemoved": 5.0
            }
        });
        let event = SessionEvent::from_json(&json).unwrap();
        if let SessionEventData::SessionSnapshotRewind(data) = &event.data {
            assert_eq!(data.up_to_event_id, "evt_100");
            assert_eq!(data.events_removed, 5.0);
        } else {
            panic!("Expected SessionSnapshotRewind");
        }
    }

    #[test]
    fn test_parse_session_usage_info() {
        let json = json!({
            "id": "evt_304",
            "timestamp": "2024-01-15T10:30:00Z",
            "type": "session.usage_info",
            "data": {
                "tokenLimit": 100000.0,
                "currentTokens": 50000.0,
                "messagesLength": 42.0
            }
        });
        let event = SessionEvent::from_json(&json).unwrap();
        if let SessionEventData::SessionUsageInfo(data) = &event.data {
            assert_eq!(data.token_limit, 100000.0);
            assert_eq!(data.current_tokens, 50000.0);
        } else {
            panic!("Expected SessionUsageInfo");
        }
    }

    #[test]
    fn test_parse_tool_execution_progress() {
        let json = json!({
            "id": "evt_305",
            "timestamp": "2024-01-15T10:30:00Z",
            "type": "tool.execution_progress",
            "data": {
                "toolCallId": "call_100",
                "progressMessage": "Processing file 3 of 10..."
            }
        });
        let event = SessionEvent::from_json(&json).unwrap();
        if let SessionEventData::ToolExecutionProgress(data) = &event.data {
            assert_eq!(data.tool_call_id, "call_100");
            assert_eq!(data.progress_message, "Processing file 3 of 10...");
        } else {
            panic!("Expected ToolExecutionProgress");
        }
    }

    #[test]
    fn test_parse_skill_invoked() {
        let json = json!({
            "id": "evt_306",
            "timestamp": "2024-01-15T10:30:00Z",
            "type": "skill.invoked",
            "data": {
                "name": "code-review",
                "path": "/skills/code-review",
                "content": "Review this code",
                "allowedTools": ["read_file", "search"]
            }
        });
        let event = SessionEvent::from_json(&json).unwrap();
        if let SessionEventData::SkillInvoked(data) = &event.data {
            assert_eq!(data.name, "code-review");
            assert_eq!(data.allowed_tools.as_ref().unwrap().len(), 2);
        } else {
            panic!("Expected SkillInvoked");
        }
    }

    #[test]
    fn test_session_error_with_code_and_provider_call_id() {
        let json = json!({
            "id": "evt_err",
            "timestamp": "2024-01-15T10:30:00Z",
            "type": "session.error",
            "data": {
                "errorType": "provider_error",
                "message": "Rate limited",
                "code": 429.0,
                "providerCallId": "call-abc-123"
            }
        });
        let event = SessionEvent::from_json(&json).unwrap();
        if let SessionEventData::SessionError(data) = &event.data {
            assert_eq!(data.error_type, "provider_error");
            assert_eq!(data.code, Some(429.0));
            assert_eq!(data.provider_call_id.as_deref(), Some("call-abc-123"));
        } else {
            panic!("Expected SessionError");
        }
    }

    #[test]
    fn test_tool_execution_complete_with_mcp_fields() {
        let json = json!({
            "id": "evt_mcp",
            "timestamp": "2024-01-15T10:30:00Z",
            "type": "tool.execution_complete",
            "data": {
                "toolCallId": "call-1",
                "success": true,
                "mcpServerName": "my-server",
                "mcpToolName": "read_file"
            }
        });
        let event = SessionEvent::from_json(&json).unwrap();
        if let SessionEventData::ToolExecutionComplete(data) = &event.data {
            assert_eq!(data.mcp_server_name.as_deref(), Some("my-server"));
            assert_eq!(data.mcp_tool_name.as_deref(), Some("read_file"));
        } else {
            panic!("Expected ToolExecutionComplete");
        }
    }

    #[test]
    fn test_session_start_data_optional_fields() {
        // All fields missing should still parse with defaults
        let json = json!({
            "id": "evt_start",
            "timestamp": "2024-01-15T10:30:00Z",
            "type": "session.start",
            "data": {}
        });
        let event = SessionEvent::from_json(&json).unwrap();
        if let SessionEventData::SessionStart(data) = &event.data {
            assert_eq!(data.session_id, "");
            assert_eq!(data.version, 0.0);
            assert_eq!(data.producer, "");
        } else {
            panic!("Expected SessionStart");
        }
    }

    #[test]
    fn test_unknown_event_type_handled_gracefully() {
        let json = json!({
            "id": "evt_unknown",
            "timestamp": "2025-01-01T00:00:00Z",
            "type": "some.future.event.type",
            "data": {"someField": "someValue"}
        });
        // Parsing an unknown event type should not panic
        let raw: RawSessionEvent = serde_json::from_value(json.clone()).unwrap();
        assert_eq!(raw.event_type, "some.future.event.type");

        // It should also parse into a SessionEvent with Unknown data
        let event = SessionEvent::from_json(&json).unwrap();
        assert_eq!(event.event_type, "some.future.event.type");
        assert!(matches!(event.data, SessionEventData::Unknown(_)));
    }

    #[test]
    fn test_session_shutdown_event_parsed() {
        let json = json!({
            "id": "evt_shutdown",
            "timestamp": "2025-01-01T00:00:00Z",
            "type": "session.shutdown",
            "data": {
                "shutdownType": "routine",
                "reason": "user requested"
            }
        });
        let raw: RawSessionEvent = serde_json::from_value(json.clone()).unwrap();
        assert_eq!(raw.event_type, "session.shutdown");

        let event = SessionEvent::from_json(&json).unwrap();
        assert_eq!(event.event_type, "session.shutdown");
    }

    #[test]
    fn test_session_usage_info_recognized() {
        let json = json!({
            "id": "evt_usage",
            "timestamp": "2025-01-01T00:00:00Z",
            "type": "session.usage_info",
            "data": {}
        });
        let raw: RawSessionEvent = serde_json::from_value(json.clone()).unwrap();
        assert_eq!(raw.event_type, "session.usage_info");

        let event = SessionEvent::from_json(&json).unwrap();
        assert_eq!(event.event_type, "session.usage_info");
    }

    // =========================================================================
    // Tests for newly-added event variants (upstream parity batch)
    // =========================================================================

    fn make_event(event_type: &str, data: serde_json::Value) -> SessionEvent {
        let json = json!({
            "id": format!("evt_{event_type}"),
            "timestamp": "2025-01-01T00:00:00Z",
            "type": event_type,
            "data": data,
        });
        SessionEvent::from_json(&json).unwrap()
    }

    #[test]
    fn test_parse_session_remote_steerable_changed() {
        let event = make_event(
            "session.remote_steerable_changed",
            json!({ "remoteSteerable": true }),
        );
        match event.data {
            SessionEventData::SessionRemoteSteerableChanged(d) => assert!(d.remote_steerable),
            other => panic!("Expected SessionRemoteSteerableChanged, got {other:?}"),
        }
    }

    #[test]
    fn test_parse_session_title_changed() {
        let event = make_event("session.title_changed", json!({ "title": "Hello" }));
        match event.data {
            SessionEventData::SessionTitleChanged(d) => assert_eq!(d.title, "Hello"),
            other => panic!("Expected SessionTitleChanged, got {other:?}"),
        }
    }

    #[test]
    fn test_parse_session_schedule_created_and_cancelled() {
        let created = make_event(
            "session.schedule_created",
            json!({ "id": 1, "prompt": "/echo", "intervalMs": 1000, "recurring": true }),
        );
        match created.data {
            SessionEventData::SessionScheduleCreated(d) => {
                assert_eq!(d.id, 1.0);
                assert_eq!(d.prompt, "/echo");
                assert_eq!(d.interval_ms, 1000.0);
                assert_eq!(d.recurring, Some(true));
            }
            other => panic!("Expected SessionScheduleCreated, got {other:?}"),
        }
        let cancelled = make_event("session.schedule_cancelled", json!({ "id": 1 }));
        assert!(matches!(
            cancelled.data,
            SessionEventData::SessionScheduleCancelled(_)
        ));
    }

    #[test]
    fn test_parse_session_warning() {
        let event = make_event(
            "session.warning",
            json!({ "warningType": "policy", "message": "heads up" }),
        );
        match event.data {
            SessionEventData::SessionWarning(d) => {
                assert_eq!(d.warning_type, "policy");
                assert_eq!(d.message, "heads up");
            }
            other => panic!("Expected SessionWarning, got {other:?}"),
        }
    }

    #[test]
    fn test_parse_session_mode_changed() {
        let event = make_event(
            "session.mode_changed",
            json!({ "previousMode": "interactive", "newMode": "plan" }),
        );
        match event.data {
            SessionEventData::SessionModeChanged(d) => {
                assert_eq!(d.previous_mode, "interactive");
                assert_eq!(d.new_mode, "plan");
            }
            other => panic!("Expected SessionModeChanged, got {other:?}"),
        }
    }

    #[test]
    fn test_parse_session_plan_changed() {
        let event = make_event("session.plan_changed", json!({ "operation": "create" }));
        match event.data {
            SessionEventData::SessionPlanChanged(d) => {
                assert_eq!(d.operation, PlanChangedOperation::Create);
            }
            other => panic!("Expected SessionPlanChanged, got {other:?}"),
        }
    }

    #[test]
    fn test_parse_session_workspace_file_changed() {
        let event = make_event(
            "session.workspace_file_changed",
            json!({ "operation": "update", "path": "notes.md" }),
        );
        match event.data {
            SessionEventData::SessionWorkspaceFileChanged(d) => {
                assert_eq!(d.operation, WorkspaceFileChangedOperation::Update);
                assert_eq!(d.path, "notes.md");
            }
            other => panic!("Expected SessionWorkspaceFileChanged, got {other:?}"),
        }
    }

    #[test]
    fn test_parse_session_context_changed() {
        let event = make_event(
            "session.context_changed",
            json!({
                "cwd": "/repo",
                "branch": "main",
                "hostType": "github",
                "repository": "owner/name"
            }),
        );
        match event.data {
            SessionEventData::SessionContextChanged(ctx) => {
                assert_eq!(ctx.cwd, "/repo");
                assert_eq!(ctx.branch.as_deref(), Some("main"));
                assert_eq!(ctx.host_type, Some(WorkingDirectoryHostType::Github));
                assert_eq!(ctx.repository.as_deref(), Some("owner/name"));
            }
            other => panic!("Expected SessionContextChanged, got {other:?}"),
        }
    }

    #[test]
    fn test_parse_session_task_complete() {
        let event = make_event(
            "session.task_complete",
            json!({ "success": true, "summary": "done" }),
        );
        match event.data {
            SessionEventData::SessionTaskComplete(d) => {
                assert_eq!(d.success, Some(true));
                assert_eq!(d.summary.as_deref(), Some("done"));
            }
            other => panic!("Expected SessionTaskComplete, got {other:?}"),
        }
    }

    #[test]
    fn test_parse_session_info_with_tip_url() {
        let event = make_event(
            "session.info",
            json!({
                "infoType": "notification",
                "message": "hi",
                "tip": "press tab",
                "url": "https://example.com"
            }),
        );
        match event.data {
            SessionEventData::SessionInfo(d) => {
                assert_eq!(d.tip.as_deref(), Some("press tab"));
                assert_eq!(d.url.as_deref(), Some("https://example.com"));
            }
            other => panic!("Expected SessionInfo, got {other:?}"),
        }
    }

    #[test]
    fn test_parse_assistant_streaming_delta() {
        let event = make_event(
            "assistant.streaming_delta",
            json!({ "totalResponseSizeBytes": 1234 }),
        );
        match event.data {
            SessionEventData::AssistantStreamingDelta(d) => {
                assert_eq!(d.total_response_size_bytes, 1234.0);
            }
            other => panic!("Expected AssistantStreamingDelta, got {other:?}"),
        }
    }

    #[test]
    fn test_parse_assistant_message_start() {
        let event = make_event(
            "assistant.message_start",
            json!({ "messageId": "msg_1", "phase": "thinking" }),
        );
        match event.data {
            SessionEventData::AssistantMessageStart(d) => {
                assert_eq!(d.message_id, "msg_1");
                assert_eq!(d.phase.as_deref(), Some("thinking"));
            }
            other => panic!("Expected AssistantMessageStart, got {other:?}"),
        }
    }

    #[test]
    fn test_parse_model_call_failure() {
        let event = make_event(
            "model.call_failure",
            json!({
                "source": "top_level",
                "model": "gpt-4",
                "statusCode": 500,
                "errorMessage": "boom"
            }),
        );
        match event.data {
            SessionEventData::ModelCallFailure(d) => {
                assert_eq!(d.source, ModelCallFailureSource::TopLevel);
                assert_eq!(d.model.as_deref(), Some("gpt-4"));
                assert_eq!(d.status_code, Some(500.0));
            }
            other => panic!("Expected ModelCallFailure, got {other:?}"),
        }
    }

    #[test]
    fn test_parse_subagent_deselected() {
        let event = make_event("subagent.deselected", json!({}));
        assert!(matches!(
            event.data,
            SessionEventData::SubagentDeselected(_)
        ));
    }

    #[test]
    fn test_parse_system_notification() {
        let event = make_event(
            "system.notification",
            json!({
                "content": "<system_notification>...</system_notification>",
                "kind": { "type": "agent_completed", "agentId": "a1", "agentType": "explore", "status": "completed" }
            }),
        );
        match event.data {
            SessionEventData::SystemNotification(d) => {
                assert!(d.content.contains("system_notification"));
                assert_eq!(d.kind["type"], "agent_completed");
            }
            other => panic!("Expected SystemNotification, got {other:?}"),
        }
    }

    #[test]
    fn test_parse_permission_completed() {
        let event = make_event(
            "permission.completed",
            json!({
                "requestId": "req_1",
                "result": { "kind": "approved" },
                "toolCallId": "call_1"
            }),
        );
        match event.data {
            SessionEventData::PermissionCompleted(d) => {
                assert_eq!(d.request_id, "req_1");
                assert_eq!(d.result["kind"], "approved");
                assert_eq!(d.tool_call_id.as_deref(), Some("call_1"));
            }
            other => panic!("Expected PermissionCompleted, got {other:?}"),
        }
    }

    #[test]
    fn test_parse_user_input_requested_and_completed() {
        let req = make_event(
            "user_input.requested",
            json!({
                "requestId": "ui_1",
                "question": "yes?",
                "choices": ["yes", "no"],
                "allowFreeform": true
            }),
        );
        match req.data {
            SessionEventData::UserInputRequested(d) => {
                assert_eq!(d.question, "yes?");
                assert_eq!(d.choices.as_ref().unwrap().len(), 2);
                assert_eq!(d.allow_freeform, Some(true));
            }
            other => panic!("Expected UserInputRequested, got {other:?}"),
        }
        let cmp = make_event(
            "user_input.completed",
            json!({ "requestId": "ui_1", "answer": "yes", "wasFreeform": false }),
        );
        match cmp.data {
            SessionEventData::UserInputCompleted(d) => {
                assert_eq!(d.answer.as_deref(), Some("yes"));
            }
            other => panic!("Expected UserInputCompleted, got {other:?}"),
        }
    }

    #[test]
    fn test_parse_elicitation_requested_and_completed() {
        let req = make_event(
            "elicitation.requested",
            json!({
                "requestId": "e1",
                "message": "fill it out",
                "mode": "form",
                "requestedSchema": { "type": "object", "properties": { "name": { "type": "string" } } }
            }),
        );
        match req.data {
            SessionEventData::ElicitationRequested(d) => {
                assert_eq!(d.mode, Some(ElicitationRequestedMode::Form));
                assert!(d.requested_schema.is_some());
            }
            other => panic!("Expected ElicitationRequested, got {other:?}"),
        }
        let cmp = make_event(
            "elicitation.completed",
            json!({ "requestId": "e1", "action": "accept", "content": { "name": "alice" } }),
        );
        match cmp.data {
            SessionEventData::ElicitationCompleted(d) => {
                assert_eq!(d.action, Some(ElicitationCompletedAction::Accept));
                assert_eq!(d.content.as_ref().unwrap()["name"], "alice");
            }
            other => panic!("Expected ElicitationCompleted, got {other:?}"),
        }
    }

    #[test]
    fn test_parse_sampling_requested_and_completed() {
        let req = make_event(
            "sampling.requested",
            json!({ "requestId": "s1", "serverName": "srv", "mcpRequestId": 7 }),
        );
        match req.data {
            SessionEventData::SamplingRequested(d) => {
                assert_eq!(d.server_name, "srv");
                assert_eq!(d.mcp_request_id, 7);
            }
            other => panic!("Expected SamplingRequested, got {other:?}"),
        }
        let cmp = make_event("sampling.completed", json!({ "requestId": "s1" }));
        assert!(matches!(cmp.data, SessionEventData::SamplingCompleted(_)));
    }

    #[test]
    fn test_parse_mcp_oauth_required_and_completed() {
        let req = make_event(
            "mcp.oauth_required",
            json!({
                "requestId": "o1",
                "serverName": "srv",
                "serverUrl": "https://srv.example",
                "staticClientConfig": { "clientId": "cid" }
            }),
        );
        match req.data {
            SessionEventData::McpOauthRequired(d) => {
                assert_eq!(d.server_name, "srv");
                assert_eq!(d.static_client_config.as_ref().unwrap().client_id, "cid");
            }
            other => panic!("Expected McpOauthRequired, got {other:?}"),
        }
        let cmp = make_event("mcp.oauth_completed", json!({ "requestId": "o1" }));
        assert!(matches!(cmp.data, SessionEventData::McpOauthCompleted(_)));
    }

    #[test]
    fn test_parse_custom_notification() {
        let event = make_event(
            "session.custom_notification",
            json!({
                "source": "my-ext",
                "name": "ping",
                "payload": { "x": 1 },
                "version": 2
            }),
        );
        match event.data {
            SessionEventData::CustomNotification(d) => {
                assert_eq!(d.source, "my-ext");
                assert_eq!(d.payload["x"], 1);
                assert_eq!(d.version, Some(2.0));
            }
            other => panic!("Expected CustomNotification, got {other:?}"),
        }
    }

    #[test]
    fn test_parse_external_tool_completed() {
        let event = make_event("external_tool.completed", json!({ "requestId": "x1" }));
        match event.data {
            SessionEventData::ExternalToolCompleted(d) => assert_eq!(d.request_id, "x1"),
            other => panic!("Expected ExternalToolCompleted, got {other:?}"),
        }
    }

    #[test]
    fn test_parse_command_lifecycle() {
        let queued = make_event(
            "command.queued",
            json!({ "requestId": "c1", "command": "/help" }),
        );
        assert!(matches!(queued.data, SessionEventData::CommandQueued(_)));
        let exec = make_event(
            "command.execute",
            json!({
                "requestId": "c2",
                "command": "/deploy prod",
                "commandName": "deploy",
                "args": "prod"
            }),
        );
        match exec.data {
            SessionEventData::CommandExecute(d) => {
                assert_eq!(d.command_name, "deploy");
                assert_eq!(d.args, "prod");
            }
            other => panic!("Expected CommandExecute, got {other:?}"),
        }
        let done = make_event("command.completed", json!({ "requestId": "c1" }));
        assert!(matches!(done.data, SessionEventData::CommandCompleted(_)));
    }

    #[test]
    fn test_parse_auto_mode_switch() {
        let req = make_event(
            "auto_mode_switch.requested",
            json!({ "requestId": "a1", "errorCode": "user_global_rate_limited", "retryAfterSeconds": 60 }),
        );
        match req.data {
            SessionEventData::AutoModeSwitchRequested(d) => {
                assert_eq!(d.retry_after_seconds, Some(60.0));
            }
            other => panic!("Expected AutoModeSwitchRequested, got {other:?}"),
        }
        let cmp = make_event(
            "auto_mode_switch.completed",
            json!({ "requestId": "a1", "response": "yes" }),
        );
        match cmp.data {
            SessionEventData::AutoModeSwitchCompleted(d) => assert_eq!(d.response, "yes"),
            other => panic!("Expected AutoModeSwitchCompleted, got {other:?}"),
        }
    }

    #[test]
    fn test_parse_commands_changed() {
        let event = make_event(
            "commands.changed",
            json!({ "commands": [{ "name": "deploy", "description": "ship it" }] }),
        );
        match event.data {
            SessionEventData::CommandsChanged(d) => {
                assert_eq!(d.commands.len(), 1);
                assert_eq!(d.commands[0].name, "deploy");
                assert_eq!(d.commands[0].description.as_deref(), Some("ship it"));
            }
            other => panic!("Expected CommandsChanged, got {other:?}"),
        }
    }

    #[test]
    fn test_parse_capabilities_changed() {
        let event = make_event(
            "capabilities.changed",
            json!({ "ui": { "elicitation": true } }),
        );
        match event.data {
            SessionEventData::CapabilitiesChanged(d) => {
                assert_eq!(d.ui.unwrap().elicitation, Some(true));
            }
            other => panic!("Expected CapabilitiesChanged, got {other:?}"),
        }
    }

    #[test]
    fn test_parse_exit_plan_mode_requested_and_completed() {
        let req = make_event(
            "exit_plan_mode.requested",
            json!({
                "requestId": "p1",
                "summary": "build it",
                "planContent": "## plan",
                "actions": ["approve", "edit"],
                "recommendedAction": "approve"
            }),
        );
        match req.data {
            SessionEventData::ExitPlanModeRequested(d) => {
                assert_eq!(d.recommended_action, "approve");
                assert_eq!(d.actions.len(), 2);
            }
            other => panic!("Expected ExitPlanModeRequested, got {other:?}"),
        }
        let cmp = make_event(
            "exit_plan_mode.completed",
            json!({ "requestId": "p1", "approved": true, "selectedAction": "autopilot" }),
        );
        match cmp.data {
            SessionEventData::ExitPlanModeCompleted(d) => {
                assert_eq!(d.approved, Some(true));
                assert_eq!(d.selected_action.as_deref(), Some("autopilot"));
            }
            other => panic!("Expected ExitPlanModeCompleted, got {other:?}"),
        }
    }

    #[test]
    fn test_parse_session_tools_updated_and_background_tasks() {
        let tu = make_event("session.tools_updated", json!({ "model": "gpt-4" }));
        match tu.data {
            SessionEventData::SessionToolsUpdated(d) => assert_eq!(d.model, "gpt-4"),
            other => panic!("Expected SessionToolsUpdated, got {other:?}"),
        }
        let bt = make_event("session.background_tasks_changed", json!({}));
        assert!(matches!(
            bt.data,
            SessionEventData::SessionBackgroundTasksChanged(_)
        ));
    }

    #[test]
    fn test_parse_session_skills_loaded() {
        let event = make_event(
            "session.skills_loaded",
            json!({
                "skills": [
                    {
                        "name": "code-review",
                        "description": "review code",
                        "source": "project",
                        "enabled": true,
                        "userInvocable": false
                    }
                ]
            }),
        );
        match event.data {
            SessionEventData::SessionSkillsLoaded(d) => {
                assert_eq!(d.skills.len(), 1);
                assert_eq!(d.skills[0].name, "code-review");
                assert!(d.skills[0].enabled);
            }
            other => panic!("Expected SessionSkillsLoaded, got {other:?}"),
        }
    }

    #[test]
    fn test_parse_session_custom_agents_updated() {
        let event = make_event(
            "session.custom_agents_updated",
            json!({
                "agents": [{
                    "id": "agent.1",
                    "name": "internal",
                    "displayName": "My Agent",
                    "description": "does things",
                    "source": "project",
                    "userInvocable": true,
                    "tools": ["read_file"]
                }],
                "errors": [],
                "warnings": ["minor"]
            }),
        );
        match event.data {
            SessionEventData::SessionCustomAgentsUpdated(d) => {
                assert_eq!(d.agents.len(), 1);
                assert_eq!(d.agents[0].id, "agent.1");
                assert_eq!(d.warnings.len(), 1);
            }
            other => panic!("Expected SessionCustomAgentsUpdated, got {other:?}"),
        }
    }

    #[test]
    fn test_parse_session_mcp_servers_loaded_and_status_changed() {
        let loaded = make_event(
            "session.mcp_servers_loaded",
            json!({
                "servers": [
                    { "name": "fs", "status": "connected" },
                    { "name": "auth", "status": "needs-auth", "error": "401" }
                ]
            }),
        );
        match loaded.data {
            SessionEventData::SessionMcpServersLoaded(d) => {
                assert_eq!(d.servers.len(), 2);
                assert_eq!(d.servers[0].status, McpServerStatus::Connected);
                assert_eq!(d.servers[1].status, McpServerStatus::NeedsAuth);
            }
            other => panic!("Expected SessionMcpServersLoaded, got {other:?}"),
        }
        let changed = make_event(
            "session.mcp_server_status_changed",
            json!({ "serverName": "fs", "status": "failed" }),
        );
        match changed.data {
            SessionEventData::SessionMcpServerStatusChanged(d) => {
                assert_eq!(d.server_name, "fs");
                assert_eq!(d.status, McpServerStatus::Failed);
            }
            other => panic!("Expected SessionMcpServerStatusChanged, got {other:?}"),
        }
    }

    #[test]
    fn test_parse_session_extensions_loaded() {
        let event = make_event(
            "session.extensions_loaded",
            json!({
                "extensions": [
                    { "id": "project:my-ext", "name": "my-ext", "source": "project", "status": "running" }
                ]
            }),
        );
        match event.data {
            SessionEventData::SessionExtensionsLoaded(d) => {
                assert_eq!(d.extensions.len(), 1);
                assert_eq!(d.extensions[0].source, ExtensionSource::Project);
                assert_eq!(d.extensions[0].status, ExtensionStatus::Running);
            }
            other => panic!("Expected SessionExtensionsLoaded, got {other:?}"),
        }
    }

    #[test]
    fn test_parse_session_idle_with_aborted() {
        let event = make_event("session.idle", json!({ "aborted": true }));
        match event.data {
            SessionEventData::SessionIdle(d) => assert_eq!(d.aborted, Some(true)),
            other => panic!("Expected SessionIdle, got {other:?}"),
        }
    }

    #[test]
    fn test_unknown_fallback_still_works_for_truly_unknown_types() {
        let event = make_event("some.totally.new.thing", json!({ "foo": "bar" }));
        match event.data {
            SessionEventData::Unknown(v) => assert_eq!(v["foo"], "bar"),
            other => panic!("Expected Unknown, got {other:?}"),
        }
    }
}
