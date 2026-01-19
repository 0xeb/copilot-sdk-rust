// Copyright (c) 2026 Elias Bachaalany
// SPDX-License-Identifier: MIT

//! Core types for the Copilot SDK.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

// =============================================================================
// Protocol Version
// =============================================================================

/// SDK protocol version - must match copilot-agent-runtime server.
pub const SDK_PROTOCOL_VERSION: u32 = 1;

// =============================================================================
// Enums
// =============================================================================

/// Connection state of the client.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ConnectionState {
    #[default]
    Disconnected,
    Connecting,
    Connected,
    Error,
}

/// System message mode for session configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SystemMessageMode {
    Append,
    Replace,
}

/// Attachment type for user messages.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AttachmentType {
    File,
    Directory,
}

/// Log level for the CLI.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LogLevel {
    Debug,
    #[default]
    Info,
    Warn,
    Error,
}

impl std::fmt::Display for LogLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LogLevel::Debug => write!(f, "debug"),
            LogLevel::Info => write!(f, "info"),
            LogLevel::Warn => write!(f, "warn"),
            LogLevel::Error => write!(f, "error"),
        }
    }
}

// =============================================================================
// Tool Types
// =============================================================================

/// Binary result from a tool execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolBinaryResult {
    pub data: String,
    pub mime_type: String,
    #[serde(rename = "type")]
    pub result_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

/// Result object returned from tool execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolResultObject {
    pub text_result_for_llm: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub binary_results_for_llm: Option<Vec<ToolBinaryResult>>,
    #[serde(default = "default_result_type")]
    pub result_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_log: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_telemetry: Option<HashMap<String, serde_json::Value>>,
}

fn default_result_type() -> String {
    "success".to_string()
}

impl ToolResultObject {
    /// Create a success result with text.
    pub fn text(result: impl Into<String>) -> Self {
        Self {
            text_result_for_llm: result.into(),
            binary_results_for_llm: None,
            result_type: "success".to_string(),
            error: None,
            session_log: None,
            tool_telemetry: None,
        }
    }

    /// Create an error result.
    pub fn error(message: impl Into<String>) -> Self {
        Self {
            text_result_for_llm: String::new(),
            binary_results_for_llm: None,
            result_type: "error".to_string(),
            error: Some(message.into()),
            session_log: None,
            tool_telemetry: None,
        }
    }
}

/// Convenient alias for tool results.
pub type ToolResult = ToolResultObject;

/// Information about a tool invocation from the server.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolInvocation {
    pub session_id: String,
    pub tool_call_id: String,
    pub tool_name: String,
    #[serde(default)]
    pub arguments: Option<serde_json::Value>,
}

impl ToolInvocation {
    /// Get an argument by name, deserializing to the specified type.
    pub fn arg<T: serde::de::DeserializeOwned>(&self, name: &str) -> crate::Result<T> {
        let args = self
            .arguments
            .as_ref()
            .ok_or_else(|| crate::CopilotError::ToolError("No arguments provided".into()))?;

        let value = args
            .get(name)
            .ok_or_else(|| crate::CopilotError::ToolError(format!("Missing argument: {}", name)))?;

        serde_json::from_value(value.clone()).map_err(|e| {
            crate::CopilotError::ToolError(format!("Invalid argument '{}': {}", name, e))
        })
    }
}

// =============================================================================
// Permission Types
// =============================================================================

/// Permission request from the server.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PermissionRequest {
    pub kind: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(flatten)]
    pub extension_data: HashMap<String, serde_json::Value>,
}

/// Result of a permission request (response to CLI).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PermissionRequestResult {
    pub kind: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rules: Option<Vec<serde_json::Value>>,
}

impl PermissionRequestResult {
    /// Create an approved permission result.
    pub fn approved() -> Self {
        Self {
            kind: "approved".to_string(),
            rules: None,
        }
    }

    /// Create a denied permission result.
    pub fn denied() -> Self {
        Self {
            kind: "denied-no-approval-rule-and-could-not-request-from-user".to_string(),
            rules: None,
        }
    }
}

// =============================================================================
// Configuration Types
// =============================================================================

/// System message configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SystemMessageConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mode: Option<SystemMessageMode>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

/// Azure-specific provider options.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AzureOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_version: Option<String>,
}

/// Provider configuration for BYOK (Bring Your Own Key).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProviderConfig {
    pub base_url: String,
    #[serde(skip_serializing_if = "Option::is_none", rename = "type")]
    pub provider_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub wire_api: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bearer_token: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub azure: Option<AzureOptions>,
}

// =============================================================================
// MCP Server Configuration
// =============================================================================

/// Configuration for a local/stdio MCP server.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct McpLocalServerConfig {
    pub tools: Vec<String>,
    pub command: String,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "type")]
    pub server_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timeout: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub env: Option<HashMap<String, String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cwd: Option<String>,
}

/// Configuration for a remote MCP server (HTTP or SSE).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct McpRemoteServerConfig {
    pub tools: Vec<String>,
    pub url: String,
    #[serde(default = "default_mcp_type", rename = "type")]
    pub server_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timeout: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub headers: Option<HashMap<String, String>>,
}

fn default_mcp_type() -> String {
    "http".to_string()
}

/// MCP server configuration (either local or remote).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum McpServerConfig {
    Local(McpLocalServerConfig),
    Remote(McpRemoteServerConfig),
}

// =============================================================================
// Custom Agent Configuration
// =============================================================================

/// Configuration for a custom agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CustomAgentConfig {
    pub name: String,
    pub prompt: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub display_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mcp_servers: Option<HashMap<String, serde_json::Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub infer: Option<bool>,
}

// =============================================================================
// Attachment Types
// =============================================================================

/// Attachment item for user messages.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UserMessageAttachment {
    #[serde(rename = "type")]
    pub attachment_type: AttachmentType,
    pub path: String,
    pub display_name: String,
}

// =============================================================================
// Tool Definition (SDK-side)
// =============================================================================

/// Tool definition for registration with a session.
///
/// Use the builder pattern to create tools:
/// ```no_run
/// use copilot_sdk::{Client, SessionConfig, Tool, ToolHandler, ToolResultObject};
/// use std::sync::Arc;
///
/// #[tokio::main]
/// async fn main() -> copilot_sdk::Result<()> {
/// let client = Client::builder().build()?;
/// client.start().await?;
///
/// let tool = Tool::new("get_weather")
///     .description("Get weather for a city")
///     .schema(serde_json::json!({
///         "type": "object",
///         "properties": { "city": { "type": "string" } },
///         "required": ["city"]
///     }));
///
/// let session = client.create_session(SessionConfig {
///     tools: vec![tool.clone()],
///     ..Default::default()
/// }).await?;
///
/// let handler: ToolHandler = Arc::new(|_name, args| {
///     let city = args.get("city").and_then(|v| v.as_str()).unwrap_or("unknown");
///     ToolResultObject::text(format!("Weather in {}: sunny", city))
/// });
/// session.register_tool_with_handler(tool, Some(handler)).await;
/// client.stop().await?;
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub struct Tool {
    pub name: String,
    pub description: String,
    pub parameters_schema: serde_json::Value,
    // Handler is stored separately in Session since it's not Clone-friendly
}

impl Tool {
    /// Create a new tool with the given name.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: String::new(),
            parameters_schema: serde_json::json!({}),
        }
    }

    /// Set the tool description.
    pub fn description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    /// Set the parameters JSON schema.
    pub fn schema(mut self, schema: serde_json::Value) -> Self {
        self.parameters_schema = schema;
        self
    }

    /// Derive the parameters JSON schema from a Rust type (requires the `schemars` feature).
    #[cfg(feature = "schemars")]
    pub fn typed_schema<T: schemars::JsonSchema>(mut self) -> Self {
        let schema = schemars::schema_for!(T);
        match serde_json::to_value(&schema) {
            Ok(value) => self.parameters_schema = value,
            Err(err) => {
                tracing::warn!("Failed to serialize schemars schema: {err}");
                self.parameters_schema = serde_json::json!({});
            }
        }
        self
    }
}

impl std::fmt::Debug for Tool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tool")
            .field("name", &self.name)
            .field("description", &self.description)
            .finish()
    }
}

// Serialization for sending tool definitions to the CLI
impl Serialize for Tool {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("Tool", 3)?;
        state.serialize_field("name", &self.name)?;
        state.serialize_field("description", &self.description)?;
        state.serialize_field("parameters", &self.parameters_schema)?;
        state.end()
    }
}

// =============================================================================
// Session Configuration
// =============================================================================

/// Configuration for creating a new session.
#[derive(Debug, Clone, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<Tool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_message: Option<SystemMessageConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub available_tools: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub excluded_tools: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider: Option<ProviderConfig>,
    #[serde(skip)]
    pub streaming: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mcp_servers: Option<HashMap<String, serde_json::Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub custom_agents: Option<Vec<CustomAgentConfig>>,
}

/// Configuration for resuming an existing session.
#[derive(Debug, Clone, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ResumeSessionConfig {
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<Tool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider: Option<ProviderConfig>,
    #[serde(skip)]
    pub streaming: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mcp_servers: Option<HashMap<String, serde_json::Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub custom_agents: Option<Vec<CustomAgentConfig>>,
}

/// Options for sending a message.
#[derive(Debug, Clone, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MessageOptions {
    pub prompt: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub attachments: Option<Vec<UserMessageAttachment>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mode: Option<String>,
}

impl From<&str> for MessageOptions {
    fn from(prompt: &str) -> Self {
        Self {
            prompt: prompt.to_string(),
            attachments: None,
            mode: None,
        }
    }
}

impl From<String> for MessageOptions {
    fn from(prompt: String) -> Self {
        Self {
            prompt,
            attachments: None,
            mode: None,
        }
    }
}

// =============================================================================
// Client Options
// =============================================================================

/// Options for creating a CopilotClient.
#[derive(Debug, Clone)]
pub struct ClientOptions {
    pub cli_path: Option<PathBuf>,
    pub cli_args: Option<Vec<String>>,
    pub cwd: Option<PathBuf>,
    pub port: u16,
    pub use_stdio: bool,
    pub cli_url: Option<String>,
    pub log_level: LogLevel,
    pub auto_start: bool,
    pub auto_restart: bool,
    pub environment: Option<HashMap<String, String>>,
}

impl Default for ClientOptions {
    fn default() -> Self {
        Self {
            cli_path: None,
            cli_args: None,
            cwd: None,
            port: 0,
            use_stdio: true,
            cli_url: None,
            log_level: LogLevel::Info,
            auto_start: true,
            auto_restart: true,
            environment: None,
        }
    }
}

// =============================================================================
// Response Types
// =============================================================================

/// Metadata about a session.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionMetadata {
    pub session_id: String,
    #[serde(default)]
    pub start_time: Option<String>,
    #[serde(default)]
    pub modified_time: Option<String>,
    #[serde(default)]
    pub summary: Option<String>,
    #[serde(default)]
    pub is_remote: bool,
}

/// Response from a ping request.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PingResponse {
    pub message: String,
    pub timestamp: i64,
    #[serde(default)]
    pub protocol_version: Option<u32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tool_result_text() {
        let result = ToolResult::text("Hello, world!");
        assert_eq!(result.text_result_for_llm, "Hello, world!");
        assert_eq!(result.result_type, "success");
    }

    #[test]
    fn test_tool_result_error() {
        let result = ToolResult::error("Something went wrong");
        assert_eq!(result.result_type, "error");
        assert_eq!(result.error, Some("Something went wrong".to_string()));
    }

    #[test]
    fn test_permission_result() {
        let approved = PermissionRequestResult::approved();
        assert_eq!(approved.kind, "approved");

        let denied = PermissionRequestResult::denied();
        assert!(denied.kind.starts_with("denied"));
    }

    #[test]
    fn test_message_options_from_str() {
        let opts: MessageOptions = "Hello".into();
        assert_eq!(opts.prompt, "Hello");
    }

    #[test]
    fn test_session_config_default() {
        let config = SessionConfig::default();
        assert!(config.model.is_none());
        assert!(config.tools.is_empty());
    }

    #[test]
    fn test_tool_builder() {
        let tool = Tool::new("my_tool")
            .description("A test tool")
            .schema(serde_json::json!({"type": "object"}));

        assert_eq!(tool.name, "my_tool");
        assert_eq!(tool.description, "A test tool");
    }
}
