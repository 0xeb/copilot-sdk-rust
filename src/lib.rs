// Copyright (c) 2026 Elias Bachaalany
// SPDX-License-Identifier: MIT

#![forbid(unsafe_code)]

//! # Copilot SDK for Rust
//!
//! A Rust SDK for interacting with the GitHub Copilot CLI.
//!
//! ## Quick Start
//!
//! ```no_run
//! use copilot_sdk::{Client, SessionConfig, SessionEventData};
//!
//! #[tokio::main]
//! async fn main() -> copilot_sdk::Result<()> {
//!     let client = Client::builder().build()?;
//!     client.start().await?;
//!
//!     let session = client.create_session(SessionConfig::default()).await?;
//!     let mut events = session.subscribe();
//!
//!     session.send("What is the capital of France?").await?;
//!
//!     while let Ok(event) = events.recv().await {
//!         match &event.data {
//!             SessionEventData::AssistantMessage(msg) => println!("{}", msg.content),
//!             SessionEventData::SessionIdle(_) => break,
//!             _ => {}
//!         }
//!     }
//!
//!     client.stop().await?;
//!     Ok(())
//! }
//! ```

pub mod client;
pub mod error;
pub mod events;
pub mod jsonrpc;
pub mod process;
pub mod session;
pub mod transport;
pub mod types;

// Re-export main types at crate root for convenience
pub use error::{CopilotError, Result};
pub use types::{
    // Enums
    AttachmentType,
    // Config types
    AzureOptions,
    ClientOptions,
    ConnectionState,
    CustomAgentConfig,
    // Response types
    GetAuthStatusResponse,
    GetStatusResponse,
    InfiniteSessionConfig,
    LogLevel,
    McpLocalServerConfig,
    McpRemoteServerConfig,
    McpServerConfig,
    MessageOptions,
    ModelBilling,
    ModelCapabilities,
    ModelInfo,
    ModelLimits,
    ModelPolicy,
    ModelSupports,
    // Permission types
    PermissionRequest,
    PermissionRequestResult,
    PingResponse,
    ProviderConfig,
    ResumeSessionConfig,
    // Constants
    SDK_PROTOCOL_VERSION,
    SessionConfig,
    SessionMetadata,
    SystemMessageConfig,
    SystemMessageMode,
    // Tool types
    Tool,
    ToolBinaryResult,
    ToolInvocation,
    ToolResult,
    ToolResultObject,
    UserMessageAttachment,
};

// Re-export event types
pub use events::{
    // Event data types
    AbortData,
    AssistantIntentData,
    AssistantMessageData,
    AssistantMessageDeltaData,
    AssistantReasoningData,
    AssistantReasoningDeltaData,
    AssistantTurnEndData,
    AssistantTurnStartData,
    AssistantUsageData,
    CustomAgentCompletedData,
    CustomAgentFailedData,
    CustomAgentSelectedData,
    CustomAgentStartedData,
    HandoffSourceType,
    HookEndData,
    HookError,
    HookStartData,
    PendingMessagesModifiedData,
    // Main event types
    RawSessionEvent,
    RepositoryInfo,
    SessionErrorData,
    SessionEvent,
    SessionEventData,
    SessionHandoffData,
    SessionIdleData,
    SessionInfoData,
    SessionModelChangeData,
    SessionResumeData,
    SessionStartData,
    SessionTruncationData,
    SystemMessageEventData,
    SystemMessageMetadata,
    SystemMessageRole,
    ToolExecutionCompleteData,
    ToolExecutionError,
    ToolExecutionPartialResultData,
    ToolExecutionStartData,
    ToolRequestItem,
    ToolResultContent,
    ToolUserRequestedData,
    UserMessageAttachmentItem,
    UserMessageData,
};

// Re-export transport types
pub use transport::{MessageFramer, StdioTransport, Transport};

// Re-export JSON-RPC types
pub use jsonrpc::{
    JsonRpcClient, JsonRpcError, JsonRpcId, JsonRpcRequest, JsonRpcResponse, NotificationHandler,
    RequestHandler,
};

// Re-export process types
pub use process::{
    CopilotProcess, ProcessOptions, find_copilot_cli, find_executable, find_node, is_node_script,
};

// Re-export session types
pub use session::{
    EventHandler, EventSubscription, InvokeFuture, PermissionHandler, RegisteredTool, Session,
    ToolHandler,
};

// Re-export client types
pub use client::{Client, ClientBuilder};
