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
//!     client.stop().await;
//!     Ok(())
//! }
//! ```

pub mod client;
pub mod error;
pub mod events;
pub mod jsonrpc;
pub mod process;
pub mod rpc_methods;
pub mod rpc_types;
pub mod session;
pub mod tools;
pub mod transport;
pub mod types;

// Re-export tool utilities
pub use tools::define_tool;

// Re-export main types at crate root for convenience
pub use error::{CopilotError, Result};
pub use types::{
    // Session lifecycle event type constants
    session_lifecycle_event_types,
    // Config types
    AgentInfo,
    // Enums
    AttachmentType,
    AutoModeSwitchResponse,
    AzureOptions,
    ClientOptions,
    ConnectionState,
    CustomAgentConfig,
    DefaultAgentConfig,
    DeliveryMode,
    ElicitationMode,
    ElicitationRequest,
    ElicitationResult,
    // Hook types
    ErrorOccurredHandler,
    ErrorOccurredHookInput,
    ErrorOccurredHookOutput,
    ExitPlanModeData,
    ExitPlanModeResult,
    FleetStartOptions,
    // Response types
    GetAuthStatusResponse,
    GetForegroundSessionResponse,
    GetStatusResponse,
    GitHubReferenceType,
    InfiniteSessionConfig,
    InputFormat,
    LogLevel,
    LogOptions,
    LogResult,
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
    ModelVisionLimits,
    // Permission types
    PermissionRequest,
    PermissionRequestKind,
    PermissionRequestResult,
    PingResponse,
    PlanData,
    PostToolUseHandler,
    PostToolUseHookInput,
    PostToolUseHookOutput,
    PreToolUseHandler,
    PreToolUseHookInput,
    PreToolUseHookOutput,
    ProviderConfig,
    // Quota types
    QuotaResult,
    QuotaSnapshot,
    RemoteSessionMode,
    ResumeSessionConfig,
    // Selection types
    SelectionAttachment,
    SelectionPosition,
    SelectionRange,
    SessionConfig,
    SessionContext,
    SessionEndHandler,
    SessionEndHookInput,
    SessionEndHookOutput,
    SessionHooks,
    // Session lifecycle types
    SessionLifecycleEvent,
    SessionLifecycleEventMetadata,
    SessionListFilter,
    SessionLogLevel,
    SessionMetadata,
    SessionMode,
    SessionStartHandler,
    SessionStartHookInput,
    SessionStartHookOutput,
    SetForegroundSessionResponse,
    SetModelOptions,
    // Shell types
    ShellExecOptions,
    ShellExecResult,
    ShellSignal,
    StopError,
    SystemMessageConfig,
    SystemMessageMode,
    // Telemetry types
    TelemetryConfig,
    // Tool types
    Tool,
    ToolBinaryResult,
    ToolInfo,
    ToolInvocation,
    ToolResult,
    ToolResultExpanded,
    ToolsListResult,
    // User input types
    UserInputInvocation,
    UserInputRequest,
    UserInputResponse,
    UserMessageAttachment,
    UserPromptSubmittedHandler,
    UserPromptSubmittedHookInput,
    UserPromptSubmittedHookOutput,
    // Workspace types
    WorkspaceFile,
    // Constants
    SDK_PROTOCOL_VERSION,
};

// Re-export event types
pub use events::{
    // Event data types
    AbortData,
    AssistantIntentData,
    AssistantMessageData,
    AssistantMessageDeltaData,
    AssistantMessageStartData,
    AssistantReasoningData,
    AssistantReasoningDeltaData,
    AssistantStreamingDeltaData,
    AssistantTurnEndData,
    AssistantTurnStartData,
    AssistantUsageData,
    AutoModeSwitchCompletedData,
    AutoModeSwitchRequestedData,
    CapabilitiesChangedData,
    CapabilitiesChangedUi,
    CommandCompletedData,
    CommandExecuteData,
    CommandQueuedData,
    CommandsChangedCommand,
    CommandsChangedData,
    CompactionTokensUsed,
    CustomAgentCompletedData,
    CustomAgentFailedData,
    CustomAgentSelectedData,
    CustomAgentStartedData,
    CustomAgentsUpdatedAgent,
    CustomNotificationData,
    ElicitationCompletedAction,
    ElicitationCompletedData,
    ElicitationRequestedData,
    ElicitationRequestedMode,
    ExitPlanModeCompletedData,
    ExitPlanModeRequestedData,
    ExtensionSource,
    ExtensionStatus,
    ExtensionsLoadedExtension,
    ExternalToolCompletedData,
    ExternalToolRequestedData,
    HandoffSourceType,
    HookEndData,
    HookError,
    HookStartData,
    McpOauthCompletedData,
    McpOauthRequiredData,
    McpOauthStaticClientConfig,
    McpServerStatus,
    McpServersLoadedServer,
    ModelCallFailureData,
    ModelCallFailureSource,
    PendingMessagesModifiedData,
    PermissionCompletedData,
    PermissionRequestedData,
    PlanChangedOperation,
    // Main event types
    RawSessionEvent,
    RepositoryInfo,
    SamplingCompletedData,
    SamplingRequestedData,
    SessionBackgroundTasksChangedData,
    SessionCompactionCompleteData,
    SessionCompactionStartData,
    SessionCustomAgentsUpdatedData,
    SessionErrorData,
    SessionEvent,
    SessionEventData,
    SessionExtensionsLoadedData,
    SessionHandoffData,
    SessionIdleData,
    SessionInfoData,
    SessionMcpServerStatusChangedData,
    SessionMcpServersLoadedData,
    SessionModeChangedData,
    SessionModelChangeData,
    SessionPlanChangedData,
    SessionRemoteSteerableChangedData,
    SessionResumeData,
    SessionScheduleCancelledData,
    SessionScheduleCreatedData,
    SessionShutdownData,
    SessionSkillsLoadedData,
    SessionSnapshotRewindData,
    SessionStartData,
    SessionTaskCompleteData,
    SessionTitleChangedData,
    SessionToolsUpdatedData,
    SessionTruncationData,
    SessionUsageInfoData,
    SessionWarningData,
    SessionWorkspaceFileChangedData,
    ShutdownCodeChanges,
    ShutdownType,
    SkillInvokedData,
    SkillsLoadedSkill,
    SubagentDeselectedData,
    SystemMessageEventData,
    SystemMessageMetadata,
    SystemMessageRole,
    SystemNotificationData,
    ToolExecutionCompleteData,
    ToolExecutionError,
    ToolExecutionPartialResultData,
    ToolExecutionProgressData,
    ToolExecutionStartData,
    ToolRequestItem,
    ToolResultContent,
    ToolUserRequestedData,
    UserInputCompletedData,
    UserInputRequestedData,
    UserMessageAttachmentItem,
    UserMessageData,
    WorkingDirectoryContext,
    WorkingDirectoryHostType,
    WorkspaceFileChangedOperation,
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
    find_copilot_cli, find_executable, find_node, is_node_script, CopilotProcess, ProcessOptions,
};

// Re-export session types
pub use session::{
    AutoModeSwitchHandler, ElicitationHandler, EventHandler, EventSubscription,
    ExitPlanModeHandler, InvokeFuture, PermissionHandler, RegisteredTool, Session, ToolHandler,
    UserInputHandler,
};

// Re-export client types
pub use client::{Client, ClientBuilder, LifecycleHandler};

// Re-export representative new RPC payload types
pub use rpc_types::{
    CommandsHandlePendingCommandRequest, CommandsHandlePendingCommandResult, CommandsInvokeRequest,
    CommandsListRequest, CommandsListResult, CommandsRespondToQueuedCommandRequest,
    CommandsRespondToQueuedCommandResult, HistoryCompactContextWindow, HistoryCompactResult,
    HistoryTruncateRequest, HistoryTruncateResult, QueuedCommandHandled, QueuedCommandNotHandled,
    QueuedCommandResult, SessionFsAppendFileRequest, SessionFsError, SessionFsErrorCode,
    SessionFsExistsRequest, SessionFsExistsResult, SessionFsMkdirRequest, SessionFsReadFileRequest,
    SessionFsReadFileResult, SessionFsReaddirRequest, SessionFsReaddirResult,
    SessionFsReaddirWithTypesEntry, SessionFsReaddirWithTypesEntryType,
    SessionFsReaddirWithTypesRequest, SessionFsReaddirWithTypesResult, SessionFsRenameRequest,
    SessionFsRmRequest, SessionFsSetProviderConventions, SessionFsSetProviderRequest,
    SessionFsSetProviderResult, SessionFsStatRequest, SessionFsStatResult,
    SessionFsWriteFileRequest, SessionsForkRequest, SessionsForkResult, SlashCommandInfo,
    SlashCommandInput, SlashCommandInputCompletion, SlashCommandKind, UIElicitationResponse,
    UIElicitationResponseAction, UIElicitationResult, UIHandlePendingElicitationRequest,
};
