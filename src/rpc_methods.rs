// Copyright (c) 2026 Elias Bachaalany
// SPDX-License-Identifier: MIT

//! JSON-RPC method-name catalog.
//!
//! Compile-time constants mirroring the wire strings emitted by the upstream
//! Copilot CLI JSON-RPC surface (see
//! `reference/copilot-sdk/nodejs/src/generated/rpc.ts`). Centralising the
//! method names keeps the rest of the crate from drifting away from upstream
//! and makes it trivial to assert exact-match parity from unit tests.
//!
//! The names are grouped by namespace and ordered roughly the same way as the
//! upstream `connection.sendRequest(...)` / `connection.onRequest(...)`
//! call sites in `rpc.ts`.

// =============================================================================
// Top-level (non-namespaced) methods
// =============================================================================

/// `ping` — Liveness check sent by the SDK to the CLI.
pub const PING: &str = "ping";

/// `connect` — Initial handshake; exchanges protocol version and client info.
pub const CONNECT: &str = "connect";

// =============================================================================
// Models / Tools / Account
// =============================================================================

/// `models.list` — Enumerate the models available to the CLI.
pub const MODELS_LIST: &str = "models.list";

/// `tools.list` — Enumerate the built-in tools known to the CLI.
pub const TOOLS_LIST: &str = "tools.list";

/// `account.getQuota` — Read the authenticated account quota snapshot.
pub const ACCOUNT_GET_QUOTA: &str = "account.getQuota";

// =============================================================================
// MCP configuration / discovery
// =============================================================================

pub const MCP_CONFIG_LIST: &str = "mcp.config.list";
pub const MCP_CONFIG_ADD: &str = "mcp.config.add";
pub const MCP_CONFIG_UPDATE: &str = "mcp.config.update";
pub const MCP_CONFIG_REMOVE: &str = "mcp.config.remove";
pub const MCP_CONFIG_ENABLE: &str = "mcp.config.enable";
pub const MCP_CONFIG_DISABLE: &str = "mcp.config.disable";
pub const MCP_DISCOVER: &str = "mcp.discover";

// =============================================================================
// Skills configuration / discovery
// =============================================================================

pub const SKILLS_CONFIG_SET_DISABLED_SKILLS: &str = "skills.config.setDisabledSkills";
pub const SKILLS_DISCOVER: &str = "skills.discover";

// =============================================================================
// Top-level session lifecycle
// =============================================================================

/// `sessionFs.setProvider` — Register the calling SDK client as the session
/// filesystem provider for subsequent `sessionFs.*` callbacks.
pub const SESSION_FS_SET_PROVIDER: &str = "sessionFs.setProvider";

/// `sessions.fork` — Fork an existing session up to an optional event boundary.
pub const SESSIONS_FORK: &str = "sessions.fork";

/// `sessions.connect` — Connect to an existing remote session.
pub const SESSIONS_CONNECT: &str = "sessions.connect";

// =============================================================================
// session.* — per-session JSON-RPC methods
// =============================================================================

/// `session.suspend`
pub const SESSION_SUSPEND: &str = "session.suspend";

// --- auth ---------------------------------------------------------------------
pub const SESSION_AUTH_GET_STATUS: &str = "session.auth.getStatus";

// --- model --------------------------------------------------------------------
pub const SESSION_MODEL_GET_CURRENT: &str = "session.model.getCurrent";
pub const SESSION_MODEL_SWITCH_TO: &str = "session.model.switchTo";

// --- mode ---------------------------------------------------------------------
pub const SESSION_MODE_GET: &str = "session.mode.get";
pub const SESSION_MODE_SET: &str = "session.mode.set";

// --- name ---------------------------------------------------------------------
pub const SESSION_NAME_GET: &str = "session.name.get";
pub const SESSION_NAME_SET: &str = "session.name.set";

// --- plan ---------------------------------------------------------------------
pub const SESSION_PLAN_READ: &str = "session.plan.read";
pub const SESSION_PLAN_UPDATE: &str = "session.plan.update";
pub const SESSION_PLAN_DELETE: &str = "session.plan.delete";

// --- workspaces ---------------------------------------------------------------
pub const SESSION_WORKSPACES_GET_WORKSPACE: &str = "session.workspaces.getWorkspace";
pub const SESSION_WORKSPACES_LIST_FILES: &str = "session.workspaces.listFiles";
pub const SESSION_WORKSPACES_READ_FILE: &str = "session.workspaces.readFile";
pub const SESSION_WORKSPACES_CREATE_FILE: &str = "session.workspaces.createFile";

// --- instructions -------------------------------------------------------------
pub const SESSION_INSTRUCTIONS_GET_SOURCES: &str = "session.instructions.getSources";

// --- fleet --------------------------------------------------------------------
pub const SESSION_FLEET_START: &str = "session.fleet.start";

// --- agent --------------------------------------------------------------------
pub const SESSION_AGENT_LIST: &str = "session.agent.list";
pub const SESSION_AGENT_GET_CURRENT: &str = "session.agent.getCurrent";
pub const SESSION_AGENT_SELECT: &str = "session.agent.select";
pub const SESSION_AGENT_DESELECT: &str = "session.agent.deselect";
pub const SESSION_AGENT_RELOAD: &str = "session.agent.reload";

// --- tasks --------------------------------------------------------------------
pub const SESSION_TASKS_START_AGENT: &str = "session.tasks.startAgent";
pub const SESSION_TASKS_LIST: &str = "session.tasks.list";
pub const SESSION_TASKS_PROMOTE_TO_BACKGROUND: &str = "session.tasks.promoteToBackground";
pub const SESSION_TASKS_CANCEL: &str = "session.tasks.cancel";
pub const SESSION_TASKS_REMOVE: &str = "session.tasks.remove";
pub const SESSION_TASKS_SEND_MESSAGE: &str = "session.tasks.sendMessage";

// --- skills (per session) -----------------------------------------------------
pub const SESSION_SKILLS_LIST: &str = "session.skills.list";
pub const SESSION_SKILLS_ENABLE: &str = "session.skills.enable";
pub const SESSION_SKILLS_DISABLE: &str = "session.skills.disable";
pub const SESSION_SKILLS_RELOAD: &str = "session.skills.reload";

// --- mcp (per session) --------------------------------------------------------
pub const SESSION_MCP_LIST: &str = "session.mcp.list";
pub const SESSION_MCP_ENABLE: &str = "session.mcp.enable";
pub const SESSION_MCP_DISABLE: &str = "session.mcp.disable";
pub const SESSION_MCP_RELOAD: &str = "session.mcp.reload";
pub const SESSION_MCP_OAUTH_LOGIN: &str = "session.mcp.oauth.login";

// --- plugins ------------------------------------------------------------------
pub const SESSION_PLUGINS_LIST: &str = "session.plugins.list";

// --- extensions ---------------------------------------------------------------
pub const SESSION_EXTENSIONS_LIST: &str = "session.extensions.list";
pub const SESSION_EXTENSIONS_ENABLE: &str = "session.extensions.enable";
pub const SESSION_EXTENSIONS_DISABLE: &str = "session.extensions.disable";
pub const SESSION_EXTENSIONS_RELOAD: &str = "session.extensions.reload";

// --- tools (per session) ------------------------------------------------------
pub const SESSION_TOOLS_HANDLE_PENDING_TOOL_CALL: &str = "session.tools.handlePendingToolCall";

// --- commands -----------------------------------------------------------------
pub const SESSION_COMMANDS_LIST: &str = "session.commands.list";
pub const SESSION_COMMANDS_INVOKE: &str = "session.commands.invoke";
pub const SESSION_COMMANDS_HANDLE_PENDING_COMMAND: &str = "session.commands.handlePendingCommand";
pub const SESSION_COMMANDS_RESPOND_TO_QUEUED_COMMAND: &str =
    "session.commands.respondToQueuedCommand";

// --- ui (elicitation) ---------------------------------------------------------
pub const SESSION_UI_ELICITATION: &str = "session.ui.elicitation";
pub const SESSION_UI_HANDLE_PENDING_ELICITATION: &str = "session.ui.handlePendingElicitation";

// --- permissions --------------------------------------------------------------
pub const SESSION_PERMISSIONS_HANDLE_PENDING_PERMISSION_REQUEST: &str =
    "session.permissions.handlePendingPermissionRequest";
pub const SESSION_PERMISSIONS_SET_APPROVE_ALL: &str = "session.permissions.setApproveAll";
pub const SESSION_PERMISSIONS_RESET_SESSION_APPROVALS: &str =
    "session.permissions.resetSessionApprovals";

// --- log ----------------------------------------------------------------------
pub const SESSION_LOG: &str = "session.log";

// --- shell --------------------------------------------------------------------
pub const SESSION_SHELL_EXEC: &str = "session.shell.exec";
pub const SESSION_SHELL_KILL: &str = "session.shell.kill";

// --- history (compaction / truncation) ---------------------------------------
pub const SESSION_HISTORY_COMPACT: &str = "session.history.compact";
pub const SESSION_HISTORY_TRUNCATE: &str = "session.history.truncate";

// --- usage --------------------------------------------------------------------
pub const SESSION_USAGE_GET_METRICS: &str = "session.usage.getMetrics";

// --- remote -------------------------------------------------------------------
pub const SESSION_REMOTE_ENABLE: &str = "session.remote.enable";
pub const SESSION_REMOTE_DISABLE: &str = "session.remote.disable";

// =============================================================================
// sessionFs.* — server-handled callbacks for the SDK-provided filesystem
//
// These method names are registered with `connection.onRequest(...)` upstream;
// the SDK implements them, the CLI invokes them. They are exposed here for
// symmetry and future server-side wiring.
// =============================================================================

pub const SESSION_FS_READ_FILE: &str = "sessionFs.readFile";
pub const SESSION_FS_WRITE_FILE: &str = "sessionFs.writeFile";
pub const SESSION_FS_APPEND_FILE: &str = "sessionFs.appendFile";
pub const SESSION_FS_EXISTS: &str = "sessionFs.exists";
pub const SESSION_FS_STAT: &str = "sessionFs.stat";
pub const SESSION_FS_MKDIR: &str = "sessionFs.mkdir";
pub const SESSION_FS_READDIR: &str = "sessionFs.readdir";
pub const SESSION_FS_READDIR_WITH_TYPES: &str = "sessionFs.readdirWithTypes";
pub const SESSION_FS_RM: &str = "sessionFs.rm";
pub const SESSION_FS_RENAME: &str = "sessionFs.rename";

// =============================================================================
// Unit tests — assert exact wire-string parity with upstream rpc.ts
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn top_level_methods_match_upstream() {
        assert_eq!(PING, "ping");
        assert_eq!(CONNECT, "connect");
        assert_eq!(MODELS_LIST, "models.list");
        assert_eq!(TOOLS_LIST, "tools.list");
        assert_eq!(ACCOUNT_GET_QUOTA, "account.getQuota");
    }

    #[test]
    fn mcp_methods_match_upstream() {
        assert_eq!(MCP_CONFIG_LIST, "mcp.config.list");
        assert_eq!(MCP_CONFIG_ADD, "mcp.config.add");
        assert_eq!(MCP_CONFIG_UPDATE, "mcp.config.update");
        assert_eq!(MCP_CONFIG_REMOVE, "mcp.config.remove");
        assert_eq!(MCP_CONFIG_ENABLE, "mcp.config.enable");
        assert_eq!(MCP_CONFIG_DISABLE, "mcp.config.disable");
        assert_eq!(MCP_DISCOVER, "mcp.discover");
    }

    #[test]
    fn skills_methods_match_upstream() {
        assert_eq!(
            SKILLS_CONFIG_SET_DISABLED_SKILLS,
            "skills.config.setDisabledSkills"
        );
        assert_eq!(SKILLS_DISCOVER, "skills.discover");
    }

    #[test]
    fn top_level_session_methods_match_upstream() {
        assert_eq!(SESSION_FS_SET_PROVIDER, "sessionFs.setProvider");
        assert_eq!(SESSIONS_FORK, "sessions.fork");
        assert_eq!(SESSIONS_CONNECT, "sessions.connect");
    }

    #[test]
    fn session_lifecycle_methods_match_upstream() {
        assert_eq!(SESSION_SUSPEND, "session.suspend");
        assert_eq!(SESSION_AUTH_GET_STATUS, "session.auth.getStatus");
    }

    #[test]
    fn session_model_methods_match_upstream() {
        assert_eq!(SESSION_MODEL_GET_CURRENT, "session.model.getCurrent");
        assert_eq!(SESSION_MODEL_SWITCH_TO, "session.model.switchTo");
    }

    #[test]
    fn session_mode_methods_match_upstream() {
        assert_eq!(SESSION_MODE_GET, "session.mode.get");
        assert_eq!(SESSION_MODE_SET, "session.mode.set");
    }

    #[test]
    fn session_name_methods_match_upstream() {
        assert_eq!(SESSION_NAME_GET, "session.name.get");
        assert_eq!(SESSION_NAME_SET, "session.name.set");
    }

    #[test]
    fn session_plan_methods_match_upstream() {
        assert_eq!(SESSION_PLAN_READ, "session.plan.read");
        assert_eq!(SESSION_PLAN_UPDATE, "session.plan.update");
        assert_eq!(SESSION_PLAN_DELETE, "session.plan.delete");
    }

    #[test]
    fn session_workspaces_methods_match_upstream() {
        assert_eq!(
            SESSION_WORKSPACES_GET_WORKSPACE,
            "session.workspaces.getWorkspace"
        );
        assert_eq!(
            SESSION_WORKSPACES_LIST_FILES,
            "session.workspaces.listFiles"
        );
        assert_eq!(SESSION_WORKSPACES_READ_FILE, "session.workspaces.readFile");
        assert_eq!(
            SESSION_WORKSPACES_CREATE_FILE,
            "session.workspaces.createFile"
        );
    }

    #[test]
    fn session_instructions_methods_match_upstream() {
        assert_eq!(
            SESSION_INSTRUCTIONS_GET_SOURCES,
            "session.instructions.getSources"
        );
    }

    #[test]
    fn session_fleet_methods_match_upstream() {
        assert_eq!(SESSION_FLEET_START, "session.fleet.start");
    }

    #[test]
    fn session_agent_methods_match_upstream() {
        assert_eq!(SESSION_AGENT_LIST, "session.agent.list");
        assert_eq!(SESSION_AGENT_GET_CURRENT, "session.agent.getCurrent");
        assert_eq!(SESSION_AGENT_SELECT, "session.agent.select");
        assert_eq!(SESSION_AGENT_DESELECT, "session.agent.deselect");
        assert_eq!(SESSION_AGENT_RELOAD, "session.agent.reload");
    }

    #[test]
    fn session_tasks_methods_match_upstream() {
        assert_eq!(SESSION_TASKS_START_AGENT, "session.tasks.startAgent");
        assert_eq!(SESSION_TASKS_LIST, "session.tasks.list");
        assert_eq!(
            SESSION_TASKS_PROMOTE_TO_BACKGROUND,
            "session.tasks.promoteToBackground"
        );
        assert_eq!(SESSION_TASKS_CANCEL, "session.tasks.cancel");
        assert_eq!(SESSION_TASKS_REMOVE, "session.tasks.remove");
        assert_eq!(SESSION_TASKS_SEND_MESSAGE, "session.tasks.sendMessage");
    }

    #[test]
    fn session_skills_methods_match_upstream() {
        assert_eq!(SESSION_SKILLS_LIST, "session.skills.list");
        assert_eq!(SESSION_SKILLS_ENABLE, "session.skills.enable");
        assert_eq!(SESSION_SKILLS_DISABLE, "session.skills.disable");
        assert_eq!(SESSION_SKILLS_RELOAD, "session.skills.reload");
    }

    #[test]
    fn session_mcp_methods_match_upstream() {
        assert_eq!(SESSION_MCP_LIST, "session.mcp.list");
        assert_eq!(SESSION_MCP_ENABLE, "session.mcp.enable");
        assert_eq!(SESSION_MCP_DISABLE, "session.mcp.disable");
        assert_eq!(SESSION_MCP_RELOAD, "session.mcp.reload");
        assert_eq!(SESSION_MCP_OAUTH_LOGIN, "session.mcp.oauth.login");
    }

    #[test]
    fn session_plugins_methods_match_upstream() {
        assert_eq!(SESSION_PLUGINS_LIST, "session.plugins.list");
    }

    #[test]
    fn session_extensions_methods_match_upstream() {
        assert_eq!(SESSION_EXTENSIONS_LIST, "session.extensions.list");
        assert_eq!(SESSION_EXTENSIONS_ENABLE, "session.extensions.enable");
        assert_eq!(SESSION_EXTENSIONS_DISABLE, "session.extensions.disable");
        assert_eq!(SESSION_EXTENSIONS_RELOAD, "session.extensions.reload");
    }

    #[test]
    fn session_tools_methods_match_upstream() {
        assert_eq!(
            SESSION_TOOLS_HANDLE_PENDING_TOOL_CALL,
            "session.tools.handlePendingToolCall"
        );
    }

    #[test]
    fn session_commands_methods_match_upstream() {
        assert_eq!(SESSION_COMMANDS_LIST, "session.commands.list");
        assert_eq!(SESSION_COMMANDS_INVOKE, "session.commands.invoke");
        assert_eq!(
            SESSION_COMMANDS_HANDLE_PENDING_COMMAND,
            "session.commands.handlePendingCommand"
        );
        assert_eq!(
            SESSION_COMMANDS_RESPOND_TO_QUEUED_COMMAND,
            "session.commands.respondToQueuedCommand"
        );
    }

    #[test]
    fn session_ui_methods_match_upstream() {
        assert_eq!(SESSION_UI_ELICITATION, "session.ui.elicitation");
        assert_eq!(
            SESSION_UI_HANDLE_PENDING_ELICITATION,
            "session.ui.handlePendingElicitation"
        );
    }

    #[test]
    fn session_permissions_methods_match_upstream() {
        assert_eq!(
            SESSION_PERMISSIONS_HANDLE_PENDING_PERMISSION_REQUEST,
            "session.permissions.handlePendingPermissionRequest"
        );
        assert_eq!(
            SESSION_PERMISSIONS_SET_APPROVE_ALL,
            "session.permissions.setApproveAll"
        );
        assert_eq!(
            SESSION_PERMISSIONS_RESET_SESSION_APPROVALS,
            "session.permissions.resetSessionApprovals"
        );
    }

    #[test]
    fn session_log_method_matches_upstream() {
        assert_eq!(SESSION_LOG, "session.log");
    }

    #[test]
    fn session_shell_methods_match_upstream() {
        assert_eq!(SESSION_SHELL_EXEC, "session.shell.exec");
        assert_eq!(SESSION_SHELL_KILL, "session.shell.kill");
    }

    #[test]
    fn session_history_methods_match_upstream() {
        assert_eq!(SESSION_HISTORY_COMPACT, "session.history.compact");
        assert_eq!(SESSION_HISTORY_TRUNCATE, "session.history.truncate");
    }

    #[test]
    fn session_usage_methods_match_upstream() {
        assert_eq!(SESSION_USAGE_GET_METRICS, "session.usage.getMetrics");
    }

    #[test]
    fn session_remote_methods_match_upstream() {
        assert_eq!(SESSION_REMOTE_ENABLE, "session.remote.enable");
        assert_eq!(SESSION_REMOTE_DISABLE, "session.remote.disable");
    }

    #[test]
    fn session_fs_callback_methods_match_upstream() {
        assert_eq!(SESSION_FS_READ_FILE, "sessionFs.readFile");
        assert_eq!(SESSION_FS_WRITE_FILE, "sessionFs.writeFile");
        assert_eq!(SESSION_FS_APPEND_FILE, "sessionFs.appendFile");
        assert_eq!(SESSION_FS_EXISTS, "sessionFs.exists");
        assert_eq!(SESSION_FS_STAT, "sessionFs.stat");
        assert_eq!(SESSION_FS_MKDIR, "sessionFs.mkdir");
        assert_eq!(SESSION_FS_READDIR, "sessionFs.readdir");
        assert_eq!(SESSION_FS_READDIR_WITH_TYPES, "sessionFs.readdirWithTypes");
        assert_eq!(SESSION_FS_RM, "sessionFs.rm");
        assert_eq!(SESSION_FS_RENAME, "sessionFs.rename");
    }
}
