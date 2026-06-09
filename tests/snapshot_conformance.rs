// Copyright (c) 2026 Elias Bachaalany
// SPDX-License-Identifier: MIT

#![cfg(feature = "snapshots")]

use copilot_sdk::transport::{MessageReader, MessageWriter};
use copilot_sdk::{Client, LogLevel, SessionConfig, Tool, ToolHandler, ToolResult};
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::Mutex as StdMutex;
use std::time::Duration;
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::Mutex;

#[derive(Debug, Clone)]
struct ToolCallExpectation {
    name: String,
    arguments: Value,
}

#[derive(Debug, Clone)]
struct TurnExpectation {
    prompt: String,
    tool_calls: Vec<ToolCallExpectation>,
}

#[derive(Debug, Clone)]
struct SnapshotTest {
    name: String,
    turns: Vec<TurnExpectation>,
}

fn snapshot_dir() -> Option<PathBuf> {
    if let Ok(dir) = std::env::var("COPILOT_SDK_RUST_SNAPSHOT_DIR") {
        let p = PathBuf::from(dir);
        if p.exists() {
            return Some(p);
        }
    }
    if let Ok(dir) = std::env::var("UPSTREAM_SNAPSHOTS") {
        let p = PathBuf::from(dir);
        if p.exists() {
            return Some(p);
        }
    }

    let default = PathBuf::from("../copilot-sdk/test/snapshots");
    if default.exists() {
        return Some(default);
    }
    None
}

fn find_snapshot_files(dir: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    for category in ["tools", "session"] {
        let cat_dir = dir.join(category);
        if !cat_dir.exists() {
            continue;
        }
        if let Ok(entries) = std::fs::read_dir(cat_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().and_then(|e| e.to_str()) == Some("yaml") {
                    files.push(path);
                }
            }
        }
    }
    files.sort();
    files
}

fn yaml_get<'a>(v: &'a serde_yaml::Value, key: &str) -> Option<&'a serde_yaml::Value> {
    let key = serde_yaml::Value::String(key.to_string());
    v.as_mapping()?.get(&key)
}

fn yaml_str(v: &serde_yaml::Value) -> Option<&str> {
    v.as_str()
}

fn json_type(v: &Value) -> &'static str {
    match v {
        Value::String(_) => "string",
        Value::Bool(_) => "boolean",
        Value::Number(n) => {
            if n.is_i64() || n.is_u64() {
                "integer"
            } else {
                "number"
            }
        }
        Value::Array(_) => "array",
        Value::Object(_) => "object",
        Value::Null => "null",
    }
}

fn parse_snapshot(path: &Path) -> Option<SnapshotTest> {
    let text = std::fs::read_to_string(path).ok()?;
    let data: serde_yaml::Value = serde_yaml::from_str(&text).ok()?;

    let conversations = yaml_get(&data, "conversations")?.as_sequence()?;
    let first = conversations.first()?;
    let messages = yaml_get(first, "messages")?.as_sequence()?;

    let mut turns: Vec<TurnExpectation> = Vec::new();

    for msg in messages {
        let role = yaml_get(msg, "role").and_then(yaml_str)?;

        match role {
            "user" => {
                let content = yaml_get(msg, "content").and_then(yaml_str).unwrap_or("");
                if content == "${system}" {
                    continue;
                }
                let turn = TurnExpectation {
                    prompt: content.to_string(),
                    tool_calls: Vec::new(),
                };
                turns.push(turn);
            }
            "assistant" => {
                let tool_calls = yaml_get(msg, "tool_calls")
                    .and_then(|v| v.as_sequence())
                    .cloned()
                    .unwrap_or_default();

                for tc in tool_calls {
                    let func = yaml_get(&tc, "function")?;
                    let name = yaml_get(func, "name").and_then(yaml_str).unwrap_or("");
                    if name.starts_with("${") {
                        continue;
                    }
                    let args_str = yaml_get(func, "arguments")
                        .and_then(yaml_str)
                        .unwrap_or("{}");
                    let args: Value = serde_json::from_str(args_str).unwrap_or_else(|_| json!({}));

                    if let Some(last) = turns.last_mut() {
                        last.tool_calls.push(ToolCallExpectation {
                            name: name.to_string(),
                            arguments: args,
                        });
                    }
                }
            }
            _ => {}
        }
    }

    if turns.is_empty() {
        return None;
    }

    // Only run snapshot tests that include at least one tool call (matches the C++ harness behavior).
    if !turns.iter().any(|t| !t.tool_calls.is_empty()) {
        return None;
    }

    Some(SnapshotTest {
        name: path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("snapshot")
            .to_string(),
        turns,
    })
}

fn build_tools_and_schemas(test: &SnapshotTest) -> Vec<Tool> {
    let mut schemas: BTreeMap<String, BTreeMap<String, &'static str>> = BTreeMap::new();

    for turn in &test.turns {
        for tc in &turn.tool_calls {
            let entry = schemas.entry(tc.name.clone()).or_default();
            if let Some(obj) = tc.arguments.as_object() {
                for (k, v) in obj {
                    entry.insert(k.clone(), json_type(v));
                }
            }
        }
    }

    schemas
        .into_iter()
        .map(|(tool_name, props)| {
            let mut properties = serde_json::Map::new();
            let mut required = Vec::new();
            for (k, ty) in props {
                required.push(Value::String(k.clone()));
                properties.insert(k, json!({ "type": ty }));
            }

            Tool::new(tool_name)
                .description("Snapshot tool")
                .schema(Value::Object(
                    [
                        ("type".to_string(), Value::String("object".to_string())),
                        ("properties".to_string(), Value::Object(properties)),
                        ("required".to_string(), Value::Array(required)),
                    ]
                    .into_iter()
                    .collect(),
                ))
        })
        .collect()
}

struct SnapshotServer {
    listener: TcpListener,
    turns: Vec<TurnExpectation>,
}

impl SnapshotServer {
    async fn bind(turns: Vec<TurnExpectation>) -> std::io::Result<(Self, u16)> {
        let listener = TcpListener::bind("127.0.0.1:0").await?;
        let port = listener.local_addr()?.port();
        Ok((Self { listener, turns }, port))
    }

    async fn run(self) -> copilot_sdk::Result<()> {
        let (stream, _) = self
            .listener
            .accept()
            .await
            .map_err(copilot_sdk::CopilotError::Transport)?;

        self.run_conn(stream).await
    }

    async fn run_conn(self, stream: TcpStream) -> copilot_sdk::Result<()> {
        let (read_half, write_half) = stream.into_split();
        let mut reader = MessageReader::new(read_half);
        let writer = Arc::new(Mutex::new(MessageWriter::new(write_half)));

        let session_id = "snapshot-session-1".to_string();
        let mut next_turn = 0usize;
        let mut next_id: i64 = 1;

        loop {
            let msg = match reader.read_message().await {
                Ok(m) => m,
                Err(copilot_sdk::CopilotError::ConnectionClosed) => return Ok(()),
                Err(e) => return Err(e),
            };
            let value: Value = serde_json::from_str(&msg)?;

            // Handle responses to server-originated requests (tool.call).
            if value.get("method").is_none()
                && value.get("id").is_some()
                && (value.get("result").is_some() || value.get("error").is_some())
            {
                continue;
            }

            let id = value.get("id").cloned().unwrap_or(Value::Null);
            let method = value.get("method").and_then(|m| m.as_str()).unwrap_or("");
            let _params = value.get("params").cloned().unwrap_or(Value::Null);

            let (result, turn_to_run) = match method {
                "ping" => (
                    json!({
                        "message": "pong",
                        "timestamp": 1,
                        "protocolVersion": copilot_sdk::SDK_PROTOCOL_VERSION,
                    }),
                    None,
                ),
                "session.create" => (json!({ "sessionId": session_id }), None),
                "session.resume" => (json!({ "sessionId": session_id }), None),
                "session.send" => {
                    let turn = self.turns.get(next_turn).cloned();
                    next_turn += 1;
                    (json!({ "messageId": format!("msg_{next_turn}") }), turn)
                }
                "session.destroy" => (json!({}), None),
                "session.abort" => (json!({}), None),
                "session.getMessages" => (json!({ "messages": [] }), None),
                _ => (json!({}), None),
            };

            let response = json!({
                "jsonrpc": "2.0",
                "id": id,
                "result": result,
            });
            let response_str = serde_json::to_string(&response)?;
            writer.lock().await.write_message(&response_str).await?;

            if method == "session.send" {
                if let Some(turn) = turn_to_run {
                    self.run_turn(&session_id, turn, &mut reader, &writer, &mut next_id)
                        .await?;
                } else {
                    send_idle(&session_id, &writer).await?;
                }
            }
        }
    }

    async fn run_turn(
        &self,
        session_id: &str,
        turn: TurnExpectation,
        reader: &mut MessageReader<tokio::net::tcp::OwnedReadHalf>,
        writer: &Arc<Mutex<MessageWriter<tokio::net::tcp::OwnedWriteHalf>>>,
        next_id: &mut i64,
    ) -> copilot_sdk::Result<()> {
        for (idx, call) in turn.tool_calls.into_iter().enumerate() {
            send_tool_execution_start(session_id, &call.name, idx, writer).await?;

            let req_id = *next_id;
            *next_id += 1;

            let request = json!({
                "jsonrpc": "2.0",
                "id": req_id,
                "method": "tool.call",
                "params": {
                    "sessionId": session_id,
                    "toolCallId": format!("toolcall_{}", req_id),
                    "toolName": call.name,
                    "arguments": call.arguments,
                }
            });
            let request_str = serde_json::to_string(&request)?;
            writer.lock().await.write_message(&request_str).await?;

            wait_for_response_id(reader, req_id).await?;
            send_tool_execution_complete(session_id, req_id, writer).await?;
        }

        send_idle(session_id, writer).await?;
        Ok(())
    }
}

async fn send_session_event(
    session_id: &str,
    event_type: &str,
    data: Value,
    writer: &Arc<Mutex<MessageWriter<tokio::net::tcp::OwnedWriteHalf>>>,
) -> copilot_sdk::Result<()> {
    let event = json!({
        "id": "evt_1",
        "timestamp": "2025-01-01T00:00:00Z",
        "type": event_type,
        "data": data,
    });

    let notif = json!({
        "jsonrpc": "2.0",
        "method": "session.event",
        "params": {
            "sessionId": session_id,
            "event": event,
        }
    });

    let msg = serde_json::to_string(&notif)?;
    writer.lock().await.write_message(&msg).await?;
    Ok(())
}

async fn send_idle(
    session_id: &str,
    writer: &Arc<Mutex<MessageWriter<tokio::net::tcp::OwnedWriteHalf>>>,
) -> copilot_sdk::Result<()> {
    send_session_event(session_id, "session.idle", json!({}), writer).await
}

async fn send_tool_execution_start(
    session_id: &str,
    tool_name: &str,
    idx: usize,
    writer: &Arc<Mutex<MessageWriter<tokio::net::tcp::OwnedWriteHalf>>>,
) -> copilot_sdk::Result<()> {
    send_session_event(
        session_id,
        "tool.execution_start",
        json!({
            "toolCallId": format!("toolcall_{idx}"),
            "toolName": tool_name,
            "arguments": {},
        }),
        writer,
    )
    .await
}

async fn send_tool_execution_complete(
    session_id: &str,
    call_id: i64,
    writer: &Arc<Mutex<MessageWriter<tokio::net::tcp::OwnedWriteHalf>>>,
) -> copilot_sdk::Result<()> {
    send_session_event(
        session_id,
        "tool.execution_complete",
        json!({
            "toolCallId": format!("toolcall_{call_id}"),
            "success": true,
            "result": { "content": "OK" },
        }),
        writer,
    )
    .await
}

async fn wait_for_response_id(
    reader: &mut MessageReader<tokio::net::tcp::OwnedReadHalf>,
    id: i64,
) -> copilot_sdk::Result<()> {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    loop {
        let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
        if remaining.is_zero() {
            return Err(copilot_sdk::CopilotError::Timeout(Duration::from_secs(10)));
        }

        let msg = tokio::time::timeout(remaining, reader.read_message())
            .await
            .map_err(|_| copilot_sdk::CopilotError::Timeout(Duration::from_secs(10)))??;
        let value: Value = serde_json::from_str(&msg)?;

        if value.get("method").is_none()
            && value.get("id").and_then(|v| v.as_i64()) == Some(id)
            && (value.get("result").is_some() || value.get("error").is_some())
        {
            return Ok(());
        }
    }
}

#[tokio::test]
async fn snapshot_conformance_tools_and_sessions() -> copilot_sdk::Result<()> {
    let dir = match snapshot_dir() {
        Some(d) => d,
        None => {
            eprintln!("Skipping snapshot conformance: snapshots dir not found");
            return Ok(());
        }
    };

    let files = find_snapshot_files(&dir);
    if files.is_empty() {
        eprintln!("Skipping snapshot conformance: no snapshot YAML files found");
        return Ok(());
    }

    let mut ran = 0usize;
    let mut skipped = 0usize;

    for path in files {
        let Some(test) = parse_snapshot(&path) else {
            skipped += 1;
            continue;
        };

        ran += 1;

        let tools = build_tools_and_schemas(&test);

        let (server, port) = SnapshotServer::bind(test.turns.clone())
            .await
            .map_err(|e| {
                copilot_sdk::CopilotError::Transport(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    e.to_string(),
                ))
            })?;
        let server_task = tokio::spawn(async move { server.run().await });

        let captured: Arc<StdMutex<Vec<(String, Value)>>> = Arc::new(StdMutex::new(Vec::new()));
        let captured_handlers = Arc::clone(&captured);

        let client = Client::builder()
            .cli_url(port.to_string())
            .log_level(LogLevel::Error)
            .build()?;

        client.start().await?;

        let session = client
            .create_session(SessionConfig {
                tools: tools.clone(),
                ..Default::default()
            })
            .await?;

        for tool in &tools {
            let tool_name = tool.name.clone();
            let captured_handlers = Arc::clone(&captured_handlers);
            let handler: ToolHandler = Arc::new(move |_name, args| {
                let mut guard = captured_handlers.lock().unwrap();
                guard.push((tool_name.clone(), args.clone()));
                ToolResult::text("OK")
            });

            session
                .register_tool_with_handler(tool.clone(), Some(handler))
                .await;
        }

        let mut expected_by_prompt: HashMap<String, Vec<(String, Value)>> = HashMap::new();
        for turn in &test.turns {
            let calls = turn
                .tool_calls
                .iter()
                .map(|c| (c.name.clone(), c.arguments.clone()))
                .collect::<Vec<_>>();
            expected_by_prompt.insert(turn.prompt.clone(), calls);
        }

        for turn in &test.turns {
            let before = captured.lock().unwrap().len();
            let _ = session.send_and_collect(turn.prompt.clone(), None).await?;
            let after = captured.lock().unwrap().len();

            let observed = captured.lock().unwrap()[before..after].to_vec();
            let expected = expected_by_prompt
                .get(&turn.prompt)
                .cloned()
                .unwrap_or_default();

            for (name, args) in expected {
                let found = observed.iter().any(|(n, a)| *n == name && *a == args);
                assert!(
                    found,
                    "snapshot {}: expected tool call '{}' with args {} not observed",
                    test.name, name, args
                );
            }
        }

        client.stop().await;

        let join = tokio::time::timeout(Duration::from_secs(5), server_task)
            .await
            .map_err(|_| copilot_sdk::CopilotError::Timeout(Duration::from_secs(5)))?;
        let server_res = join.map_err(|e| copilot_sdk::CopilotError::Protocol(e.to_string()))?;
        server_res?;
    }

    eprintln!("Snapshot conformance: ran={ran} skipped={skipped}");
    Ok(())
}
