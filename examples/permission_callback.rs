// Copyright (c) 2026 Elias Bachaalany
// SPDX-License-Identifier: MIT

//! Permission callback example demonstrating custom permission handling.

use copilot_sdk::{
    Client, PermissionRequest, PermissionRequestResult, SessionConfig, SessionEventData, Tool,
    ToolHandler, ToolResult,
};
use std::io::{self, Write};
use std::sync::Arc;

fn is_safe_tool(name: &str) -> bool {
    ["Read", "Glob", "Grep", "echo", "calculate"].contains(&name)
}

#[tokio::main]
async fn main() -> copilot_sdk::Result<()> {
    println!("=== Permission Callback Example ===\n");

    let client = Client::builder().use_stdio(true).build()?;
    client.start().await?;

    // Define tools
    let echo_tool = Tool::new("echo")
        .description("Echo a message (safe)")
        .schema(serde_json::json!({"type": "object", "properties": {"message": {"type": "string"}}, "required": ["message"]}));

    let delete_tool = Tool::new("delete_file")
        .description("Delete a file (dangerous)")
        .schema(serde_json::json!({"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}));

    let config = SessionConfig {
        tools: vec![echo_tool.clone(), delete_tool.clone()],
        ..Default::default()
    };

    let session = client.create_session(config).await?;

    // Register handlers
    let echo_handler: ToolHandler = Arc::new(|_, args| {
        let msg = args.get("message").and_then(|v| v.as_str()).unwrap_or("");
        ToolResult::text(format!("Echo: {}", msg))
    });
    session
        .register_tool_with_handler(echo_tool, Some(echo_handler))
        .await;

    let delete_handler: ToolHandler = Arc::new(|_, args| {
        let path = args.get("path").and_then(|v| v.as_str()).unwrap_or("");
        ToolResult::text(format!("Would delete: {} (simulated)", path))
    });
    session
        .register_tool_with_handler(delete_tool, Some(delete_handler))
        .await;

    // Register permission handler
    session
        .register_permission_handler(|req: &PermissionRequest| {
            let tool_name = req
                .extension_data
                .get("toolName")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");
            println!(
                "[Permission] Tool: {} -> {}",
                tool_name,
                if is_safe_tool(tool_name) {
                    "APPROVED"
                } else {
                    "DENIED"
                }
            );
            if is_safe_tool(tool_name) {
                PermissionRequestResult::approved()
            } else {
                PermissionRequestResult::denied()
            }
        })
        .await;

    let mut events = session.subscribe();

    // Test safe tool
    println!("\nTesting safe tool (echo):");
    session.send("Echo 'Hello World'").await?;

    print!("Assistant: ");
    io::stdout().flush().unwrap();
    while let Ok(event) = events.recv().await {
        match &event.data {
            SessionEventData::AssistantMessageDelta(d) => {
                print!("{}", d.delta_content);
                io::stdout().flush().unwrap();
            }
            SessionEventData::SessionIdle(_) => {
                println!("\n");
                break;
            }
            _ => {}
        }
    }

    client.stop().await;
    Ok(())
}
