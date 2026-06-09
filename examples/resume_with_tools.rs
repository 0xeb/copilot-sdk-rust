// Copyright (c) 2026 Elias Bachaalany
// SPDX-License-Identifier: MIT

//! Resume session example with new tools.

use copilot_sdk::{
    Client, ResumeSessionConfig, SessionConfig, SessionEventData, Tool, ToolHandler, ToolResult,
};
use std::io::{self, Write};
use std::sync::Arc;

#[tokio::main]
async fn main() -> copilot_sdk::Result<()> {
    println!("=== Resume with Tools Example ===\n");

    let client = Client::builder().use_stdio(true).build()?;
    client.start().await?;

    // Phase 1: Create session with greet tool
    let greet_tool = Tool::new("greet")
        .description("Greet someone")
        .schema(serde_json::json!({"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}));

    let session1 = client
        .create_session(SessionConfig {
            tools: vec![greet_tool.clone()],
            ..Default::default()
        })
        .await?;

    let session_id = session1.session_id().to_string();
    println!("Created session: {}\n", session_id);

    let greet_handler: ToolHandler = Arc::new(|_, args| {
        let name = args
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or("friend");
        ToolResult::text(format!("Hello, {}!", name))
    });
    session1
        .register_tool_with_handler(greet_tool, Some(greet_handler))
        .await;

    let mut events1 = session1.subscribe();
    session1.send("My name is Alice. Please greet me.").await?;

    print!("Turn 1: ");
    io::stdout().flush().unwrap();
    while let Ok(event) = events1.recv().await {
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

    // Phase 2: Resume with farewell tool
    println!("Resuming session with new tool...\n");

    let farewell_tool = Tool::new("farewell")
        .description("Say goodbye")
        .schema(serde_json::json!({"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}));

    let session2 = client
        .resume_session(
            &session_id,
            ResumeSessionConfig {
                tools: vec![farewell_tool.clone()],
                ..Default::default()
            },
        )
        .await?;

    let farewell_handler: ToolHandler = Arc::new(|_, args| {
        let name = args
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or("friend");
        ToolResult::text(format!("Goodbye, {}!", name))
    });
    session2
        .register_tool_with_handler(farewell_tool, Some(farewell_handler))
        .await;

    let mut events2 = session2.subscribe();
    session2
        .send("What was my name? Say goodbye to me.")
        .await?;

    print!("Turn 2: ");
    io::stdout().flush().unwrap();
    while let Ok(event) = events2.recv().await {
        match &event.data {
            SessionEventData::AssistantMessageDelta(d) => {
                print!("{}", d.delta_content);
                io::stdout().flush().unwrap();
            }
            SessionEventData::SessionIdle(_) => {
                println!();
                break;
            }
            _ => {}
        }
    }

    client.stop().await;
    Ok(())
}
