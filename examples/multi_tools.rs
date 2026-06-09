// Copyright (c) 2026 Elias Bachaalany
// SPDX-License-Identifier: MIT

//! Multi-tools example demonstrating multiple custom tools.

use copilot_sdk::{Client, SessionConfig, SessionEventData, Tool, ToolHandler, ToolResult};
use std::io::{self, Write};
use std::sync::Arc;

#[tokio::main]
async fn main() -> copilot_sdk::Result<()> {
    println!("=== Multi-Tools Example ===\n");

    let client = Client::builder().use_stdio(true).build()?;
    client.start().await?;

    // Define multiple tools
    let calculator = Tool::new("calculate")
        .description("Perform arithmetic")
        .schema(serde_json::json!({
            "type": "object",
            "properties": {
                "a": {"type": "number"},
                "b": {"type": "number"},
                "op": {"type": "string", "enum": ["add", "subtract", "multiply", "divide"]}
            },
            "required": ["a", "b", "op"]
        }));

    let echo = Tool::new("echo")
        .description("Echo a message")
        .schema(serde_json::json!({
            "type": "object",
            "properties": {"message": {"type": "string"}},
            "required": ["message"]
        }));

    let random = Tool::new("random")
        .description("Generate random number")
        .schema(serde_json::json!({
            "type": "object",
            "properties": {
                "min": {"type": "integer", "default": 0},
                "max": {"type": "integer", "default": 100}
            }
        }));

    let config = SessionConfig {
        tools: vec![calculator.clone(), echo.clone(), random.clone()],
        ..Default::default()
    };

    let session = client.create_session(config).await?;
    println!("Registered 3 tools: calculate, echo, random\n");

    // Register handlers
    let calc_handler: ToolHandler = Arc::new(|_, args| {
        let a = args.get("a").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let b = args.get("b").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let op = args.get("op").and_then(|v| v.as_str()).unwrap_or("add");
        let result = match op {
            "add" => a + b,
            "subtract" => a - b,
            "multiply" => a * b,
            "divide" => {
                if b != 0.0 {
                    a / b
                } else {
                    f64::NAN
                }
            }
            _ => f64::NAN,
        };
        ToolResult::text(format!("{}", result))
    });
    session
        .register_tool_with_handler(calculator, Some(calc_handler))
        .await;

    let echo_handler: ToolHandler = Arc::new(|_, args| {
        let msg = args.get("message").and_then(|v| v.as_str()).unwrap_or("");
        ToolResult::text(format!("Echo: {}", msg))
    });
    session
        .register_tool_with_handler(echo, Some(echo_handler))
        .await;

    let random_handler: ToolHandler = Arc::new(|_, args| {
        let min = args.get("min").and_then(|v| v.as_i64()).unwrap_or(0) as i32;
        let max = args.get("max").and_then(|v| v.as_i64()).unwrap_or(100) as i32;
        let result = min + (rand::random::<u32>() % (max - min + 1) as u32) as i32;
        ToolResult::text(format!("{}", result))
    });
    session
        .register_tool_with_handler(random, Some(random_handler))
        .await;

    let mut events = session.subscribe();

    // Test
    session
        .send("What is 123 * 456? Then generate a random number 1-10.")
        .await?;

    print!("Assistant: ");
    io::stdout().flush().unwrap();
    while let Ok(event) = events.recv().await {
        match &event.data {
            SessionEventData::AssistantMessageDelta(d) => {
                print!("{}", d.delta_content);
                io::stdout().flush().unwrap();
            }
            SessionEventData::ToolExecutionStart(t) => {
                println!("\n[Tool: {}]", t.tool_name);
            }
            SessionEventData::ToolExecutionComplete(_) => {
                print!("Assistant: ");
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
