// Copyright (c) 2026 Elias Bachaalany
// SPDX-License-Identifier: MIT

//! Fluent tools example demonstrating concise tool definition with headless E2E test.
//!
//! This example shows the idiomatic Rust way to define tools using the builder API.
//! Compare with tool_usage.rs which shows the verbose approach.
//!
//! Key benefits:
//! - Builder pattern with `Tool::new()` + `.parameter()` chaining
//! - Type-safe parameter extraction
//! - Concise, readable syntax

use copilot_sdk::{Client, SessionConfig, SessionEventData, Tool, ToolResult};
use std::sync::Arc;

#[tokio::main]
async fn main() -> copilot_sdk::Result<()> {
    println!("=== Fluent Tool Builder E2E Example ===\n");

    // Define tools using the fluent builder API
    let calculator = Tool::new("calculate")
        .description("Perform basic arithmetic operations")
        .parameter("a", "number", "First operand", true)
        .parameter("b", "number", "Second operand", true)
        .parameter(
            "operation",
            "string",
            "The operation: add, subtract, multiply, divide, power",
            true,
        );

    let get_time = Tool::new("get_time")
        .description("Get the current date and time")
        .parameter(
            "timezone",
            "string",
            "Timezone (e.g., 'UTC', 'local')",
            false,
        );

    let echo = Tool::new("echo")
        .description("Echo back the input message")
        .parameter("message", "string", "The message to echo", true);

    let random = Tool::new("random")
        .description("Generate a random number in a range")
        .parameter("min", "integer", "Minimum value (inclusive)", false)
        .parameter("max", "integer", "Maximum value (inclusive)", false);

    // Print generated schemas
    println!("Generated tool schemas:\n");
    println!(
        "1. {}:\n{}\n",
        calculator.name,
        serde_json::to_string_pretty(&calculator.parameters_schema).unwrap()
    );
    println!(
        "2. {}:\n{}\n",
        get_time.name,
        serde_json::to_string_pretty(&get_time.parameters_schema).unwrap()
    );

    // Connect to Copilot CLI
    println!("=== Starting Copilot Session ===\n");

    let client = Client::builder().build()?;

    println!("Connecting to Copilot CLI...");
    client.start().await?;
    println!("Connected!\n");

    let config = SessionConfig {
        tools: vec![calculator, get_time, echo, random],
        ..Default::default()
    };

    let session = client.create_session(config).await?;
    println!("Session created: {}", session.session_id());
    println!("Registered 4 fluent tools\n");

    // Register tool handlers
    session
        .register_tool_with_handler(
            Tool::new("calculate"),
            Some(Arc::new(|_name, args| {
                let a = args.get("a").and_then(|v| v.as_f64()).unwrap_or(0.0);
                let b = args.get("b").and_then(|v| v.as_f64()).unwrap_or(0.0);
                let op = args
                    .get("operation")
                    .and_then(|v| v.as_str())
                    .unwrap_or("add");

                let (result, symbol) = match op {
                    "add" => (a + b, "+"),
                    "subtract" => (a - b, "-"),
                    "multiply" => (a * b, "*"),
                    "divide" => {
                        if b == 0.0 {
                            return ToolResult::text("Error: Division by zero");
                        }
                        (a / b, "/")
                    }
                    "power" => (a.powf(b), "^"),
                    _ => return ToolResult::text(format!("Unknown operation: {op}")),
                };

                ToolResult::text(format!("{a} {symbol} {b} = {result}"))
            })),
        )
        .await;

    session
        .register_tool_with_handler(
            Tool::new("get_time"),
            Some(Arc::new(|_name, args| {
                let tz = args
                    .get("timezone")
                    .and_then(|v| v.as_str())
                    .unwrap_or("local");
                let now = chrono::Local::now();
                ToolResult::text(format!("Current time ({tz}): {now}"))
            })),
        )
        .await;

    session
        .register_tool_with_handler(
            Tool::new("echo"),
            Some(Arc::new(|_name, args| {
                let msg = args
                    .get("message")
                    .and_then(|v| v.as_str())
                    .unwrap_or("(empty)");
                ToolResult::text(format!("Echo: {msg}"))
            })),
        )
        .await;

    session
        .register_tool_with_handler(
            Tool::new("random"),
            Some(Arc::new(|_name, args| {
                let min = args.get("min").and_then(|v| v.as_i64()).unwrap_or(0);
                let max = args.get("max").and_then(|v| v.as_i64()).unwrap_or(100);
                let result = min + (rand::random::<i64>().abs() % (max - min + 1));
                ToolResult::text(format!("{result}"))
            })),
        )
        .await;

    // Event handling
    let idle_notify = Arc::new(tokio::sync::Notify::new());
    let idle_clone = Arc::clone(&idle_notify);

    let _unsub = session
        .on(move |event| match &event.data {
            SessionEventData::AssistantMessage(msg) => {
                println!("Assistant: {}", msg.content);
            }
            SessionEventData::ToolExecutionStart(start) => {
                print!("[Tool: {}]", start.tool_name);
                if let Some(ref args) = start.arguments {
                    print!(" args={args}");
                }
                println!();
            }
            SessionEventData::ToolExecutionComplete(complete) => {
                print!("[Result: ");
                if let Some(ref result) = complete.result {
                    print!("{}", result.content);
                }
                println!("]");
            }
            SessionEventData::SessionIdle(_) => {
                idle_clone.notify_one();
            }
            _ => {}
        })
        .await;

    // Headless E2E tests
    println!("=== Running Headless E2E Tests ===");

    let prompts = [
        "What is 123 times 456? Use the calculate tool.",
        "Divide 1000 by 8 using the calculate tool.",
        "What time is it right now? Use the get_time tool.",
        "Echo the message 'Hello from fluent tools!'",
        "Generate a random number between 1 and 10.",
        "Calculate 2 to the power of 10 using the calculate tool.",
    ];

    for prompt in &prompts {
        println!("\n--- Prompt: {prompt} ---");
        session.send(*prompt).await?;
        idle_notify.notified().await;
    }

    // Cleanup
    println!("\n=== Fluent Tools E2E Complete ===");

    session.destroy().await?;
    client.stop().await;

    println!("Session destroyed, client stopped.");

    Ok(())
}
