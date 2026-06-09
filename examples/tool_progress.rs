// Copyright (c) 2026 Elias Bachaalany
// SPDX-License-Identifier: MIT

//! Tool progress example demonstrating tool execution lifecycle monitoring.
//!
//! This example shows how to:
//! 1. Register custom tools and subscribe to tool lifecycle events
//! 2. Monitor ToolExecutionStart, ToolExecutionProgress, and ToolExecutionComplete
//! 3. Display real-time progress updates during tool execution

use copilot_sdk::{Client, SessionConfig, SessionEventData, Tool, ToolResult};
use std::io::Write;
use std::sync::Arc;

#[tokio::main]
async fn main() -> copilot_sdk::Result<()> {
    let client = Client::builder().build()?;

    println!("=== Tool Execution Progress Example ===\n");
    println!("This example shows the full tool lifecycle:");
    println!("  ToolExecutionStart -> ToolExecutionProgress -> ToolExecutionComplete\n");

    client.start().await?;

    // Define custom tools
    let word_count = Tool::new("word_count")
        .description("Count the number of words in text")
        .parameter("text", "string", "The text to count words in", true);

    let search_files = Tool::new("search_files")
        .description("Search for files containing a query string")
        .parameter("query", "string", "The search query", true);

    let config = SessionConfig {
        tools: vec![word_count, search_files],
        ..Default::default()
    };

    let session = client.create_session(config).await?;
    println!("Session created with 2 tools: word_count, search_files\n");

    // Register tool handlers
    session
        .register_tool_with_handler(
            Tool::new("word_count"),
            Some(Arc::new(|_name, args| {
                let text = args
                    .get("text")
                    .and_then(|v| v.as_str())
                    .unwrap_or("Hello World");
                let count = text.split_whitespace().count();
                ToolResult::text(format!("The text contains {} word(s).", count))
            })),
        )
        .await;

    session
        .register_tool_with_handler(
            Tool::new("search_files"),
            Some(Arc::new(|_name, args| {
                let query = args
                    .get("query")
                    .and_then(|v| v.as_str())
                    .unwrap_or("example");
                ToolResult::text(format!(
                    "Search results for '{query}':\n\
                     1. README.md - line 42: contains '{query}'\n\
                     2. main.rs - line 7: references '{query}'\n\
                     3. docs/guide.md - line 15: explains '{query}'"
                ))
            })),
        )
        .await;

    let idle_notify = Arc::new(tokio::sync::Notify::new());
    let idle_clone = Arc::clone(&idle_notify);

    let _unsub = session
        .on(move |event| match &event.data {
            SessionEventData::AssistantMessage(msg) => {
                println!("\nAssistant: {}", msg.content);
            }
            SessionEventData::AssistantMessageDelta(delta) => {
                print!("{}", delta.delta_content);
                std::io::stdout().flush().unwrap();
            }
            SessionEventData::ToolExecutionStart(start) => {
                print!("\n[Tool Start] {}", start.tool_name);
                if let Some(ref args) = start.arguments {
                    print!(" | Args: {args}");
                }
                println!();
            }
            SessionEventData::ToolExecutionProgress(progress) => {
                println!(
                    "[Tool Progress] {}: {}",
                    progress.tool_call_id, progress.progress_message
                );
            }
            SessionEventData::ToolExecutionComplete(complete) => {
                print!(
                    "[Tool Complete] {} | {}",
                    complete.tool_call_id,
                    if complete.success {
                        "Success"
                    } else {
                        "Failed"
                    }
                );
                if let Some(ref result) = complete.result {
                    print!(" | {}", result.content);
                }
                if let Some(ref error) = complete.error {
                    print!(" | Error: {}", error.message);
                }
                println!();
            }
            SessionEventData::SessionError(error) => {
                eprintln!("\nError: {}", error.message);
            }
            SessionEventData::SessionIdle(_) => {
                idle_clone.notify_one();
            }
            _ => {}
        })
        .await;

    println!("Try asking questions that use the tools!");
    println!("Examples:");
    println!("  - How many words are in 'The quick brown fox jumps over the lazy dog'?");
    println!("  - Search for files containing 'main'");
    println!("\nType 'quit' to exit.\n");
    print!("> ");
    std::io::stdout().flush().unwrap();

    let stdin = std::io::stdin();
    let mut line = String::new();

    loop {
        line.clear();
        if stdin.read_line(&mut line).unwrap() == 0 {
            break;
        }
        let input = line.trim();

        if input == "quit" || input == "exit" {
            break;
        }
        if input.is_empty() {
            print!("> ");
            std::io::stdout().flush().unwrap();
            continue;
        }

        session.send(input).await?;
        idle_notify.notified().await;
        print!("\n> ");
        std::io::stdout().flush().unwrap();
    }

    println!("\nCleaning up...");
    session.destroy().await?;
    client.stop().await;

    Ok(())
}
