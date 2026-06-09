// Copyright (c) 2026 Elias Bachaalany
// SPDX-License-Identifier: MIT

//! Hooks example demonstrating the hooks system for tool lifecycle interception.
//!
//! This example shows how to:
//! 1. Register PreToolUse hooks to inspect/modify/deny tool calls
//! 2. Register PostToolUse hooks to inspect/modify tool results
//! 3. Register session lifecycle hooks (start, end, error)

use copilot_sdk::{Client, PreToolUseHookOutput, SessionConfig, SessionEventData, SessionHooks};
use std::io::Write;
use std::sync::{Arc, Mutex};

#[tokio::main]
async fn main() -> copilot_sdk::Result<()> {
    let hook_log: Arc<Mutex<Vec<(String, String)>>> = Arc::new(Mutex::new(Vec::new()));

    let log = Arc::clone(&hook_log);
    let log_hook = move |hook_type: &str, detail: &str| {
        let mut log = log.lock().unwrap();
        println!("[HOOK:{hook_type}] {detail}");
        log.push((hook_type.to_string(), detail.to_string()));
    };

    let client = Client::builder().build()?;

    println!("=== Hooks System Example ===\n");
    println!("Demonstrates PreToolUse, PostToolUse, and session lifecycle hooks.\n");

    client.start().await?;

    // Set up hooks
    let log1 = log_hook.clone();
    let log2 = log_hook.clone();
    let log3 = log_hook.clone();
    let log4 = log_hook.clone();
    let log5 = log_hook.clone();

    let config = SessionConfig {
        session_hooks: Some(SessionHooks {
            on_pre_tool_use: Some(Arc::new(move |input| {
                log1("PreToolUse", &format!("Tool: {}", input.tool_name));

                // Example: deny any tool named "dangerous_tool"
                if input.tool_name == "dangerous_tool" {
                    log1("PreToolUse", &format!("DENIED: {}", input.tool_name));
                    return PreToolUseHookOutput {
                        permission_decision: Some("deny".into()),
                        permission_decision_reason: Some("This tool is blocked by policy".into()),
                        ..Default::default()
                    };
                }

                // Allow all other tools
                PreToolUseHookOutput::default()
            })),
            on_post_tool_use: Some(Arc::new(move |input| {
                let result_str = format!("{:.100}", input.tool_result);
                log2(
                    "PostToolUse",
                    &format!("Tool: {} => {result_str}", input.tool_name),
                );
                Default::default()
            })),
            on_session_start: Some(Arc::new(move |_input| {
                log3("SessionStart", "Session is starting");
                Default::default()
            })),
            on_session_end: Some(Arc::new(move |_input| {
                log4("SessionEnd", "Session is ending");
                Default::default()
            })),
            on_error_occurred: Some(Arc::new(move |input| {
                log5(
                    "ErrorOccurred",
                    &format!("{} (context: {})", input.error, input.error_context),
                );
                Default::default()
            })),
            ..Default::default()
        }),
        ..Default::default()
    };

    let session = client.create_session(config).await?;
    println!("Session created: {}\n", session.session_id());

    // Subscribe to events
    let session_clone = Arc::clone(&session);
    let idle_notify = Arc::new(tokio::sync::Notify::new());
    let idle_notify_clone = Arc::clone(&idle_notify);

    let _unsub = session_clone
        .on(move |event| {
            if let SessionEventData::AssistantMessage(msg) = &event.data {
                println!("\nAssistant: {}", msg.content);
            } else if matches!(&event.data, SessionEventData::SessionIdle(_)) {
                idle_notify_clone.notify_one();
            }
        })
        .await;

    println!("Chat with hooks enabled. Type 'log' to see hook log, 'quit' to exit.\n");
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

        if input == "log" {
            let log = hook_log.lock().unwrap();
            println!("\n=== Hook Log ({} entries) ===", log.len());
            for (hook_type, detail) in log.iter() {
                println!("  [{hook_type}] {detail}");
            }
            println!("==============================\n");
            print!("> ");
            std::io::stdout().flush().unwrap();
            continue;
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

    // Final hook log
    println!("\n=== Final Hook Log ===");
    {
        let log = hook_log.lock().unwrap();
        for (hook_type, detail) in log.iter() {
            println!("  [{hook_type}] {detail}");
        }
    }
    println!("======================");

    session.destroy().await?;
    client.stop().await;

    Ok(())
}
