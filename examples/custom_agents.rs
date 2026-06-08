// Copyright (c) 2026 Elias Bachaalany
// SPDX-License-Identifier: MIT

//! Custom agents example demonstrating agent configuration.

use copilot_sdk::{Client, CustomAgentConfig, SessionConfig, SessionEventData};
use std::io::{self, Write};

#[tokio::main]
async fn main() -> copilot_sdk::Result<()> {
    println!("=== Custom Agents Example ===\n");

    let client = Client::builder().use_stdio(true).build()?;
    client.start().await?;

    // Define custom agents
    let code_reviewer = CustomAgentConfig::new(
        "code-reviewer",
        "You are a code reviewer. Look for bugs, security issues, and suggest improvements.",
    )
    .with_display_name("Code Reviewer")
    .with_description("Reviews code for bugs and best practices")
    .with_tools(["Read", "Glob", "Grep"])
    .with_infer(true);

    let security_auditor = CustomAgentConfig::new(
        "security-auditor",
        "You are a security expert. Focus on OWASP Top 10, input validation, and auth issues.",
    )
    .with_display_name("Security Auditor")
    .with_description("Audits code for security vulnerabilities")
    .with_tools(["Read", "Grep"])
    .with_infer(false);

    let config = SessionConfig {
        custom_agents: Some(vec![code_reviewer, security_auditor]),
        ..Default::default()
    };

    let session = client.create_session(config).await?;
    let mut events = session.subscribe();

    println!("Registered agents: code-reviewer (auto), security-auditor (@mention)\n");

    // Test auto-inference
    session
        .send("Review: fn divide(a: i32, b: i32) -> i32 { a / b }")
        .await?;

    print!("Assistant: ");
    io::stdout().flush().unwrap();
    while let Ok(event) = events.recv().await {
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
