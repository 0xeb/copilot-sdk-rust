// Copyright (c) 2026 Elias Bachaalany
// SPDX-License-Identifier: MIT

//! Escape room example (tools + streaming + a bit of fun).
//!
//! Run:
//! `cargo run --example escape_room`

use copilot_sdk::{
    find_copilot_cli, Client, LogLevel, SessionConfig, SessionEventData, SystemMessageConfig,
    SystemMessageMode, Tool, ToolHandler, ToolResult,
};
use std::io::{self, Write};
use std::sync::Arc;

fn prime_factors(mut n: u64) -> Vec<u64> {
    let mut factors = Vec::new();
    if n < 2 {
        return factors;
    }

    while n % 2 == 0 {
        factors.push(2);
        n /= 2;
    }

    let mut d = 3u64;
    while d * d <= n {
        while n % d == 0 {
            factors.push(d);
            n /= d;
        }
        d += 2;
    }

    if n > 1 {
        factors.push(n);
    }

    factors
}

fn caesar_shift(text: &str, shift: i32) -> String {
    let shift = shift.rem_euclid(26);
    text.chars()
        .map(|ch| match ch {
            'a'..='z' => {
                let base = b'a';
                let idx = (ch as u8) - base;
                let out = (idx as i32 + shift) as u8 % 26;
                (base + out) as char
            }
            'A'..='Z' => {
                let base = b'A';
                let idx = (ch as u8) - base;
                let out = (idx as i32 + shift) as u8 % 26;
                (base + out) as char
            }
            _ => ch,
        })
        .collect()
}

fn checksum8(text: &str) -> u8 {
    text.as_bytes()
        .iter()
        .fold(0u8, |acc, b| acc.wrapping_add(*b))
}

#[tokio::main]
async fn main() -> copilot_sdk::Result<()> {
    println!("=== Copilot SDK Rust: Escape Room ===\n");

    if find_copilot_cli().is_none() {
        println!(
            "Copilot CLI not found. Set `COPILOT_CLI_PATH` or install/authenticate `copilot`."
        );
        return Ok(());
    }

    let tool_prime_factors = Tool::new("prime_factors")
        .description("Return the prime factorization of an integer n.")
        .schema(serde_json::json!({
            "type": "object",
            "properties": { "n": { "type": "integer" } },
            "required": ["n"]
        }));

    let tool_caesar = Tool::new("caesar_shift")
        .description("Caesar-shift letters in `text` by `shift` (can be negative).")
        .schema(serde_json::json!({
            "type": "object",
            "properties": {
                "text": { "type": "string" },
                "shift": { "type": "integer" }
            },
            "required": ["text", "shift"]
        }));

    let tool_checksum8 = Tool::new("checksum8")
        .description("Compute a tiny 8-bit checksum (sum of bytes mod 256) for `text`.")
        .schema(serde_json::json!({
            "type": "object",
            "properties": { "text": { "type": "string" } },
            "required": ["text"]
        }));

    let config = SessionConfig {
        tools: vec![
            tool_prime_factors.clone(),
            tool_caesar.clone(),
            tool_checksum8.clone(),
        ],
        streaming: true,
        available_tools: Some(vec![
            "prime_factors".to_string(),
            "caesar_shift".to_string(),
            "checksum8".to_string(),
        ]),
        excluded_tools: Some(vec!["powershell".to_string(), "shell".to_string()]),
        system_message: Some(SystemMessageConfig {
            mode: Some(SystemMessageMode::Replace),
            content: Some(
                "You are an escape-room engineer. Use the provided tools whenever asked. \
                 Never use shell/powershell. Do not do the math in your head; call tools. \
                 Keep it concise."
                    .to_string(),
            ),
        }),
        ..Default::default()
    };

    let client = Client::builder().log_level(LogLevel::Info).build()?;
    client.start().await?;

    let session = client.create_session(config).await?;

    let prime_factors_handler: ToolHandler = Arc::new(|_name, args| {
        let Some(n) = args.get("n").and_then(|v| v.as_u64()) else {
            return ToolResult::error(r#"Expected arguments: {"n": integer}"#);
        };
        let factors = prime_factors(n);
        ToolResult::text(
            serde_json::json!({ "n": n, "factors": factors, "count": factors.len() }).to_string(),
        )
    });
    session
        .register_tool_with_handler(tool_prime_factors, Some(prime_factors_handler))
        .await;

    let caesar_handler: ToolHandler = Arc::new(|_name, args| {
        let Some(text) = args.get("text").and_then(|v| v.as_str()) else {
            return ToolResult::error(r#"Expected arguments: {"text": string, "shift": integer}"#);
        };
        let Some(shift) = args.get("shift").and_then(|v| v.as_i64()) else {
            return ToolResult::error(r#"Expected arguments: {"text": string, "shift": integer}"#);
        };
        let shift = shift as i32;
        ToolResult::text(caesar_shift(text, shift))
    });
    session
        .register_tool_with_handler(tool_caesar, Some(caesar_handler))
        .await;

    let checksum_handler: ToolHandler = Arc::new(|_name, args| {
        let Some(text) = args.get("text").and_then(|v| v.as_str()) else {
            return ToolResult::error(r#"Expected arguments: {"text": string}"#);
        };
        ToolResult::text(format!("{}", checksum8(text)))
    });
    session
        .register_tool_with_handler(tool_checksum8, Some(checksum_handler))
        .await;

    session
        .register_permission_handler(|req| {
            eprintln!("[permission request] kind={}", req.kind);
            copilot_sdk::PermissionRequestResult::denied()
        })
        .await;

    let mut events = session.subscribe();

    let prompt = r#"
You are in a vault with a 3-part code. Compute each part using tools:

1) prime_factors for n=99991 and return the number of prime factors.
2) caesar_shift the text "Ymj vznhp gwtbs ktc ozrux tajw ymj qfed itl." by shift=-5.
3) checksum8 for the exact string: "Rust > C++? Discuss." (including punctuation)

Finally, output the code as: PART1-PART2-PART3.
"#
    .trim();

    println!("Puzzle prompt:\n{prompt}\n");
    session.send(prompt).await?;

    print!("Assistant: ");
    io::stdout().flush().ok();
    let mut printed_delta = false;

    loop {
        match events.recv().await {
            Ok(event) => match &event.data {
                SessionEventData::AssistantMessageDelta(d) => {
                    printed_delta = true;
                    print!("{}", d.delta_content);
                    io::stdout().flush().ok();
                }
                SessionEventData::AssistantMessage(msg) => {
                    if !printed_delta {
                        print!("{}", msg.content);
                        io::stdout().flush().ok();
                    }
                }
                SessionEventData::ToolExecutionStart(t) => {
                    println!("\n[tool start] {} ({})", t.tool_name, t.tool_call_id);
                    print!("Assistant: ");
                    io::stdout().flush().ok();
                }
                SessionEventData::ToolExecutionComplete(t) => {
                    println!("\n[tool done] {} success={}", t.tool_call_id, t.success);
                    if let Some(result) = &t.result {
                        println!("[tool result] {}", result.content.trim());
                    }
                    if let Some(err) = &t.error {
                        eprintln!("[tool error] {}", err.message);
                    }
                    print!("Assistant: ");
                    io::stdout().flush().ok();
                }
                SessionEventData::SessionIdle(_) => {
                    println!();
                    break;
                }
                SessionEventData::SessionError(err) => {
                    eprintln!("\n[session error] {}", err.message);
                    break;
                }
                _ => {}
            },
            Err(e) => {
                eprintln!("\n[event stream error] {e:?}");
                break;
            }
        }
    }

    client.stop().await;
    Ok(())
}
