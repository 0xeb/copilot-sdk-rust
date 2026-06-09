// Copyright (c) 2026 Elias Bachaalany
// SPDX-License-Identifier: MIT

//! Tool usage example demonstrating custom tools with the Copilot SDK.
//!
//! This example shows how to:
//! - Define custom tools with JSON schemas
//! - Register tool handlers
//! - Have the assistant use your tools

use copilot_sdk::{Client, SessionConfig, SessionEventData, Tool, ToolHandler, ToolResult};
use std::sync::Arc;

#[cfg(feature = "schemars")]
#[derive(Debug, Clone, serde::Deserialize, schemars::JsonSchema)]
struct GetWeatherArgs {
    location: String,
    #[serde(default)]
    unit: Option<TemperatureUnit>,
}

#[cfg(feature = "schemars")]
#[derive(Debug, Clone, Copy, serde::Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "lowercase")]
enum TemperatureUnit {
    Celsius,
    Fahrenheit,
}

#[tokio::main]
async fn main() -> copilot_sdk::Result<()> {
    // Create and start client
    let client = Client::builder().use_stdio(true).build()?;
    println!("Starting Copilot client...");
    client.start().await?;

    // Define a custom tool
    #[cfg(feature = "schemars")]
    let weather_tool = Tool::new("get_weather")
        .description("Get the current weather for a location")
        .typed_schema::<GetWeatherArgs>();

    #[cfg(not(feature = "schemars"))]
    let weather_tool = Tool::new("get_weather")
        .description("Get the current weather for a location")
        .schema(serde_json::json!({
            "type": "object",
            "properties": {
                "location": { "type": "string" },
                "unit": { "type": "string", "enum": ["celsius", "fahrenheit"] }
            },
            "required": ["location"]
        }));

    // Create session with the tool
    let config = SessionConfig {
        tools: vec![weather_tool.clone()],
        ..Default::default()
    };

    let session = client.create_session(config).await?;
    println!("Session created: {}", session.session_id());

    // Register tool handler
    #[cfg(feature = "schemars")]
    let handler: ToolHandler = Arc::new(|_name, args| {
        let parsed: GetWeatherArgs =
            serde_json::from_value(args.clone()).unwrap_or_else(|_| GetWeatherArgs {
                location: "Unknown".to_string(),
                unit: None,
            });

        let (temp, symbol) = match parsed.unit.unwrap_or(TemperatureUnit::Fahrenheit) {
            TemperatureUnit::Celsius => (22, "°C"),
            TemperatureUnit::Fahrenheit => (72, "°F"),
        };

        ToolResult::text(format!(
            "The weather in {} is sunny with a temperature of {}{}",
            parsed.location, temp, symbol
        ))
    });

    #[cfg(not(feature = "schemars"))]
    let handler: ToolHandler = Arc::new(|_name, args| {
        let location = args
            .get("location")
            .and_then(|v| v.as_str())
            .unwrap_or("Unknown");
        let unit = args
            .get("unit")
            .and_then(|v| v.as_str())
            .unwrap_or("fahrenheit");

        // Simulate weather lookup
        let (temp, symbol) = if unit == "celsius" {
            (22, "°C")
        } else {
            (72, "°F")
        };

        ToolResult::text(format!(
            "The weather in {} is sunny with a temperature of {}{}",
            location, temp, symbol
        ))
    });

    session
        .register_tool_with_handler(weather_tool, Some(handler))
        .await;

    // Subscribe to events
    let mut events = session.subscribe();

    // Ask a question that requires the tool
    let prompt = "What's the weather like in San Francisco?";
    println!("\nYou: {}\n", prompt);
    session.send(prompt).await?;

    // Process events
    print!("Assistant: ");
    while let Ok(event) = events.recv().await {
        match &event.data {
            SessionEventData::AssistantMessageDelta(delta) => {
                print!("{}", delta.delta_content);
            }
            SessionEventData::AssistantMessage(msg) => {
                println!("{}", msg.content);
            }
            SessionEventData::ToolExecutionStart(tool) => {
                println!(
                    "\n[Tool called: {} ({})]",
                    tool.tool_name, tool.tool_call_id
                );
            }
            SessionEventData::ToolExecutionComplete(tool) => {
                println!("[Tool completed: {}]\n", tool.tool_call_id);
                print!("Assistant: ");
            }
            SessionEventData::SessionIdle(_) => {
                println!("\n");
                break;
            }
            SessionEventData::SessionError(err) => {
                eprintln!("\nError: {}", err.message);
                break;
            }
            _ => {}
        }
    }

    client.stop().await;
    println!("Done!");
    Ok(())
}
