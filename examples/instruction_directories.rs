// Copyright (c) 2026 Elias Bachaalany
// SPDX-License-Identifier: MIT

//! Demonstrates the v0.1.49 `instruction_directories` field on
//! [`SessionConfig`] (upstream PR #1190). The example builds a config and
//! prints the serialized JSON the SDK would send to the CLI, so it runs
//! offline without requiring a Copilot CLI installation.
//!
//! Run with:
//!
//! ```bash
//! cargo run --example instruction_directories
//! ```

use copilot_sdk::{RemoteSessionMode, SessionConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = SessionConfig {
        client_name: Some("instruction-dirs-demo".into()),
        agent: Some("default".into()),
        working_directory: Some("./my-project".into()),
        // Per-session instruction directories are merged with the global set
        // configured for the CLI. Paths are resolved by the server.
        instruction_directories: Some(vec![
            "./docs/instructions".into(),
            "/etc/copilot/instructions".into(),
        ]),
        enable_config_discovery: Some(true),
        remote_session: Some(RemoteSessionMode::Off),
        ..Default::default()
    };

    let wire = serde_json::to_string_pretty(&config)?;
    println!("// SessionConfig payload sent to session.create:");
    println!("{wire}");
    Ok(())
}
