// Copyright (c) 2026 Elias Bachaalany
// SPDX-License-Identifier: MIT

//! Demonstrates the v0.1.49 `remote_session` field on [`SessionConfig`]
//! (upstream PR #1295, Mission Control integration). The example prints
//! the serialized JSON for each `RemoteSessionMode` variant so the wire
//! shape can be inspected without spinning up a real CLI.
//!
//! Run with:
//!
//! ```bash
//! cargo run --example remote_session
//! ```

use copilot_sdk::{ClientOptions, RemoteSessionMode, SessionConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Mission Control integration is opt-in at two layers:
    //
    //   * `ClientOptions::remote = true` enables the CLI feature
    //     (`--remote` flag).
    //   * `SessionConfig::remote_session` controls per-session behavior
    //     (`off` | `export` | `on`).
    let client_opts = ClientOptions {
        remote: true,
        ..Default::default()
    };
    println!(
        "// Client-level remote flag (would be forwarded via `--remote`): {}",
        client_opts.remote
    );

    for mode in [
        RemoteSessionMode::Off,
        RemoteSessionMode::Export,
        RemoteSessionMode::On,
    ] {
        let config = SessionConfig {
            client_name: Some("remote-session-demo".into()),
            working_directory: Some("./repo".into()),
            remote_session: Some(mode),
            ..Default::default()
        };
        let wire = serde_json::to_string_pretty(&config)?;
        println!("\n// --- SessionConfig with remote_session = {mode:?} ---");
        println!("{wire}");
    }

    Ok(())
}
