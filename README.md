# copilot-sdk-rust

[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-blue)](https://github.com/copilot-community-sdk/copilot-sdk-rust)
[![Rust](https://img.shields.io/badge/rust-1.85%2B%20(Edition%202024)-orange)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Rust SDK for interacting with the GitHub Copilot CLI agent runtime (JSON-RPC over stdio or TCP).

This is an **unofficial** Rust port of the upstream SDKs.

**Disclaimer:** This project is not affiliated with, endorsed by, or sponsored by GitHub. "GitHub" and "Copilot" are trademarks of their respective owners.

## Requirements

- Rust 1.85+ (Edition 2024)
- GitHub Copilot CLI installed and authenticated
- `copilot` available in `PATH`, or set `COPILOT_CLI_PATH` to the CLI executable/script

## Related Projects

| Project | Language | Description |
|---------|----------|-------------|
| [copilot-sdk-cpp](https://github.com/copilot-community-sdk/copilot-sdk-cpp) | C++ | C++ SDK for Copilot CLI |
| [claude-agent-sdk-cpp](https://github.com/copilot-community-sdk/claude-agent-sdk-cpp) | C++ | C++ SDK for Claude Agent |
| [claude-agent-sdk-dotnet](https://github.com/copilot-community-sdk/claude-agent-sdk-dotnet) | C# | .NET SDK for Claude Agent |

## License

MIT License - see [LICENSE](LICENSE).

Copyright (c) 2026 Elias Bachaalany

Based on GitHub's [copilot-sdk](https://github.com/github/copilot-sdk).
