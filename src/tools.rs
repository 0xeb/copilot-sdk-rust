// Copyright (c) 2026 Elias Bachaalany
// SPDX-License-Identifier: MIT

//! Tool definition utilities for the Copilot SDK.
//!
//! Provides convenience functions for defining tools with automatic
//! result normalization and error handling.

use crate::types::{Tool, ToolResult, ToolResultExpanded};
use serde_json::Value;

/// Normalize any result into a [`ToolResult`].
///
/// - `None` / null → empty text result
/// - `String` → text result
/// - expanded result objects (dict with resultType + textResultForLlm) → pass-through
/// - Everything else → JSON serialize
pub fn normalize_result(result: Value) -> ToolResult {
    match result {
        Value::Null => ToolResult::Text(String::new()),
        Value::String(s) => ToolResult::Text(s),
        Value::Object(ref map)
            if map.contains_key("resultType") && map.contains_key("textResultForLlm") =>
        {
            serde_json::from_value::<ToolResultExpanded>(result)
                .map(ToolResult::Expanded)
                .unwrap_or_else(|_| {
                    ToolResult::Expanded(ToolResultExpanded {
                        text_result_for_llm: "Failed to parse tool result".to_string(),
                        binary_results_for_llm: None,
                        result_type: "failure".to_string(),
                        error: None,
                        session_log: None,
                        tool_telemetry: None,
                    })
                })
        }
        other => ToolResult::Text(serde_json::to_string(&other).unwrap_or_default()),
    }
}

/// Define a tool with metadata for registration on a session.
///
/// Returns a `Tool` struct with name, description, and parameters schema.
/// The handler must be registered separately on the session via
/// `session.register_tool_with_handler()`.
///
/// # Example
/// ```rust,no_run
/// use copilot_sdk::tools::define_tool;
/// use serde_json::json;
///
/// let tool = define_tool(
///     "my_tool",
///     "A description of my tool",
///     Some(json!({"type": "object", "properties": {"query": {"type": "string"}}})),
/// );
/// // Register on session: session.register_tool_with_handler(tool, Some(handler)).await;
/// ```
pub fn define_tool(name: &str, description: &str, parameters_schema: Option<Value>) -> Tool {
    Tool::new(name)
        .with_description(description)
        .with_parameters(parameters_schema.unwrap_or(serde_json::json!({})))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_normalize_null() {
        let result = normalize_result(Value::Null);
        assert!(matches!(result, ToolResult::Text(ref s) if s.is_empty()));
    }

    #[test]
    fn test_normalize_string() {
        let result = normalize_result(Value::String("hello".to_string()));
        assert!(matches!(result, ToolResult::Text(ref s) if s == "hello"));
    }

    #[test]
    fn test_normalize_tool_result_passthrough() {
        let val = json!({
            "resultType": "success",
            "textResultForLlm": "tool output"
        });
        let result = normalize_result(val);
        assert!(matches!(
            result,
            ToolResult::Expanded(ToolResultExpanded {
                ref result_type,
                ref text_result_for_llm,
                ..
            }) if result_type == "success" && text_result_for_llm == "tool output"
        ));
    }

    #[test]
    fn test_normalize_other_value() {
        let val = json!({"key": "value"});
        let result = normalize_result(val);
        assert!(matches!(result, ToolResult::Text(ref s) if s.contains("key")));
    }

    #[test]
    fn test_define_tool_basic() {
        let tool = define_tool("test_tool", "A test tool", None);
        assert_eq!(tool.name, "test_tool");
        assert_eq!(tool.description, "A test tool");
    }

    #[test]
    fn test_define_tool_with_schema() {
        let schema = json!({"type": "object", "properties": {"q": {"type": "string"}}});
        let tool = define_tool("search", "Search tool", Some(schema.clone()));
        assert_eq!(tool.name, "search");
        assert_eq!(tool.parameters_schema, schema);
    }
}
