//! The OpenAI tool-call envelope must survive deserialisation.
//!
//! `ToolCall` used to be derived, which produced two stacked internally-tagged
//! enums: the outer one consumed `"type": "function"` and the inner one then
//! looked for a second `type` and found `"function"` again — not one of its
//! variants. Every response that carried tool calls failed to parse as a
//! `ChatMessage`, and the caller saw `finish_reason: "tool_calls"` with no
//! calls attached.

use serde::Deserialize;
use serde_json::json;
use vincents_llm::types::{ChatMessage, FunctionCall, ToolCall};

fn only_call(message: ChatMessage) -> ToolCall {
    match message {
        ChatMessage::Assistant { tool_calls, .. } => {
            let mut calls = tool_calls.expect("the tool calls must survive parsing");
            assert_eq!(calls.len(), 1, "exactly one call was sent");
            calls.remove(0)
        }
        other => panic!("expected an assistant message, got {other:?}"),
    }
}

#[test]
fn an_openai_assistant_message_keeps_its_tool_calls() {
    let wire = json!({
        "role": "assistant",
        "content": null,
        "tool_calls": [{
            "id": "call_1",
            "type": "function",
            "function": { "name": "done", "arguments": "{\"result\":\"READY\"}" }
        }]
    });

    let message: ChatMessage = serde_json::from_value(wire).expect("the message must parse");

    match only_call(message) {
        ToolCall::Function(FunctionCall::Custom(call)) => {
            assert_eq!(call.id.as_deref(), Some("call_1"));
            assert_eq!(call.name, "done");
            assert_eq!(call.arguments, "{\"result\":\"READY\"}");
        }
        other => panic!("expected a custom function call, got {other:?}"),
    }
}

#[test]
fn arguments_sent_as_an_object_are_stringified() {
    let wire = json!({
        "role": "assistant",
        "content": null,
        "tool_calls": [{
            "type": "function",
            "function": { "name": "done", "arguments": { "result": "READY" } }
        }]
    });

    let message: ChatMessage = serde_json::from_value(wire).expect("the message must parse");

    match only_call(message) {
        ToolCall::Function(FunctionCall::Custom(call)) => {
            assert_eq!(call.arguments, "{\"result\":\"READY\"}");
        }
        other => panic!("expected a custom function call, got {other:?}"),
    }
}

#[test]
fn an_unknown_tool_call_type_is_named_in_the_error() {
    let wire = json!({
        "role": "assistant",
        "content": null,
        "tool_calls": [{ "type": "telepathy", "function": { "name": "done" } }]
    });

    let error = serde_json::from_value::<ChatMessage>(wire)
        .expect_err("an unknown tool call type must be refused, not dropped");
    let text = error.to_string();
    assert!(
        text.contains("telepathy"),
        "the error must name the type it refused, got: {text}"
    );
}

#[test]
fn a_message_without_tool_calls_is_unchanged() {
    let wire = json!({ "role": "assistant", "content": "hello" });
    let message: ChatMessage = serde_json::from_value(wire).expect("the message must parse");
    match message {
        ChatMessage::Assistant { tool_calls, .. } => assert!(tool_calls.is_none()),
        other => panic!("expected an assistant message, got {other:?}"),
    }
}

#[test]
fn a_tool_call_round_trips_through_serde() {
    let original = ChatMessage::deserialize(serde_json::json!({
        "role": "assistant",
        "content": null,
        "tool_calls": [{
            "id": "call_1",
            "type": "function",
            "function": { "name": "done", "arguments": "{}" }
        }]
    }))
    .expect("the wire form must parse");

    let encoded = serde_json::to_value(&original).expect("it must serialize");
    println!("encoded: {encoded}");

    let decoded: ChatMessage =
        serde_json::from_value(encoded).expect("what we serialize must parse back");
    assert_eq!(decoded, original);
}
