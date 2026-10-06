//! Streaming golden tests: §61.
//!
//! Each case is one thing that can go wrong in an SSE stream. They are written
//! as whole recorded frames rather than unit tests of the parser, because the
//! parser's correctness is a property of its behaviour against bytes, not of
//! its match arms.

use futures::StreamExt;
use serde_json::{json, Value};
use vincents_llm::types::{ChatStreamEvent, ToolCallAssembler, Usage};
use vincents_llm::error::LLMError;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn sse(frames: &[Value]) -> String {
    frames
        .iter()
        .map(|frame| format!("data: {frame}\n\n"))
        .collect::<Vec<_>>()
        .join("")
}

async fn streaming(body: String) -> MockServer {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(body, "text/event-stream"))
        .mount(&server)
        .await;
    server
}

fn provider(base: &str) -> vincents_llm::openai::OpenAIProvider {
    vincents_llm::openai::OpenAIProvider::with_config(vincents_llm::openai::OpenAIConfig {
        api_key: "k".into(),
        base_url: Some(base.to_string()),
        default_model: Some("gpt-4o".into()),
        ..Default::default()
    })
    .expect("provider")
}

fn request() -> vincents_llm::types::ChatCompletionRequest {
    vincents_llm::types::ChatCompletionRequest {
        model: "gpt-4o".into(),
        messages: vec![vincents_llm::types::ChatMessage::user("hi")],
        ..Default::default()
    }
}

/// One collected item: an event, or the terminal error.
#[derive(Debug)]
enum Item {
    Event(ChatStreamEvent),
    Error(String),
}

impl Item {
    fn as_event(&self) -> Option<&ChatStreamEvent> {
        match self {
            Item::Event(event) => Some(event),
            Item::Error(_) => None,
        }
    }

    fn error_message(&self) -> Option<&str> {
        match self {
            Item::Error(message) => Some(message),
            Item::Event(_) => None,
        }
    }
}

async fn collect(server: &MockServer) -> Vec<Item> {
    use vincents_llm::provider::LLMProvider;
    let mut stream = provider(&server.uri())
        .chat_completion_stream_events(request())
        .await
        .expect("stream opens");
    let mut items = Vec::new();
    while let Some(item) = stream.next().await {
        match item {
            Ok(event) => items.push(Item::Event(event)),
            Err(error) => {
                items.push(Item::Error(error.to_string()));
                break;
            }
        }
    }
    items
}

fn frame(id: &str, model: &str, created: u64, choices: Value) -> Value {
    json!({
        "id": id, "object": "chat.completion.chunk", "created": created,
        "model": model, "choices": choices
    })
}

fn delta(content: &str) -> Value {
    json!([{"index": 0, "delta": {"content": content}, "finish_reason": null}])
}

// ── Plain text deltas ──────────────────────────────────────────────────────

#[tokio::test]
async fn plain_text_deltas_arrive_in_order() {
    let server = streaming(
        sse(&[
            frame("chatcmpl-1", "gpt-4o", 100, delta("Hello")),
            frame("chatcmpl-1", "gpt-4o", 100, delta(", ")),
            frame("chatcmpl-1", "gpt-4o", 100, delta("world")),
            {
                let mut last = frame("chatcmpl-1", "gpt-4o", 100, delta(""));
                last["choices"][0]["finish_reason"] = json!("stop");
                last
            },
        ]) + "data: [DONE]\n\n",
    )
    .await;

    let events = collect(&server).await;
    let text: String = events
        .iter()
        .filter_map(|e| match e.as_event() {
            Some(ChatStreamEvent::TextDelta { text, .. }) => Some(text.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(text, "Hello, world");
}

#[tokio::test]
async fn multiple_choices_are_all_reported() {
    // Only `choices[0]` is a common bug: a `n > 1` request silently loses every
    // candidate but the first.
    let server = streaming(
        sse(&[json!({
            "id": "chatcmpl-2", "created": 1, "model": "gpt-4o",
            "choices": [
                {"index": 0, "delta": {"content": "first"}, "finish_reason": null},
                {"index": 1, "delta": {"content": "second"}, "finish_reason": null},
                {"index": 2, "delta": {"content": "third"}, "finish_reason": null}
            ]
        })]) + "data: [DONE]\n\n",
    )
    .await;

    let events = collect(&server).await;
    let by_index: std::collections::BTreeMap<u32, String> = events
        .iter()
        .filter_map(|e| match e.as_event() {
            Some(ChatStreamEvent::TextDelta { choice_index, text }) => {
                Some((*choice_index, text.clone()))
            }
            _ => None,
        })
        .collect();
    assert_eq!(by_index.len(), 3, "every choice must survive: {by_index:?}");
    assert_eq!(by_index[&0], "first");
    assert_eq!(by_index[&2], "third");
}

// ── Tool calls ─────────────────────────────────────────────────────────────

#[tokio::test]
async fn a_tool_call_start_and_fragmented_arguments_assemble() {
    let server = streaming(
        sse(&[
            frame(
                "chatcmpl-3",
                "gpt-4o",
                1,
                json!([{"index": 0, "finish_reason": null, "delta": {"tool_calls": [
                    {"index": 0, "id": "call_a", "function": {"name": "get_weather",
                                                               "arguments": ""}}
                ]}}]),
            ),
            frame(
                "chatcmpl-3",
                "gpt-4o",
                1,
                json!([{"index": 0, "finish_reason": null, "delta": {"tool_calls": [
                    {"index": 0, "function": {"arguments": "{\"city\":"}}
                ]}}]),
            ),
            frame(
                "chatcmpl-3",
                "gpt-4o",
                1,
                json!([{"index": 0, "finish_reason": null, "delta": {"tool_calls": [
                    {"index": 0, "function": {"arguments": "\"Oslo\"}"}}
                ]}}]),
            ),
        ]) + "data: [DONE]\n\n",
    )
    .await;

    let events = collect(&server).await;
    let mut assembler = ToolCallAssembler::new();
    for event in &events {
        match event.as_event() {
            Some(ChatStreamEvent::ToolCallStart { tool_index, id, name, .. }) => {
                assembler.start(*tool_index, id.clone(), name.clone())
            }
            Some(ChatStreamEvent::ToolCallArgumentsDelta { tool_index, delta, .. }) => {
                assembler.push_arguments(*tool_index, delta)
            }
            _ => {}
        }
    }
    let calls = assembler.finish();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].id.as_deref(), Some("call_a"));
    assert_eq!(calls[0].name.as_deref(), Some("get_weather"));
    assert_eq!(calls[0].arguments, r#"{"city":"Oslo"}"#);
}

#[tokio::test]
async fn parallel_interleaved_tool_calls_assemble_by_index() {
    let server = streaming(
        sse(&[
            frame("chatcmpl-4", "gpt-4o", 1, json!([{"index": 0, "finish_reason": null,
                "delta": {"tool_calls": [
                    {"index": 0, "id": "call_a", "function": {"name": "get_weather", "arguments": ""}},
                    {"index": 1, "id": "call_b", "function": {"name": "get_time", "arguments": ""}}
                ]}}])),
            frame("chatcmpl-4", "gpt-4o", 1, json!([{"index": 0, "finish_reason": null,
                "delta": {"tool_calls": [{"index": 0, "function": {"arguments": "{\"city\":"}}]}}])),
            frame("chatcmpl-4", "gpt-4o", 1, json!([{"index": 0, "finish_reason": null,
                "delta": {"tool_calls": [{"index": 1, "function": {"arguments": "{\"tz\":"}}]}}])),
            frame("chatcmpl-4", "gpt-4o", 1, json!([{"index": 0, "finish_reason": null,
                "delta": {"tool_calls": [{"index": 0, "function": {"arguments": "\"Oslo\"}"}}]}}])),
            frame("chatcmpl-4", "gpt-4o", 1, json!([{"index": 0, "finish_reason": null,
                "delta": {"tool_calls": [{"index": 1, "function": {"arguments": "\"UTC\"}"}}]}}])),
        ]) + "data: [DONE]\n\n",
    )
    .await;

    let events = collect(&server).await;
    let mut assembler = ToolCallAssembler::new();
    for event in &events {
        match event.as_event() {
            Some(ChatStreamEvent::ToolCallStart { tool_index, id, name, .. }) => {
                assembler.start(*tool_index, id.clone(), name.clone())
            }
            Some(ChatStreamEvent::ToolCallArgumentsDelta { tool_index, delta, .. }) => {
                assembler.push_arguments(*tool_index, delta)
            }
            _ => {}
        }
    }
    let calls = assembler.finish();
    let by_name: std::collections::BTreeMap<&str, &str> = calls
        .iter()
        .map(|c| (c.name.as_deref().unwrap_or_default(), c.arguments.as_str()))
        .collect();
    // Arrival-order assembly would yield `{"city":{"tz":"Oslo"}` for the first
    // call: valid JSON, wrong call.
    assert_eq!(by_name["get_weather"], r#"{"city":"Oslo"}"#);
    assert_eq!(by_name["get_time"], r#"{"tz":"UTC"}"#);
}

#[tokio::test]
async fn a_refusal_delta_is_distinct_from_text() {
    // A refusal is not an answer. Conflating them with text would bill it as a
    // successful completion.
    let server = streaming(
        sse(&[frame("chatcmpl-5", "gpt-4o", 1,
            json!([{"index": 0, "finish_reason": null,
                    "delta": {"refusal": "I can't help with that"}}]))])
            + "data: [DONE]\n\n",
    )
    .await;

    let events = collect(&server).await;
    assert!(events.iter().any(|e| matches!(
        e.as_event(),
        Some(ChatStreamEvent::RefusalDelta { delta, .. }) if delta == "I can't help with that"
    )));
    assert!(
        !events
            .iter()
            .any(|e| matches!(e.as_event(), Some(ChatStreamEvent::TextDelta { .. }))),
        "a refusal must not also appear as text: {events:?}"
    );
}

// ── Finish reason and usage ────────────────────────────────────────────────

#[tokio::test]
async fn the_providers_finish_reason_is_reported_without_a_synthetic_stop() {
    let server = streaming(
        sse(&[frame("chatcmpl-6", "gpt-4o", 1,
            json!([{"index": 0, "delta": {}, "finish_reason": "length"}]))])
            + "data: [DONE]\n\n",
    )
    .await;

    let events = collect(&server).await;
    let reasons: Vec<Option<String>> = events
        .iter()
        .filter_map(|e| match e.as_event() {
            Some(ChatStreamEvent::ChoiceFinished { reason, .. }) => Some(reason.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(reasons, vec![Some("length".to_string())]);
}

#[tokio::test]
async fn usage_arrives_in_a_final_event_and_keeps_its_detail() {
    let mut final_frame =
        frame("chatcmpl-7", "gpt-4o", 1, json!([{"index": 0, "delta": {}, "finish_reason": "stop"}]));
    final_frame["usage"] = json!({
        "prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150,
        "prompt_tokens_details": {"cached_tokens": 20},
        "completion_tokens_details": {"reasoning_tokens": 30}
    });
    let server = streaming(sse(&[final_frame]) + "data: [DONE]\n\n").await;

    let events = collect(&server).await;
    let usage: Usage = events
        .iter()
        .find_map(|e| match e.as_event() {
            Some(ChatStreamEvent::Usage(u)) => Some(u.clone()),
            _ => None,
        })
        .expect("usage arrives in a final event");
    assert_eq!(usage.prompt_tokens, 100);
    assert_eq!(usage.reasoning_tokens, Some(30));
    assert_eq!(usage.cached_tokens, Some(20));
}

#[tokio::test]
async fn a_stream_without_usage_does_not_invent_it() {
    // §45: absent usage must stay absent. Zero tokens is a different fact.
    let server = streaming(
        sse(&[frame("chatcmpl-8", "gpt-4o", 1,
            json!([{"index": 0, "delta": {}, "finish_reason": "stop"}]))])
            + "data: [DONE]\n\n",
    )
    .await;

    let events = collect(&server).await;
    assert!(
        !events
            .iter()
            .any(|e| matches!(e.as_event(), Some(ChatStreamEvent::Usage(_)))),
        "no usage event may be synthesized: {events:?}"
    );
}

// ── Identity is preserved, not fabricated ──────────────────────────────────

#[tokio::test]
async fn the_provider_id_and_model_are_preserved() {
    let server = streaming(
        sse(&[frame("chatcmpl-real-id", "gpt-4o-2024-08-06", 1_700_000_000, delta("hi"))])
            + "data: [DONE]\n\n",
    )
    .await;

    let events = collect(&server).await;
    match events.iter().filter_map(Item::as_event).find(|e| matches!(e, ChatStreamEvent::ResponseStart { .. })) {
        Some(ChatStreamEvent::ResponseStart {
            id,
            model,
            created,
            system_fingerprint,
        }) => {
            assert_eq!(id, "chatcmpl-real-id");
            // The *returned* model, which differs from the requested one here.
            assert_eq!(model, "gpt-4o-2024-08-06");
            assert_eq!(*created, Some(1_700_000_000));
            assert!(system_fingerprint.is_none());
        }
        other => panic!("no response start: {other:?}"),
    }
}

#[tokio::test]
async fn the_system_fingerprint_is_captured_when_sent() {
    let mut f = frame("chatcmpl-9", "gpt-4o", 1, delta("hi"));
    f["system_fingerprint"] = json!("fp_44709d6fcb");
    let server = streaming(sse(&[f]) + "data: [DONE]\n\n").await;

    let events = collect(&server).await;
    assert!(events.iter().any(|e| matches!(
        e.as_event(),
        Some(ChatStreamEvent::ResponseStart { system_fingerprint: Some(f), .. }) if f == "fp_44709d6fcb"
    )));
}

// ── Failure modes ──────────────────────────────────────────────────────────

#[tokio::test]
async fn a_disconnect_before_done_is_an_error() {
    // The single most dangerous case: a partial answer reported as complete.
    let server = streaming(sse(&[frame("chatcmpl-10", "gpt-4o", 1, delta("half"))])).await;

    use vincents_llm::provider::LLMProvider;
    let mut stream = provider(&server.uri())
        .chat_completion_stream_events(request())
        .await
        .expect("stream opens");
    let mut text = String::new();
    let mut error = None;
    while let Some(event) = stream.next().await {
        match event {
            Ok(ChatStreamEvent::TextDelta { text: t, .. }) => text.push_str(&t),
            Ok(_) => {}
            Err(e) => {
                error = Some(e);
                break;
            }
        }
    }
    assert_eq!(text, "half", "the partial text is still delivered");
    let error = error.expect("a truncated stream must surface an error");
    assert!(matches!(error, LLMError::StreamInterrupted { .. }), "{error:?}");
    assert!(error.is_retryable(), "the request was understood; transit failed");
}

#[tokio::test]
async fn a_malformed_frame_is_skipped_without_ending_the_stream() {
    // One bad frame must not end a live stream: the provider's terminator is
    // what ends it.
    let body = format!(
        "{}{}{}{}",
        sse(&[frame("chatcmpl-11", "gpt-4o", 1, delta("before"))]),
        "data: {not valid json at all}\n\n",
        sse(&[frame("chatcmpl-11", "gpt-4o", 1, delta("after"))]),
        "data: [DONE]\n\n",
    );
    let server = streaming(body).await;

    let events = collect(&server).await;
    let text: String = events
        .iter()
        .filter_map(|e| match e.as_event() {
            Some(ChatStreamEvent::TextDelta { text, .. }) => Some(text.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(text, "beforeafter", "a bad frame is skipped, not fatal");
}

#[tokio::test]
async fn an_empty_delta_is_skipped_rather_than_emitted_as_empty_text() {
    // A role-only opening delta is normal. Emitting it as `""` would put a
    // meaningless event in the stream.
    let server = streaming(
        sse(&[
            frame("chatcmpl-12", "gpt-4o", 1,
                json!([{"index": 0, "delta": {"role": "assistant"}, "finish_reason": null}])),
            frame("chatcmpl-12", "gpt-4o", 1, delta("real")),
        ]) + "data: [DONE]\n\n",
    )
    .await;

    let events = collect(&server).await;
    let deltas: Vec<&String> = events
        .iter()
        .filter_map(|e| match e.as_event() {
            Some(ChatStreamEvent::TextDelta { text, .. }) => Some(text),
            _ => None,
        })
        .collect();
    assert_eq!(deltas, vec!["real"]);
}

#[tokio::test]
async fn a_frame_with_no_id_does_not_suppress_a_later_valid_start() {
    // Some providers open with an event that has no identity. Requiring an id on
    // the *first* frame would lose the real one.
    let body = format!(
        "{}{}",
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"lead\"}}]}\n\n",
        sse(&[frame("chatcmpl-13", "gpt-4o", 1, delta("body"))]) + "data: [DONE]\n\n",
    );
    let server = streaming(body).await;

    let events = collect(&server).await;
    let text: String = events
        .iter()
        .filter_map(|e| match e.as_event() {
            Some(ChatStreamEvent::TextDelta { text, .. }) => Some(text.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(text, "leadbody");
    assert!(events.iter().any(|e| matches!(
        e.as_event(),
        Some(ChatStreamEvent::ResponseStart { id, .. }) if id == "chatcmpl-13"
    )));
    for item in &events {
        assert!(
            item.error_message().is_none(),
            "an id-less first frame is not an error, got {:?}",
            item.error_message()
        );
    }
}