//! Wire-conformance harness for the OpenAI and OpenRouter chat providers.
//!
//! The point of this file is to pin **actual wire behaviour**, not "the Rust
//! structs compile". Every test drives a real provider against a mock HTTP
//! server and asserts on the exact request body or the parsed response. A test
//! that only checks types cannot catch a field being dropped on the way to the
//! network, which is the entire class of bug this brief exists to close.
//!
//! The two regression tests at the top are deliberate: they are **known-broken
//! today** and are written to fail. They are the P0 bugs from the brief,
//! reproduced precisely so the fix has something to turn green.
//!
//! `wiremock` is already a dev-dependency.

use serde_json::{json, Value};
use vincents_llm::provider::LLMProvider;
use vincents_llm::types::{ChatCompletionRequest, ChatMessage};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// A request body captured by the mock, parsed.
async fn captured_body(server: &MockServer) -> Value {
    let requests = server
        .received_requests()
        .await
        .expect("wiremock records requests");
    assert_eq!(requests.len(), 1, "expected exactly one request");
    serde_json::from_slice(&requests[0].body).expect("the body must be JSON")
}

async fn captured_headers(server: &MockServer) -> Vec<(String, String)> {
    let requests = server.received_requests().await.expect("recorded");
    requests[0]
        .headers
        .iter()
        .map(|(k, v)| {
            (
                k.as_str().to_string(),
                v.as_str().to_string(),
            )
        })
        .collect()
}

fn openai(server: &MockServer) -> vincents_llm::openai::OpenAIProvider {
    vincents_llm::openai::OpenAIProvider::with_config(vincents_llm::openai::OpenAIConfig {
        api_key: "test-key".into(),
        base_url: Some(server.uri()),
        default_model: Some("gpt-4o".into()),
        ..Default::default()
    })
    .expect("provider constructs")
}

fn openrouter(server: &MockServer) -> vincents_llm::openrouter::OpenRouterProvider {
    vincents_llm::openrouter::OpenRouterProvider::with_config(
        vincents_llm::openrouter::OpenRouterConfig {
        api_key: "test-key".into(),
        base_url: Some(server.uri()),
            default_model: Some("openai/gpt-4o".into()),
            ..Default::default()
        },
    )
    .expect("provider constructs")
}

fn text_request() -> ChatCompletionRequest {
    ChatCompletionRequest {
        model: "gpt-4o".into(),
        messages: vec![ChatMessage::user("hello")],
        ..Default::default()
    }
}

fn completion_body(content: &str) -> Value {
    json!({
        "id": "chatcmpl-real-id",
        "object": "chat.completion",
        "created": 1_700_000_000,
        "model": "gpt-4o-2024-08-06",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 9, "completion_tokens": 5, "total_tokens": 14}
    })
}

/// An SSE body from a list of pre-serialized event payloads.
fn sse(events: &[&str]) -> String {
    let mut body = String::new();
    for event in events {
        body.push_str("data: ");
        body.push_str(event);
        body.push_str("\n\n");
    }
    body.push_str("data: [DONE]\n\n");
    body
}

/// A realistic streaming text response, with the provider's own id and created.
fn text_sse() -> String {
    sse(&[
        &json!({
            "id": "chatcmpl-stream-real",
            "object": "chat.completion.chunk",
            "created": 1_700_000_123,
            "model": "gpt-4o-2024-08-06",
            "system_fingerprint": "fp_44709d6fcb",
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": null}]
        })
        .to_string(),
        &json!({
            "id": "chatcmpl-stream-real",
            "object": "chat.completion.chunk",
            "created": 1_700_000_123,
            "model": "gpt-4o-2024-08-06",
            "choices": [{"index": 0, "delta": {"content": "Hello"}, "finish_reason": null}]
        })
        .to_string(),
        &json!({
            "id": "chatcmpl-stream-real",
            "object": "chat.completion.chunk",
            "created": 1_700_000_123,
            "model": "gpt-4o-2024-08-06",
            "choices": [{"index": 0, "delta": {"content": " world"}, "finish_reason": null}]
        })
        .to_string(),
        &json!({
            "id": "chatcmpl-stream-real",
            "object": "chat.completion.chunk",
            "created": 1_700_000_123,
            "model": "gpt-4o-2024-08-06",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 9, "completion_tokens": 2, "total_tokens": 11}
        })
        .to_string(),
    ])
}

// ═══════════════════════════════════════════════════════════════════════════
// P0 regression tests — these FAIL today. That is the point.
// ═══════════════════════════════════════════════════════════════════════════

/// P0: OpenRouter converts `Tool` and `Function` messages into ordinary
/// `user` messages, which destroys a native tool-call conversation.
#[tokio::test]
async fn p0_openrouter_preserves_tool_and_function_roles() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(completion_body("ok")))
        .mount(&server)
        .await;

    let provider = openrouter(&server);
    let request = ChatCompletionRequest {
        model: "openai/gpt-4o".into(),
        messages: vec![
            ChatMessage::user("what is the weather?"),
            ChatMessage::assistant("let me check"),
            ChatMessage::Tool {
                tool_call_id: "call_1".into(),
                content: "72F and sunny".into(),
            },
        ],
        ..Default::default()
    };
    provider
        .chat_completion(request)
        .await
        .expect("request succeeds");

    let body = captured_body(&server).await;
    let messages = body["messages"].as_array().expect("messages array");

    assert_eq!(
        messages[2]["role"], "tool",
        "a Tool message must stay role=tool, not be rewritten to user: {}",
        body["messages"]
    );

    // And a `Function` message is the legacy spelling of the same thing.
    let server2 = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(completion_body("ok")))
        .mount(&server2)
        .await;
    let provider2 = openrouter(&server2);
    provider2
        .chat_completion(ChatCompletionRequest {
            model: "openai/gpt-4o".into(),
            messages: vec![ChatMessage::Function {
                name: "get_weather".into(),
                arguments: "{}".into(),
            }],
            ..Default::default()
        })
        .await
        .expect("request succeeds");
    let body2 = captured_body(&server2).await;
    assert_eq!(
        body2["messages"][0]["role"], "function",
        "a Function message must stay role=function: {}",
        body2["messages"]
    );
}

/// P0: OpenAI streaming builds a body that keeps only `max_tokens`,
/// `temperature` and `top_p`. Every other typed field is silently dropped.
#[tokio::test]
async fn p0_openai_streaming_request_keeps_every_typed_parameter() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(text_sse(), "text/event-stream"))
        .mount(&server)
        .await;

    let mut request = text_request();
    request.max_tokens = Some(256);
    request.temperature = Some(0.3);
    request.top_p = Some(0.9);
    request.n = Some(2);
    request.stop = vec!["END".into()];
    request.presence_penalty = Some(0.1);
    request.frequency_penalty = Some(0.2);
    request.user = Some("user-42".into());

    let mut stream = openai(&server)
        .chat_completion_stream(request)
        .await
        .expect("stream opens");
    // Drain so the request is fully sent and the mock has recorded it.
    use futures::StreamExt;
    while let Some(chunk) = stream.next().await {
        let _ = chunk;
    }

    let body = captured_body(&server).await;
    assert_eq!(body["stream"], json!(true));
    assert_eq!(body["max_tokens"], json!(256));
    assert_eq!(body["n"], json!(2), "n is dropped by the streaming builder");
    assert_eq!(body["stop"], json!(["END"]), "stop is dropped by the streaming builder");
    assert_eq!(
        body["presence_penalty"],
        json!(0.1),
        "presence_penalty is dropped by the streaming builder"
    );
    assert_eq!(
        body["frequency_penalty"],
        json!(0.2),
        "frequency_penalty is dropped by the streaming builder"
    );
    assert_eq!(body["user"], json!("user-42"), "user is dropped by the streaming builder");
}

/// P0: the streaming response fabricates its own id and timestamp, and appends
/// a synthetic `finish_reason: "stop"` chunk after the provider stream ends.
#[tokio::test]
async fn p0_openai_stream_preserves_provider_metadata_and_finish_reason() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(text_sse(), "text/event-stream"))
        .mount(&server)
        .await;

    let mut stream = openai(&server)
        .chat_completion_stream(text_request())
        .await
        .expect("stream opens");
    use futures::StreamExt;

    let mut ids = Vec::new();
    let mut createds = Vec::new();
    let mut models = Vec::new();
    let mut finish_reasons = Vec::new();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.expect("chunk");
        let vincents_llm::types::Chunk::Completion(c) = chunk;
        {
            ids.push(c.id.clone());
            createds.push(c.created);
            models.push(c.model.clone());
            for choice in &c.choices {
                if let Some(reason) = &choice.finish_reason {
                    finish_reasons.push(reason.clone());
                }
            }
        }
    }

    assert!(
        ids.iter().all(|id| id == "chatcmpl-stream-real"),
        "every chunk must carry the provider's id, not a fabricated one: {ids:?}"
    );
    assert!(
        createds.iter().all(|c| *c == 1_700_000_123),
        "the provider's timestamp must be preserved, not regenerated: {createds:?}"
    );
    assert!(
        models.iter().all(|m| m == "gpt-4o-2024-08-06"),
        "the provider's model must be preserved: {models:?}"
    );
    assert_eq!(
        finish_reasons,
        vec!["stop".to_string()],
        "exactly the provider's finish reason, with no synthetic trailing chunk: {finish_reasons:?}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Streaming that must survive: tool calls
// ═══════════════════════════════════════════════════════════════════════════

/// Two parallel tool calls, interleaved exactly as OpenAI emits them, with
/// fragmented argument JSON. Nothing may be dropped or reordered.
#[tokio::test]
async fn tool_call_deltas_are_preserved_across_interleaved_parallel_calls() {
    let server = MockServer::start().await;
    let events = sse(&[
        &json!({
            "id": "chatcmpl-tools", "object": "chat.completion.chunk",
            "created": 1_700_000_500, "model": "gpt-4o-2024-08-06",
            "choices": [{"index": 0, "delta": {"role": "assistant", "tool_calls": [
                {"index": 0, "id": "call_a", "type": "function", "function": {"name": "get_weather", "arguments": ""}},
                {"index": 1, "id": "call_b", "type": "function", "function": {"name": "get_time", "arguments": ""}}
            ]}, "finish_reason": null}]
        })
        .to_string(),
        &json!({
            "id": "chatcmpl-tools", "object": "chat.completion.chunk",
            "created": 1_700_000_500, "model": "gpt-4o-2024-08-06",
            "choices": [{"index": 0, "delta": {"tool_calls": [
                {"index": 0, "function": {"arguments": "{\"city\":"}}
            ]}, "finish_reason": null}]
        })
        .to_string(),
        &json!({
            "id": "chatcmpl-tools", "object": "chat.completion.chunk",
            "created": 1_700_000_500, "model": "gpt-4o-2024-08-06",
            "choices": [{"index": 0, "delta": {"tool_calls": [
                {"index": 1, "function": {"arguments": "{\"tz\":"}}
            ]}, "finish_reason": null}]
        })
        .to_string(),
        &json!({
            "id": "chatcmpl-tools", "object": "chat.completion.chunk",
            "created": 1_700_000_500, "model": "gpt-4o-2024-08-06",
            "choices": [{"index": 0, "delta": {"tool_calls": [
                {"index": 0, "function": {"arguments": "\"Oslo\"}"}}
            ]}, "finish_reason": null}]
        })
        .to_string(),
        &json!({
            "id": "chatcmpl-tools", "object": "chat.completion.chunk",
            "created": 1_700_000_500, "model": "gpt-4o-2024-08-06",
            "choices": [{"index": 0, "delta": {"tool_calls": [
                {"index": 1, "function": {"arguments": "\"UTC\"}"}}
            ]}, "finish_reason": null}]
        })
        .to_string(),
        &json!({
            "id": "chatcmpl-tools", "object": "chat.completion.chunk",
            "created": 1_700_000_500, "model": "gpt-4o-2024-08-06",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
            "usage": {"prompt_tokens": 30, "completion_tokens": 18, "total_tokens": 48}
        })
        .to_string(),
    ]);
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(events, "text/event-stream"))
        .mount(&server)
        .await;

    let mut stream = openai(&server)
        .chat_completion_stream(text_request())
        .await
        .expect("stream opens");
    use futures::StreamExt;

    // Whatever the provider emitted must survive, in order.
    let mut raw: Vec<Value> = Vec::new();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.expect("chunk");
        let vincents_llm::types::Chunk::Completion(c) = chunk;
        {
            raw.push(serde_json::to_value(&c).unwrap_or(Value::Null));
        }
    }

    // A hard check that this is not vacuous: there must be at least one chunk
    // whose delta mentions a tool.
    let joined = serde_json::to_string(&raw).unwrap_or_default();
    assert!(
        joined.contains("call_a") && joined.contains("call_b"),
        "both tool call ids must appear in the emitted chunks. A tool-call-only \
         response currently yields {} chunk(s), all of them synthetic: {joined}",
        raw.len()
    );
    assert!(
        joined.contains("get_weather") && joined.contains("get_time"),
        "both tool names must appear: {joined}"
    );
    assert!(
        joined.contains("Oslo") && joined.contains("UTC"),
        "argument fragments must appear: {joined}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Non-streaming conformance — the shape streaming must match
// ═══════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn non_streaming_request_carries_every_typed_parameter() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(completion_body("hi")))
        .mount(&server)
        .await;

    let mut request = text_request();
    request.max_tokens = Some(256);
    request.temperature = Some(0.3);
    request.top_p = Some(0.9);
    request.n = Some(2);
    request.stop = vec!["END".into()];
    request.presence_penalty = Some(0.1);
    request.frequency_penalty = Some(0.2);
    request.logit_bias.insert("50256".into(), -100);
    request.user = Some("user-42".into());
    request.extra_params.insert("custom".into(), json!(true));

    openai(&server)
        .chat_completion(request)
        .await
        .expect("request succeeds");

    let body = captured_body(&server).await;
    for (field, expected) in [
        ("max_tokens", json!(256)),
        ("temperature", json!(0.3)),
        ("top_p", json!(0.9)),
        ("n", json!(2)),
        ("stop", json!(["END"])),
        ("presence_penalty", json!(0.1)),
        ("frequency_penalty", json!(0.2)),
        ("logit_bias", json!({"50256": -100})),
        ("user", json!("user-42")),
        ("custom", json!(true)),
    ] {
        assert_eq!(body[field], expected, "non-streaming body lost {field}");
    }
}

/// The same request through both paths must produce the same body apart from
/// `stream`. This is the §17 structural requirement, expressed as a test.
#[tokio::test]
async fn streaming_and_non_streaming_bodies_differ_only_by_stream() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(completion_body("hi")))
        .mount(&server)
        .await;

    let mut request = text_request();
    request.max_tokens = Some(64);
    request.temperature = Some(0.1);
    request.n = Some(1);
    request.stop = vec!["X".into()];

    openai(&server)
        .chat_completion(request.clone())
        .await
        .expect("non-streaming");
    let non_stream_body = captured_body(&server).await;

    let server2 = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(text_sse(), "text/event-stream"))
        .mount(&server2)
        .await;
    let mut stream = openai(&server2)
        .chat_completion_stream(request)
        .await
        .expect("stream opens");
    use futures::StreamExt;
    while let Some(c) = stream.next().await {
        let _ = c;
    }
    let stream_body = captured_body(&server2).await;

    let mut a = non_stream_body.clone();
    let mut b = stream_body.clone();
    a.as_object_mut().unwrap().remove("stream");
    b.as_object_mut().unwrap().remove("stream");
    assert_eq!(
        a, b,
        "the two paths build different bodies: one of them is dropping fields"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Headers
// ═══════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn openai_sends_authorization_and_organization() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(completion_body("hi")))
        .mount(&server)
        .await;

    let provider =
        vincents_llm::openai::OpenAIProvider::with_config(vincents_llm::openai::OpenAIConfig {
            api_key: "sk-test".into(),
            organization_id: Some("org-123".into()),
            base_url: Some(server.uri()),
            default_model: Some("gpt-4o".into()),
            ..Default::default()
        })
        .expect("provider");

    provider
        .chat_completion(text_request())
        .await
        .expect("request succeeds");

    let headers = captured_headers(&server).await;
    let has = |name: &str| {
        headers
            .iter()
            .any(|(k, v)| k == name && v.to_lowercase().contains("sk-test"))
    };
    assert!(has("authorization"), "auth header missing: {headers:?}");
    let org = headers
        .iter()
        .find(|(k, _)| k == "openai-organization")
        .map(|(_, v)| v.as_str());
    assert_eq!(org, Some("org-123"), "organization header missing: {headers:?}");
}

// ═══════════════════════════════════════════════════════════════════════════
// Usage
// ═══════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn non_streaming_usage_is_parsed() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(completion_body("hi")))
        .mount(&server)
        .await;

    let response = openai(&server)
        .chat_completion(text_request())
        .await
        .expect("request succeeds");

    let usage = response.usage.expect("usage must be present, not defaulted");
    assert_eq!(usage.prompt_tokens, 9);
    assert_eq!(usage.completion_tokens, 5);
    assert_eq!(usage.total_tokens, 14);
}

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Guard against the mock silently never being hit, which would make every
/// assertion in this file vacuously true.
#[tokio::test]
async fn the_harness_actually_reaches_the_server() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(completion_body("hi")))
        .mount(&server)
        .await;

    openai(&server)
        .chat_completion(text_request())
        .await
        .expect("request succeeds");
    // Assert the request was actually observed, so a misrouted URL cannot make
    // the rest of this file pass vacuously.
    let seen = server.received_requests().await.expect("recorded");
    assert_eq!(seen.len(), 1, "the harness must reach the mock server");
    assert_eq!(seen[0].url.path(), "/chat/completions");
}

/// The brief's §5 required round-trip matrix, asserted on the wire.
///
/// The P0 test above checks the two roles that were being rewritten. This one
/// covers the whole set, because a fix that restores `tool` while dropping
/// `tool_call_id` — or dropping assistant `tool_calls` — would pass that test and
/// still break the conversation.
#[tokio::test]
async fn openrouter_preserves_the_whole_message_vocabulary() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(completion_body("ok")))
        .mount(&server)
        .await;

    use vincents_llm::types::{CustomFunctionCall, FunctionCall, ToolCall};
    let request = ChatCompletionRequest {
        model: "openai/gpt-4o".into(),
        messages: vec![
            ChatMessage::system("be brief"),
            ChatMessage::user("what is the weather in Oslo?"),
            ChatMessage::Assistant {
                content: None,
                tool_calls: Some(vec![ToolCall::Function(FunctionCall::Custom(
                    CustomFunctionCall {
                        id: Some("call_a".into()),
                        name: "get_weather".into(),
                        arguments: r#"{"city":"Oslo"}"#.into(),
                    },
                ))]),
                name: None,
            },
            ChatMessage::Tool {
                tool_call_id: "call_a".into(),
                content: "72F and sunny".into(),
            },
            ChatMessage::Function {
                name: "get_time".into(),
                arguments: r#"{"tz":"UTC"}"#.into(),
            },
        ],
        ..Default::default()
    };

    openrouter(&server)
        .chat_completion(request)
        .await
        .expect("request succeeds");

    let body = captured_body(&server).await;
    let messages = body["messages"].as_array().expect("messages array");
    assert_eq!(messages[0]["role"], "system", "{body}");
    assert_eq!(messages[1]["role"], "user", "{body}");
    assert_eq!(messages[2]["role"], "assistant", "{body}");
    assert_eq!(
        messages[2]["tool_calls"][0]["id"],
        "call_a",
        "assistant tool_calls must survive: {body}"
    );
    assert_eq!(
        messages[2]["tool_calls"][0]["function"]["name"],
        "get_weather",
        "{body}"
    );
    assert_eq!(
        messages[2]["content"],
        Value::Null,
        "a tool-call turn has null content, not an empty string: {body}"
    );
    assert_eq!(messages[3]["role"], "tool", "{body}");
    assert_eq!(
        messages[3]["tool_call_id"], "call_a",
        "tool_call_id links the result back to the call and must survive: {body}"
    );
    assert_eq!(messages[4]["role"], "function", "{body}");
    assert_eq!(messages[4]["name"], "get_time", "{body}");
}

/// The same conversation through both providers must serialise identically.
///
/// OpenRouter speaks the OpenAI dialect, so any difference here is a bug in one
/// of them rather than a legitimate protocol difference.
#[tokio::test]
async fn openai_and_openrouter_serialise_the_same_conversation_identically() {
    let conversation = vec![
        ChatMessage::system("be brief"),
        ChatMessage::user("hi"),
        ChatMessage::Tool {
            tool_call_id: "call_a".into(),
            content: "result".into(),
        },
    ];

    let oa_server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(completion_body("ok")))
        .mount(&oa_server)
        .await;
    openai(&oa_server)
        .chat_completion(ChatCompletionRequest {
            model: "gpt-4o".into(),
            messages: conversation.clone(),
            ..Default::default()
        })
        .await
        .expect("openai request");
    let oa_messages = captured_body(&oa_server).await["messages"].clone();

    let or_server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(completion_body("ok")))
        .mount(&or_server)
        .await;
    openrouter(&or_server)
        .chat_completion(ChatCompletionRequest {
            model: "openai/gpt-4o".into(),
            messages: conversation,
            ..Default::default()
        })
        .await
        .expect("openrouter request");
    let or_messages = captured_body(&or_server).await["messages"].clone();

    assert_eq!(
        oa_messages, or_messages,
        "the two providers diverge on the same conversation"
    );
}
