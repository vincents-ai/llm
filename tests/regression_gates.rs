//! §75. Hard regression gates.
//!
//! "Block merge if any of these occur." Each named gate in the brief is one test
//! here, so a regression is a red build rather than something a reviewer has to
//! remember to look for.
//!
//! Every gate here corresponds to a defect that was actually live at some point
//! during this work. That is the point: a gate named after a bug that shipped is
//! worth more than a gate named after a rule.

use serde_json::{json, Value};
use std::collections::BTreeMap;
use vincents_llm::types::{
    ChatCompletionRequest, ChatMessage, ContentPart, FunctionDefinitionWire, MessageContent,
    ReasoningConfig, ReasoningEffort, ResponseFormat, ToolCall, ToolChoice, ToolDefinition,
};
use vincents_llm::wire::{build_chat_request, ChatRequestParts};
use vincents_llm::provider::LLMProvider as _;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn messages() -> Vec<Value> {
    vec![json!({"role": "user", "content": "hi"})]
}

fn build(request: &ChatCompletionRequest) -> Result<Value, vincents_llm::error::LLMError> {
    build_chat_request(ChatRequestParts {
        model: "gpt-4o",
        messages: &messages(),
        request,
        stream: false,
        tools: None,
        default_service_tier: None,
        provider_params: None,
    })
}

fn base() -> ChatCompletionRequest {
    ChatCompletionRequest {
        model: "gpt-4o".into(),
        messages: vec![ChatMessage::user("hi")],
        ..Default::default()
    }
}

fn provider(base_url: &str) -> vincents_llm::openai::OpenAIProvider {
    vincents_llm::openai::OpenAIProvider::with_config(vincents_llm::openai::OpenAIConfig {
        api_key: "k".into(),
        base_url: Some(base_url.to_string()),
        default_model: Some("gpt-4o".into()),
        ..Default::default()
    })
    .expect("provider")
}

// ═══ 1. A tool message must not become user text ═══════════════════════════

#[test]
fn gate_tool_message_never_becomes_user_text() {
    let wire = ChatMessage::Tool {
        tool_call_id: "call_a".into(),
        content: "42".into(),
    }
    .to_openai_json();
    assert_eq!(wire["role"], "tool", "{wire}");
    assert_ne!(wire["role"], "user", "this was the OpenRouter P0 defect");
}

// ═══ 2. `tool_call_id` must survive ════════════════════════════════════════

#[test]
fn gate_tool_call_id_survives() {
    let wire = ChatMessage::Tool {
        tool_call_id: "call_abc".into(),
        content: "42".into(),
    }
    .to_openai_json();
    assert_eq!(wire["tool_call_id"], "call_abc");
}

// ═══ 3. Assistant tool calls must survive ═════════════════════════════════

#[test]
fn gate_assistant_tool_calls_survive() {
    let wire = ChatMessage::assistant_with_tool_calls(vec![ToolCall::Function(
        vincents_llm::types::FunctionCall::Custom(vincents_llm::types::CustomFunctionCall {
            id: Some("call_a".into()),
            name: "get_weather".into(),
            arguments: r#"{"city":"Oslo"}"#.into(),
        }),
    )])
    .to_openai_json();
    assert_eq!(wire["role"], "assistant");
    assert_eq!(wire["tool_calls"][0]["id"], "call_a");
    assert_eq!(wire["tool_calls"][0]["function"]["name"], "get_weather");
    assert_eq!(wire["tool_calls"][0]["function"]["arguments"], r#"{"city":"Oslo"}"#);
    // `content: null`, not `""`: the API distinguishes "no content" from "empty".
    assert!(wire["content"].is_null(), "{wire}");
}

// ═══ 4. A streaming request must not drop a typed parameter ════════════════

#[tokio::test]
async fn gate_a_streaming_request_drops_no_typed_parameter() {
    // The streaming path builds its own body. If it ever stops using the shared
    // builder, this is what catches it.
    // A real SSE body: a JSON body would make the stream truncated, and the test
    // would fail for the wrong reason.
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_raw("data: [DONE]\n\n", "text/event-stream"),
        )
        .mount(&server)
        .await;
    let mut request = base();
    request.max_completion_tokens = Some(256);
    request.temperature = Some(0.3);
    request.seed = Some(7);
    request.logprobs = Some(true);
    request.top_logprobs = Some(2);
    request.reasoning = Some(ReasoningConfig {
        effort: Some(ReasoningEffort("high".into())),
        max_tokens: None,
        exclude: None,
        summary: None,
    });
    request.response_format = Some(ResponseFormat::JsonObject);

    use futures::StreamExt;
    let mut stream = provider(&server.uri())
        .chat_completion_stream_events(request.clone())
        .await
        .expect("stream opens");
    while let Some(event) = stream.next().await {
        event.expect("event");
    }

    let sent = server.received_requests().await.expect("recorded");
    let streamed: Value = serde_json::from_slice(&sent[0].body).expect("json");
    let non_streamed = build(&request).expect("built");

    for key in [
        "max_completion_tokens",
        "temperature",
        "seed",
        "logprobs",
        "top_logprobs",
        "reasoning",
        "response_format",
    ] {
        assert_eq!(
            streamed.get(key),
            non_streamed.get(key),
            "the streaming path dropped or altered '{key}': {streamed}"
        );
    }
    assert_eq!(streamed["stream"], true);
}

// ═══ 5. Tool arguments must not be reordered or corrupted ═══════════════════

#[tokio::test]
async fn gate_streaming_tool_arguments_are_not_reordered() {
    // Arrival-order assembly yields `{"city":{"tz":"Oslo"}` for the first call:
    // valid JSON, wrong call.
    let frames = [
        json!({"id": "c", "created": 1, "model": "m", "choices": [
            {"index": 0, "finish_reason": null, "delta": {"tool_calls": [
                {"index": 0, "id": "a", "function": {"name": "weather", "arguments": ""}},
                {"index": 1, "id": "b", "function": {"name": "time", "arguments": ""}}
            ]}}]}),
        json!({"id": "c", "created": 1, "model": "m", "choices": [
            {"index": 0, "finish_reason": null, "delta": {"tool_calls": [
                {"index": 0, "function": {"arguments": "{\"city\":"}}]}}]}),
        json!({"id": "c", "created": 1, "model": "m", "choices": [
            {"index": 0, "finish_reason": null, "delta": {"tool_calls": [
                {"index": 1, "function": {"arguments": "{\"tz\":"}}]}}]}),
        json!({"id": "c", "created": 1, "model": "m", "choices": [
            {"index": 0, "finish_reason": null, "delta": {"tool_calls": [
                {"index": 0, "function": {"arguments": "\"Oslo\"}"}}]}}]}),
        json!({"id": "c", "created": 1, "model": "m", "choices": [
            {"index": 0, "finish_reason": null, "delta": {"tool_calls": [
                {"index": 1, "function": {"arguments": "\"UTC\"}"}}]}}]}),
    ]
    .iter()
    .map(|frame| format!("data: {frame}\n\n"))
    .collect::<String>()
        + "data: [DONE]\n\n";

    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(frames, "text/event-stream"))
        .mount(&server)
        .await;

    use futures::StreamExt;
    use vincents_llm::types::{ChatStreamEvent, ToolCallAssembler};
    let mut stream = provider(&server.uri())
        .chat_completion_stream_events(base())
        .await
        .expect("stream opens");
    let mut assembler = ToolCallAssembler::new();
    while let Some(event) = stream.next().await {
        match event.expect("event") {
            ChatStreamEvent::ToolCallStart { tool_index, id, name, .. } => {
                assembler.start(tool_index, id, name)
            }
            ChatStreamEvent::ToolCallArgumentsDelta { tool_index, delta, .. } => {
                assembler.push_arguments(tool_index, &delta)
            }
            _ => {}
        }
    }

    let assembled = assembler.finish();
    let by_name: BTreeMap<&str, &str> = assembled
        .iter()
        .map(|call| {
            (
                call.name.as_deref().unwrap_or_default(),
                call.arguments.as_str(),
            )
        })
        .collect();
    assert_eq!(by_name["weather"], r#"{"city":"Oslo"}"#);
    assert_eq!(by_name["time"], r#"{"tz":"UTC"}"#);
    // And each assembled argument string is the JSON the model meant.
    for arguments in by_name.values() {
        serde_json::from_str::<Value>(arguments)
            .unwrap_or_else(|e| panic!("corrupted arguments {arguments:?}: {e}"));
    }
}

// ═══ 6. Final usage must not be silently discarded ═══════════════════════

#[tokio::test]
async fn gate_final_usage_is_never_discarded() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "c", "object": "chat.completion", "created": 1, "model": "gpt-4o",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 500, "total_tokens": 600,
                      "completion_tokens_details": {"reasoning_tokens": 480}}
        })))
        .mount(&server)
        .await;

    use vincents_llm::provider::LLMProvider;
    let response = provider(&server.uri())
        .chat_completion(base())
        .await
        .expect("ok");
    let usage = response.usage.expect("usage reported");
    assert_eq!(usage.total_tokens, 600);
    assert_eq!(usage.reasoning_tokens, Some(480));
    assert_eq!(usage.visible_completion_tokens(), 20);
}

// ═══ 7. Reasoning tokens must count in session/quota accounting ═══════════

#[tokio::test]
async fn gate_reasoning_tokens_are_inside_the_total() {
    // They are a subset of completion, not an addition. Quota and session totals
    // must include them, or a reasoning model appears to cost almost nothing.
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "c", "object": "chat.completion", "created": 1, "model": "o3",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "42"},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 500, "total_tokens": 510,
                      "completion_tokens_details": {"reasoning_tokens": 480}}
        })))
        .mount(&server)
        .await;

    use vincents_llm::provider::LLMProvider;
    let response = provider(&server.uri())
        .chat_completion(base())
        .await
        .expect("ok");
    let usage = response.usage.expect("usage");
    assert_eq!(
        usage.total_tokens, 510,
        "the total already includes reasoning; it must not be added again"
    );
    assert!(usage.reasoning_tokens.unwrap_or(0) <= usage.completion_tokens);
}

// ═══ 8. A structured-output schema must not be altered ═════════════════════

#[test]
fn gate_structured_output_schema_is_not_altered() {
    let schema = json!({
        "type": "object",
        "properties": {"answer": {"type": "number"}},
        "required": ["answer"],
        "additionalProperties": false
    });
    let mut request = base();
    request.response_format = Some(ResponseFormat::JsonSchema {
        name: "answer".into(),
        description: None,
        schema: schema.clone(),
        strict: Some(true),
    });
    let body = build(&request).expect("built");
    let sent = &body["response_format"];
    assert_eq!(sent["type"], "json_schema");
    assert_eq!(sent["json_schema"]["name"], "answer");
    assert_eq!(sent["json_schema"]["strict"], true);
    assert_eq!(
        sent["json_schema"]["schema"], schema,
        "the schema the caller wrote must be the schema that is sent"
    );
}

// ═══ 9. Provider routing/privacy policy must not be dropped ═══════════════

#[tokio::test]
async fn gate_openrouter_privacy_policy_is_never_dropped() {
    // A silently dropped `data_collection: "deny"` is the worst gate on this
    // list: the deployment believes it is protecting data and is not.
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "c", "object": "chat.completion", "created": 1, "model": "m",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"},
                         "finish_reason": "stop"}]
        })))
        .mount(&server)
        .await;

    let mut provider = vincents_llm::openrouter::OpenRouterProvider::with_config(
        vincents_llm::openrouter::OpenRouterConfig {
            api_key: "k".into(),
            base_url: Some(server.uri()),
            ..Default::default()
        },
    )
    .expect("provider");
    provider.set_extensions(vincents_llm::openrouter::OpenRouterExtensions {
        provider: Some(vincents_llm::openrouter::OpenRouterProviderRouting {
            data_collection: Some(vincents_llm::openrouter::DataCollectionPolicy::Deny),
            zdr: Some(true),
            ..Default::default()
        }),
        ..Default::default()
    });

    use vincents_llm::provider::LLMProvider;
    provider
        .chat_completion(base())
        .await
        .expect("ok");

    let sent = server.received_requests().await.expect("recorded");
    let body: Value = serde_json::from_slice(&sent[0].body).expect("json");
    assert_eq!(body["provider"]["data_collection"], "deny", "{body}");
    assert_eq!(body["provider"]["zdr"], true);
}

// ═══ 10. Responses output must not be assumed text-only ═══════════════════

#[test]
fn gate_responses_output_is_not_assumed_to_be_text() {
    let response: vincents_llm::responses::Response = serde_json::from_value(json!({
        "id": "r", "object": "response", "created_at": 1, "model": "gpt-5",
        "status": "completed",
        "output": [
            {"type": "reasoning", "id": "rs", "summary": []},
            {"type": "function_call", "id": "fc", "call_id": "c", "name": "f", "arguments": "{}"}
        ]
    }))
    .expect("parses");

    // No message item at all. Indexing `output[0]` as text yields nothing.
    assert_eq!(response.output.len(), 2);
    assert!(matches!(
        response.output[0],
        vincents_llm::responses::ResponseOutputItem::Reasoning(_)
    ));
    assert_eq!(response.output_text(), "", "no message means empty, not a panic");
    assert_eq!(response.function_calls().len(), 1);
}

// ═══ 11. An OpenRouter fallback must not silently change the model ═════════

#[tokio::test]
async fn gate_a_fallback_is_visible_in_the_request() {
    // The fallback list must be in the body, so a reader can see the model may
    // have changed; and the *returned* model must be whatever actually served.
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "c", "object": "chat.completion", "created": 1,
            // The provider fell back.
            "model": "anthropic/claude-sonnet-4",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"},
                         "finish_reason": "stop"}]
        })))
        .mount(&server)
        .await;

    let mut provider = vincents_llm::openrouter::OpenRouterProvider::with_config(
        vincents_llm::openrouter::OpenRouterConfig {
            api_key: "k".into(),
            base_url: Some(server.uri()),
            ..Default::default()
        },
    )
    .expect("provider");
    provider.set_extensions(vincents_llm::openrouter::OpenRouterExtensions {
        models: vec!["anthropic/claude-sonnet-4".into()],
        ..Default::default()
    });

    use vincents_llm::provider::LLMProvider;
    let mut request = base();
    request.model = "openai/gpt-4o".into();
    let response = provider.chat_completion(request).await.expect("ok");

    let sent = server.received_requests().await.expect("recorded");
    let body: Value = serde_json::from_slice(&sent[0].body).expect("json");
    assert_eq!(
        body["models"][0], "anthropic/claude-sonnet-4",
        "the fallback must be visible in the request"
    );
    assert_eq!(
        response.model, "anthropic/claude-sonnet-4",
        "and the returned model is whatever actually served, not what was asked for"
    );
    assert_ne!(response.model, "openai/gpt-4o");
}

// ═══ 12. A raw provider param must not silently override a typed field ═════

#[test]
fn gate_a_raw_param_cannot_silently_override_a_typed_field() {
    let mut request = base();
    request.temperature = Some(0.7);
    request.extra_params.insert("temperature".into(), json!(0.1));
    let error = build(&request).expect_err("collision");
    assert!(
        matches!(
            error,
            vincents_llm::error::LLMError::ConfigurationError { .. }
        ),
        "{error:?}"
    );
}

// ═══ Additional gate: a tool_choice naming an absent tool ══════════════════

#[test]
fn gate_tool_choice_survives_and_names_a_real_tool() {
    let mut request = base();
    request.tools = Some(vec![ToolDefinition::Function(FunctionDefinitionWire {
        name: "get_weather".into(),
        description: None,
        parameters: None,
        strict: None,
    })]);
    request.tool_choice = Some(ToolChoice::Function { name: "get_weather".into() });
    let body = build(&request).expect("built");
    assert_eq!(
        body["tool_choice"],
        json!({"type": "function", "function": {"name": "get_weather"}})
    );
}

// ═══ Additional gate: multimodal content nesting ═══════════════════════════

#[test]
fn gate_multimodal_content_nests_and_is_never_flattened() {
    let wire = ChatMessage::User {
        content: MessageContent::Parts(vec![
            ContentPart::Text { text: "what is this?".into() },
            ContentPart::ImageUrl {
                url: "https://example.com/c.png".into(),
                detail: Some(vincents_llm::types::ImageDetail::High),
            },
        ]),
        name: None,
    }
    .to_openai_json();
    assert_eq!(wire["content"][1]["type"], "image_url");
    assert_eq!(wire["content"][1]["image_url"]["detail"], "high");
    // A flat shape is what a derived serializer produces, and it is rejected.
    assert!(wire["content"][1].get("url").is_none(), "{wire}");
}