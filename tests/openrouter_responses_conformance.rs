//! OpenRouter Responses parity: §57.
//!
//! The same portable `ResponseRequest` sent to a second provider. This is the
//! test that decides whether "portable" means anything: a type is only portable
//! if it produces correct requests and correct results against a provider that
//! is not the one it was modelled on.
//!
//! Provider-specific extensions must be opt-in -- they go through
//! `provider_params`, and a portable request must not carry any.

use serde_json::{json, Value};
use vincents_llm::responses::*;
use vincents_llm::responses_provider::{
    OpenAIResponsesConfig, OpenAIResponsesProvider, ResponseStreamEvent,
};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn response_body() -> Value {
    json!({
        "id": "gen-or-1",
        "object": "response",
        "created_at": 1_700_000_000,
        "model": "openai/gpt-4o",
        "status": "completed",
        "output": [
            {"type": "reasoning", "id": "rs_1", "summary": []},
            {"type": "message", "id": "msg_1", "role": "assistant",
             "status": "completed",
             "content": [{"type": "output_text", "text": "42", "annotations": []}]},
            {"type": "function_call", "id": "fc_1", "call_id": "call_1",
             "name": "get_weather", "arguments": "{\"city\":\"Oslo\"}"}
        ],
        "usage": {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150,
                  "output_tokens_details": {"reasoning_tokens": 30}}
    })
}

fn config(base: &str) -> OpenAIResponsesConfig {
    OpenAIResponsesConfig {
        api_key: "or-key".into(),
        base_url: Some(base.to_string()),
        ..Default::default()
    }
}

fn provider(base: &str) -> OpenAIResponsesProvider {
    OpenAIResponsesProvider::with_config(config(base)).expect("provider constructs")
}

fn request() -> ResponseRequest {
    ResponseRequest {
        model: "openai/gpt-4o".into(),
        input: ResponseInput::text("What is the answer?"),
        ..Default::default()
    }
}

async fn server() -> MockServer {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/responses"))
        .respond_with(ResponseTemplate::new(200).set_body_json(response_body()))
        .mount(&server)
        .await;
    server
}

async fn body_of(server: &MockServer) -> Value {
    let requests = server.received_requests().await.expect("recorded");
    serde_json::from_slice(&requests[0].body).expect("json body")
}

// ── Portable cases (§57's list) ────────────────────────────────────────────

#[tokio::test]
async fn plain_text_round_trips() {
    let server = server().await;
    let response = provider(&server.uri())
        .create(request())
        .await
        .expect("ok");
    let body = body_of(&server).await;

    assert_eq!(body["model"], "openai/gpt-4o");
    assert!(body["input"].is_string(), "{body}");
    // Identical typed-output handling on a different provider: the reasoning
    // item is still first, and the text is still found.
    assert!(matches!(response.output[0], ResponseOutputItem::Reasoning(_)));
    assert_eq!(response.output_text(), "42");
}

#[tokio::test]
async fn a_function_tool_round_trips_with_its_call_id() {
    let server = server().await;
    let mut req = request();
    req.tools = vec![ResponseTool::Function {
        name: "get_weather".into(),
        description: Some("weather for a city".into()),
        parameters: Some(json!({"type": "object"})),
        strict: Some(true),
    }];
    provider(&server.uri())
        .create(req)
        .await
        .expect("ok");

    let body = body_of(&server).await;
    // The envelope is provider-portable: top-level `name`, `parameters` nested
    // under the tool rather than under `function`.
    assert_eq!(body["tools"][0]["type"], "function");
    assert_eq!(body["tools"][0]["name"], "get_weather");
    assert_eq!(body["tools"][0]["strict"], true);
    assert!(body["tools"][0]["function"].is_null());

    let response = provider(&server.uri()).create(request()).await.expect("ok");
    assert_eq!(response.function_calls()[0].call_id.as_deref(), Some("call_1"));
}

#[tokio::test]
async fn structured_output_config_round_trips() {
    let server = server().await;
    let mut req = request();
    req.text = Some(ResponseTextConfig {
        format: Some(vincents_llm::types::ResponseFormat::JsonSchema {
            name: "answer".into(),
            description: None,
            schema: json!({"type": "object", "properties": {"answer": {"type": "number"}}}),
            strict: Some(true),
        }),
        verbosity: Some(vincents_llm::types::Verbosity("low".into())),
    });
    provider(&server.uri())
        .create(req)
        .await
        .expect("ok");
    let body = body_of(&server).await;

    // `json_schema` nests, which is exactly the place a hand-written serializer
    // gets it wrong and the API accepts a flat shape without complaint.
    let schema = &body["text"]["format"];
    assert_eq!(schema["type"], "json_schema");
    // `name` and `strict` live *inside* `json_schema`, not beside it. A
    // serializer that put them at the top level looks right in review and is
    // rejected by the API.
    assert!(schema["json_schema"].is_object());
    assert_eq!(schema["json_schema"]["name"], "answer");
    assert_eq!(schema["json_schema"]["strict"], true);
    assert_eq!(schema["json_schema"]["schema"]["type"], "object");
    assert!(schema["name"].is_null(), "not at the top level: {schema}");
    assert_eq!(body["text"]["verbosity"], "low");
}

#[tokio::test]
async fn reasoning_config_round_trips() {
    let server = server().await;
    let mut req = request();
    req.reasoning = Some(vincents_llm::types::ReasoningConfig {
        effort: Some(vincents_llm::types::ReasoningEffort("high".into())),
        max_tokens: None,
        exclude: None,
        summary: Some(vincents_llm::types::ReasoningSummary::Auto),
    });
    provider(&server.uri())
        .create(req)
        .await
        .expect("ok");
    let body = body_of(&server).await;
    assert_eq!(body["reasoning"]["effort"], "high");
    assert_eq!(body["reasoning"]["summary"], "auto");
}

#[tokio::test]
async fn previous_response_id_round_trips() {
    let server = server().await;
    let mut req = request();
    req.previous_response_id = Some("gen-or-prev".into());
    provider(&server.uri())
        .create(req)
        .await
        .expect("ok");
    let body = body_of(&server).await;
    assert_eq!(body["previous_response_id"], "gen-or-prev");
}

#[tokio::test]
async fn usage_keeps_reasoning_detail_on_the_second_provider() {
    let server = server().await;
    let response = provider(&server.uri())
        .create(request())
        .await
        .expect("ok");
    let usage = response.usage.expect("usage").as_usage();
    assert_eq!(usage.prompt_tokens, 100);
    assert_eq!(usage.completion_tokens, 50);
    assert_eq!(usage.reasoning_tokens, Some(30));
    // This provider reports no cache detail. `None`, not zero.
    assert_eq!(usage.cached_tokens, None);
    assert!(usage.reasoning_tokens.is_some());
}

#[tokio::test]
async fn streaming_round_trips_on_the_second_provider() {
    let body = [
        json!({"type": "response.created", "response": {
            "id": "gen-or-s", "created_at": 1, "model": "openai/gpt-4o",
            "status": "in_progress", "output": []}}),
        json!({"type": "response.output_text.delta", "output_index": 0, "delta": "42"}),
        json!({"type": "response.completed", "response": {
            "id": "gen-or-s", "created_at": 1, "model": "openai/gpt-4o",
            "status": "completed", "output": [],
            "usage": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12}}}),
    ]
    .iter()
    .map(|e| format!("data: {e}\n\n"))
    .collect::<String>()
        + "data: [DONE]\n\n";

    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/responses"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(body, "text/event-stream"))
        .mount(&server)
        .await;

    use futures::StreamExt;
    let mut stream = provider(&server.uri())
        .create_stream(request())
        .await
        .expect("stream opens");
    let mut text = String::new();
    let mut done = false;
    let mut usage = None;
    while let Some(event) = stream.next().await {
        match event.expect("event") {
            ResponseStreamEvent::TextDelta { text: t, .. } => text.push_str(&t),
            ResponseStreamEvent::Completed { response } => usage = response.usage,
            ResponseStreamEvent::Done => done = true,
            // The envelope and any other informational event is expected.
            _ => {}
        }
    }
    assert_eq!(text, "42");
    assert!(done);
    assert_eq!(usage.expect("usage").as_usage().total_tokens, 12);
}

// ── Provider-specific extensions are opt-in ───────────────────────────────

#[tokio::test]
async fn a_portable_request_carries_no_provider_specific_fields() {
    let server = server().await;
    provider(&server.uri())
        .create(request())
        .await
        .expect("ok");
    let body = body_of(&server).await;
    // Routing is a Chat Completions concept and must not leak into a portable
    // Responses body by default.
    assert!(body.get("provider").is_none(), "{body}");
    assert!(body.get("models").is_none(), "{body}");
    assert!(body.get("extra_params").is_none(), "{body}");
}

#[tokio::test]
async fn provider_extensions_reach_the_second_provider_when_asked() {
    let server = server().await;
    let mut req = request();
    req.provider_params
        .insert("provider".into(), json!({"zdr": true, "data_collection": "deny"}));
    provider(&server.uri())
        .create(req)
        .await
        .expect("ok");
    let body = body_of(&server).await;
    assert_eq!(body["provider"]["zdr"], true);
    assert_eq!(body["provider"]["data_collection"], "deny");
}

// ── Errors classify the same way on either provider ───────────────────────

#[tokio::test]
async fn a_rejected_parameter_is_classified_on_the_second_provider_too() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/responses"))
        .respond_with(
            ResponseTemplate::new(400)
                .set_body_json(json!({"error": {
                    "message": "unsupported model",
                    "param": "model",
                    "type": "invalid_request_error"}}))
                .insert_header("x-request-id", "or_req_1"),
        )
        .mount(&server)
        .await;

    let error = provider(&server.uri())
        .create(request())
        .await
        .expect_err("must fail");
    match error {
        vincents_llm::error::LLMError::ProviderRejectedParameter { parameter, .. } => {
            assert_eq!(parameter, "model");
        }
        other => panic!("same classification as the first provider: {other:?}"),
    }
}

#[tokio::test]
async fn validation_is_identical_on_the_second_provider() {
    let server = server().await;
    let mut req = request();
    req.previous_response_id = Some("resp_p".into());
    req.conversation = Some(ConversationRef::Id("conv".into()));
    assert!(provider(&server.uri()).create(req).await.is_err());
    assert!(server.received_requests().await.expect("recorded").is_empty());
}