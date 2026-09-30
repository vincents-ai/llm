//! Strict request validation: §51.
//!
//! Two properties matter, and the second is the one usually missed:
//!
//! 1. A conflict is caught **before** network I/O, so the caller gets a named
//!    error instead of an opaque 400 after a round trip.
//! 2. When a request has more than one problem, the **most fundamental** one is
//!    reported. A request that sets two conflicting parameters *and* shadows a
//!    typed field with `extra_params` should say what is wrong with its own
//!    terms, not complain about a field the caller never got to use.


use serde_json::{json, Value};
use vincents_llm::error::LLMError;
use vincents_llm::types::{ChatCompletionRequest, ChatMessage};
use vincents_llm::wire::{build_chat_request, ChatRequestParts};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn messages() -> Vec<Value> {
    vec![json!({"role": "user", "content": "hi"})]
}

fn build(request: &ChatCompletionRequest, stream: bool) -> Result<Value, LLMError> {
    build_chat_request(ChatRequestParts {
        model: "gpt-4o",
        messages: &messages(),
        request,
        stream,
        tools: None,
        provider_params: None,
    })
}

fn request() -> ChatCompletionRequest {
    ChatCompletionRequest {
        model: "gpt-4o".into(),
        messages: vec![ChatMessage::user("hi")],
        ..Default::default()
    }
}

// ── Existing rules, now in one place ───────────────────────────────────────

#[test]
fn max_tokens_and_max_completion_tokens_together_are_rejected() {
    let mut r = request();
    r.max_tokens = Some(100);
    r.max_completion_tokens = Some(100);
    let error = build(&r, false).expect_err("conflicting token limits");
    assert!(matches!(error, LLMError::ConfigurationError { .. }), "{error:?}");
}

#[test]
fn top_logprobs_without_logprobs_is_rejected() {
    // Providers differ: some ignore `top_logprobs`, some 400. Sending a request
    // whose outcome depends on which you got is not acceptable.
    let mut r = request();
    r.top_logprobs = Some(5);
    let error = build(&r, false).expect_err("top_logprobs needs logprobs");
    assert!(matches!(error, LLMError::ConfigurationError { .. }), "{error:?}");
}

#[test]
fn tool_choice_naming_an_absent_tool_is_rejected() {
    use vincents_llm::types::{ToolChoice, ToolDefinition, FunctionDefinitionWire};
    let mut r = request();
    r.tools = Some(vec![ToolDefinition::Function(FunctionDefinitionWire {
        name: "get_weather".into(),
        description: None,
        parameters: None,
        strict: None,
    })]);
    r.tool_choice = Some(ToolChoice::Function { name: "get_stock".into() });
    let error = build(&r, false).expect_err("the named tool is not offered");
    let text = error.to_string();
    assert!(text.contains("get_stock"), "{text}");
    assert!(text.contains("get_weather"), "the message should list what *is* offered: {text}");
}

#[test]
fn tool_choice_naming_a_tool_with_no_tools_at_all_is_rejected() {
    use vincents_llm::types::ToolChoice;
    let mut r = request();
    r.tool_choice = Some(ToolChoice::Function { name: "get_weather".into() });
    assert!(build(&r, false).is_err(), "naming a tool while offering none is a mistake");
}

// ── New: stream_options on a non-streaming request ─────────────────────────

#[test]
fn stream_options_on_a_non_streaming_request_is_rejected() {
    // `include_usage` is what tells the caller what a stream cost. On a
    // non-streaming request it is either ignored or 400'd depending on the
    // provider, and the caller's response is unaffected either way -- so a
    // silent ignore looks like the setting simply does not work.
    let mut r = request();
    r.extra_params
        .insert("stream_options".into(), json!({"include_usage": true}));
    let error = build(&r, false).expect_err("stream_options requires stream = true");
    assert!(
        matches!(error, LLMError::InvalidParameterCombination { .. }),
        "{error:?}"
    );
}

#[test]
fn stream_options_on_a_streaming_request_is_allowed() {
    let mut r = request();
    r.extra_params
        .insert("stream_options".into(), json!({"include_usage": true}));
    let body = build(&r, true).expect("legal on a stream");
    assert_eq!(body["stream"], true);
    assert_eq!(body["stream_options"]["include_usage"], true);
}

// ── The ordering property ──────────────────────────────────────────────────

#[test]
fn a_validity_conflict_is_reported_before_a_collision() {
    // The caller has two problems. Reporting the collision would be technically
    // true and useless: it points at a field they set *because* of the first
    // problem.
    let mut r = request();
    r.max_tokens = Some(100);
    r.max_completion_tokens = Some(100);
    r.extra_params.insert("temperature".into(), json!(0.5));

    let error = build(&r, false).expect_err("two problems");
    assert!(
        matches!(error, LLMError::ConfigurationError { ref message, .. }
            if message.contains("max_completion_tokens")),
        "the token-limit conflict must win: {error:?}"
    );
}

#[test]
fn a_stream_options_conflict_is_reported_before_a_collision() {
    let mut r = request();
    r.extra_params.insert("stream_options".into(), json!({"include_usage": true}));
    r.extra_params.insert("temperature".into(), json!(0.5));

    let error = build(&r, false).expect_err("two problems");
    assert!(
        matches!(error, LLMError::InvalidParameterCombination { .. }),
        "{error:?}"
    );
}

// ── Validation happens before network I/O ─────────────────────────────────

#[tokio::test]
async fn an_invalid_request_never_reaches_the_provider() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "x", "object": "chat.completion", "created": 1, "model": "gpt-4o",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"},
                         "finish_reason": "stop"}]
        })))
        .mount(&server)
        .await;

    let provider = vincents_llm::openai::OpenAIProvider::with_config(
        vincents_llm::openai::OpenAIConfig {
            api_key: "k".into(),
            base_url: Some(server.uri()),
            default_model: Some("gpt-4o".into()),
            ..Default::default()
        },
    )
    .expect("provider");

    use vincents_llm::provider::LLMProvider;
    let mut r = request();
    r.top_logprobs = Some(3);
    provider
        .chat_completion(r)
        .await
        .expect_err("rejected locally");

    // The decisive assertion: the mock recorded nothing. A 400 from the
    // provider would also fail the call, but only after a round trip.
    assert!(
        server.received_requests().await.expect("recorded").is_empty(),
        "validation must run before the request is sent"
    );
}