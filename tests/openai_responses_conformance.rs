//! OpenAI Responses API wire conformance: §25–§29, §58.
//!
//! This file is the specification. It was written before the implementation and
//! is expected to be red until the types and the provider exist.
//!
//! The properties that matter:
//!
//! - The Responses API is spoken **natively**. `output[0]` is not assistant
//!   text: a response can lead with a reasoning item or a function call, and
//!   code that assumes index 0 is text silently drops calls.
//! - Unknown output items survive. A provider adding an item type must not make
//!   an older build lose data.
//! - Built-in tools are not forced into `FunctionDefinition`. They execute
//!   remotely and have different authority semantics, so they are distinct
//!   types rather than a function with a magic name.
//! - `previous_response_id` and `conversation` have validity rules.

use serde_json::{json, Value};
use vincents_llm::responses::*;
use vincents_llm::responses_provider::{
    OpenAIResponsesConfig, OpenAIResponsesProvider, ResponseStreamEvent,
};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn response_body() -> Value {
    json!({
        "id": "resp_abc123",
        "object": "response",
        "created_at": 1_700_000_000,
        "model": "gpt-5-2025-01-01",
        "status": "completed",
        "output": [
            {"type": "reasoning", "id": "rs_1", "summary": []},
            {"type": "message", "id": "msg_1", "role": "assistant",
             "status": "completed",
             "content": [{"type": "output_text", "text": "The answer is 42.",
                          "annotations": []}]},
            {"type": "function_call", "id": "fc_1", "call_id": "call_1",
             "name": "get_weather", "arguments": "{\"city\":\"Oslo\"}",
             "status": "completed"}
        ],
        "usage": {
            "input_tokens": 100, "output_tokens": 50, "total_tokens": 150,
            "input_tokens_details": {"cached_tokens": 20},
            "output_tokens_details": {"reasoning_tokens": 30}
        }
    })
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

fn provider(base: &str) -> OpenAIResponsesProvider {
    OpenAIResponsesProvider::with_config(
        OpenAIResponsesConfig {
            api_key: "test-key".into(),
            base_url: Some(base.to_string()),
            ..Default::default()
        },
    )
    .expect("provider constructs")
}

fn request() -> ResponseRequest {
    ResponseRequest {
        model: "gpt-5".into(),
        input: ResponseInput::text("What is the answer?"),
        ..Default::default()
    }
}

async fn body_of(server: &MockServer) -> Value {
    let requests = server.received_requests().await.expect("recorded");
    serde_json::from_slice(&requests[0].body).expect("json body")
}

// ── Typed output items (§26) ──────────────────────────────────────────────

#[tokio::test]
async fn output_items_are_typed_and_none_is_assumed_to_be_text() {
    let server = server().await;
    let response = provider(&server.uri())
        .create(request())
        .await
        .expect("request succeeds");

    // The provider leads with a *reasoning* item. Code reading `output[0]` as
    // assistant text gets an empty string and no idea why.
    assert_eq!(response.output.len(), 3);
    assert!(matches!(response.output[0], ResponseOutputItem::Reasoning(_)));
    assert!(matches!(response.output[1], ResponseOutputItem::Message(_)));
    assert!(matches!(response.output[2], ResponseOutputItem::FunctionCall(_)));
}

#[tokio::test]
async fn output_text_finds_the_message_without_discarding_typed_items() {
    let server = server().await;
    let response = provider(&server.uri())
        .create(request())
        .await
        .expect("request succeeds");

    // The convenience accessor returns text and keeps everything else. A
    // provider that returns no message is not an error, and must not panic.
    assert_eq!(response.output_text(), "The answer is 42.");
    assert_eq!(response.output.len(), 3, "typed items are retained");
}

#[tokio::test]
async fn a_function_call_is_readable_with_its_call_id() {
    let server = server().await;
    let response = provider(&server.uri())
        .create(request())
        .await
        .expect("request succeeds");

    let call = response.function_calls();
    assert_eq!(call.len(), 1);
    assert_eq!(call[0].name, "get_weather");
    // `call_id` links the call to its result; `id` is the item's own identity.
    // Conflating them makes the follow-up request unanswerable.
    assert_eq!(call[0].call_id.as_deref(), Some("call_1"));
    assert_eq!(call[0].id.as_deref(), Some("fc_1"));
    assert_eq!(call[0].arguments, "{\"city\":\"Oslo\"}");
}

#[tokio::test]
async fn an_unknown_output_item_is_preserved_not_dropped() {
    // Forward compatibility: a provider shipping a new item type must not make
    // an older build silently lose it.
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/responses"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "resp_x", "object": "response", "created_at": 1, "model": "gpt-5",
            "status": "completed",
            "output": [
                {"type": "mcp_call", "id": "mcp_1", "server_label": "docs",
                 "name": "search", "arguments": "{\"q\":\"x\"}"},
                {"type": "message", "id": "m", "role": "assistant", "status": "completed",
                 "content": [{"type": "output_text", "text": "done", "annotations": []}]}
            ]
        })))
        .mount(&server)
        .await;

    let response = provider(&server.uri())
        .create(request())
        .await
        .expect("request succeeds");

    match &response.output[0] {
        ResponseOutputItem::Unknown(value) => {
            assert_eq!(value["type"], "mcp_call");
            assert_eq!(value["server_label"], "docs");
        }
        other => panic!("an unrecognised item must be preserved verbatim: {other:?}"),
    }
    // And the recognised item still works.
    assert_eq!(response.output_text(), "done");
}

#[tokio::test]
async fn a_response_with_no_message_item_has_empty_text_not_a_panic() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/responses"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "resp_x", "object": "response", "created_at": 1, "model": "gpt-5",
            "status": "completed",
            "output": [{"type": "function_call", "id": "fc", "call_id": "c",
                        "name": "f", "arguments": "{}"}]
        })))
        .mount(&server)
        .await;

    let response = provider(&server.uri())
        .create(request())
        .await
        .expect("request succeeds");
    assert_eq!(response.output_text(), "");
    assert_eq!(response.function_calls().len(), 1);
}

// ── The request body (§25) ─────────────────────────────────────────────────

#[tokio::test]
async fn every_portable_field_reaches_the_body() {
    let server = server().await;
    let mut req = request();
    req.instructions = Some("be terse".into());
    req.max_output_tokens = Some(512);
    req.max_tool_calls = Some(3);
    req.parallel_tool_calls = Some(false);
    req.temperature = Some(0.3);
    req.top_p = Some(0.9);
    req.top_logprobs = Some(5);
    req.logprobs = Some(true);
    req.truncation = Some(TruncationMode::Auto);
    req.store = Some(true);
    req.service_tier = Some(vincents_llm::types::ServiceTier("flex".into()));
    req.reasoning = Some(vincents_llm::types::ReasoningConfig {
        effort: Some(vincents_llm::types::ReasoningEffort("high".into())),
        max_tokens: None,
        exclude: None,
        summary: None,
    });
    req.metadata.insert("tenant".into(), "acme".into());
    req.include
        .push(ResponseInclude::reasoning_encrypted_content());

    provider(&server.uri())
        .create(req)
        .await
        .expect("request succeeds");

    let body = body_of(&server).await;
    assert_eq!(body["model"], "gpt-5");
    assert_eq!(body["instructions"], "be terse");
    assert_eq!(body["max_output_tokens"], 512);
    assert_eq!(body["max_tool_calls"], 3);
    assert_eq!(body["parallel_tool_calls"], false);
    assert_eq!(body["temperature"], 0.3);
    assert_eq!(body["top_p"], 0.9);
    assert_eq!(body["top_logprobs"], 5);
    assert_eq!(body["logprobs"], true);
    assert_eq!(body["truncation"], "auto");
    assert_eq!(body["store"], true);
    assert_eq!(body["service_tier"], "flex");
    assert_eq!(body["reasoning"]["effort"], "high");
    assert_eq!(body["metadata"]["tenant"], "acme");
    assert_eq!(body["include"][0], "reasoning.encrypted_content");
}

#[tokio::test]
async fn text_input_is_a_plain_string_and_items_input_is_a_list() {
    let server = server().await;
    provider(&server.uri())
        .create(request())
        .await
        .expect("ok");
    let body = body_of(&server).await;
    assert!(body["input"].is_string(), "{body}");

    let server2 = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/responses"))
        .respond_with(ResponseTemplate::new(200).set_body_json(response_body()))
        .mount(&server2)
        .await;
    let mut req = request();
    req.input = ResponseInput::items(vec![ResponseInputItem::Message {
        role: "user".into(),
        content: vec![ResponseContent::InputText { text: "hi".into() }],
    }]);
    provider(&server2.uri())
        .create(req)
        .await
        .expect("ok");
    let body2: Value = {
        let requests = server2.received_requests().await.expect("recorded");
        serde_json::from_slice(&requests[0].body).expect("json")
    };
    assert!(body2["input"].is_array());
    assert_eq!(body2["input"][0]["role"], "user");
    assert_eq!(body2["input"][0]["content"][0]["text"], "hi");
}

// ── Built-in tools (§27) ──────────────────────────────────────────────────

#[tokio::test]
async fn builtin_tools_are_typed_and_not_shaped_like_functions() {
    let server = server().await;
    let mut req = request();
    req.tools = vec![
        ResponseTool::Function {
            name: "get_weather".into(),
            description: Some("weather for a city".into()),
            parameters: Some(json!({"type": "object"})),
            strict: Some(true),
        },
        ResponseTool::WebSearch {
            search_context_size: Some(SearchContextSize::High),
            user_location: None,
        },
        ResponseTool::FileSearch {
            vector_store_ids: vec!["vs_1".into()],
            max_num_results: Some(5),
            filters: None,
            ranking_options: None,
        },
        ResponseTool::CodeInterpreter {
            container: CodeInterpreterContainer::Auto,
        },
        ResponseTool::ComputerUse {
            display_width: 1024,
            display_height: 768,
            environment: Some("browser".into()),
        },
        ResponseTool::Mcp {
            server_label: "docs".into(),
            server_url: Some("https://mcp.example.com".into()),
            allowed_tools: vec!["search".into()],
            authorization: None,
            require_approval: None,
        },
    ];

    provider(&server.uri())
        .create(req)
        .await
        .expect("request succeeds");

    let body = body_of(&server).await;
    let tools = body["tools"].as_array().expect("tools array");
    assert_eq!(tools.len(), 6);

    // A local function is the familiar `{type: function, name, ...}`.
    assert_eq!(tools[0]["type"], "function");
    assert_eq!(tools[0]["name"], "get_weather");
    assert_eq!(tools[0]["strict"], true);
    // Wrapped in the documented envelope, so `parameters` is not top-level.
    assert_eq!(tools[0]["parameters"]["type"], "object");
    assert!(tools[0]["function"].is_null(), "not nested under `function`");

    // Provider-operated tools are NOT functions. They execute remotely, which
    // is a different authority question, and forcing them into
    // FunctionDefinition would make that invisible to any policy check.
    assert_eq!(tools[1]["type"], "web_search");
    assert_eq!(tools[1]["search_context_size"], "high");
    assert!(tools[1]["name"].is_null(), "a web search is not a named function");

    assert_eq!(tools[2]["type"], "file_search");
    assert_eq!(tools[2]["vector_store_ids"][0], "vs_1");
    assert_eq!(tools[2]["max_num_results"], 5);

    assert_eq!(tools[3]["type"], "code_interpreter");
    assert_eq!(tools[3]["container"]["type"], "auto");

    assert_eq!(tools[4]["type"], "computer_use_preview");
    assert_eq!(tools[4]["display_width"], 1024);
    assert_eq!(tools[4]["display_height"], 768);

    assert_eq!(tools[5]["type"], "mcp");
    assert_eq!(tools[5]["server_label"], "docs");
    assert_eq!(tools[5]["allowed_tools"][0], "search");
}

#[tokio::test]
async fn tool_choice_is_typed_across_local_and_provider_operated_tools() {
    let server = server().await;
    let mut req = request();
    req.tools = vec![ResponseTool::WebSearch {
        search_context_size: None,
        user_location: None,
    }];
    req.tool_choice = Some(ResponseToolChoice::ProviderOperated(json!({
        "type": "web_search",
        "search_context_size": "low"
    })));
    provider(&server.uri())
        .create(req)
        .await
        .expect("ok");

    let body = body_of(&server).await;
    assert_eq!(body["tool_choice"]["type"], "web_search");
    assert_eq!(body["tool_choice"]["search_context_size"], "low");
}

// ── Conversation state (§28) ──────────────────────────────────────────────

#[tokio::test]
async fn previous_response_id_and_conversation_are_independent_knobs() {
    let server = server().await;
    let mut req = request();
    req.previous_response_id = Some("resp_prev".into());
    provider(&server.uri())
        .create(req)
        .await
        .expect("ok");
    let body = body_of(&server).await;
    assert_eq!(body["previous_response_id"], "resp_prev");
    assert!(body.get("conversation").is_none());
}

#[tokio::test]
async fn conversation_is_serialized_as_a_reference() {
    let server = server().await;
    let mut req = request();
    req.conversation = Some(ConversationRef::Id("conv_1".into()));
    provider(&server.uri())
        .create(req)
        .await
        .expect("ok");
    let body = body_of(&server).await;
    assert_eq!(body["conversation"], "conv_1");
}

#[tokio::test]
async fn a_local_transcript_alongside_provider_state_is_flagged_as_intentional() {
    // §28's real hazard: resending the whole local session *and* pointing at
    // provider-side state duplicates every turn. It must be an explicit choice.
    let server = server().await;
    let mut req = request();
    req.conversation = Some(ConversationRef::Id("conv_1".into()));
    req.previous_response_id = Some("resp_prev".into());

    let error = provider(&server.uri())
        .create(req)
        .await
        .expect_err("an ambiguous request is rejected locally");
    assert!(
        matches!(error, vincents_llm::error::LLMError::InvalidParameterCombination { .. }),
        "{error:?}"
    );
    assert!(server.received_requests().await.expect("recorded").is_empty());
}

// ── Status model (§29) ────────────────────────────────────────────────────

#[test]
fn every_background_status_is_representable() {
    // The core type must be able to express the states even before the
    // background lifecycle helpers exist.
    for status in ["queued", "in_progress", "completed", "failed", "cancelled", "incomplete"] {
        let parsed: ResponseStatus = serde_json::from_value(json!(status))
            .unwrap_or_else(|e| panic!("{status} must parse: {e}"));
        assert_eq!(serde_json::to_value(&parsed).unwrap(), json!(status));
    }
}

#[tokio::test]
async fn an_unknown_status_is_preserved_rather_than_rejected() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/responses"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "resp_x", "object": "response", "created_at": 1, "model": "gpt-5",
            "status": "some_future_status", "output": []
        })))
        .mount(&server)
        .await;
    let response = provider(&server.uri())
        .create(request())
        .await
        .expect("a new status must not break the build");
    assert!(matches!(
        response.status,
        ResponseStatus::Unknown(ref v) if v == "some_future_status"
    ));
}

#[tokio::test]
async fn background_is_sent_and_the_status_is_read_back() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/responses"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "resp_bg", "object": "response", "created_at": 1, "model": "gpt-5",
            "status": "queued", "background": true, "output": []
        })))
        .mount(&server)
        .await;
    let mut req = request();
    req.background = Some(true);
    let response = provider(&server.uri())
        .create(req)
        .await
        .expect("ok");
    assert_eq!(response.status, ResponseStatus::Queued);
    let requests = server.received_requests().await.expect("recorded");
    let body: Value = serde_json::from_slice(&requests[0].body).expect("json");
    assert_eq!(body["background"], true);
}

// ── Usage (§47 continues to apply) ────────────────────────────────────────

#[tokio::test]
async fn responses_usage_carries_reasoning_and_cache_detail() {
    let server = server().await;
    let response = provider(&server.uri())
        .create(request())
        .await
        .expect("ok");
    // The Responses API names these `input_tokens`/`output_tokens`. Read
    // through `as_usage` so a Responses call and a Chat call report the same
    // numbers -- and so a mapping error cannot hide behind a plausible value.
    let usage = response.usage.expect("usage reported").as_usage();
    assert_eq!(usage.prompt_tokens, 100);
    assert_eq!(usage.completion_tokens, 50);
    assert_eq!(usage.reasoning_tokens, Some(30));
    assert_eq!(usage.cached_tokens, Some(20));
    assert!(usage.has_token_details());
}

#[tokio::test]
async fn absent_usage_stays_absent() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/responses"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "resp_x", "object": "response", "created_at": 1, "model": "gpt-5",
            "status": "completed", "output": []
        })))
        .mount(&server)
        .await;
    let response = provider(&server.uri())
        .create(request())
        .await
        .expect("ok");
    assert!(response.usage.is_none(), "absent usage must not become zero");
}

// ── Provider extensions (§54) ─────────────────────────────────────────────

#[tokio::test]
async fn a_provider_extension_cannot_shadow_a_typed_field() {
    let server = server().await;
    let mut req = request();
    // The typed field is *also* set: that is what makes it a collision. A fresh
    // key through the escape hatch is the forward-compatibility path working as
    // intended, not a shadowing bug.
    req.temperature = Some(0.7);
    req.provider_params
        .insert("temperature".into(), json!(0.1));
    let error = provider(&server.uri())
        .create(req)
        .await
        .expect_err("a collision is a configuration error");
    assert!(
        matches!(error, vincents_llm::error::LLMError::ConfigurationError { .. }),
        "{error:?}"
    );
    assert!(server.received_requests().await.expect("recorded").is_empty());
}

#[tokio::test]
async fn an_extension_for_an_unset_typed_field_is_allowed() {
    // The other half of the collision rule, and the reason it is "collides with
    // what this request set" rather than "collides with any known field": using
    // the escape hatch for a field the typed API left alone must work.
    let server = server().await;
    let mut req = request();
    req.provider_params.insert("top_k".into(), json!(40));
    provider(&server.uri())
        .create(req)
        .await
        .expect("ok");
    let body = body_of(&server).await;
    assert_eq!(body["top_k"], 40);
}

#[tokio::test]
async fn a_fresh_provider_extension_is_forwarded() {
    // The forward-compatibility path: usable immediately, promoted to a typed
    // field later, with nothing lost in between.
    let server = server().await;
    let mut req = request();
    req.provider_params
        .insert("a_parameter_this_crate_has_never_heard_of".into(), json!(42));
    provider(&server.uri())
        .create(req)
        .await
        .expect("ok");
    let body = body_of(&server).await;
    assert_eq!(body["a_parameter_this_crate_has_never_heard_of"], 42);
}
// ── Streaming (§25, §26) ──────────────────────────────────────────────────

#[tokio::test]
async fn a_stream_emits_typed_items_not_assumed_text() {
    // The same hazard as the non-streaming path, and the reason `output[0]`
    // cannot be assumed to be a message: a streamed tool call arrives as an
    // item-done event, not as text.
    let body = [
        json!({"type": "response.created", "response": {
            "id": "resp_s", "object": "response", "created_at": 1, "model": "gpt-5",
            "status": "in_progress", "output": []}}),
        json!({"type": "response.reasoning_summary_text.delta", "output_index": 0,
               "delta": "thinking"}),
        json!({"type": "response.output_text.delta", "output_index": 1, "delta": "The "}),
        json!({"type": "response.output_text.delta", "output_index": 1, "delta": "answer."}),
        json!({"type": "response.output_item.done", "output_index": 2, "item": {
            "type": "function_call", "id": "fc_1", "call_id": "call_1",
            "name": "get_weather", "arguments": "{\"city\":\"Oslo\"}"}}),
        json!({"type": "response.completed", "response": {
            "id": "resp_s", "object": "response", "created_at": 1, "model": "gpt-5",
            "status": "completed", "output": [],
            "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}}}),
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
    let mut reasoning = String::new();
    let mut calls = Vec::new();
    let mut usage = None;
    let mut done = false;
    let mut saw_created = false;

    while let Some(event) = stream.next().await {
        match event.expect("event") {
            ResponseStreamEvent::Created { .. } => saw_created = true,
            ResponseStreamEvent::TextDelta { text: t, .. } => text.push_str(&t),
            ResponseStreamEvent::ReasoningDelta { text: t, .. } => reasoning.push_str(&t),
            ResponseStreamEvent::OutputItemDone { item, .. } => {
                if let ResponseOutputItem::FunctionCall(call) = item {
                    calls.push(call);
                }
            }
            ResponseStreamEvent::Completed { response } => {
                usage = response.usage.map(|u| u.as_usage());
            }
            ResponseStreamEvent::Done => done = true,
            other => panic!("unexpected event: {other:?}"),
        }
    }

    assert!(saw_created, "the envelope arrives first");
    assert_eq!(text, "The answer.");
    assert_eq!(reasoning, "thinking", "reasoning is not text and not dropped");
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].name, "get_weather");
    assert_eq!(calls[0].call_id.as_deref(), Some("call_1"));
    let usage = usage.expect("usage from the final event");
    assert_eq!(usage.total_tokens, 15);
    assert!(done, "[DONE] terminates the stream");
}

#[tokio::test]
async fn a_truncated_response_stream_is_an_error() {
    let body = "data: {\"type\":\"response.created\",\"response\":{\"id\":\"r\",\"created_at\":1,\"model\":\"m\",\"status\":\"in_progress\",\"output\":[]}}\n\n";
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
    let mut error = None;
    while let Some(event) = stream.next().await {
        if let Err(e) = event {
            error = Some(e);
        }
    }
    let error = error.expect("a truncated stream must surface an error");
    assert!(matches!(error, vincents_llm::error::LLMError::StreamInterrupted { .. }), "{error:?}");
}

#[tokio::test]
async fn an_unknown_stream_event_is_preserved() {
    let body = "data: {\"type\":\"response.some_future_event\",\"detail\":1}\n\ndata: [DONE]\n\n";
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
    let mut unknown = None;
    while let Some(event) = stream.next().await {
        if let ResponseStreamEvent::Unknown(value) = event.expect("event") {
            unknown = Some(value);
        }
    }
    let value = unknown.expect("an unknown event must be preserved, not dropped");
    assert_eq!(value["type"], "response.some_future_event");
}

#[tokio::test]
async fn streaming_sets_the_stream_flag_and_keeps_validation() {
    let validating = server().await;
    let mut req = request();
    req.previous_response_id = Some("resp_p".into());
    req.conversation = Some(ConversationRef::Id("conv".into()));
    // A `Stream` is not `Debug`, so the error is matched rather than unwrapped.
    match provider(&validating.uri()).create_stream(req).await {
        Err(error) => assert!(
            matches!(error, vincents_llm::error::LLMError::InvalidParameterCombination { .. }),
            "{error:?}"
        ),
        Ok(_) => panic!("validation must still apply on the streaming path"),
    }
    assert!(validating.received_requests().await.expect("recorded").is_empty());

    // A *streaming* response, this time: the non-streaming body would be a
    // truncated stream and would fail for the right reason at the wrong point.
    let streaming = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/responses"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(
            "data: [DONE]\n\n",
            "text/event-stream",
        ))
        .mount(&streaming)
        .await;

    use futures::StreamExt;
    let mut stream = provider(&streaming.uri())
        .create_stream(request())
        .await
        .expect("stream opens");
    while let Some(event) = stream.next().await {
        event.expect("event");
    }
    let requests = streaming.received_requests().await.expect("recorded");
    let body: Value = serde_json::from_slice(&requests[0].body).expect("json");
    assert_eq!(body["stream"], true);
}

// ── Lifecycle helpers (§56) ───────────────────────────────────────────────

#[tokio::test]
async fn retrieve_and_cancel_hit_the_right_endpoints() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/responses/resp_1"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "resp_1", "object": "response", "created_at": 1, "model": "gpt-5",
            "status": "completed", "output": []
        })))
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/responses/resp_2/cancel"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "resp_2", "object": "response", "created_at": 1, "model": "gpt-5",
            "status": "cancelled", "output": []
        })))
        .mount(&server)
        .await;

    let p = provider(&server.uri());
    let response = p.retrieve("resp_1").await.expect("retrieve");
    assert_eq!(response.status, ResponseStatus::Completed);
    let cancelled = p.cancel("resp_2").await.expect("cancel");
    assert_eq!(cancelled.status, ResponseStatus::Cancelled);
}

// ── §62: the remaining named golden cases ─────────────────────────────────

fn test_config(base: &str) -> OpenAIResponsesConfig {
    OpenAIResponsesConfig {
        api_key: "k".into(),
        base_url: Some(base.to_string()),
        default_model: Some("gpt-5".into()),
        ..Default::default()
    }
}


#[tokio::test]
async fn a_tool_result_continuation_round_trips() {
    // §62: "tool result continuation". The assistant's function call, the caller's
    // `function_call_output`, then the model's answer -- and the `call_id` must
    // still match, because a mismatch here makes the request unanswerable.
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/responses"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "resp_cont", "object": "response", "created_at": 1, "model": "gpt-5",
            "status": "completed",
            "output": [{"type": "message", "id": "m2", "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "It is 12C.",
                                     "annotations": []}]}]
        })))
        .mount(&server)
        .await;

    let response = OpenAIResponsesProvider::with_config(test_config(&server.uri()))
        .expect("provider")
        .create(ResponseRequest {
            model: "gpt-5".into(),
            input: ResponseInput::items(vec![
                ResponseInputItem::Message {
                    role: "user".into(),
                    content: vec![ResponseContent::InputText { text: "weather in Oslo?".into() }],
                },
                ResponseInputItem::FunctionCall {
                    id: Some("fc_1".into()),
                    call_id: "call_a".into(),
                    name: "get_weather".into(),
                    arguments: r#"{"city":"Oslo"}"#.into(),
                },
                ResponseInputItem::FunctionCallOutput {
                    call_id: "call_a".into(),
                    output: "12C and sunny".into(),
                },
            ]),
            ..Default::default()
        })
        .await
        .expect("request succeeds");

    let requests = server.received_requests().await.expect("recorded");
    let body: Value = serde_json::from_slice(&requests[0].body).expect("json");
    let input = body["input"].as_array().expect("items");

    assert_eq!(input[1]["type"], "function_call");
    assert_eq!(input[1]["call_id"], "call_a");
    assert_eq!(input[2]["type"], "function_call_output");
    // The link is what makes the continuation work; a renamed field would leave
    // the provider unable to match the result to the call.
    assert_eq!(
        input[2]["call_id"], input[1]["call_id"],
        "the output must reference the call it answers"
    );
    assert_eq!(input[2]["output"], "12C and sunny");

    assert_eq!(response.output_text(), "It is 12C.");
}

#[tokio::test]
async fn an_incomplete_response_is_reported_as_incomplete() {
    // §62: "status incomplete". A response cut off by the output-token limit is
    // not a completed response, and treating it as one ships a half-answer.
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/responses"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "resp_inc", "object": "response", "created_at": 1, "model": "gpt-5",
            "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"},
            "output": [{"type": "message", "id": "m", "role": "assistant",
                        "status": "incomplete",
                        "content": [{"type": "output_text", "text": "It is 1",
                                     "annotations": []}]}]
        })))
        .mount(&server)
        .await;

    let response = OpenAIResponsesProvider::with_config(test_config(&server.uri()))
        .expect("provider")
        .create(ResponseRequest {
            model: "gpt-5".into(),
            input: ResponseInput::text("weather?"),
            ..Default::default()
        })
        .await
        .expect("request succeeds");

    assert_eq!(response.status, ResponseStatus::Incomplete);
    assert!(response.status.is_terminal(), "a caller must be able to stop waiting");
    // The reason is retained: `incomplete` alone does not say *why*, and the
    // remedy differs between "hit the token limit" and "was filtered".
    assert_eq!(
        response
            .incomplete_details
            .as_ref()
            .expect("details")["reason"],
        "max_output_tokens"
    );
    // The partial text is still delivered -- it is the only thing the caller has.
    assert_eq!(response.output_text(), "It is 1");
}

#[tokio::test]
async fn a_completed_response_is_not_terminal_confused_with_incomplete() {
    // The two statuses differ in what a caller should do: `completed` means
    // done, `incomplete` means the answer may be short.
    assert!(ResponseStatus::Completed.is_terminal());
    assert!(ResponseStatus::Incomplete.is_terminal());
    assert!(ResponseStatus::Failed.is_terminal());
    assert!(ResponseStatus::Cancelled.is_terminal());
    // A queued or in-progress background response is *not*, so a caller does not
    // stop waiting on one.
    assert!(!ResponseStatus::Queued.is_terminal());
    assert!(!ResponseStatus::InProgress.is_terminal());
}

#[tokio::test]
async fn an_input_content_array_preserves_order_and_nesting() {
    // §62: "input content array". A message whose parts are reordered reads as a
    // different question, and this is the shape that carries them.
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/responses"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "r", "object": "response", "created_at": 1, "model": "gpt-5",
            "status": "completed", "output": []
        })))
        .mount(&server)
        .await;

    OpenAIResponsesProvider::with_config(test_config(&server.uri()))
        .expect("provider")
        .create(ResponseRequest {
            model: "gpt-5".into(),
            input: ResponseInput::items(vec![ResponseInputItem::Message {
                role: "user".into(),
                content: vec![
                    ResponseContent::InputText { text: "what is in these?".into() },
                    ResponseContent::InputImage {
                        image_url: "https://example.com/a.png".into(),
                        detail: Some(vincents_llm::types::ImageDetail::High),
                    },
                    ResponseContent::InputText { text: "be specific".into() },
                ],
            }]),
            ..Default::default()
        })
        .await
        .expect("ok");

    let requests = server.received_requests().await.expect("recorded");
    let body: Value = serde_json::from_slice(&requests[0].body).expect("json");
    let content = body["input"][0]["content"].as_array().expect("array");
    assert_eq!(content.len(), 3);
    assert_eq!(content[0]["type"], "input_text");
    assert_eq!(content[0]["text"], "what is in these?");
    assert_eq!(content[1]["type"], "input_image");
    assert_eq!(content[1]["image_url"], "https://example.com/a.png");
    assert_eq!(content[1]["detail"], "high");
    assert_eq!(content[2]["text"], "be specific", "order is the message");
}
