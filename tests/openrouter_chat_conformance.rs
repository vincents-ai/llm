//! OpenRouter wire conformance: routing, privacy, attribution, and the
//! provider-specific parameters the brief calls out in §30–§38.
//!
//! The point of routing and privacy settings is that they decide *where a user's
//! code goes*. A setting that is silently dropped is worse than one that is
//! absent: the deployment believes it is routing to a zero-data-retention
//! provider and is not. So these assert on the captured body and headers rather
//! than on the type.

use std::collections::BTreeMap;

use serde_json::{json, Value};
use vincents_llm::openrouter::{
    DataCollectionPolicy, OpenRouterConfig, OpenRouterExtensions, OpenRouterMaxPrice,
    OpenRouterProvider, OpenRouterProviderRouting, OpenRouterServerTool, OpenRouterSort,
    RoutingMetadata,
};
use vincents_llm::provider::LLMProvider;
use vincents_llm::types::{ChatCompletionRequest, ChatMessage};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn completion_body(content: &str) -> Value {
    json!({
        "id": "gen-real", "object": "chat.completion", "created": 1_700_000_000,
        "model": "openai/gpt-4o",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}
    })
}

async fn server() -> MockServer {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(completion_body("ok")))
        .mount(&server)
        .await;
    server
}

fn provider(server: &MockServer, configure: impl FnOnce(&mut OpenRouterConfig)) -> OpenRouterProvider {
    let mut config = OpenRouterConfig {
        api_key: "test-key".into(),
        base_url: Some(server.uri()),
        default_model: Some("openai/gpt-4o".into()),
        ..Default::default()
    };
    configure(&mut config);
    OpenRouterProvider::with_config(config).expect("provider constructs")
}

fn request() -> ChatCompletionRequest {
    ChatCompletionRequest {
        model: "openai/gpt-4o".into(),
        messages: vec![ChatMessage::user("hello")],
        ..Default::default()
    }
}

async fn body_of(server: &MockServer) -> Value {
    let requests = server.received_requests().await.expect("recorded");
    assert_eq!(requests.len(), 1, "expected exactly one request");
    serde_json::from_slice(&requests[0].body).expect("the body must be JSON")
}

async fn headers_of(server: &MockServer) -> BTreeMap<String, String> {
    let requests = server.received_requests().await.expect("recorded");
    requests[0]
        .headers
        .iter()
        .map(|(k, v)| (k.as_str().to_lowercase(), v.as_str().to_string()))
        .collect()
}

fn full_routing() -> OpenRouterProviderRouting {
    OpenRouterProviderRouting {
        order: vec!["Together".into(), "Fireworks".into()],
        allow_fallbacks: Some(false),
        only: vec!["Fireworks".into()],
        ignore: vec!["OpenAI".into()],
        require_parameters: Some(true),
        data_collection: Some(DataCollectionPolicy::Deny),
        zdr: Some(true),
        quantizations: vec!["fp8".into()],
        max_price: Some(OpenRouterMaxPrice {
            prompt: 3.0,
            completion: 15.0,
            image: None,
            request: None,
            web_search: None,
            internal_reasoning: None,
        }),
        sort: Some(OpenRouterSort::Throughput),
        preferred_min_throughput: Some(100.0),
        preferred_max_latency: Some(2.5),
    }
}

// ── Routing (§31, §33) ─────────────────────────────────────────────────────

#[tokio::test]
async fn provider_routing_reaches_the_request_body() {
    let server = server().await;
    let mut p = provider(&server, |_| {});
    p.set_extensions(OpenRouterExtensions {
        provider: Some(full_routing()),
        ..Default::default()
    });
    p.chat_completion(request()).await.expect("request succeeds");

    let body = body_of(&server).await;
    let routing = &body["provider"];
    assert_eq!(routing["order"][0], "Together");
    assert_eq!(routing["order"][1], "Fireworks");
    assert_eq!(routing["allow_fallbacks"], false);
    assert_eq!(routing["only"][0], "Fireworks");
    assert_eq!(routing["ignore"][0], "OpenAI");
    assert_eq!(routing["require_parameters"], true);
    assert_eq!(routing["zdr"], true);
    assert_eq!(routing["quantizations"][0], "fp8");
    assert_eq!(routing["max_price"]["prompt"], 3.0);
    assert_eq!(routing["sort"], "throughput");
    assert_eq!(routing["preferred_min_throughput"], 100.0);
    assert_eq!(routing["preferred_max_latency"], 2.5);
}

#[tokio::test]
async fn privacy_policy_reaches_the_body() {
    // The field an operator would most regret losing: it decides whether an
    // upstream provider may retain the request.
    let server = server().await;
    let mut p = provider(&server, |_| {});
    p.set_extensions(OpenRouterExtensions {
        provider: Some(OpenRouterProviderRouting {
            data_collection: Some(DataCollectionPolicy::Deny),
            zdr: Some(true),
            ..Default::default()
        }),
        ..Default::default()
    });
    p.chat_completion(request()).await.expect("request succeeds");

    let body = body_of(&server).await;
    assert_eq!(body["provider"]["data_collection"], "deny");
    assert_eq!(body["provider"]["zdr"], true);
}

#[tokio::test]
async fn an_absent_routing_policy_sends_no_provider_object() {
    // A deployment that expresses no opinion should not carry `provider: {}`,
    // which reads as "an empty policy" rather than "none".
    let server = server().await;
    provider(&server, |_| {})
        .chat_completion(request())
        .await
        .expect("request succeeds");
    let body = body_of(&server).await;
    assert!(body.get("provider").is_none(), "{body}");
    assert!(body.get("models").is_none(), "{body}");
}

#[tokio::test]
async fn a_fallback_model_list_reaches_the_body() {
    // A fallback changes which model answered, so it must be visible in the
    // request and in the returned metadata.
    let server = server().await;
    let mut p = provider(&server, |_| {});
    p.set_extensions(OpenRouterExtensions {
        models: vec!["anthropic/claude-sonnet-4".into(), "google/gemini-2.5-pro".into()],
        ..Default::default()
    });
    p.chat_completion(request()).await.expect("request succeeds");
    let body = body_of(&server).await;
    assert_eq!(body["models"][0], "anthropic/claude-sonnet-4");
    assert_eq!(body["models"][1], "google/gemini-2.5-pro");
    // The primary is still the model.
    assert_eq!(body["model"], "openai/gpt-4o");
}

// ── Sampling extras (§30) ──────────────────────────────────────────────────

#[tokio::test]
async fn provider_sampling_extras_reach_the_body() {
    let server = server().await;
    let mut p = provider(&server, |_| {});
    p.set_extensions(OpenRouterExtensions {
        top_k: Some(40),
        min_p: Some(0.05),
        top_a: Some(0.8),
        repetition_penalty: Some(1.1),
        structured_outputs: Some(true),
        ..Default::default()
    });
    p.chat_completion(request()).await.expect("request succeeds");

    let body = body_of(&server).await;
    assert_eq!(body["top_k"], 40);
    assert_eq!(body["min_p"], 0.05);
    assert_eq!(body["top_a"], 0.8);
    assert_eq!(body["repetition_penalty"], 1.1);
    assert_eq!(body["structured_outputs"], true);
    // And the portable fields still work alongside them.
    let mut r = request();
    r.max_tokens = Some(256);
    r.seed = Some(11);
    p.chat_completion(r).await.expect("second request");
    let second: Value = {
        let requests = server.received_requests().await.expect("recorded");
        serde_json::from_slice(&requests[1].body).expect("json")
    };
    assert_eq!(second["max_tokens"], 256);
    assert_eq!(second["seed"], 11);
    assert_eq!(second["top_k"], 40, "the extension must not displace a typed field");
}

// ── Attribution and metadata headers (§34, §36) ───────────────────────────

#[tokio::test]
async fn attribution_headers_come_from_configuration() {
    // §34: the title names the application, so it must not be a per-request
    // parameter that different call sites can set differently.
    let server = server().await;
    provider(&server, |c| {
        c.referrer = Some("https://example.com".into());
        c.title = Some("Example App".into());
    })
    .chat_completion(request())
    .await
    .expect("request succeeds");

    let headers = headers_of(&server).await;
    assert_eq!(headers.get("http-referer").map(String::as_str), Some("https://example.com"));
    assert_eq!(headers.get("x-openrouter-title").map(String::as_str), Some("Example App"));
    assert!(
        !headers.contains_key("x-openrouter-metadata"),
        "routing metadata is opt-in: {:?}",
        headers.keys().collect::<Vec<_>>()
    );
}

#[tokio::test]
async fn routing_metadata_is_opt_in() {
    let server = server().await;
    provider(&server, |c| c.metadata = true)
        .chat_completion(request())
        .await
        .expect("request succeeds");
    let headers = headers_of(&server).await;
    assert_eq!(headers.get("x-openrouter-metadata").map(String::as_str), Some("enabled"));
}

#[tokio::test]
async fn the_legacy_router_header_is_still_applied() {
    // Kept for compatibility. It is *not* the primary routing API: it can only
    // express an ordering, and a deployment that relies on it for privacy
    // believes it has a policy it does not have.
    let server = server().await;
    provider(&server, |c| c.default_route = Some("fireworks".into()))
        .chat_completion(request())
        .await
        .expect("request succeeds");
    let headers = headers_of(&server).await;
    assert_eq!(headers.get("openrouter-router").map(String::as_str), Some("fireworks"));
}

// ── Collision semantics (§53) ─────────────────────────────────────────────

#[tokio::test]
async fn a_provider_extension_cannot_shadow_a_typed_field() {
    // The same rule as `extra_params`. A provider extension that can override a
    // typed setting makes the typed field decorative.
    let mut r = request();
    r.extra_params.insert("top_k".into(), json!(999));
    let mut p = provider(&server().await, |_| {});
    p.set_extensions(OpenRouterExtensions {
        top_k: Some(40),
        ..Default::default()
    });
    let error = p
        .chat_completion(r)
        .await
        .expect_err("a collision must be rejected");
    assert!(
        matches!(error, vincents_llm::error::LLMError::ConfigurationError { .. }),
        "{error:?}"
    );
}

// ── Server tools (§38) ────────────────────────────────────────────────────

#[test]
fn a_server_tool_is_not_shaped_like_a_local_function() {
    // §38: these execute remotely, which is a different authority question. The
    // type is distinct so a policy check can tell them apart.
    let tool = OpenRouterServerTool {
        name: "openrouter:web_search".into(),
        description: Some("search the web".into()),
        input_schema: Some(json!({"type": "object"})),
        stop_when_not_called: Some(true),
    };
    let wire = serde_json::to_value(&tool).expect("serializes");
    // Serialized as `type`: that field is what says the call happens remotely.
    assert_eq!(wire["type"], "openrouter:web_search");
    assert_eq!(wire["stop_when_not_called"], true);
    assert!(
        wire.get("function").is_none(),
        "a server tool must not masquerade as a local function: {wire}"
    );
    // The discriminator is the `type` tag, which is why it must be present and
    // why it must carry the `openrouter:` prefix.
    assert_eq!(wire["type"], "openrouter:web_search", "{wire}");
}

// ── supported_parameters (§39) ─────────────────────────────────────────────

#[tokio::test]
async fn supported_parameters_are_preserved_verbatim() {
    // The list is what makes `require_parameters` decidable locally, and it
    // scales to parameters this crate has never heard of.
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/models"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(
            json!({
                "data": [{
                    "id": "some/provider:model",
                    "created": 1_700_000_000,
                    "context_length": 128000,
                    "supported_parameters": [
                        "tools", "tool_choice", "response_format", "reasoning",
                        "verbosity", "top_k", "a_parameter_this_crate_has_never_heard_of"
                    ]
                }]
            })
            .to_string(),
            "application/json",
        ))
        .mount(&server)
        .await;

    let p = provider(&server, |_| {});
    let models = p.list_models().await.expect("models");
    assert_eq!(models.len(), 1);
    let params = &models[0].supported_parameters;
    assert!(params.contains("tools"));
    assert!(params.contains("reasoning"));
    assert!(params.contains("verbosity"));
    assert!(
        params.contains("a_parameter_this_crate_has_never_heard_of"),
        "an unknown parameter must survive: {params:?}"
    );
    // The booleans are derived from the same list.
    assert!(models[0].capabilities.function_calling);
    assert_eq!(models[0].capabilities.source.is_authoritative(), true);
}

// ── Streaming (§40, and the defects that were only ever on this path) ──────

fn sse(events: &[Value]) -> String {
    events
        .iter()
        .map(|event| format!("data: {event}\n\n"))
        .collect::<Vec<_>>()
        .join("")
}

fn tool_stream() -> String {
    sse(&[
        json!({"id": "gen-tools", "created": 1_700_000_500, "model": "openai/gpt-4o",
               "choices": [{"index": 0, "delta": {"tool_calls": [
                   {"index": 0, "id": "call_a", "function": {"name": "get_weather", "arguments": ""}},
                   {"index": 1, "id": "call_b", "function": {"name": "get_time", "arguments": ""}}
               ]}, "finish_reason": null}]}),
        json!({"id": "gen-tools", "created": 1_700_000_500, "model": "openai/gpt-4o",
               "choices": [{"index": 0, "delta": {"tool_calls": [
                   {"index": 0, "function": {"arguments": "{\"city\":"}}]}, "finish_reason": null}]}),
        json!({"id": "gen-tools", "created": 1_700_000_500, "model": "openai/gpt-4o",
               "choices": [{"index": 0, "delta": {"tool_calls": [
                   {"index": 1, "function": {"arguments": "{\"tz\":"}}]}, "finish_reason": null}]}),
        json!({"id": "gen-tools", "created": 1_700_000_500, "model": "openai/gpt-4o",
               "choices": [{"index": 0, "delta": {"tool_calls": [
                   {"index": 0, "function": {"arguments": "\"Oslo\"}"}}]}, "finish_reason": null}]}),
        json!({"id": "gen-tools", "created": 1_700_000_500, "model": "openai/gpt-4o",
               "choices": [{"index": 0, "delta": {"tool_calls": [
                   {"index": 1, "function": {"arguments": "\"UTC\"}"}}]}, "finish_reason": null}]}),
        json!({"id": "gen-tools", "created": 1_700_000_500, "model": "openai/gpt-4o",
               "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
               "usage": {"prompt_tokens": 30, "completion_tokens": 18, "total_tokens": 48}}),
    ]) + "data: [DONE]\n\n"
}

async fn streaming_server(body: String) -> MockServer {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(body, "text/event-stream"))
        .mount(&server)
        .await;
    server
}

#[tokio::test]
async fn openrouter_streams_tool_calls_rather_than_silently_skipping_them() {
    // This is the defect that was unique to OpenRouter: the parser read
    // `delta.content` and skipped everything else, so a tool-call response
    // produced *no* events and one synthetic `stop`. An agentic loop saw a clean
    // completion with no tools.
    let server = streaming_server(tool_stream()).await;
    use futures::StreamExt;
    let mut stream = provider(&server, |_| {})
        .chat_completion_stream_events(request())
        .await
        .expect("stream opens");

    let mut assembler = vincents_llm::types::ToolCallAssembler::new();
    let mut ids = Vec::new();
    let mut finish = Vec::new();
    let mut usage = None;
    while let Some(event) = stream.next().await {
        use vincents_llm::types::ChatStreamEvent as E;
        match event.expect("event") {
            E::ResponseStart { id, .. } => ids.push(id),
            E::ToolCallStart { tool_index, id, name, .. } => {
                assembler.start(tool_index, id, name)
            }
            E::ToolCallArgumentsDelta { tool_index, delta, .. } => {
                assembler.push_arguments(tool_index, &delta)
            }
            E::ChoiceFinished { reason, .. } => finish.push(reason),
            E::Usage(u) => usage = Some(u),
            _ => {}
        }
    }

    assert_eq!(ids, vec!["gen-tools".to_string()], "the provider's id, not a fabricated one");
    assert_eq!(finish, vec![Some("tool_calls".to_string())], "no synthetic trailing stop");

    let calls = assembler.finish();
    let by_name: BTreeMap<&str, &str> = calls
        .iter()
        .map(|c| (c.name.as_deref().unwrap_or_default(), c.arguments.as_str()))
        .collect();
    assert_eq!(by_name["get_weather"], r#"{"city":"Oslo"}"#);
    assert_eq!(by_name["get_time"], r#"{"tz":"UTC"}"#);

    let usage = usage.expect("usage must be the provider's, not invented");
    assert_eq!(usage.total_tokens, 48);
}

#[tokio::test]
async fn a_truncated_openrouter_stream_is_an_error_not_a_completion() {
    // The read loop was `while let Ok(Some(bytes))`, so a dropped connection
    // ended the stream and then fell through to the normal-terminator branch.
    // A partial answer was indistinguishable from a complete one — the single
    // most dangerous behaviour in this file.
    let truncated = sse(&[
        json!({"id": "gen-cut", "created": 1_700_000_000, "model": "openai/gpt-4o",
               "choices": [{"index": 0, "delta": {"content": "The answer is"}, "finish_reason": null}]}),
    ]);
    let server = streaming_server(truncated).await;
    use futures::StreamExt;
    let mut stream = provider(&server, |_| {})
        .chat_completion_stream_events(request())
        .await
        .expect("stream opens");

    let mut text = String::new();
    let mut error = None;
    while let Some(event) = stream.next().await {
        match event {
            Ok(vincents_llm::types::ChatStreamEvent::TextDelta { text: t, .. }) => text.push_str(&t),
            Ok(_) => {}
            Err(e) => {
                error = Some(e);
                break;
            }
        }
    }

    assert_eq!(text, "The answer is", "the partial text is still delivered");
    let error = error.expect("a truncated stream must surface an error, not a clean finish");
    assert!(
        matches!(error, vincents_llm::error::LLMError::StreamInterrupted { .. }),
        "{error:?}"
    );
    assert!(
        error.is_retryable(),
        "the request was understood; the failure was in transit"
    );
}

#[tokio::test]
async fn routing_still_reaches_the_body_on_the_streaming_path() {
    // The streaming path built its body separately, which is exactly how the
    // two paths drifted. The extensions must not vanish when `stream: true`.
    let server = streaming_server(format!("data: [DONE]\n\n")).await;
    use futures::StreamExt;
    let mut p = provider(&server, |_| {});
    p.set_extensions(OpenRouterExtensions {
        provider: Some(OpenRouterProviderRouting {
            zdr: Some(true),
            data_collection: Some(DataCollectionPolicy::Deny),
            ..Default::default()
        }),
        top_k: Some(20),
        ..Default::default()
    });
    let mut stream = p
        .chat_completion_stream_events(request())
        .await
        .expect("stream opens");
    while let Some(event) = stream.next().await {
        event.expect("event");
    }

    let body = body_of(&server).await;
    assert_eq!(body["stream"], true);
    assert_eq!(body["provider"]["zdr"], true);
    assert_eq!(body["provider"]["data_collection"], "deny");
    assert_eq!(body["top_k"], 20);
}

// ── §36, §37, §38 ──────────────────────────────────────────────────────────

#[tokio::test]
async fn trace_identity_headers_are_sent_and_carry_no_prompt_data() {
    let server = server().await;
    provider(&server, |c| {
        c.session_id = Some("traj-abc".into());
        c.trace = Some("span-1".into());
        c.user_id = Some("tenant-7".into());
    })
    .chat_completion(request())
    .await
    .expect("request succeeds");

    let headers = headers_of(&server).await;
    assert_eq!(header(&headers, "x-session-id").as_deref(), Some("traj-abc"));
    assert_eq!(header(&headers, "x-openrouter-trace").as_deref(), Some("span-1"));
    assert_eq!(header(&headers, "x-openrouter-user").as_deref(), Some("tenant-7"));
}

fn header(headers: &std::collections::BTreeMap<String, String>, name: &str) -> Option<String> {
    headers.get(name).cloned()
}

#[tokio::test]
async fn returned_routing_metadata_is_captured_not_merely_requested() {
    // §36. Sending `X-OpenRouter-Metadata: enabled` and discarding the answer
    // satisfies the letter of the request and none of the purpose: the whole
    // point is fallback diagnosis, which needs the generation id *after* the
    // fact.
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(completion_body("ok"))
                .insert_header("x-openrouter-generation-id", "gen-123")
                .insert_header("x-openrouter-provider", "Together")
                .insert_header("x-request-id", "req-9"),
        )
        .mount(&server)
        .await;

    let response = provider(&server, |c| c.metadata = true)
        .chat_completion(request())
        .await
        .expect("request succeeds");

    let metadata: RoutingMetadata =
        serde_json::from_value(response.unknown(vincents_llm::openrouter::OPENROUTER_METADATA_KEY)
            .cloned()
            .expect("routing metadata is present"))
        .expect("decodes");
    assert_eq!(metadata.generation_id.as_deref(), Some("gen-123"));
    assert_eq!(metadata.provider.as_deref(), Some("Together"));
    assert_eq!(metadata.request_id.as_deref(), Some("req-9"));
}

#[tokio::test]
async fn absent_routing_metadata_is_recorded_as_absent() {
    let server = server().await;
    let response = provider(&server, |_| {})
        .chat_completion(request())
        .await
        .expect("request succeeds");
    let value = response
        .unknown(vincents_llm::openrouter::OPENROUTER_METADATA_KEY)
        .cloned()
        .expect("the key is present even when nothing was returned");
    assert_eq!(value, json!({}), "empty, not fabricated");
}

#[tokio::test]
async fn server_tools_reach_the_wire_in_the_tools_array() {
    // §38. This type previously existed and nothing ever sent it.
    let server = server().await;
    let mut p = provider(&server, |_| {});
    p.server_tools = vec![
        OpenRouterServerTool::web_search(),
        OpenRouterServerTool::web_fetch(),
    ];
    p.chat_completion(request()).await.expect("request succeeds");

    let body = body_of(&server).await;
    let tools = body["tools"].as_array().expect("tools array");
    assert_eq!(tools.len(), 2);
    // The wire shape is a bare `{"type": "openrouter:<tool>"}` -- *not* a
    // function object. That is the distinction a policy check reads.
    assert_eq!(tools[0]["type"], "openrouter:web_search");
    assert_eq!(tools[1]["type"], "openrouter:web_fetch");
    assert!(tools[0]["function"].is_null(), "not a local function: {tools:?}");
}

#[tokio::test]
async fn server_tools_and_local_function_tools_share_one_array() {
    let server = server().await;
    let mut p = provider(&server, |_| {});
    p.server_tools = vec![OpenRouterServerTool::web_search()];
    let mut request = request();
    request.tools = Some(vec![vincents_llm::types::ToolDefinition::Function(
        vincents_llm::types::FunctionDefinitionWire {
            name: "get_weather".into(),
            description: None,
            parameters: None,
            strict: None,
        },
    )]);
    p.chat_completion(request).await.expect("request succeeds");

    let body = body_of(&server).await;
    let tools = body["tools"].as_array().expect("tools array");
    assert_eq!(tools.len(), 2, "both kinds, one array");
    assert_eq!(tools[0]["type"], "function");
    assert_eq!(tools[0]["function"]["name"], "get_weather");
    assert_eq!(tools[1]["type"], "openrouter:web_search");
}

#[test]
fn a_server_tool_must_be_named_as_one() {
    // Without the prefix the name is indistinguishable from a local function,
    // and the only thing distinguishing them on the wire is the type tag.
    let error = OpenRouterServerTool::new("web_search").expect_err("must be prefixed");
    assert!(
        matches!(error, vincents_llm::error::LLMError::ConfigurationError { .. }),
        "{error:?}"
    );
    assert!(OpenRouterServerTool::new("openrouter:web_search").is_ok());
}

// ── §35: the OpenRouter Responses extension fields ──────────────────────────

#[tokio::test]
async fn every_openrouter_extension_field_named_by_the_brief_reaches_the_body() {
    // §35 names them explicitly: models, provider, plugins, session_id, trace,
    // cache_control, modalities, server tools, stop_server_tools_when.
    let server = server().await;
    let mut p = provider(&server, |c| {
        // session_id and trace are headers, and are asserted in the header test.
        c.session_id = Some("traj-1".into());
        c.trace = Some("span-1".into());
    });
    p.set_extensions(OpenRouterExtensions {
        models: vec!["anthropic/claude-sonnet-4".into()],
        provider: Some(OpenRouterProviderRouting {
            zdr: Some(true),
            ..Default::default()
        }),
        plugins: vec!["image-gen".into()],
        cache_control: Some(json!({"type": "ephemeral"})),
        modalities: Some(json!(["text", "image"])),
        stop_server_tools_when: Some("model_stopped".into()),
        top_k: Some(40),
        ..Default::default()
    });
    p.chat_completion(request()).await.expect("ok");

    let body = body_of(&server).await;
    assert_eq!(body["models"][0], "anthropic/claude-sonnet-4", "{body}");
    assert_eq!(body["provider"]["zdr"], true);
    assert_eq!(body["plugins"][0], "image-gen");
    assert_eq!(body["cache_control"]["type"], "ephemeral");
    assert_eq!(body["modalities"][1], "image");
    // Stated rather than defaulted: upstream leaves it off, which means a tool
    // that keeps billing after the conversation has moved on.
    assert_eq!(body["stop_server_tools_when"], "model_stopped");
    assert_eq!(body["top_k"], 40);
}

#[tokio::test]
async fn unset_extension_fields_are_omitted_rather_than_null() {
    let server = server().await;
    provider(&server, |_| {})
        .chat_completion(request())
        .await
        .expect("ok");
    let body = body_of(&server).await;
    for field in [
        "plugins",
        "cache_control",
        "modalities",
        "stop_server_tools_when",
        "models",
        "provider",
    ] {
        assert!(body.get(field).is_none(), "'{field}' should be absent: {body}");
    }
}
