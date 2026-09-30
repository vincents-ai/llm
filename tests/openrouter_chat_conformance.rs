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
    assert_eq!(wire["name"], "openrouter:web_search");
    assert_eq!(wire["stop_when_not_called"], true);
    assert!(
        wire.get("function").is_none(),
        "a server tool must not masquerade as a local function: {wire}"
    );
    assert!(wire.get("type").is_none(), "{wire}");
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
