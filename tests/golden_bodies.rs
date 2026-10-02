//! Serialization precedence and golden bodies: §53, §59, §60.
//!
//! The contract under test is that the *order* fields are merged in is fixed by
//! this crate and cannot be decided by iteration order of a hash map. A
//! request body whose content depends on insertion order is a body nobody can
//! reproduce from a transcript, which defeats the point of storing one.
//!
//! ```text
//! 1. required protocol fields
//! 2. typed common fields
//! 3. typed provider extension fields
//! 4. raw provider_params / extra_params
//!
//! collision at any step -> error, never override
//! ```

use serde_json::{json, Value};
use std::collections::BTreeMap;
use vincents_llm::responses::{
    ConversationRef, ResponseInput, ResponseRequest, ResponseTool,
};
use vincents_llm::types::{
    ChatCompletionRequest, ChatMessage, ContentPart, FunctionDefinitionWire, ResponseFormat,
    ToolChoice, ToolDefinition,
};
use vincents_llm::wire::{build_chat_request, ChatRequestParts};

fn messages() -> Vec<Value> {
    vec![json!({"role": "user", "content": "hi"})]
}

fn body(request: &ChatCompletionRequest) -> Value {
    build(request, false).expect("valid request")
}

/// Build a body for either path, so a test can compare the two.
fn build(request: &ChatCompletionRequest, stream: bool) -> Result<Value, vincents_llm::error::LLMError> {
    build_chat_request(ChatRequestParts {
        model: "gpt-4o",
        messages: &messages(),
        request,
        stream,
        tools: None,
        provider_params: None,
        default_service_tier: None,
    })
}

fn request() -> ChatCompletionRequest {
    ChatCompletionRequest {
        model: "gpt-4o".into(),
        messages: vec![ChatMessage::user("hi")],
        ..Default::default()
    }
}

/// The body with every key removed except those named, for a readable diff.
fn only(body: &Value, keys: &[&str]) -> BTreeMap<String, Value> {
    body.as_object()
        .expect("object")
        .iter()
        .filter(|(k, _)| keys.contains(&k.as_str()))
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect()
}

// ═══════════════════════════════════════════════════════════════════════════
// §53 — precedence
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn the_body_is_byte_identical_across_repeated_builds() {
    // The property that `HashMap` iteration order would break. Same input, same
    // bytes, every time and on every platform.
    let mut request = request();
    request.temperature = Some(0.3);
    request.seed = Some(7);
    request.extra_params
        .insert("z_first".into(), json!(1));
    request.extra_params
        .insert("a_second".into(), json!(2));
    request.extra_params
        .insert("m_third".into(), json!(3));

    let first = serde_json::to_string(&body(&request)).expect("serializes");
    for _ in 0..50 {
        assert_eq!(serde_json::to_string(&body(&request)).expect("serializes"), first);
    }
    // And input order cannot change the output. This is the property a
    // `HashMap`-backed `extra_params` would break, and it matters because a
    // stored transcript has to reproduce a byte-identical request.
    let mut forward = self::request();
    let mut reverse = self::request();
    for (key, value) in [("a_second", 2), ("m_third", 3), ("z_first", 1)] {
        forward.extra_params.insert(key.into(), json!(value));
    }
    for (key, value) in [("z_first", 1), ("m_third", 3), ("a_second", 2)] {
        reverse.extra_params.insert(key.into(), json!(value));
    }
    assert_eq!(
        serde_json::to_string(&body(&forward)).expect("serializes"),
        serde_json::to_string(&body(&reverse)).expect("serializes"),
        "insertion order must not affect the body"
    );

    // The same for the Responses builder's extension map.
    let responses_forward = |pairs: Vec<(&str, i32)>| {
        let mut request = ResponseRequest {
            model: "gpt-5".into(),
            input: ResponseInput::text("hi"),
            ..Default::default()
        };
        for (key, value) in pairs {
            request.provider_params.insert(key.into(), json!(value));
        }
        request.to_body().expect("valid")
    };
    assert_eq!(
        responses_forward(vec![("a", 1), ("b", 2), ("c", 3)]).to_string(),
        responses_forward(vec![("c", 3), ("b", 2), ("a", 1)]).to_string(),
        "insertion order must not affect the Responses body either"
    );
}

#[test]
fn required_protocol_fields_are_present_first_and_cannot_be_removed() {
    let body = body(&request());
    assert_eq!(body["model"], "gpt-4o");
    assert_eq!(body["messages"][0]["role"], "user");
    // Nothing else: unset typed fields must be absent, not null. A `null` for an
    // unset field is a different request to most providers.
    assert_eq!(body.as_object().expect("object").len(), 2, "{body}");
}

#[test]
fn a_collision_errors_rather_than_overriding_in_either_direction() {
    // Typed set, extension set: the typed value must win by *erroring*, not by
    // being overwritten. An override would make the typed field decorative.
    let mut request = request();
    request.temperature = Some(0.7);
    request.extra_params.insert("temperature".into(), json!(0.1));
    let error = build_chat_request(ChatRequestParts {
        model: "gpt-4o",
        messages: &messages(),
        request: &request,
        stream: false,
        tools: None,
        provider_params: None,
        default_service_tier: None,
    })
    .expect_err("collision");
    assert!(
        matches!(error, vincents_llm::error::LLMError::ConfigurationError { .. }),
        "{error:?}"
    );
}

#[test]
fn responses_precedence_follows_the_same_rule() {
    // The Responses builder is a second implementation of the same contract, so
    // it gets the same assertions.
    let mut request = ResponseRequest {
        model: "gpt-5".into(),
        input: ResponseInput::text("hi"),
        temperature: Some(0.7),
        ..Default::default()
    };
    request
        .provider_params
        .insert("temperature".into(), json!(0.1));
    let error = request.to_body().expect_err("collision");
    assert!(
        matches!(error, vincents_llm::error::LLMError::ConfigurationError { .. }),
        "{error:?}"
    );

    let mut ok = ResponseRequest {
        model: "gpt-5".into(),
        input: ResponseInput::text("hi"),
        ..Default::default()
    };
    ok.provider_params
        .insert("brand_new".into(), json!(1));
    let body = ok.to_body().expect("no collision");
    assert_eq!(body["brand_new"], 1);
    assert_eq!(body["model"], "gpt-5");
}

// ═══════════════════════════════════════════════════════════════════════════
// §59 — OpenAI Chat golden bodies
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn golden_basic_text() {
    assert_eq!(
        body(&request()),
        json!({"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]})
    );
}

#[test]
fn golden_developer_role() {
    let request = ChatCompletionRequest {
        model: "gpt-4o".into(),
        messages: vec![
            ChatMessage::developer("instructions"),
            ChatMessage::user("hi"),
        ],
        ..Default::default()
    };
    let built = build_chat_request(ChatRequestParts {
        model: "gpt-4o",
        messages: &request.messages.iter().map(ChatMessage::to_openai_json).collect::<Vec<_>>(),
        request: &request,
        stream: false,
        tools: None,
        provider_params: None,
        default_service_tier: None,
    })
    .expect("valid");
    assert_eq!(
        built["messages"][0],
        json!({"role": "developer", "content": "instructions"})
    );
}

#[test]
fn golden_image_content() {
    let request = ChatCompletionRequest {
        model: "gpt-4o".into(),
        messages: vec![ChatMessage::User {
            content: vincents_llm::types::MessageContent::Parts(vec![
                ContentPart::Text { text: "what is this?".into() },
                ContentPart::ImageUrl {
                    url: "https://example.com/c.png".into(),
                    detail: Some(vincents_llm::types::ImageDetail::High),
                },
            ]),
            name: None,
        }],
        ..Default::default()
    };
    let built = build_chat_request(ChatRequestParts {
        model: "gpt-4o",
        messages: &request.messages.iter().map(ChatMessage::to_openai_json).collect::<Vec<_>>(),
        request: &request,
        stream: false,
        tools: None,
        provider_params: None,
        default_service_tier: None,
    })
    .expect("valid");
    assert_eq!(
        built["messages"][0]["content"][1],
        json!({"type": "image_url",
               "image_url": {"url": "https://example.com/c.png", "detail": "high"}})
    );
}

#[test]
fn golden_audio_and_file_content() {
    let content = vincents_llm::types::MessageContent::Parts(vec![
        ContentPart::InputAudio {
            data: "UklGRg==".into(),
            format: vincents_llm::types::AudioFormat::Wav,
        },
        ContentPart::File {
            file_id: Some("file-1".into()),
            file_data: None,
            filename: Some("a.pdf".into()),
        },
    ]);
    let wire = ChatMessage::User { content, name: None }.to_openai_json();
    assert_eq!(
        wire["content"][0],
        json!({"type": "input_audio", "input_audio": {"data": "UklGRg==", "format": "wav"}})
    );
    assert_eq!(
        wire["content"][1],
        json!({"type": "file", "file": {"file_id": "file-1", "filename": "a.pdf"}})
    );
}

#[test]
fn golden_generation_parameters() {
    let mut request = request();
    request.max_completion_tokens = Some(256);
    request.seed = Some(42);
    request.logprobs = Some(true);
    request.top_logprobs = Some(3);
    request.service_tier = Some(vincents_llm::types::ServiceTier("flex".into()));

    let built = body(&request);
    assert_eq!(
        only(
            &built,
            &["max_completion_tokens", "seed", "logprobs", "top_logprobs", "service_tier"]
        ),
        BTreeMap::from([
            ("logprobs".to_string(), json!(true)),
            ("max_completion_tokens".to_string(), json!(256)),
            ("seed".to_string(), json!(42)),
            ("service_tier".to_string(), json!("flex")),
            ("top_logprobs".to_string(), json!(3)),
        ])
    );
    // `max_tokens` is *absent*, not null: the two are not interchangeable and
    // sending both is rejected.
    assert!(built.get("max_tokens").is_none());
}

#[test]
fn golden_structured_output() {
    let mut request = request();
    request.response_format = Some(ResponseFormat::JsonObject);
    assert_eq!(body(&request)["response_format"], json!({"type": "json_object"}));

    request.response_format = Some(ResponseFormat::JsonSchema {
        name: "answer".into(),
        description: None,
        schema: json!({"type": "object"}),
        strict: Some(true),
    });
    let built = body(&request);
    assert_eq!(built["response_format"]["type"], "json_schema");
    // Nested, with `name` and `strict` *inside* `json_schema`.
    assert_eq!(built["response_format"]["json_schema"]["name"], "answer");
    assert_eq!(built["response_format"]["json_schema"]["strict"], true);
    assert!(built["response_format"]["name"].is_null());
}

#[test]
fn golden_function_tool_and_choice() {
    let mut request = request();
    request.tools = Some(vec![
        ToolDefinition::Function(FunctionDefinitionWire {
            name: "get_weather".into(),
            description: Some("weather".into()),
            parameters: Some(json!({"type": "object"})),
            strict: Some(true),
        }),
        ToolDefinition::Function(FunctionDefinitionWire {
            name: "get_time".into(),
            description: None,
            parameters: None,
            strict: None,
        }),
    ]);

    let tool_wire: Vec<Value> = request
        .tools
        .as_ref()
        .expect("tools")
        .iter()
        .map(|t| serde_json::to_value(t).expect("tool serializes"))
        .collect();
    let built = build_chat_request(ChatRequestParts {
        model: "gpt-4o",
        messages: &messages(),
        request: &request,
        stream: false,
        tools: Some(&tool_wire),
        default_service_tier: None,
        provider_params: None,
    })
    .expect("valid");

    assert_eq!(built["tools"][0]["type"], "function");
    assert_eq!(built["tools"][0]["function"]["name"], "get_weather");
    assert_eq!(built["tools"][0]["function"]["strict"], true);
    // An absent description is omitted, not null.
    assert!(built["tools"][1]["function"].get("description").is_none());

    request.tool_choice = Some(ToolChoice::Required);
    assert_eq!(body(&request)["tool_choice"], json!("required"));
    request.tool_choice = Some(ToolChoice::Function { name: "get_time".into() });
    assert_eq!(
        body(&request)["tool_choice"],
        json!({"type": "function", "function": {"name": "get_time"}})
    );
    request.tool_choice = Some(ToolChoice::Auto);
    assert_eq!(body(&request)["tool_choice"], json!("auto"));
}

#[test]
fn golden_parallel_tool_calls_and_reasoning() {
    let mut request = request();
    request.parallel_tool_calls = Some(true);
    request.reasoning = Some(vincents_llm::types::ReasoningConfig {
        effort: Some(vincents_llm::types::ReasoningEffort("medium".into())),
        max_tokens: None,
        exclude: None,
        summary: Some(vincents_llm::types::ReasoningSummary::Concise),
    });
    request.verbosity = Some(vincents_llm::types::Verbosity("low".into()));

    let built = body(&request);
    assert_eq!(built["parallel_tool_calls"], true);
    assert_eq!(built["reasoning"]["effort"], "medium");
    assert_eq!(built["reasoning"]["summary"], "concise");
    assert_eq!(built["verbosity"], "low");
    // An unknown effort from a newer provider round-trips rather than being
    // dropped or rejected.
    request.reasoning = Some(vincents_llm::types::ReasoningConfig {
        effort: Some(vincents_llm::types::ReasoningEffort("extreme".into())),
        max_tokens: None,
        exclude: None,
        summary: None,
    });
    assert_eq!(body(&request)["reasoning"]["effort"], "extreme");
}

#[test]
fn golden_extra_param_without_and_with_collision() {
    let mut request = request();
    request.extra_params.insert("seed".into(), json!(1));
    let built = body(&request);
    assert_eq!(built["seed"], 1);

    request.extra_params.insert("model".into(), json!("other"));
    let error = build_chat_request(ChatRequestParts {
        model: "gpt-4o",
        messages: &messages(),
        request: &request,
        stream: false,
        tools: None,
        provider_params: None,
        default_service_tier: None,
    })
    .expect_err("collision with a required field");
    assert!(matches!(
        error,
        vincents_llm::error::LLMError::ConfigurationError { .. }
    ));
}

#[test]
fn golden_tool_role_round_trip() {
    // A tool result is not a user message. Rewording the role breaks the
    // conversation rather than merely changing its shape.
    let messages = vec![
        ChatMessage::assistant_with_tool_calls(vec![vincents_llm::types::ToolCall::Function(
            vincents_llm::types::FunctionCall::Custom(vincents_llm::types::CustomFunctionCall {
                id: Some("call_a".into()),
                name: "get_weather".into(),
                arguments: r#"{"city":"Oslo"}"#.into(),
            }),
        )]),
        ChatMessage::Tool {
            tool_call_id: "call_a".into(),
            content: "12C".into(),
        },
    ];
    let wire: Vec<Value> = messages.iter().map(ChatMessage::to_openai_json).collect();
    assert_eq!(wire[0]["role"], "assistant");
    assert!(wire[0]["content"].is_null(), "no text, not empty text");
    assert_eq!(wire[0]["tool_calls"][0]["id"], "call_a");
    assert_eq!(wire[0]["tool_calls"][0]["type"], "function");
    assert_eq!(wire[1]["role"], "tool");
    assert_eq!(wire[1]["tool_call_id"], "call_a");
    assert_eq!(wire[1]["content"], "12C");
}

// ═══════════════════════════════════════════════════════════════════════════
// §60 — OpenRouter golden bodies
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn golden_openrouter_extensions_serialize_as_expected() {
    use vincents_llm::openrouter::{
        DataCollectionPolicy, OpenRouterExtensions, OpenRouterMaxPrice, OpenRouterProviderRouting,
        OpenRouterSort,
    };
    let extensions = OpenRouterExtensions {
        top_k: Some(40),
        min_p: Some(0.05),
        top_a: Some(0.8),
        repetition_penalty: Some(1.1),
        structured_outputs: Some(true),
        web_search_options: Some(json!({"search_context_size": "high"})),
        provider: Some(OpenRouterProviderRouting {
            order: vec!["Together".into()],
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
            preferred_max_latency: Some(2.5,
            ),
        }),
        models: vec!["anthropic/claude-sonnet-4".into()],
    };

    let params = extensions.to_params();
    let body = build_chat_request(ChatRequestParts {
        model: "openai/gpt-4o",
        messages: &messages(),
        request: &request(),
        stream: false,
        tools: None,
        provider_params: Some(&params),
        default_service_tier: None,
    })
    .expect("valid");

    assert_eq!(body["top_k"], 40);
    assert_eq!(body["min_p"], 0.05);
    assert_eq!(body["top_a"], 0.8);
    assert_eq!(body["repetition_penalty"], 1.1);
    assert_eq!(body["structured_outputs"], true);
    assert_eq!(body["web_search_options"]["search_context_size"], "high");
    assert_eq!(body["models"][0], "anthropic/claude-sonnet-4");

    let provider = &body["provider"];
    assert_eq!(provider["order"][0], "Together");
    assert_eq!(provider["allow_fallbacks"], false);
    assert_eq!(provider["only"][0], "Fireworks");
    assert_eq!(provider["ignore"][0], "OpenAI");
    assert_eq!(provider["require_parameters"], true);
    assert_eq!(provider["data_collection"], "deny");
    assert_eq!(provider["zdr"], true);
    assert_eq!(provider["quantizations"][0], "fp8");
    assert_eq!(provider["max_price"]["prompt"], 3.0);
    assert_eq!(provider["max_price"]["completion"], 15.0);
    assert_eq!(provider["sort"], "throughput");
    assert_eq!(provider["preferred_min_throughput"], 100.0);
    assert_eq!(provider["preferred_max_latency"], 2.5);

    // Unset optional price fields are omitted, not null.
    assert!(provider["max_price"].get("image").is_none());
}

#[test]
fn golden_responses_tool_envelope() {
    let request = ResponseRequest {
        model: "gpt-5".into(),
        input: ResponseInput::text("hi"),
        tools: vec![
            ResponseTool::Function {
                name: "f".into(),
                description: None,
                parameters: Some(json!({"type": "object"})),
                strict: Some(true),
            },
            ResponseTool::WebSearch {
                search_context_size: Some(vincents_llm::responses::SearchContextSize::High),
                user_location: None,
            },
        ],
        conversation: Some(ConversationRef::Id("conv_1".into())),
        ..Default::default()
    };
    let body = request.to_body().expect("valid");
    assert_eq!(body["tools"][0]["type"], "function");
    assert_eq!(body["tools"][0]["name"], "f");
    assert!(body["tools"][0]["function"].is_null());
    assert_eq!(body["tools"][1]["type"], "web_search");
    assert_eq!(body["tools"][1]["search_context_size"], "high");
    assert!(body["tools"][1]["name"].is_null(), "not a named function");
    assert_eq!(body["conversation"], "conv_1");
}
// ═══════════════════════════════════════════════════════════════════════════
// §55 — the compatibility adapter must be indistinguishable from the typed path
// ═══════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn the_legacy_functions_adapter_produces_the_typed_body() {
    use vincents_llm::provider::LLMProvider;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    async fn body_of(server: &MockServer) -> Value {
        let requests = server.received_requests().await.expect("recorded");
        serde_json::from_slice::<Value>(&requests[0].body).expect("json")
    }

    async fn mount(server: &MockServer) {
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": "c", "object": "chat.completion", "created": 1, "model": "gpt-4o",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
            })))
            .mount(server)
            .await;
    }

    let make = |uri: &str| {
        vincents_llm::openai::OpenAIProvider::with_config(vincents_llm::openai::OpenAIConfig {
            api_key: "k".into(),
            base_url: Some(uri.to_string()),
            default_model: Some("gpt-4o".into()),
            ..Default::default()
        })
        .expect("provider")
    };

    let legacy = MockServer::start().await;
    mount(&legacy).await;
    let functions = vec![vincents_llm::types::FunctionDefinition {
        name: "get_weather".into(),
        description: Some("weather for a city".into()),
        parameters: Some(json!({"type": "object"})),
    }];
    make(&legacy.uri())
        .chat_completion_with_functions(
            ChatCompletionRequest {
                model: "gpt-4o".into(),
                messages: vec![ChatMessage::user("hi")],
                ..Default::default()
            },
            functions.clone(),
        )
        .await
        .expect("legacy path");

    let typed = MockServer::start().await;
    mount(&typed).await;
    make(&typed.uri())
        .chat_completion(
            ChatCompletionRequest {
                model: "gpt-4o".into(),
                messages: vec![ChatMessage::user("hi")],
                tools: Some(vec![ToolDefinition::Function(FunctionDefinitionWire {
                    name: "get_weather".into(),
                    description: Some("weather for a city".into()),
                    parameters: Some(json!({"type": "object"})),
                    strict: None,
                })]),
                ..Default::default()
            },
        )
        .await
        .expect("typed path");

    // The whole point of the migration: the legacy entry point is now an
    // adapter, so the two produce the same bytes. A caller migrating from one to
    // the other must not see the request change on the wire.
    assert_eq!(body_of(&legacy).await, body_of(&typed).await);
}

#[tokio::test]
async fn the_adapter_appends_to_tools_rather_than_replacing_them() {
    use vincents_llm::provider::LLMProvider;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "c", "object": "chat.completion", "created": 1, "model": "gpt-4o",
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

    provider
        .chat_completion_with_functions(
            ChatCompletionRequest {
                model: "gpt-4o".into(),
                messages: vec![ChatMessage::user("hi")],
                tools: Some(vec![ToolDefinition::Function(FunctionDefinitionWire {
                    name: "already_set".into(),
                    description: None,
                    parameters: None,
                    strict: None,
                })]),
                ..Default::default()
            },
            vec![vincents_llm::types::FunctionDefinition {
                name: "from_legacy".into(),
                description: None,
                parameters: None,
            }],
        )
        .await
        .expect("request");

    let requests = server.received_requests().await.expect("recorded");
    let body: Value = serde_json::from_slice(&requests[0].body).expect("json");
    let names: Vec<&str> = body["tools"]
        .as_array()
        .expect("tools")
        .iter()
        .map(|t| t["function"]["name"].as_str().unwrap_or_default())
        .collect();
    assert_eq!(
        names,
        vec!["already_set", "from_legacy"],
        "the union, not a silent replacement"
    );
}

// ── §3: the fields the brief names as missing, all first-class ────────────

#[test]
fn golden_prompt_cache_key_reaches_the_body() {
    // A per-request key derived by hand produces a different key each time and
    // measures no hit rate at all -- which is the reason it is a typed field
    // beside the other request-shaping fields rather than an `extra_params` key.
    let mut request = request();
    request.prompt_cache_key = Some("task-42".into());
    assert_eq!(
        body(&request)["prompt_cache_key"],
        "task-42"
    );
}

#[test]
fn an_unset_prompt_cache_key_is_absent_not_null() {
    // A null on an unsupported field is rejected by some providers, and a key
    // that is present-but-empty groups every request into one cache entry.
    let built = body(&request());
    assert!(built.get("prompt_cache_key").is_none(), "{built}");
}

#[test]
fn every_field_the_brief_names_as_missing_is_first_class() {
    // §3 lists these as absent from the original `ChatCompletionRequest`. Each is
    // a named field now, not an `extra_params` key, so a caller gets a compile
    // error rather than a silently ignored parameter.
    use vincents_llm::types::{ChatCompletionRequest as R, StreamOptions};

    let mut request: R = request();
    request.max_completion_tokens = Some(1);
    request.reasoning = Some(vincents_llm::types::ReasoningConfig {
        effort: Some(vincents_llm::types::ReasoningEffort("low".into())),
        max_tokens: None,
        exclude: None,
        summary: None,
    });
    request.verbosity = Some(vincents_llm::types::Verbosity("low".into()));
    request.seed = Some(1);
    request.logprobs = Some(true);
    request.top_logprobs = Some(1);
    request.response_format = Some(vincents_llm::types::ResponseFormat::JsonObject);
    request.tools = Some(vec![ToolDefinition::Function(FunctionDefinitionWire {
        name: "f".into(),
        description: None,
        parameters: None,
        strict: Some(true),
    })]);
    request.tool_choice = Some(ToolChoice::Auto);
    request.parallel_tool_calls = Some(true);
    request.stream_options = Some(StreamOptions {
        include_usage: Some(true),
    });
    request.service_tier = Some(vincents_llm::types::ServiceTier("flex".into()));
    request.metadata.insert("k".into(), "v".into());
    request.store = Some(true);
    request.prompt_cache_key = Some("c".into());
    request.messages = vec![
        ChatMessage::developer("d"),
        ChatMessage::User {
            content: vincents_llm::types::MessageContent::Parts(vec![ContentPart::Text {
                text: "hi".into(),
            }]),
            name: None,
        },
    ];

    // Every one of them reaches the wire. Built as a streaming request, because
    // `stream_options` is only meaningful there -- and the rejection on a
    // non-streaming request is itself a thing worth having asserted elsewhere.
    //
    // Messages come from the request rather than the shared fixture, because two
    // of the fields under test are message-shaped.
    let serialized = request
        .messages
        .iter()
        .map(ChatMessage::to_openai_json)
        .collect::<Vec<_>>();
    let built = build_chat_request(ChatRequestParts {
        model: "gpt-4o",
        messages: &serialized,
        request: &request,
        stream: true,
        tools: None,
        default_service_tier: None,
        provider_params: None,
    })
    .expect("valid");
    for field in [
        "max_completion_tokens",
        "reasoning",
        "verbosity",
        "seed",
        "logprobs",
        "top_logprobs",
        "response_format",
        "tool_choice",
        "parallel_tool_calls",
        "stream_options",
        "service_tier",
        "metadata",
        "store",
        "prompt_cache_key",
    ] {
        assert!(built.get(field).is_some(), "'{field}' is not first-class: {built}");
    }
    assert_eq!(built["stream"], true);
    assert_eq!(built["messages"][0]["role"], "developer");
    assert!(built["messages"][1]["content"].is_array());
}
