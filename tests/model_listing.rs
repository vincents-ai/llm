#[cfg(test)]
mod tests {
    use vincents_llm::anthropic::AnthropicProvider;
    use vincents_llm::ollama::OllamaProvider;
    use vincents_llm::openai::OpenAIProvider;
    use vincents_llm::provider::LLMProvider;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    /// The model listings each provider is pointed at.
    ///
    /// These tests previously called the live Anthropic, OpenAI and Ollama
    /// endpoints with the literal key `"test-key"`, so they only ever passed on
    /// a machine with valid credentials and no network. Worse, they asserted on
    /// whatever those APIs happened to return that day, which means a provider
    /// adding a model broke the suite and a provider *removing* one broke it too.
    ///
    /// Fixtures invert that: the response is the specification, so the test
    /// pins parsing rather than today's catalogue.
    async fn mock_server() -> MockServer {
        let server = MockServer::start().await;

        // Anthropic reads `{base}/v1/models`.
        Mock::given(method("GET"))
            .and(path("/v1/models"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(
                serde_json::json!({
                    "data": [
                        {"id": "claude-3-5-sonnet-20241022", "display_name": "Claude 3.5 Sonnet"},
                        {"id": "claude-3-opus-20240229", "display_name": "Claude 3 Opus"},
                        {"id": "claude-3-haiku-20240307", "display_name": "Claude 3 Haiku"}
                    ]
                })
                .to_string(),
                "application/json",
            ))
            .mount(&server)
            .await;

        // OpenAI reads `{base}/models`.
        Mock::given(method("GET"))
            .and(path("/models"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(
                serde_json::json!({
                    "data": [
                        {
                            "id": "gpt-4", "created": 1_000_000,
                            "input_modalities": ["text"],
                            "context_length": 8192, "supported_parameters": ["tools"]
                        },
                        {
                            "id": "gpt-4-turbo", "created": 1_100_000,
                            "input_modalities": ["text", "image"],
                            "context_length": 128000,
                            "supported_parameters": ["tools", "response_format", "vision"]
                        },
                        {
                            "id": "gpt-3.5-turbo", "created": 900_000,
                            "input_modalities": ["text"],
                            "context_length": 4096,
                            "supported_parameters": ["tools", "response_format"]
                        }
                    ]
                })
                .to_string(),
                "application/json",
            ))
            .mount(&server)
            .await;

        // Ollama reads `{host}/api/tags`, with its own shape.
        Mock::given(method("GET"))
            .and(path("/api/tags"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(
                serde_json::json!({
                    "models": [
                        {"name": "llama3.1:8b", "size": "4661224676"},
                        {"name": "qwen2.5-coder:7b", "size": "4683077139"}
                    ]
                })
                .to_string(),
                "application/json",
            ))
            .mount(&server)
            .await;

        server
    }

    async fn anthropic(server: &MockServer) -> AnthropicProvider {
        AnthropicProvider::with_config(vincents_llm::anthropic::AnthropicConfig {
            api_key: "test-key".into(),
            base_url: Some(server.uri()),
            ..Default::default()
        })
        .expect("Should create Anthropic provider")
    }

    async fn openai(server: &MockServer) -> OpenAIProvider {
        OpenAIProvider::with_config(vincents_llm::openai::OpenAIConfig {
            api_key: "test-key".into(),
            base_url: Some(server.uri()),
            ..Default::default()
        })
        .expect("Should create OpenAI provider")
    }

    async fn ollama(server: &MockServer) -> OllamaProvider {
        OllamaProvider::with_config(vincents_llm::ollama::OllamaConfig {
            host: server.uri(),
            default_model: Some("llama3.1:8b".into()),
        })
        .expect("Should create Ollama provider")
    }

    #[tokio::test]
    async fn test_anthropic_list_models() {
        let server = mock_server().await;
        let provider = anthropic(&server).await;

        let models = provider.list_models().await.expect("Should list models");

        assert!(
            !models.is_empty(),
            "Anthropic should have at least one model"
        );

        // Verify Claude 3.5 Sonnet
        let sonnet = models
            .iter()
            .find(|m| m.id == "claude-3-5-sonnet-20241022")
            .expect("Should have Claude 3.5 Sonnet");

        assert_eq!(sonnet.name, "Claude 3.5 Sonnet");
        assert_eq!(sonnet.context_window, 200000);
        assert!(sonnet.capabilities.vision);
        assert!(sonnet.capabilities.function_calling);
        assert!(sonnet.capabilities.caching);

        let pricing = sonnet.pricing.as_ref().expect("Should have pricing");
        assert_eq!(pricing.prompt_tokens, 0.003);
        assert_eq!(pricing.completion_tokens, 0.015);
    }

    #[tokio::test]
    async fn test_anthropic_claude_opus() {
        let server = mock_server().await;
        let provider = anthropic(&server).await;

        let models = provider.list_models().await.expect("Should list models");

        let opus = models
            .iter()
            .find(|m| m.id == "claude-3-opus-20240229")
            .expect("Should have Claude 3 Opus");

        assert_eq!(opus.name, "Claude 3 Opus");
        assert_eq!(opus.context_window, 200000);

        let pricing = opus.pricing.as_ref().expect("Should have pricing");
        assert_eq!(pricing.prompt_tokens, 0.015);
        assert_eq!(pricing.completion_tokens, 0.075);
    }

    #[tokio::test]
    async fn test_anthropic_claude_haiku() {
        let server = mock_server().await;
        let provider = anthropic(&server).await;

        let models = provider.list_models().await.expect("Should list models");

        let haiku = models
            .iter()
            .find(|m| m.id == "claude-3-haiku-20240307")
            .expect("Should have Claude 3 Haiku");

        assert_eq!(haiku.name, "Claude 3 Haiku");
        assert_eq!(haiku.context_window, 200000);
        assert!(
            !haiku.capabilities.caching,
            "Haiku should not support caching"
        );

        let pricing = haiku.pricing.as_ref().expect("Should have pricing");
        assert!(pricing.prompt_tokens < 0.001);
        assert!(pricing.completion_tokens < 0.01);
    }

    #[tokio::test]
    async fn test_openai_list_models() {
        let server = mock_server().await;
        let provider = openai(&server).await;

        let models = provider.list_models().await.expect("Should list models");

        assert!(!models.is_empty(), "OpenAI should have at least one model");

        // Verify GPT-4
        let gpt4 = models
            .iter()
            .find(|m| m.id == "gpt-4")
            .expect("Should have GPT-4");

        // The id is the only name the API provides, so the parser returns it
        // unchanged. The pretty names in the old test came from the hardcoded
        // fallback list, which the live path never used.
        assert_eq!(gpt4.name, "gpt-4");
        assert_eq!(gpt4.context_window, 8192);
        assert!(gpt4.capabilities.function_calling);
        // `gpt-4` is text-only, and the fixture says so: `input_modalities` has
        // no "image". This assertion used to be `assert!(vision)` on the strength
        // of a substring match, which happened to be false for this model.
        assert!(
            !gpt4.capabilities.vision,
            "the fixture declares gpt-4 as text-only"
        );
    }

    #[tokio::test]
    async fn test_openai_gpt4_turbo() {
        let server = mock_server().await;
        let provider = openai(&server).await;

        let models = provider.list_models().await.expect("Should list models");

        let turbo = models
            .iter()
            .find(|m| m.id == "gpt-4-turbo")
            .expect("Should have GPT-4 Turbo");

        // The raw id, for the reason given in the GPT-4 case above.
        assert_eq!(turbo.name, "gpt-4-turbo");
        assert_eq!(turbo.context_window, 128000);
        assert!(turbo.capabilities.vision, "the fixture declares image support");
        // The old assertion was `caching`, which the OpenAI parser reports from
        // a substring match and reports `false` for this model. `caching` is not
        // in the fixture's `supported_parameters`, so the honest answer is false.
        assert!(
            !turbo.capabilities.caching,
            "the fixture does not list caching as supported"
        );
    }

    #[tokio::test]
    async fn test_openai_gpt35_turbo() {
        let server = mock_server().await;
        let provider = openai(&server).await;

        let models = provider.list_models().await.expect("Should list models");

        let gpt35 = models
            .iter()
            .find(|m| m.id == "gpt-3.5-turbo")
            .expect("Should have GPT-3.5 Turbo");

        assert_eq!(gpt35.name, "gpt-3.5-turbo");
        assert_eq!(gpt35.context_window, 4096);
        assert!(
            !gpt35.capabilities.vision,
            "GPT-3.5 should not support vision"
        );
        assert!(gpt35.capabilities.function_calling);

        let pricing = gpt35.pricing.as_ref().expect("Should have pricing");
        assert_eq!(pricing.prompt_tokens, 0.0005);
        assert_eq!(pricing.completion_tokens, 0.0015);
    }

    #[tokio::test]
    async fn test_ollama_list_models() {
        let server = mock_server().await;
        let provider = ollama(&server).await;

        // Previously this asserted `models.is_ok() || models.is_err()`, which
        // cannot fail, and pointed at `localhost:11434` with a comment admitting
        // it needed a running Ollama. Against a fixture it can assert something:
        // that the `{"models": [{"name", "size"}]}` shape is parsed, and that the
        // `name` field — the only identifier Ollama returns — survives.
        let models = provider.list_models().await.expect("Should list models");
        assert_eq!(models.len(), 2, "both fixture models should be listed");
        let ids: Vec<&str> = models.iter().map(|m| m.id.as_str()).collect();
        assert!(
            ids.contains(&"llama3.1:8b"),
            "Ollama's name field is the model id: {ids:?}"
        );
        assert!(ids.contains(&"qwen2.5-coder:7b"), "{ids:?}");
    }

    #[tokio::test]
    async fn test_model_info_includes_all_required_fields() {
        let server = mock_server().await;
        let provider = anthropic(&server).await;

        let models = provider.list_models().await.expect("Should list models");

        for model in models {
            assert!(!model.id.is_empty(), "Model id should not be empty");
            assert!(!model.name.is_empty(), "Model name should not be empty");
            assert_eq!(model.provider, "anthropic");
            assert!(
                model.context_window > 0,
                "Context window should be positive"
            );
            assert!(model.pricing.is_some(), "Pricing should be present");

            if let Some(pricing) = model.pricing {
                assert!(
                    pricing.prompt_tokens > 0.0 || pricing.prompt_tokens == 0.0,
                    "Pricing should be valid"
                );
                assert!(
                    pricing.completion_tokens > 0.0 || pricing.completion_tokens == 0.0,
                    "Pricing should be valid"
                );
            }
        }
    }

    #[tokio::test]
    async fn test_model_capabilities_structure() {
        let server = mock_server().await;
        let provider = openai(&server).await;

        let models = provider.list_models().await.expect("Should list models");

        for model in models {
            // Capabilities should always be present
            let caps = &model.capabilities;

            // These are boolean fields, so they should be validly initialized
            let _ = caps.vision;
            let _ = caps.function_calling;
            let _ = caps.streaming;
            let _ = caps.json_mode;
            let _ = caps.caching;
        }
    }

    #[tokio::test]
    async fn test_anthropic_models_sorted_by_price() {
        let server = mock_server().await;
        let provider = anthropic(&server).await;

        let mut models = provider.list_models().await.expect("Should list models");

        // Sort by prompt token cost
        models.sort_by(|a, b| {
            let a_cost = a.pricing.as_ref().map(|p| p.prompt_tokens).unwrap_or(0.0);
            let b_cost = b.pricing.as_ref().map(|p| p.prompt_tokens).unwrap_or(0.0);
            a_cost
                .partial_cmp(&b_cost)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Haiku should be cheapest
        assert_eq!(
            models.first().map(|m| m.id.as_str()),
            Some("claude-3-haiku-20240307")
        );

        // Opus should be most expensive
        assert_eq!(
            models.last().map(|m| m.id.as_str()),
            Some("claude-3-opus-20240229")
        );
    }

    #[tokio::test]
    async fn test_openai_models_by_context_window() {
        let server = mock_server().await;
        let provider = openai(&server).await;

        let mut models = provider.list_models().await.expect("Should list models");

        // Sort by context window descending
        models.sort_by(|a, b| b.context_window.cmp(&a.context_window));

        // GPT-4-turbo has the largest context
        assert!(models[0].context_window >= 128000);

        // GPT-3.5-turbo has the smallest context
        assert!(models.last().map(|m| m.context_window).unwrap_or(0) <= 4096);
    }

    #[tokio::test]
    async fn test_model_availability_flag() {
        let server = mock_server().await;
        let provider = anthropic(&server).await;

        let models = provider.list_models().await.expect("Should list models");

        // All returned models should be available
        for model in models {
            assert!(
                model.available,
                "All returned models should be marked as available"
            );
        }
    }

    #[tokio::test]
    async fn test_get_full_model_info_async() {
        let server = mock_server().await;
        let provider = anthropic(&server).await;

        let model_opt = provider
            .get_full_model_info("claude-3-5-sonnet-20241022")
            .await
            .expect("Should not error");

        assert!(model_opt.is_some(), "Should find model");

        let model = model_opt.unwrap();
        assert_eq!(model.name, "Claude 3.5 Sonnet");
        assert_eq!(model.context_window, 200000);
    }

    #[tokio::test]
    async fn test_get_full_model_info_not_found() {
        let server = mock_server().await;
        let provider = anthropic(&server).await;

        let model_opt = provider
            .get_full_model_info("nonexistent-model")
            .await
            .expect("Should not error");

        assert!(model_opt.is_none(), "Should not find nonexistent model");
    }
}

// ── §40: capabilities are known or unknown, never guessed ─────────────────

#[tokio::test]
async fn a_live_listing_reports_unknown_capabilities_rather_than_guessing() {
    // §40 names this defect exactly: "the current OpenAI provider guesses
    // tools/vision/context from substrings". The listing establishes that a
    // model exists. Nothing else follows from the name.
    use serde_json::json;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "data": [
                // Names the old substring matcher would have "recognised".
                {"id": "gpt-4o-2024-11-20-vision-preview", "created": 1},
                {"id": "brand-new-model-released-yesterday", "created": 2},
                {"id": "some-128k-model", "created": 3}
            ]
        })))
        .mount(&server)
        .await;

    let provider = vincents_llm::openai::OpenAIProvider::with_config(
        vincents_llm::openai::OpenAIConfig {
            api_key: "k".into(),
            base_url: Some(server.uri()),
            ..Default::default()
        },
    )
    .expect("provider");

    // `fetch_models_live` is the listing path itself; `list_models` prefers the
    // catalog and would not exercise the parse.
    let models = provider.fetch_models_live().await.expect("listing");
    // The listing is merged with the catalog, so assert on the three names the
    // old substring matcher would have "recognised" rather than on a count.
    let listed: Vec<&vincents_llm::types::FullModelInfo> = models
        .iter()
        .filter(|m| {
            m.id == "gpt-4o-2024-11-20-vision-preview"
                || m.id == "brand-new-model-released-yesterday"
                || m.id == "some-128k-model"
        })
        .collect();
    assert_eq!(listed.len(), 3, "a model we know nothing about still exists");

    for model in &listed {
        let capabilities = &model.capabilities;
        // Vision, tools and json are *not* determined by a name.
        assert!(
            !capabilities.is_known(vincents_llm::types::Capability::Vision),
            "{}: vision must not be inferred from the name",
            model.id
        );
        assert!(!capabilities.is_known(vincents_llm::types::Capability::FunctionCalling));
        assert_eq!(
            capabilities.get(vincents_llm::types::Capability::Vision),
            None,
            "{}: `get` must return None, not Some(false), for an unknown",
            model.id
        );
        // And the context window is not `contains("128k")`.
        assert_eq!(
            model.context_window, 0,
            "{}: a digit run in the name is not a context window",
            model.id
        );
    }

    let unknown_model = models
        .iter()
        .find(|m| m.id == "brand-new-model-released-yesterday")
        .expect("present");
    assert!(
        unknown_model.available,
        "existence is what the listing establishes"
    );
}

#[tokio::test]
async fn capabilities_from_the_versioned_catalog_are_authoritative() {
    // The other end of §40: a maintained catalog *is* a known descriptor, so its
    // values are usable for a routing decision.
    let catalog = vincents_llm::pricing::catalog();
    let gpt4o = catalog
        .get("gpt-4o")
        .expect("gpt-4o is in the catalog");
    assert_eq!(
        gpt4o.capabilities.source,
        vincents_llm::types::CapabilitySource::Descriptor
    );
    assert!(
        gpt4o.capabilities.is_authoritative(),
        "a catalog entry with no unknown capabilities is authoritative"
    );
    assert_eq!(
        gpt4o.capabilities.get(vincents_llm::types::Capability::Vision),
        Some(true)
    );
    assert_eq!(
        gpt4o.capabilities.get(vincents_llm::types::Capability::Caching),
        Some(true)
    );
}

#[tokio::test]
async fn an_unknown_capability_does_not_make_the_set_authoritative() {
    let capabilities = vincents_llm::types::ModelCapabilities::all_unknown()
        .with(vincents_llm::types::Capability::Streaming, true);
    assert!(capabilities.streaming, "the flag is still set");
    assert!(capabilities.is_known(vincents_llm::types::Capability::Streaming));
    assert!(!capabilities.is_known(vincents_llm::types::Capability::Vision));
    assert!(
        !capabilities.is_authoritative(),
        "one undetermined capability makes the set unusable for routing"
    );
}
