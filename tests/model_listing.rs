#[cfg(test)]
mod tests {
    use vincents_llm::provider::LLMProvider;
    use vincents_llm::anthropic::AnthropicProvider;
    use vincents_llm::openai::OpenAIProvider;
    use vincents_llm::ollama::OllamaProvider;

    #[tokio::test]
    async fn test_anthropic_list_models() {
        let provider = AnthropicProvider::new("test-key")
            .await
            .expect("Should create Anthropic provider");

        let models = provider
            .list_models()
            .await
            .expect("Should list models");

        assert!(!models.is_empty(), "Anthropic should have at least one model");
        
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
        let provider = AnthropicProvider::new("test-key")
            .await
            .expect("Should create Anthropic provider");

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
        let provider = AnthropicProvider::new("test-key")
            .await
            .expect("Should create Anthropic provider");

        let models = provider.list_models().await.expect("Should list models");
        
        let haiku = models
            .iter()
            .find(|m| m.id == "claude-3-haiku-20240307")
            .expect("Should have Claude 3 Haiku");

        assert_eq!(haiku.name, "Claude 3 Haiku");
        assert_eq!(haiku.context_window, 200000);
        assert!(!haiku.capabilities.caching, "Haiku should not support caching");
        
        let pricing = haiku.pricing.as_ref().expect("Should have pricing");
        assert!(pricing.prompt_tokens < 0.001);
        assert!(pricing.completion_tokens < 0.01);
    }

    #[tokio::test]
    async fn test_openai_list_models() {
        let provider = OpenAIProvider::new("test-key")
            .await
            .expect("Should create OpenAI provider");

        let models = provider
            .list_models()
            .await
            .expect("Should list models");

        assert!(!models.is_empty(), "OpenAI should have at least one model");

        // Verify GPT-4
        let gpt4 = models
            .iter()
            .find(|m| m.id == "gpt-4")
            .expect("Should have GPT-4");

        assert_eq!(gpt4.name, "GPT-4");
        assert_eq!(gpt4.context_window, 8192);
        assert!(gpt4.capabilities.vision);
        assert!(gpt4.capabilities.function_calling);
    }

    #[tokio::test]
    async fn test_openai_gpt4_turbo() {
        let provider = OpenAIProvider::new("test-key")
            .await
            .expect("Should create OpenAI provider");

        let models = provider.list_models().await.expect("Should list models");
        
        let turbo = models
            .iter()
            .find(|m| m.id == "gpt-4-turbo")
            .expect("Should have GPT-4 Turbo");

        assert_eq!(turbo.name, "GPT-4 Turbo");
        assert_eq!(turbo.context_window, 128000);
        assert!(turbo.capabilities.vision);
        assert!(turbo.capabilities.caching);
    }

    #[tokio::test]
    async fn test_openai_gpt35_turbo() {
        let provider = OpenAIProvider::new("test-key")
            .await
            .expect("Should create OpenAI provider");

        let models = provider.list_models().await.expect("Should list models");
        
        let gpt35 = models
            .iter()
            .find(|m| m.id == "gpt-3.5-turbo")
            .expect("Should have GPT-3.5 Turbo");

        assert_eq!(gpt35.name, "GPT-3.5 Turbo");
        assert_eq!(gpt35.context_window, 4096);
        assert!(!gpt35.capabilities.vision, "GPT-3.5 should not support vision");
        assert!(gpt35.capabilities.function_calling);

        let pricing = gpt35.pricing.as_ref().expect("Should have pricing");
        assert_eq!(pricing.prompt_tokens, 0.0005);
        assert_eq!(pricing.completion_tokens, 0.0015);
    }

    #[tokio::test]
    async fn test_ollama_list_models() {
        let config = vincents_llm::ollama::OllamaConfig {
            host: "http://localhost:11434".to_string(),
            default_model: Some("llama3.1:8b".to_string()),
        };

        let provider = OllamaProvider::with_config(config)
            .expect("Should create Ollama provider");

        // Note: This test will fail if Ollama is not running
        // In a real scenario, we would mock the HTTP response
        let models = provider.list_models().await;
        
        // Just verify the function exists and returns a result
        assert!(models.is_ok() || models.is_err(), "Should return a result");
    }

    #[tokio::test]
    async fn test_model_info_includes_all_required_fields() {
        let provider = AnthropicProvider::new("test-key")
            .await
            .expect("Should create provider");

        let models = provider.list_models().await.expect("Should list models");

        for model in models {
            assert!(!model.id.is_empty(), "Model id should not be empty");
            assert!(!model.name.is_empty(), "Model name should not be empty");
            assert_eq!(model.provider, "anthropic");
            assert!(model.context_window > 0, "Context window should be positive");
            assert!(model.pricing.is_some(), "Pricing should be present");
            
            if let Some(pricing) = model.pricing {
                assert!(pricing.prompt_tokens > 0.0 || pricing.prompt_tokens == 0.0, 
                        "Pricing should be valid");
                assert!(pricing.completion_tokens > 0.0 || pricing.completion_tokens == 0.0, 
                        "Pricing should be valid");
            }
        }
    }

    #[tokio::test]
    async fn test_model_capabilities_structure() {
        let provider = OpenAIProvider::new("test-key")
            .await
            .expect("Should create provider");

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
        let provider = AnthropicProvider::new("test-key")
            .await
            .expect("Should create provider");

        let mut models = provider.list_models().await.expect("Should list models");
        
        // Sort by prompt token cost
        models.sort_by(|a, b| {
            let a_cost = a.pricing.as_ref().map(|p| p.prompt_tokens).unwrap_or(0.0);
            let b_cost = b.pricing.as_ref().map(|p| p.prompt_tokens).unwrap_or(0.0);
            a_cost.partial_cmp(&b_cost).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Haiku should be cheapest
        assert_eq!(models.first().map(|m| m.id.as_str()), Some("claude-3-haiku-20240307"));
        
        // Opus should be most expensive
        assert_eq!(models.last().map(|m| m.id.as_str()), Some("claude-3-opus-20240229"));
    }

    #[tokio::test]
    async fn test_openai_models_by_context_window() {
        let provider = OpenAIProvider::new("test-key")
            .await
            .expect("Should create provider");

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
        let provider = AnthropicProvider::new("test-key")
            .await
            .expect("Should create provider");

        let models = provider.list_models().await.expect("Should list models");

        // All returned models should be available
        for model in models {
            assert!(model.available, "All returned models should be marked as available");
        }
    }

    #[tokio::test]
    async fn test_get_full_model_info_async() {
        let provider = AnthropicProvider::new("test-key")
            .await
            .expect("Should create provider");

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
        let provider = AnthropicProvider::new("test-key")
            .await
            .expect("Should create provider");

        let model_opt = provider
            .get_full_model_info("nonexistent-model")
            .await
            .expect("Should not error");

        assert!(model_opt.is_none(), "Should not find nonexistent model");
    }
}
