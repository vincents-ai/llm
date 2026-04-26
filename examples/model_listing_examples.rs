//! Model Listing Usage Examples
//!
//! This module demonstrates how to use the model listing functionality
//! across different LLM providers.

#[cfg(test)]
mod examples {
    use vincents_llm::provider::LLMProvider;
    use vincents_llm::anthropic::AnthropicProvider;
    use vincents_llm::openai::OpenAIProvider;
    use vincents_llm::ollama::OllamaProvider;

    /// Example 1: List all models from Anthropic
    #[tokio::test]
    async fn example_list_anthropic_models() {
        let provider = AnthropicProvider::new("your-api-key")
            .await
            .expect("Failed to initialize Anthropic provider");

        // List all available models
        let models = provider
            .list_models()
            .await
            .expect("Failed to list models");

        println!("Available Anthropic Models:");
        println!("==========================");
        for model in models {
            println!("\nModel: {}", model.name);
            println!("  ID: {}", model.id);
            println!("  Context Window: {} tokens", model.context_window);
            println!("  Provider: {}", model.provider);

            if let Some(pricing) = &model.pricing {
                println!("  Pricing:");
                println!("    Input: ${}/1K tokens", pricing.prompt_tokens);
                println!("    Output: ${}/1K tokens", pricing.completion_tokens);
            }

            println!("  Capabilities:");
            println!("    Vision: {}", model.capabilities.vision);
            println!("    Function Calling: {}", model.capabilities.function_calling);
            println!("    JSON Mode: {}", model.capabilities.json_mode);
            println!("    Caching: {}", model.capabilities.caching);
            println!("    Streaming: {}", model.capabilities.streaming);
        }
    }

    /// Example 2: Get specific model details
    #[tokio::test]
    async fn example_get_model_details() {
        let provider = AnthropicProvider::new("your-api-key")
            .await
            .expect("Failed to initialize provider");

        // Get detailed information about a specific model
        let model_info = provider
            .get_full_model_info("claude-3-5-sonnet-20241022")
            .await
            .expect("Failed to get model info");

        if let Some(model) = model_info {
            println!("Model Details");
            println!("=============");
            println!("Name: {}", model.name);
            println!("ID: {}", model.id);
            println!("Context Window: {} tokens", model.context_window);
            println!("Max Output Tokens: {}", model.max_output_tokens);

            if let Some(desc) = &model.description {
                println!("Description: {}", desc);
            }
        } else {
            println!("Model not found");
        }
    }

    /// Example 3: Compare models across providers
    #[tokio::test]
    async fn example_compare_providers() {
        let anthropic = AnthropicProvider::new("anthropic-key")
            .await
            .expect("Failed to init Anthropic");

        let openai = OpenAIProvider::new("openai-key")
            .await
            .expect("Failed to init OpenAI");

        let anthropic_models = anthropic
            .list_models()
            .await
            .expect("Failed to list Anthropic models");

        let openai_models = openai
            .list_models()
            .await
            .expect("Failed to list OpenAI models");

        println!("Model Comparison");
        println!("================");
        println!("\nAnthropic Models: {}", anthropic_models.len());
        println!("OpenAI Models: {}", openai_models.len());

        // Find most expensive input pricing
        if let Some(expensive) = anthropic_models
            .iter()
            .max_by(|a, b| {
                let a_cost = a.pricing.as_ref().map(|p| p.prompt_tokens).unwrap_or(0.0);
                let b_cost = b.pricing.as_ref().map(|p| p.prompt_tokens).unwrap_or(0.0);
                a_cost.partial_cmp(&b_cost).unwrap_or(std::cmp::Ordering::Equal)
            })
        {
            println!("\nMost Expensive Anthropic Model:");
            println!("  Name: {}", expensive.name);
            if let Some(pricing) = &expensive.pricing {
                println!("  Input Cost: ${}/1K", pricing.prompt_tokens);
            }
        }

        // Find largest context window
        if let Some(largest) = openai_models
            .iter()
            .max_by(|a, b| a.context_window.cmp(&b.context_window))
        {
            println!("\nLargest OpenAI Context Window:");
            println!("  Name: {}", largest.name);
            println!("  Context: {} tokens", largest.context_window);
        }
    }

    /// Example 4: Filter models by capability
    #[tokio::test]
    async fn example_filter_by_capability() {
        let provider = OpenAIProvider::new("your-api-key")
            .await
            .expect("Failed to initialize provider");

        let all_models = provider
            .list_models()
            .await
            .expect("Failed to list models");

        // Filter for vision-capable models
        let vision_models: Vec<_> = all_models
            .iter()
            .filter(|m| m.capabilities.vision)
            .collect();

        println!("Vision-Capable OpenAI Models:");
        println!("============================");
        for model in vision_models {
            println!("  - {}", model.name);
        }

        // Filter for models with function calling
        let function_models: Vec<_> = all_models
            .iter()
            .filter(|m| m.capabilities.function_calling)
            .collect();

        println!("\nModels with Function Calling:");
        println!("===========================");
        for model in function_models {
            println!("  - {}", model.name);
        }
    }

    /// Example 5: Sort models by context window
    #[tokio::test]
    async fn example_sort_by_context() {
        let provider = OpenAIProvider::new("your-api-key")
            .await
            .expect("Failed to initialize provider");

        let mut models = provider
            .list_models()
            .await
            .expect("Failed to list models");

        // Sort by context window (descending)
        models.sort_by(|a, b| b.context_window.cmp(&a.context_window));

        println!("OpenAI Models Sorted by Context Window (Largest First):");
        println!("====================================================");
        for model in models {
            println!("  {} - {} tokens", model.name, model.context_window);
        }
    }

    /// Example 6: Find most cost-effective model
    #[tokio::test]
    async fn example_find_cheapest_model() {
        let provider = AnthropicProvider::new("your-api-key")
            .await
            .expect("Failed to initialize provider");

        let models = provider
            .list_models()
            .await
            .expect("Failed to list models");

        // Find cheapest input pricing
        if let Some(cheapest) = models
            .iter()
            .filter(|m| m.pricing.is_some())
            .min_by(|a, b| {
                let a_cost = a.pricing.as_ref().unwrap().prompt_tokens;
                let b_cost = b.pricing.as_ref().unwrap().prompt_tokens;
                a_cost.partial_cmp(&b_cost).unwrap_or(std::cmp::Ordering::Equal)
            })
        {
            println!("Cheapest Anthropic Model");
            println!("========================");
            println!("  Name: {}", cheapest.name);
            println!("  ID: {}", cheapest.id);
            if let Some(pricing) = &cheapest.pricing {
                println!("  Input Cost: ${}/1K tokens", pricing.prompt_tokens);
                println!("  Output Cost: ${}/1K tokens", pricing.completion_tokens);
            }
        }
    }

    /// Example 7: Analyze model capabilities
    #[tokio::test]
    async fn example_analyze_capabilities() {
        let provider = AnthropicProvider::new("your-api-key")
            .await
            .expect("Failed to initialize provider");

        let models = provider
            .list_models()
            .await
            .expect("Failed to list models");

        println!("Anthropic Models Capability Matrix:");
        println!("===================================");
        println!("Model                      | Vision | Functions | Caching");
        println!("---------------------------|--------|-----------|--------");

        for model in models {
            let vision = if model.capabilities.vision { "Yes" } else { "No  " };
            let functions = if model.capabilities.function_calling {
                "Yes"
            } else {
                "No "
            };
            let caching = if model.capabilities.caching { "Yes" } else { "No " };

            println!(
                "{:<26} | {:<6} | {:<9} | {:<7}",
                model.name, vision, functions, caching
            );
        }
    }

    /// Example 8: Work with Ollama local models
    #[tokio::test]
    async fn example_ollama_local_models() {
        let config = vincents_llm::ollama::OllamaConfig {
            host: "http://localhost:11434".to_string(),
            default_model: Some("llama3.1:8b".to_string()),
        };

        let provider = OllamaProvider::with_config(config)
            .expect("Failed to create Ollama provider");

        // List available local models
        match provider.list_models().await {
            Ok(models) => {
                println!("Available Ollama Local Models:");
                println!("=============================");
                for model in models {
                    println!("\nModel: {}", model.name);
                    println!("  ID: {}", model.id);
                    println!("  Context: {} tokens", model.context_window);
                    println!("  Local: Yes (Free)");
                }
            }
            Err(e) => {
                println!("Error fetching Ollama models (is Ollama running?): {}", e);
            }
        }
    }

    /// Example 9: Calculate costs for different models
    #[tokio::test]
    async fn example_calculate_usage_costs() {
        let provider = OpenAIProvider::new("your-api-key")
            .await
            .expect("Failed to initialize provider");

        let models = provider
            .list_models()
            .await
            .expect("Failed to list models");

        // Estimate cost for 1000 input tokens + 500 output tokens
        let input_tokens = 1000.0;
        let output_tokens = 500.0;

        println!("Cost Estimate for 1000 input + 500 output tokens:");
        println!("==================================================");

        let mut cost_estimates: Vec<_> = models
            .iter()
            .filter_map(|m| {
                m.pricing.as_ref().map(|p| {
                    let input_cost = (input_tokens / 1000.0) * p.prompt_tokens;
                    let output_cost = (output_tokens / 1000.0) * p.completion_tokens;
                    let total_cost = input_cost + output_cost;
                    (m.name.clone(), total_cost)
                })
            })
            .collect();

        // Sort by cost
        cost_estimates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        for (name, cost) in cost_estimates {
            println!("  {} - ${:.6}", name, cost);
        }
    }

    /// Example 10: Build provider selector
    #[tokio::test]
    async fn example_provider_selector() {
        // Helper function to select best provider for use case
        fn select_provider(use_case: &str) -> &'static str {
            match use_case {
                "vision" => "anthropic", // All Anthropic models support vision
                "fastest" => "openai",   // GPT-3.5-turbo is fastest
                "cheapest" => "anthropic", // Haiku is cheapest
                "most_capable" => "openai", // GPT-4 is most capable
                "local" => "ollama",    // Local models
                _ => "openai",
            }
        }

        let selected = select_provider("vision");
        println!("For vision tasks, use: {}", selected);

        let selected = select_provider("cheapest");
        println!("For cost efficiency, use: {}", selected);

        let selected = select_provider("local");
        println!("For local/offline work, use: {}", selected);
    }
}
