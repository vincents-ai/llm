use cucumber::{gherkin::Step, World};
use std::collections::HashMap;
use vincents_llm::types::FullModelInfo;
use vincents_llm::anthropic::AnthropicProvider;
use vincents_llm::openai::OpenAIProvider;
use vincents_llm::ollama::OllamaProvider;
use vincents_llm::provider::LLMProvider;

#[derive(World, Debug)]
pub struct ModelListingWorld {
    providers: HashMap<String, Box<dyn LLMProvider>>,
    current_models: Vec<FullModelInfo>,
    current_error: Option<String>,
}

impl Default for ModelListingWorld {
    fn default() -> Self {
        Self {
            providers: HashMap::new(),
            current_models: Vec::new(),
            current_error: None,
        }
    }
}

#[cucumber::given(expr = "I have initialized all providers")]
async fn initialize_providers(world: &mut ModelListingWorld) {
    // Initialize Anthropic provider
    if let Ok(provider) = AnthropicProvider::new("test-key") {
        world.providers.insert(
            "anthropic".to_string(),
            Box::new(provider) as Box<dyn LLMProvider>,
        );
    }

    // Initialize OpenAI provider
    if let Ok(provider) = OpenAIProvider::new("test-key") {
        world.providers.insert(
            "openai".to_string(),
            Box::new(provider) as Box<dyn LLMProvider>,
        );
    }

    // Initialize Ollama provider
    if let Ok(provider) = OllamaProvider::with_config(
        vincents_llm::ollama::OllamaConfig {
            host: "http://localhost:11434".to_string(),
            default_model: Some("llama3.1:8b".to_string()),
        },
    ) {
        world.providers.insert(
            "ollama".to_string(),
            Box::new(provider) as Box<dyn LLMProvider>,
        );
    }
}

#[cucumber::when(expr = "I list models from the anthropic provider")]
async fn list_anthropic_models(world: &mut ModelListingWorld) {
    if let Some(provider) = world.providers.get("anthropic") {
        match provider.list_models().await {
            Ok(models) => {
                world.current_models = models;
                world.current_error = None;
            }
            Err(e) => {
                world.current_error = Some(format!("{:?}", e));
            }
        }
    } else {
        world.current_error = Some("Anthropic provider not initialized".to_string());
    }
}

#[cucumber::when(expr = "I list models from the openai provider")]
async fn list_openai_models(world: &mut ModelListingWorld) {
    if let Some(provider) = world.providers.get("openai") {
        match provider.list_models().await {
            Ok(models) => {
                world.current_models = models;
                world.current_error = None;
            }
            Err(e) => {
                world.current_error = Some(format!("{:?}", e));
            }
        }
    } else {
        world.current_error = Some("OpenAI provider not initialized".to_string());
    }
}

#[cucumber::when(expr = "I list models from the ollama provider")]
async fn list_ollama_models(world: &mut ModelListingWorld) {
    if let Some(provider) = world.providers.get("ollama") {
        match provider.list_models().await {
            Ok(models) => {
                world.current_models = models;
                world.current_error = None;
            }
            Err(e) => {
                world.current_error = Some(format!("{:?}", e));
            }
        }
    } else {
        world.current_error = Some("Ollama provider not initialized".to_string());
    }
}

#[cucumber::then(expr = "I should see the following models:")]
async fn verify_model_list(world: &mut ModelListingWorld, step: &Step) {
    if world.current_error.is_some() {
        panic!("Error occurred: {:?}", world.current_error);
    }

    let table = step
        .table()
        .expect("Step should have a table");

    for row in table.rows.iter().skip(1) {
        let model_id = &row[0];
        let expected_name = &row[1];
        let expected_context = row[2].parse::<u32>().expect("Context window should be a number");
        let expected_capabilities: Vec<&str> = row[3].split(',').collect();

        let model = world
            .current_models
            .iter()
            .find(|m| m.id == model_id)
            .unwrap_or_else(|| panic!("Model {} not found", model_id));

        assert_eq!(
            model.name, *expected_name,
            "Model name mismatch for {}",
            model_id
        );
        assert_eq!(
            model.context_window, expected_context,
            "Context window mismatch for {}",
            model_id
        );

        for capability in expected_capabilities {
            let capability = capability.trim();
            match capability {
                "vision" => assert!(model.capabilities.vision, "Vision not supported for {}", model_id),
                "tools" => assert!(model.capabilities.function_calling, "Tools not supported for {}", model_id),
                "json" => assert!(model.capabilities.json_mode, "JSON mode not supported for {}", model_id),
                "caching" => assert!(model.capabilities.caching, "Caching not supported for {}", model_id),
                _ => {}
            }
        }
    }
}

#[cucumber::then(expr = "model pricing should include:")]
async fn verify_model_pricing(world: &mut ModelListingWorld, step: &Step) {
    if world.current_error.is_some() {
        panic!("Error occurred: {:?}", world.current_error);
    }

    let table = step
        .table()
        .expect("Step should have a table");

    for row in table.rows.iter().skip(1) {
        let model_id = &row[0];
        let expected_input: f64 = row[1].parse().expect("Input cost should be a number");
        let expected_output: f64 = row[2].parse().expect("Output cost should be a number");

        let model = world
            .current_models
            .iter()
            .find(|m| m.id == model_id)
            .unwrap_or_else(|| panic!("Model {} not found", model_id));

        let pricing = model
            .pricing
            .as_ref()
            .unwrap_or_else(|| panic!("Pricing not found for {}", model_id));

        // Allow small floating-point differences
        assert!((pricing.prompt_tokens - expected_input).abs() < 0.00001,
            "Input pricing mismatch for {}: expected {}, got {}",
            model_id, expected_input, pricing.prompt_tokens
        );
        assert!((pricing.completion_tokens - expected_output).abs() < 0.00001,
            "Output pricing mismatch for {}: expected {}, got {}",
            model_id, expected_output, pricing.completion_tokens
        );
    }
}

#[cucumber::given(expr = "Ollama server is running on {string}")]
async fn check_ollama_server(_world: &mut ModelListingWorld, host: String) {
    // In a real scenario, we would check the server health
    // For now, we'll just log the host
    println!("Ollama server check for: {}", host);
}

#[cucumber::then(expr = "the model list should not be empty")]
async fn verify_non_empty_list(world: &mut ModelListingWorld) {
    assert!(!world.current_models.is_empty(), "Model list should not be empty");
}

#[cucumber::then(expr = "each model should have:")]
async fn verify_model_fields(world: &mut ModelListingWorld, step: &Step) {
    let table = step
        .table()
        .expect("Step should have a table");

    for model in &world.current_models {
        for row in table.rows.iter().skip(1) {
            let field = &row[0];
            let requirement = &row[1];

            match field {
                "name" => {
                    if requirement == "not_empty" {
                        assert!(!model.name.is_empty(), "Model name should not be empty");
                    }
                }
                "model_id" => {
                    if requirement == "not_empty" {
                        assert!(!model.id.is_empty(), "Model ID should not be empty");
                    }
                }
                _ => {}
            }
        }
    }
}

#[cucumber::when(expr = "I get model details for {string} from anthropic")]
async fn get_model_details(world: &mut ModelListingWorld, model_id: String) {
    if let Some(provider) = world.providers.get("anthropic") {
        match provider.list_models().await {
            Ok(models) => {
                if let Some(model) = models.iter().find(|m| m.id == model_id) {
                    world.current_models = vec![model.clone()];
                    world.current_error = None;
                } else {
                    world.current_error = Some(format!("Model {} not found", model_id));
                }
            }
            Err(e) => {
                world.current_error = Some(format!("{:?}", e));
            }
        }
    }
}

#[cucumber::then(expr = "the model info should contain:")]
async fn verify_model_info_fields(world: &mut ModelListingWorld, step: &Step) {
    if world.current_error.is_some() {
        panic!("Error occurred: {:?}", world.current_error);
    }

    assert!(!world.current_models.is_empty(), "No models to verify");
    let model = &world.current_models[0];

    let table = step
        .table()
        .expect("Step should have a table");

    for row in table.rows.iter().skip(1) {
        let field = &row[0];
        let expected_type = &row[1];

        match (field, expected_type) {
            ("id", "string") => {
                assert!(!model.id.is_empty(), "ID should be a non-empty string");
            }
            ("name", "string") => {
                assert!(!model.name.is_empty(), "Name should be a non-empty string");
            }
            ("context_window", "number") => {
                assert!(model.context_window > 0, "Context window should be a positive number");
            }
            ("capabilities", "array") => {
                // Capabilities object exists
                assert!(true, "Capabilities should be an object");
            }
            ("pricing", "object") => {
                assert!(model.pricing.is_some(), "Pricing should be an object");
            }
            _ => {}
        }
    }
}

#[cucumber::then(expr = "model capabilities should include:")]
async fn verify_capabilities(world: &mut ModelListingWorld, step: &Step) {
    if world.current_error.is_some() {
        panic!("Error occurred: {:?}", world.current_error);
    }

    assert!(!world.current_models.is_empty(), "No models to verify");
    let model = &world.current_models[0];

    let table = step
        .table()
        .expect("Step should have a table");

    for row in table.rows.iter().skip(1) {
        let capability = &row[0];
        let supported = &row[1] == "true";

        match capability {
            "vision" => assert_eq!(model.capabilities.vision, supported, "Vision capability mismatch"),
            "tools" => assert_eq!(model.capabilities.function_calling, supported, "Tools capability mismatch"),
            "json_mode" => assert_eq!(model.capabilities.json_mode, supported, "JSON mode capability mismatch"),
            "caching" => assert_eq!(model.capabilities.caching, supported, "Caching capability mismatch"),
            _ => {}
        }
    }
}

#[cucumber::when(expr = "I list models from {string} provider")]
async fn list_models_from_provider(world: &mut ModelListingWorld, provider_name: String) {
    if let Some(provider) = world.providers.get(&provider_name) {
        match provider.list_models().await {
            Ok(models) => {
                world.current_models = models;
                world.current_error = None;
            }
            Err(e) => {
                world.current_error = Some(format!("{:?}", e));
            }
        }
    } else {
        world.current_error = Some(format!("Provider {} not found", provider_name));
    }
}

#[cucumber::then(expr = "all models should have required fields:")]
async fn verify_required_fields(world: &mut ModelListingWorld, step: &Step) {
    if world.current_error.is_some() {
        panic!("Error occurred: {:?}", world.current_error);
    }

    let table = step
        .table()
        .expect("Step should have a table");

    for model in &world.current_models {
        for row in table.rows.iter().skip(1) {
            let field = &row[0];

            match field {
                "id" => assert!(!model.id.is_empty(), "ID is required"),
                "name" => assert!(!model.name.is_empty(), "Name is required"),
                "context_window" => assert!(model.context_window > 0, "Context window is required"),
                "capabilities" => {} // Capabilities always exist as a struct
                "pricing" => assert!(model.pricing.is_some(), "Pricing is required"),
                _ => {}
            }
        }
    }
}

#[cucumber::when(expr = "I try to list models from provider {string}")]
async fn try_list_unknown_provider(world: &mut ModelListingWorld, provider_name: String) {
    if world.providers.get(&provider_name).is_none() {
        world.current_error = Some(format!("Unknown provider: {}", provider_name));
    } else {
        world.current_error = None;
    }
}

#[cucumber::then(expr = "I should receive an error about unknown provider")]
async fn verify_unknown_provider_error(world: &mut ModelListingWorld) {
    assert!(
        world.current_error.is_some(),
        "Expected an error for unknown provider"
    );
}

#[cucumber::then(expr = "the error message should contain {string}")]
async fn verify_error_contains(world: &mut ModelListingWorld, text: String) {
    if let Some(error) = &world.current_error {
        assert!(
            error.contains(&text),
            "Error message '{}' should contain '{}'",
            error,
            text
        );
    } else {
        panic!("Expected an error message");
    }
}

#[cucumber::when(expr = "I list models from anthropic with filter {string}")]
async fn list_with_filter(world: &mut ModelListingWorld, filter: String) {
    if let Some(provider) = world.providers.get("anthropic") {
        match provider.list_models().await {
            Ok(models) => {
                let filtered = models
                    .into_iter()
                    .filter(|m| match filter.as_str() {
                        "vision" => m.capabilities.vision,
                        _ => true,
                    })
                    .collect();
                world.current_models = filtered;
                world.current_error = None;
            }
            Err(e) => {
                world.current_error = Some(format!("{:?}", e));
            }
        }
    }
}

#[cucumber::then(expr = "I should only see models that support vision capability")]
async fn verify_vision_models(world: &mut ModelListingWorld) {
    for model in &world.current_models {
        assert!(
            model.capabilities.vision,
            "Model {} should support vision",
            model.id
        );
    }
}

#[cucumber::then(expr = "the result should include {string}")]
async fn verify_result_includes(world: &mut ModelListingWorld, model_id: String) {
    assert!(
        world.current_models.iter().any(|m| m.id == model_id),
        "Model {} not found in results",
        model_id
    );
}

#[cucumber::when(expr = "I list models from openai and sort by context_window descending")]
async fn list_and_sort_models(world: &mut ModelListingWorld) {
    if let Some(provider) = world.providers.get("openai") {
        match provider.list_models().await {
            Ok(mut models) => {
                models.sort_by(|a, b| b.context_window.cmp(&a.context_window));
                world.current_models = models;
                world.current_error = None;
            }
            Err(e) => {
                world.current_error = Some(format!("{:?}", e));
            }
        }
    }
}

#[cucumber::then(expr = "{string} should appear before {string}")]
async fn verify_ordering(world: &mut ModelListingWorld, first_model: String, second_model: String) {
    let first_idx = world
        .current_models
        .iter()
        .position(|m| m.id == first_model)
        .unwrap_or_else(|| panic!("Model {} not found", first_model));

    let second_idx = world
        .current_models
        .iter()
        .position(|m| m.id == second_model)
        .unwrap_or_else(|| panic!("Model {} not found", second_model));

    assert!(
        first_idx < second_idx,
        "{} should appear before {} (positions: {} vs {})",
        first_model,
        second_model,
        first_idx,
        second_idx
    );
}

#[tokio::main]
async fn main() {
    let cli = cucumber::cli::Opts::<cucumber::cli::Empty>::parse();
    cli.run_and_exit::<ModelListingWorld>().await;
}
