/*!
 * Core LLM Provider Trait
 *
 * Defines the unified interface that all LLM providers must implement.
 */

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;
use futures::Stream;

use crate::types::*;
use crate::error::{LLMError, Result};
use crate::config::ProviderConfig;
use crate::RateLimitStatus;
use crate::CostEstimate;

#[derive(Debug, Clone)]
pub struct ModelListRequest {
    pub filter: Option<String>,
    pub sort_by: Option<String>,
    pub sort_order: Option<SortOrder>,
}

/// Core trait that all LLM providers must implement
///
/// This trait defines the standard interface for integrating different LLM providers
/// (OpenAI, Claude, Gemini, etc.) into any application. All providers must implement
/// these methods to ensure consistent behavior.
///
/// # Implementing a Provider
///
/// ```rust,no_run
/// use async_trait::async_trait;
/// use vincents_llm::{LLMProvider, ChatCompletionRequest, ChatCompletionResponse, LLMError};
///
/// struct MyProvider {
///     api_key: String,
/// }
///
/// #[async_trait]
/// impl LLMProvider for MyProvider {
///     fn name(&self) -> &str {
///         "my-provider"
///     }
///
///     fn supported_models(&self) -> Vec<String> {
///         vec!["my-model-v1".to_string()]
///     }
///
///     async fn chat_completion(
///         &self,
///         request: ChatCompletionRequest,
///     ) -> Result<ChatCompletionResponse, LLMError> {
///         // Implementation here
///         todo!()
///     }
/// }
/// ```
#[async_trait]
pub trait LLMProvider: Send + Sync {
    /// Get the provider name
    fn name(&self) -> &str;

    /// Get provider display name
    fn display_name(&self) -> &str {
        self.name()
    }

    /// Get the list of supported models
    fn supported_models(&self) -> Vec<String>;

    /// Check if a specific model is supported
    fn supports_model(&self, model: &str) -> bool {
        self.supported_models().contains(&model.to_string())
    }

    /// List all models from this provider with full information
    async fn list_models(&self) -> Result<Vec<FullModelInfo>> {
        // Default implementation - providers should override
        Ok(self.supported_models()
            .iter()
            .map(|model_id| FullModelInfo {
                id: model_id.clone(),
                name: model_id.clone(),
                provider: self.name().to_string(),
                description: None,
                context_window: 0,
                max_output_tokens: 0,
                capabilities: self.get_model_capabilities(model_id)
                    .unwrap_or_default(),
                pricing: self.get_model_pricing(model_id),
                created: 0,
                available: true,
            })
            .collect())
    }

    /// Get model information
    fn get_model_info(&self, model: &str) -> Option<ModelInfo> {
        if self.supports_model(model) {
            Some(ModelInfo {
                id: model.to_string(),
                object: "model".to_string(),
                created: 0,
                owned_by: self.name().to_string(),
                permission: None,
            })
        } else {
            None
        }
    }

    /// Get comprehensive model information
    async fn get_full_model_info(&self, model_id: &str) -> Result<Option<FullModelInfo>> {
        let models = self.list_models().await?;
        Ok(models.into_iter().find(|m| m.id == model_id))
    }

    /// Get pricing for a model
    fn get_model_pricing(&self, _model: &str) -> Option<ModelPricing> {
        None
    }

    /// Get capabilities for a model
    fn get_model_capabilities(&self, _model: &str) -> Option<ModelCapabilities> {
        Some(ModelCapabilities::default())
    }

    /// Generate chat completion
    async fn chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse>;

    /// Generate chat completion with streaming
    async fn chat_completion_stream(
        &self,
        _request: ChatCompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<Chunk>> + Send>>> {
        Err(LLMError::UnsupportedError {
            operation: "streaming".to_string(),
            provider: Some(self.name().to_string()),
            model: None,
        })
    }

    /// Generate chat completion with function calling
    async fn chat_completion_with_functions(
        &self,
        request: ChatCompletionRequest,
        functions: Vec<FunctionDefinition>,
    ) -> Result<ChatCompletionResponse>;

    /// Check if function calling is supported
    fn supports_function_calling(&self, _model: &str) -> bool {
        false
    }

    /// Get current rate limit status
    async fn rate_limit_status(&self) -> Result<RateLimitStatus>;

    /// Estimate cost for a request
    async fn estimate_cost(&self, request: &ChatCompletionRequest) -> Result<CostEstimate>;

    /// Validate the provider configuration
    async fn validate_config(&self) -> Result<()> {
        Ok(())
    }

    /// Get provider health status
    async fn health_check(&self) -> Result<ProviderHealth> {
        self.rate_limit_status()
            .await
            .map(|status| ProviderHealth {
                name: self.name().to_string(),
                healthy: true,
                latency_ms: None,
                error: None,
                rate_limit_remaining: status.remaining,
                rate_limit_total: status.limit,
            })
    }

    /// Get the provider configuration
    fn config(&self) -> &ProviderConfig;
}

/// Extension trait for additional provider functionality
#[async_trait]
pub trait LLMProviderExt: LLMProvider + Sized {
    /// Create a boxed version of this provider
    fn into_boxed(self) -> Box<dyn LLMProvider>
    where
        Self: 'static,
    {
        Box::new(self)
    }

    /// Create an Arc version of this provider
    fn into_arc(self) -> Arc<dyn LLMProvider>
    where
        Self: 'static,
    {
        Arc::new(self)
    }
}

#[async_trait]
impl<T: LLMProvider> LLMProviderExt for T {}

/// Provider health status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderHealth {
    pub name: String,
    pub healthy: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub latency_ms: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rate_limit_remaining: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rate_limit_total: Option<u64>,
}

/// Factory trait for creating providers from configuration
#[async_trait]
pub trait ProviderFactory: Send + Sync {
    /// Provider type identifier
    fn provider_type(&self) -> &str;

    /// Create a provider from configuration
    async fn create(&self, config: &ProviderConfig) -> Result<Box<dyn LLMProvider>>;

    /// Get default configuration for this provider type
    fn default_config(&self) -> ProviderConfig;
}

/// Dynamic provider registry
#[derive(Default)]
pub struct ProviderRegistry {
    factories: HashMap<String, Box<dyn ProviderFactory>>,
}

impl ProviderRegistry {
    /// Create a new registry
    pub fn new() -> Self {
        Self {
            factories: HashMap::new(),
        }
    }

    /// Register a provider factory
    pub fn register<F: ProviderFactory + 'static>(&mut self, factory: F) {
        self.factories
            .insert(factory.provider_type().to_string(), Box::new(factory));
    }

    /// Create a provider from configuration
    pub async fn create(
        &self,
        provider_type: &str,
        config: &ProviderConfig,
    ) -> Result<Box<dyn LLMProvider>> {
        match self.factories.get(provider_type) {
            Some(factory) => factory.create(config).await,
            None => Err(LLMError::ConfigurationError {
                message: format!("Unknown provider type: {}", provider_type),
                field: Some("provider_type".to_string()),
            }),
        }
    }

    /// Check if a provider type is registered
    pub fn is_registered(&self, provider_type: &str) -> bool {
        self.factories.contains_key(provider_type)
    }

    /// Get list of registered provider types
    pub fn registered_types(&self) -> Vec<&str> {
        self.factories.keys().map(|s| s.as_str()).collect()
    }
}
