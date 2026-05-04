use crate::circuit_breaker::{CircuitBreaker, CircuitBreakerConfig, CircuitBreakerMetrics, CircuitOpenError, CircuitState};
use crate::provider::{LLMProvider, ProviderHealth, ProviderRegistry};
use crate::error::{LLMError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn, debug};

/// Configuration for provider manager
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderManagerConfig {
    /// Default provider to use
    pub default_provider: String,
    /// Enable automatic failover
    pub failover_enabled: bool,
    /// Health check interval in seconds
    pub health_check_interval_sec: u64,
    /// Default circuit breaker config (used when no per-provider override exists)
    pub circuit_breaker_default: CircuitBreakerConfig,
}

impl Default for ProviderManagerConfig {
    fn default() -> Self {
        Self {
            default_provider: "openai".to_string(),
            failover_enabled: true,
            health_check_interval_sec: 60,
            circuit_breaker_default: CircuitBreakerConfig::default(),
        }
    }
}

/// Per-provider state held in the manager.
#[derive(Clone)]
struct ProviderEntry {
    provider: Arc<dyn LLMProvider>,
    circuit_breaker: Arc<CircuitBreaker>,
}

/// Provider manager for multi-provider coordination with circuit breakers.
pub struct ProviderManager {
    config: ProviderManagerConfig,
    providers: RwLock<HashMap<String, ProviderEntry>>,
    registry: ProviderRegistry,
    _health_check_task: RwLock<Option<tokio::task::JoinHandle<()>>>,
}

impl ProviderManager {
    /// Create a new provider manager
    pub fn new(config: ProviderManagerConfig) -> Self {
        Self {
            registry: ProviderRegistry::new(),
            providers: RwLock::new(HashMap::new()),
            config,
            _health_check_task: RwLock::new(None),
        }
    }

    /// Register a provider with its own circuit breaker.
    ///
    /// If the provider's `ProviderConfig` includes `circuit_breaker` overrides,
    /// those are merged with the default config. Otherwise, the default config
    /// is used.
    pub async fn register_provider(&self, provider: Arc<dyn LLMProvider>) {
        let name = provider.name().to_string();
        let provider_config = provider.config();

        // Resolve per-provider circuit breaker config
        let breaker_config = match &provider_config.circuit_breaker {
            Some(overrides) => overrides.merge_into(&self.config.circuit_breaker_default),
            None => self.config.circuit_breaker_default.clone(),
        };

        let breaker = Arc::new(CircuitBreaker::new(name.clone(), breaker_config));

        let entry = ProviderEntry {
            provider,
            circuit_breaker: breaker,
        };

        {
            let mut providers = self.providers.write().await;
            providers.insert(name.clone(), entry);
        }

        info!(
            provider = %name,
            "Registered provider with circuit breaker"
        );
    }

    /// Get a provider by name.
    pub async fn get_provider(&self, name: &str) -> Option<Arc<dyn LLMProvider>> {
        let providers = self.providers.read().await;
        providers.get(name).map(|e| e.provider.clone())
    }

    /// Get the default provider.
    pub async fn get_default_provider(&self) -> Option<Arc<dyn LLMProvider>> {
        self.get_provider(&self.config.default_provider).await
    }

    /// Select the best available provider for a given model.
    ///
    /// Respects circuit breaker state — providers with open breakers are
    /// skipped (unless all are open, in which case half-open providers are
    /// tried first).
    pub async fn select_provider(
        &self,
        model: &str,
    ) -> Result<Arc<dyn LLMProvider>> {
        let providers = self.providers.read().await;

        // Collect provider states upfront (need async for each)
        let mut closed: Vec<&ProviderEntry> = Vec::new();
        let mut half_open: Vec<&ProviderEntry> = Vec::new();
        let mut open: Vec<&ProviderEntry> = Vec::new();

        for entry in providers.values() {
            if !entry.provider.supports_model(model) {
                continue;
            }

            let state = entry.circuit_breaker.state().await;
            match state {
                CircuitState::Closed => closed.push(entry),
                CircuitState::HalfOpen => half_open.push(entry),
                CircuitState::Open => open.push(entry),
            }
        }

        // Prefer default provider if available and Closed
        if let Some(default_entry) = providers.get(&self.config.default_provider) {
            if default_entry.provider.supports_model(model) {
                let state = default_entry.circuit_breaker.state().await;
                if state == CircuitState::Closed {
                    return Ok(default_entry.provider.clone());
                }
            }
        }

        // Then any Closed provider
        if let Some(entry) = closed.first() {
            return Ok(entry.provider.clone());
        }

        // Then HalfOpen (probe)
        if let Some(entry) = half_open.first() {
            return Ok(entry.provider.clone());
        }

        // Finally, any provider that supports the model (even if Open)
        if let Some(entry) = open.first() {
            warn!(
                provider = %entry.provider.name(),
                model = %model,
                "All providers are Open — attempting request anyway on least-recently-tripped provider"
            );
            return Ok(entry.provider.clone());
        }

        // Nothing supports this model
        let all_models: Vec<String> = providers
            .values()
            .flat_map(|e| e.provider.supported_models())
            .collect();

        Err(LLMError::ModelNotAvailable {
            model: model.to_string(),
            available_models: all_models,
        })
    }

    /// Execute a request with automatic failover across providers.
    ///
    /// For each provider that supports the requested model:
    /// 1. Check the circuit breaker — skip if Open.
    /// 2. Execute the request.
    /// 3. On success, record success with the breaker.
    /// 4. On retryable error, record failure with the breaker and try the next provider.
    /// 5. On non-retryable error, propagate immediately.
    pub async fn execute_with_failover<F, T>(
        &self,
        request: &crate::types::ChatCompletionRequest,
        mut execute: impl FnMut(Arc<dyn LLMProvider>) -> F,
    ) -> Result<T>
    where
        F: std::future::Future<Output = Result<T>>,
    {
        let model = request.model.clone();
        let providers = self.providers.read().await;

        // Collect candidates sorted by breaker state priority
        let mut candidates: Vec<(&ProviderEntry, CircuitState)> = Vec::new();
        for entry in providers.values() {
            if entry.provider.supports_model(&model) {
                let state = entry.circuit_breaker.state().await;
                candidates.push((entry, state));
            }
        }

        // Sort: Closed first, then HalfOpen, then Open
        candidates.sort_by_key(|(_, state)| match state {
            CircuitState::Closed => 0,
            CircuitState::HalfOpen => 1,
            CircuitState::Open => 2,
        });

        let mut last_error: Option<LLMError> = None;

        for (entry, _) in &candidates {
            let name = entry.provider.name();

            // Check circuit breaker
            match entry.circuit_breaker.allow_request().await {
                Ok(()) => {}
                Err(CircuitOpenError { remaining_timeout, .. }) => {
                    debug!(
                        provider = %name,
                        remaining = ?remaining_timeout,
                        "Circuit breaker open — skipping provider"
                    );
                    continue;
                }
            }

            // Execute
            match execute(entry.provider.clone()).await {
                Ok(result) => {
                    entry.circuit_breaker.record_success().await;
                    return Ok(result);
                }
                Err(e) if is_breaker_failure(&e) => {
                    entry.circuit_breaker.record_failure().await;
                    warn!(
                        provider = %name,
                        error = %e,
                        "Provider failed — recording circuit breaker failure"
                    );
                    last_error = Some(e);
                    continue;
                }
                Err(e) => {
                    // Non-retryable error — don't penalise the breaker
                    debug!(
                        provider = %name,
                        error = %e,
                        "Non-retryable error — not counting against circuit breaker"
                    );
                    return Err(e);
                }
            }
        }

        // All providers exhausted
        Err(match last_error {
            Some(e) => e,
            None => LLMError::ProviderError {
                provider: "all".to_string(),
                message: format!("No providers available for model '{}'", model),
                code: None,
            },
        })
    }

    /// Get health status for all providers, enriched with circuit breaker state.
    pub async fn health_status(&self) -> HashMap<String, ProviderHealth> {
        let providers = self.providers.read().await;
        let mut status = HashMap::new();

        for (name, entry) in providers.iter() {
            let mut health = entry.provider.health_check().await.unwrap_or_else(|e| ProviderHealth {
                name: name.clone(),
                healthy: false,
                latency_ms: None,
                error: Some(e.to_string()),
                rate_limit_remaining: None,
                rate_limit_total: None,
            });

            // Enrich with circuit breaker state
            let breaker_state = entry.circuit_breaker.state().await;
            if breaker_state == CircuitState::Open {
                health.healthy = false;
                health.error = Some(format!("Circuit breaker open ({})", health.error.as_deref().unwrap_or("")));
            }

            status.insert(name.clone(), health);
        }

        status
    }

    /// Get circuit breaker metrics for all providers.
    pub async fn circuit_breaker_metrics(&self) -> HashMap<String, CircuitBreakerMetrics> {
        let providers = self.providers.read().await;
        let mut metrics = HashMap::new();

        for (name, entry) in providers.iter() {
            metrics.insert(name.clone(), entry.circuit_breaker.metrics().await);
        }

        metrics
    }

    /// Get circuit breaker metrics for a specific provider.
    pub async fn provider_circuit_breaker_metrics(&self, provider_name: &str) -> Option<CircuitBreakerMetrics> {
        let providers = self.providers.read().await;
        match providers.get(provider_name) {
            Some(entry) => Some(entry.circuit_breaker.metrics().await),
            None => None,
        }
    }

    /// Force the circuit breaker state for a provider (admin/testing).
    pub async fn force_breaker_state(&self, provider_name: &str, state: CircuitState) -> Result<()> {
        let providers = self.providers.read().await;
        match providers.get(provider_name) {
            Some(entry) => {
                entry.circuit_breaker.force_state(state).await;
                Ok(())
            }
            None => Err(LLMError::ConfigurationError {
                message: format!("Provider '{}' not found", provider_name),
                field: Some("provider_name".to_string()),
            }),
        }
    }

    /// Reset the circuit breaker for a specific provider.
    pub async fn reset_breaker(&self, provider_name: &str) -> Result<()> {
        let providers = self.providers.read().await;
        match providers.get(provider_name) {
            Some(entry) => {
                entry.circuit_breaker.reset().await;
                Ok(())
            }
            None => Err(LLMError::ConfigurationError {
                message: format!("Provider '{}' not found", provider_name),
                field: Some("provider_name".to_string()),
            }),
        }
    }

    /// Get all provider names.
    pub async fn provider_names(&self) -> Vec<String> {
        let providers = self.providers.read().await;
        providers.keys().cloned().collect()
    }
}

/// Determine whether an error should count as a circuit breaker failure.
///
/// Only transient/retryable errors penalise the breaker. Auth errors,
/// invalid request errors, and content filter errors are not the provider's
/// fault and should not trip the breaker.
fn is_breaker_failure(error: &LLMError) -> bool {
    match error {
        LLMError::RateLimitError { .. } => true,
        LLMError::HttpError { status_code, .. } => {
            matches!(status_code, Some(429 | 500..=599))
        }
        LLMError::TimeoutError { .. } => true,
        LLMError::ProviderError { .. } => true,
        LLMError::StreamingError { .. } => true,
        // Non-retryable — don't penalise
        LLMError::AuthenticationError { .. } => false,
        LLMError::ModelNotAvailable { .. } => false,
        LLMError::InvalidRequestError { .. } => false,
        LLMError::ConfigurationError { .. } => false,
        LLMError::SerializationError { .. } => false,
        LLMError::ContentFilterError { .. } => false,
        LLMError::UnsupportedError { .. } => false,
        LLMError::ContextLengthExceeded { .. } => false,
        LLMError::ResponseFormatError { .. } => false,
        LLMError::QuotaExceeded { .. } => true, // quota issues may indicate provider degradation
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ProviderConfig;
    use crate::provider::LLMProvider;
    use crate::types::*;
    use crate::error::{LLMError, Result};
    use crate::RateLimitStatus;
    use crate::CostEstimate;
    use async_trait::async_trait;

    /// Minimal mock provider for testing provider manager behaviour.
    struct MockProvider {
        name: String,
        models: Vec<String>,
        config: ProviderConfig,
    }

    impl MockProvider {
        fn new(name: &str, models: Vec<&str>) -> Self {
            Self {
                name: name.to_string(),
                models: models.into_iter().map(String::from).collect(),
                config: ProviderConfig::default(),
            }
        }
    }

    #[async_trait]
    impl LLMProvider for MockProvider {
        fn name(&self) -> &str { &self.name }
        fn supported_models(&self) -> Vec<String> { self.models.clone() }
        async fn chat_completion(&self, _request: ChatCompletionRequest) -> Result<ChatCompletionResponse> {
            Ok(ChatCompletionResponse {
                id: "test".to_string(),
                object: "chat.completion".to_string(),
                created: 0,
                model: "test".to_string(),
                choices: vec![],
                usage: None,
                system_fingerprint: None,
            })
        }
        async fn chat_completion_with_functions(
            &self,
            _request: ChatCompletionRequest,
            _functions: Vec<FunctionDefinition>,
        ) -> Result<ChatCompletionResponse> {
            Ok(ChatCompletionResponse {
                id: "test".to_string(),
                object: "chat.completion".to_string(),
                created: 0,
                model: "test".to_string(),
                choices: vec![],
                usage: None,
                system_fingerprint: None,
            })
        }
        async fn rate_limit_status(&self) -> Result<RateLimitStatus> {
            Ok(RateLimitStatus {
                remaining: Some(100),
                limit: Some(100),
                tokens_remaining: None,
                tokens_limit: None,
                reset_at: None,
                retry_after: None,
            })
        }
        async fn estimate_cost(&self, _request: &ChatCompletionRequest) -> Result<CostEstimate> {
            Ok(CostEstimate {
                input_cost: 0.0,
                output_cost: 0.0,
                total_cost: 0.0,
                input_tokens: 0,
                output_tokens: 0,
                model: "test".to_string(),
                provider: self.name.clone(),
            })
        }
        fn config(&self) -> &ProviderConfig { &self.config }
    }

    #[tokio::test]
    async fn test_register_provider_creates_breaker() {
        let manager = ProviderManager::new(ProviderManagerConfig::default());
        let provider = Arc::new(MockProvider::new("test-prov", vec!["gpt-4"]));

        manager.register_provider(provider).await;

        let metrics = manager.circuit_breaker_metrics().await;
        assert!(metrics.contains_key("test-prov"));
        assert_eq!(metrics["test-prov"].state, CircuitState::Closed);
    }

    #[tokio::test]
    async fn test_provider_names() {
        let manager = ProviderManager::new(ProviderManagerConfig::default());
        assert!(manager.provider_names().await.is_empty());

        let provider = Arc::new(MockProvider::new("prov-a", vec!["gpt-4"]));
        manager.register_provider(provider).await;
        let names = manager.provider_names().await;
        assert_eq!(names, vec!["prov-a"]);
    }

    #[tokio::test]
    async fn test_force_breaker_state() {
        let manager = ProviderManager::new(ProviderManagerConfig::default());
        let provider = Arc::new(MockProvider::new("test-prov", vec!["gpt-4"]));

        manager.register_provider(provider).await;
        manager.force_breaker_state("test-prov", CircuitState::Open).await.unwrap();

        let metrics = manager.circuit_breaker_metrics().await;
        assert_eq!(metrics["test-prov"].state, CircuitState::Open);
    }

    #[tokio::test]
    async fn test_reset_breaker() {
        let manager = ProviderManager::new(ProviderManagerConfig::default());
        let provider = Arc::new(MockProvider::new("test-prov", vec!["gpt-4"]));

        manager.register_provider(provider).await;
        manager.force_breaker_state("test-prov", CircuitState::Open).await.unwrap();
        manager.reset_breaker("test-prov").await.unwrap();

        let metrics = manager.circuit_breaker_metrics().await;
        assert_eq!(metrics["test-prov"].state, CircuitState::Closed);
        assert_eq!(metrics["test-prov"].trip_count, 0);
    }

    #[tokio::test]
    async fn test_execute_with_failover_skips_open_breaker() {
        let config = ProviderManagerConfig {
            circuit_breaker_default: CircuitBreakerConfig {
                failure_threshold: 1,
                ..Default::default()
            },
            ..Default::default()
        };
        let manager = ProviderManager::new(config);

        let provider_a = Arc::new(MockProvider::new("prov-a", vec!["gpt-4"]));
        let provider_b = Arc::new(MockProvider::new("prov-b", vec!["gpt-4"]));

        manager.register_provider(provider_a).await;
        manager.register_provider(provider_b).await;

        // Trip breaker on prov-a
        let _ = manager.force_breaker_state("prov-a", CircuitState::Open).await;

        let request = ChatCompletionRequest {
            model: "gpt-4".to_string(),
            messages: vec![ChatMessage::user("hello")],
            ..Default::default()
        };

        let result = manager.execute_with_failover(&request, |prov| {
            // prov-b should be selected (prov-a is Open)
            let name = prov.name().to_string();
            async move {
                if name == "prov-b" {
                    Ok(ChatCompletionResponse {
                        id: "test".to_string(),
                        object: "chat.completion".to_string(),
                        created: 0,
                        model: "gpt-4".to_string(),
                        choices: vec![],
                        usage: None,
                        system_fingerprint: None,
                    })
                } else {
                    Err(LLMError::ProviderError {
                        provider: name.clone(),
                        message: "unexpected".to_string(),
                        code: None,
                    })
                }
            }
        }).await;

        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_execute_with_failover_records_failures() {
        let config = ProviderManagerConfig {
            circuit_breaker_default: CircuitBreakerConfig {
                failure_threshold: 2,
                ..Default::default()
            },
            ..Default::default()
        };
        let manager = ProviderManager::new(config);

        let provider = Arc::new(MockProvider::new("prov-a", vec!["gpt-4"]));
        manager.register_provider(provider).await;

        let request = ChatCompletionRequest {
            model: "gpt-4".to_string(),
            messages: vec![ChatMessage::user("hello")],
            ..Default::default()
        };

        // First failure
        let _: Result<()> = manager.execute_with_failover(&request, |_| async {
            Err(LLMError::HttpError {
                message: "timeout".to_string(),
                status_code: Some(500),
                body: None,
            })
        }).await;

        let metrics = manager.circuit_breaker_metrics().await;
        assert_eq!(metrics["prov-a"].total_failures, 1);

        // Second failure trips the breaker
        let _: Result<()> = manager.execute_with_failover(&request, |_| async {
            Err(LLMError::HttpError {
                message: "timeout".to_string(),
                status_code: Some(503),
                body: None,
            })
        }).await;

        let metrics = manager.circuit_breaker_metrics().await;
        assert_eq!(metrics["prov-a"].state, CircuitState::Open);
        assert_eq!(metrics["prov-a"].trip_count, 1);
    }

    #[tokio::test]
    async fn test_non_retryable_errors_dont_trip_breaker() {
        let config = ProviderManagerConfig {
            circuit_breaker_default: CircuitBreakerConfig {
                failure_threshold: 1,
                ..Default::default()
            },
            ..Default::default()
        };
        let manager = ProviderManager::new(config);

        let provider = Arc::new(MockProvider::new("prov-a", vec!["gpt-4"]));
        manager.register_provider(provider).await;

        let request = ChatCompletionRequest {
            model: "gpt-4".to_string(),
            messages: vec![ChatMessage::user("hello")],
            ..Default::default()
        };

        // Auth error — should NOT trip breaker
        let result: Result<()> = manager.execute_with_failover(&request, |_| async {
            Err(LLMError::AuthenticationError {
                message: "bad key".to_string(),
                retry_after: None,
            })
        }).await;

        assert!(result.is_err());
        let metrics = manager.circuit_breaker_metrics().await;
        assert_eq!(metrics["prov-a"].state, CircuitState::Closed);
        assert_eq!(metrics["prov-a"].total_failures, 0);
    }
}
