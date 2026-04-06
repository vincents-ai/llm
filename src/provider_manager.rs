use crate::provider::{LLMProvider, ProviderHealth, ProviderRegistry};
use crate::error::{LLMError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use tokio::sync::Mutex;
use tracing::info;

/// Configuration for provider manager
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderManagerConfig {
    /// Default provider to use
    pub default_provider: String,
    /// Enable automatic failover
    pub failover_enabled: bool,
    /// Health check interval in seconds
    pub health_check_interval_sec: u64,
    /// Circuit breaker threshold (errors before opening)
    pub circuit_breaker_threshold: u32,
    /// Circuit breaker timeout in seconds
    pub circuit_breaker_timeout_sec: u64,
}

impl Default for ProviderManagerConfig {
    fn default() -> Self {
        Self {
            default_provider: "openai".to_string(),
            failover_enabled: true,
            health_check_interval_sec: 60,
            circuit_breaker_threshold: 5,
            circuit_breaker_timeout_sec: 30,
        }
    }
}

/// Provider manager for multi-provider coordination
pub struct ProviderManager {
    config: ProviderManagerConfig,
    providers: RwLock<HashMap<String, Arc<dyn LLMProvider>>>,
    registry: ProviderRegistry,
    errors: Mutex<HashMap<String, u32>>,
    circuit_breakers: RwLock<HashMap<String, CircuitBreakerState>>,
    _health_check_task: Mutex<Option<tokio::task::JoinHandle<()>>>,
}

impl ProviderManager {
    /// Create a new provider manager
    pub fn new(config: ProviderManagerConfig) -> Self {
        Self {
            registry: ProviderRegistry::new(),
            providers: RwLock::new(HashMap::new()),
            circuit_breakers: RwLock::new(HashMap::new()),
            config,
            errors: Mutex::new(HashMap::new()),
            _health_check_task: Mutex::new(None),
        }
    }

    /// Register a provider
    pub async fn register_provider(&self, provider: Arc<dyn LLMProvider>) {
        let name = provider.name().to_string();
        let _config = provider.config();

        match self.providers.write() {
            Ok(mut p) => { p.insert(name.clone(), provider); }
            Err(poisoned) => { poisoned.into_inner().insert(name.clone(), provider); }
        }

        info!("Registered provider: {}", name);
    }

    /// Get a provider by name
    pub fn get_provider(&self, name: &str) -> Option<Arc<dyn LLMProvider>> {
        match self.providers.read() {
            Ok(p) => p.get(name).cloned(),
            Err(poisoned) => poisoned.into_inner().get(name).cloned(),
        }
    }

    /// Get the default provider
    pub fn get_default_provider(&self) -> Option<Arc<dyn LLMProvider>> {
        self.get_provider(&self.config.default_provider)
    }

    /// Select best provider for a request
    pub async fn select_provider(
        &self,
        model: &str,
    ) -> Result<Arc<dyn LLMProvider>> {
        let providers = match self.providers.read() {
            Ok(p) => p,
            Err(poisoned) => poisoned.into_inner(),
        };

        // Find available providers that support the model
        let available: Vec<_> = providers
            .iter()
            .filter(|(_, p)| {
                p.supports_model(model) && self.is_provider_available(p.name())
            })
            .collect();

        if available.is_empty() {
            // Try any provider supporting the model
            let fallback: Vec<_> = providers
                .iter()
                .filter(|(_, p)| p.supports_model(model))
                .collect();

            if fallback.is_empty() {
                return Err(LLMError::ModelNotAvailable {
                    model: model.to_string(),
                    available_models: providers
                        .values()
                        .flat_map(|p| p.supported_models())
                        .collect(),
                });
            }

            if fallback.len() == 1 {
                return Ok(fallback[0].1.clone());
            }

            // Return first available or default
            return Ok(fallback[0].1.clone());
        }

        if available.len() == 1 {
            return Ok(available[0].1.clone());
        }

        // If multiple available, prefer default
        if let Some(default) = providers.get(&self.config.default_provider) {
            if default.supports_model(model) && self.is_provider_available(default.name()) {
                return Ok(default.clone());
            }
        }

        // Return first available
        Ok(available[0].1.clone())
    }

    /// Execute with failover
    pub async fn execute_with_failover<F, T>(
        &self,
        request: &crate::types::ChatCompletionRequest,
        mut execute: impl FnMut(Arc<dyn LLMProvider>) -> F,
    ) -> Result<T>
    where
        F: std::future::Future<Output = Result<T>>,
    {
        let model = request.model.clone();
        let providers = match self.providers.read() {
            Ok(p) => p.clone(),
            Err(poisoned) => poisoned.into_inner().clone(),
        };

        for (name, provider) in providers {
            if !provider.supports_model(&model) {
                continue;
            }

            if !self.is_provider_available(&name) {
                continue;
            }

            match execute(provider.clone()).await {
                Ok(result) => return Ok(result),
                Err(LLMError::RateLimitError { .. }) => {
                    self.record_error(&name);
                    continue;
                }
                Err(LLMError::HttpError { .. }) => {
                    self.record_error(&name);
                    continue;
                }
                Err(e) => return Err(e),
            }
        }

        Err(LLMError::ProviderError {
            provider: "all".to_string(),
            message: "All providers failed".to_string(),
            code: None,
        })
    }

    /// Get health status for all providers
    pub async fn health_status(&self) -> HashMap<String, ProviderHealth> {
        let providers = match self.providers.read() {
            Ok(p) => p,
            Err(poisoned) => poisoned.into_inner(),
        };
        let mut status = HashMap::new();

        for (name, provider) in providers.iter() {
            let health = provider.health_check().await.unwrap_or_else(|e| ProviderHealth {
                name: name.clone(),
                healthy: false,
                latency_ms: None,
                error: Some(e.to_string()),
                rate_limit_remaining: None,
                rate_limit_total: None,
            });
            status.insert(name.clone(), health);
        }

        status
    }

    /// Check if provider is available
    fn is_provider_available(&self, name: &str) -> bool {
        let breakers = match self.circuit_breakers.read() {
            Ok(b) => b,
            Err(poisoned) => poisoned.into_inner(),
        };
        match breakers.get(name) {
            Some(state) => !state.is_open(),
            None => true,
        }
    }

    /// Record an error for a provider
    fn record_error(&self, name: &str) {
        let mut breakers = match self.circuit_breakers.write() {
            Ok(b) => b,
            Err(poisoned) => poisoned.into_inner(),
        };
        if let Some(state) = breakers.get_mut(name) {
            state.record_failure();
        }
    }

    /// Get all provider names
    pub fn provider_names(&self) -> Vec<String> {
        match self.providers.read() {
            Ok(p) => p.keys().cloned().collect(),
            Err(poisoned) => poisoned.into_inner().keys().cloned().collect(),
        }
    }
}

/// Circuit breaker state
#[derive(Debug, Clone)]
pub struct CircuitBreakerState {
    failures: u32,
    threshold: u32,
    last_failure: Option<std::time::Instant>,
    timeout: std::time::Duration,
    state: CircuitBreaker,
}

impl CircuitBreakerState {
    fn new(threshold: u32) -> Self {
        Self {
            failures: 0,
            threshold,
            last_failure: None,
            timeout: std::time::Duration::from_secs(30),
            state: CircuitBreaker::Closed,
        }
    }

    fn record_failure(&mut self) {
        self.failures += 1;
        self.last_failure = Some(std::time::Instant::now());

        if self.failures >= self.threshold {
            self.state = CircuitBreaker::Open;
        }
    }

    fn is_open(&self) -> bool {
        if self.state == CircuitBreaker::Open {
            if let Some(last_failure) = self.last_failure {
                if last_failure.elapsed() > self.timeout {
                    return false;
                }
            }
            return true;
        }
        false
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitBreaker {
    Closed,
    Open,
    HalfOpen,
}
