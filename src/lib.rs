/*!
 * Vincents LLM Provider Abstraction
 *
 * A multi-LLM provider abstraction layer for Rust applications.
 * Provides a unified interface for OpenAI, Claude, Gemini, Ollama, and OpenRouter.
 */
#![allow(dead_code)]

pub mod config;
pub mod error;
pub mod types;

#[cfg(feature = "openai")]
pub mod openai;

#[cfg(feature = "anthropic")]
pub mod anthropic;

#[cfg(feature = "gemini")]
pub mod gemini;

#[cfg(feature = "ollama")]
pub mod ollama;

#[cfg(feature = "openrouter")]
pub mod openrouter;

// Re-exports
pub use config::{LLMConfig, ProviderConfig};
pub use error::{LLMError, Result};
pub use types::*;

// Provider trait and implementations
pub use provider::{LLMProvider, ProviderHealth, ProviderRegistry};

#[cfg(feature = "openai")]
pub use openai::OpenAIProvider;

#[cfg(feature = "anthropic")]
pub use anthropic::AnthropicProvider;

#[cfg(feature = "gemini")]
pub use gemini::GeminiProvider;

#[cfg(feature = "ollama")]
pub use ollama::OllamaProvider;

#[cfg(feature = "openrouter")]
pub use openrouter::OpenRouterProvider;

// Supporting infrastructure
pub use circuit_breaker::{
    CircuitBreaker, CircuitBreakerConfig, CircuitBreakerMetrics, CircuitOpenError, CircuitState,
};
pub use cost_tracker::{CostEstimate, CostTracker};
pub use provider_manager::ProviderManager;
pub use rate_limiter::{RateLimitStatus, RateLimiter, RetryConfig, RetryPolicy, RetryStrategy};
pub use telemetry::{
    LlmTelemetryAttributes, LlmTelemetryRecorder, NoopTelemetryRecorder, TelemetrySpan,
    TracingTelemetryRecorder,
};
pub use token_counter::{TokenCount, TokenCounter, TokenCounterConfig, TokenCountingError};

pub mod circuit_breaker;
mod cost_tracker;
pub mod provider;
pub mod provider_manager;
mod rate_limiter;
mod telemetry;
mod token_counter;
