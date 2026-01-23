use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Provider configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderConfig {
    /// Provider type (e.g., "openai", "anthropic", "gemini")
    pub provider_type: String,
    /// Provider name (for display)
    pub name: String,
    /// API key (or env var name like "OPENAI_API_KEY")
    pub api_key: Option<String>,
    /// Base URL for API (for custom endpoints)
    pub base_url: Option<String>,
    /// Organization ID (provider-specific)
    pub organization_id: Option<String>,
    /// Default model
    pub default_model: Option<String>,
    /// Timeout in seconds
    pub timeout_secs: u64,
    /// Rate limits
    pub rate_limits: Option<RateLimitConfig>,
    /// Extra parameters
    pub extra_params: HashMap<String, serde_json::Value>,
}

impl Default for ProviderConfig {
    fn default() -> Self {
        Self {
            provider_type: String::new(),
            name: String::new(),
            api_key: None,
            base_url: None,
            organization_id: None,
            default_model: None,
            timeout_secs: 60,
            rate_limits: None,
            extra_params: HashMap::new(),
        }
    }
}

/// Rate limit configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitConfig {
    /// Requests per minute
    pub requests_per_minute: u64,
    /// Tokens per minute
    pub tokens_per_minute: u64,
    /// Requests per day
    pub requests_per_day: Option<u64>,
}

/// LLM configuration for the application
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMConfig {
    /// Default provider
    pub default_provider: String,
    /// Providers configuration
    pub providers: HashMap<String, ProviderConfig>,
    /// Global settings
    pub global: GlobalLLMSettings,
}

impl Default for LLMConfig {
    fn default() -> Self {
        Self {
            default_provider: "openai".to_string(),
            providers: HashMap::new(),
            global: GlobalLLMSettings::default(),
        }
    }
}

/// Global LLM settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalLLMSettings {
    /// Enable cost tracking
    pub cost_tracking: bool,
    /// Enable rate limiting
    pub rate_limiting: bool,
    /// Enable provider failover
    pub failover_enabled: bool,
    /// Maximum retries per request
    pub max_retries: u32,
    /// Default timeout
    pub default_timeout_secs: u64,
    /// Enable streaming
    pub streaming_enabled: bool,
    /// Max concurrent requests
    pub max_concurrent_requests: u32,
}

impl Default for GlobalLLMSettings {
    fn default() -> Self {
        Self {
            cost_tracking: true,
            rate_limiting: true,
            failover_enabled: true,
            max_retries: 3,
            default_timeout_secs: 60,
            streaming_enabled: true,
            max_concurrent_requests: 10,
        }
    }
}

/// Provider-specific settings
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "provider")]
pub enum ProviderSettings {
    OpenAI(OpenAISettings),
    Anthropic(AnthropicSettings),
    Gemini(GeminiSettings),
    Ollama(OllamaSettings),
    Custom(CustomProviderSettings),
}

/// OpenAI-specific settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAISettings {
    /// API base URL (custom endpoint)
    pub base_url: Option<String>,
    /// Organization ID
    pub organization_id: Option<String>,
    /// Project ID
    pub project_id: Option<String>,
}

/// Anthropic-specific settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicSettings {
    /// API base URL
    pub base_url: Option<String>,
    /// API version
    pub api_version: Option<String>,
}

/// Gemini-specific settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeminiSettings {
    /// API base URL
    pub base_url: Option<String>,
    /// Project ID
    pub project_id: Option<String>,
    /// Region
    pub region: Option<String>,
}

/// Ollama-specific settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaSettings {
    /// Server URL
    pub base_url: String,
    /// Model to use
    pub default_model: String,
    /// Timeout
    pub timeout_secs: u64,
}

/// Custom provider settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomProviderSettings {
    /// API type identifier
    pub api_type: String,
    /// Base URL
    pub base_url: String,
    /// API version
    pub api_version: Option<String>,
}
