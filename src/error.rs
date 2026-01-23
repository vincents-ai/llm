/*!
 * LLM Provider Errors
 *
 * Comprehensive error types for LLM operations including authentication,
 * rate limiting, model availability, and network issues.
 */

use thiserror::Error;
use serde::{Deserialize, Serialize};

/// Result type for LLM operations
pub type Result<T> = std::result::Result<T, LLMError>;

/// Comprehensive error type for LLM operations
#[derive(Error, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "error_type", rename_all = "snake_case")]
pub enum LLMError {
    /// Authentication failed - invalid or missing API key
    #[error("Authentication failed: {message}")]
    #[serde(rename = "authentication_error")]
    AuthenticationError {
        message: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        retry_after: Option<u64>,
    },

    /// Rate limit exceeded
    #[error("Rate limit exceeded: {message}")]
    #[serde(rename = "rate_limit_error")]
    RateLimitError {
        message: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        retry_after: Option<u64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        limit: Option<u64>,
    },

    /// Requested model is not available
    #[error("Model not available: {model}")]
    #[serde(rename = "model_not_available")]
    ModelNotAvailable {
        model: String,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        available_models: Vec<String>,
    },

    /// Invalid request parameters
    #[error("Invalid request: {message}")]
    #[serde(rename = "invalid_request_error")]
    InvalidRequestError {
        message: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        param: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        code: Option<u32>,
    },

    /// HTTP/network error
    #[error("HTTP error: {message}")]
    #[serde(rename = "http_error")]
    HttpError {
        message: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        status_code: Option<u16>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        body: Option<String>,
    },

    /// Request timeout
    #[error("Request timeout after {timeout_secs} seconds")]
    #[serde(rename = "timeout_error")]
    TimeoutError {
        timeout_secs: u64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        message: Option<String>,
    },

    /// Provider-specific error
    #[error("Provider error: {message}")]
    #[serde(rename = "provider_error")]
    ProviderError {
        provider: String,
        message: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        code: Option<String>,
    },

    /// Configuration error
    #[error("Configuration error: {message}")]
    #[serde(rename = "configuration_error")]
    ConfigurationError {
        message: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        field: Option<String>,
    },

    /// Serialization/deserialization error
    #[error("Serialization error: {message}")]
    #[serde(rename = "serialization_error")]
    SerializationError {
        message: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        context: Option<String>,
    },

    /// Streaming error
    #[error("Streaming error: {message}")]
    #[serde(rename = "streaming_error")]
    StreamingError {
        message: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        bytes_received: Option<u64>,
    },

    /// Content moderation triggered
    #[error("Content moderation triggered: {reason}")]
    #[serde(rename = "content_filter_error")]
    ContentFilterError {
        reason: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        flagged_categories: Option<Vec<String>>,
    },

    /// Unsupported operation
    #[error("Unsupported operation: {operation}")]
    #[serde(rename = "unsupported_error")]
    UnsupportedError {
        operation: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        provider: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        model: Option<String>,
    },

    /// Context window exceeded
    #[error("Context window exceeded: {message}")]
    #[serde(rename = "context_length_exceeded")]
    ContextLengthExceeded {
        message: String,
        #[serde(default)]
        max_tokens: u32,
        #[serde(default)]
        requested_tokens: u32,
    },

    /// Unexpected response format
    #[error("Unexpected response format: {message}")]
    #[serde(rename = "response_format_error")]
    ResponseFormatError {
        message: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        expected: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        actual: Option<String>,
    },

    /// Quota exceeded
    #[error("Quota exceeded: {message}")]
    #[serde(rename = "quota_exceeded")]
    QuotaExceeded {
        message: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        reset_at: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        limit: Option<f64>,
    },
}

impl LLMError {
    /// Check if this error suggests retrying
    pub fn is_retryable(&self) -> bool {
        match self {
            LLMError::RateLimitError { .. } => true,
            LLMError::HttpError { status_code, .. } => {
                matches!(status_code, Some(500..=599))
            }
            LLMError::TimeoutError { .. } => true,
            LLMError::AuthenticationError { .. } => false,
            LLMError::ModelNotAvailable { .. } => false,
            LLMError::InvalidRequestError { .. } => false,
            LLMError::ProviderError { .. } => true,
            LLMError::ConfigurationError { .. } => false,
            LLMError::SerializationError { .. } => false,
            LLMError::StreamingError { .. } => true,
            LLMError::ContentFilterError { .. } => false,
            LLMError::UnsupportedError { .. } => false,
            LLMError::ContextLengthExceeded { .. } => false,
            LLMError::ResponseFormatError { .. } => false,
            LLMError::QuotaExceeded { .. } => false,
        }
    }

    /// Get recommended retry delay in seconds
    pub fn retry_delay_secs(&self) -> Option<u64> {
        match self {
            LLMError::RateLimitError { retry_after, .. } => *retry_after,
            LLMError::AuthenticationError { retry_after, .. } => *retry_after,
            LLMError::HttpError { status_code, .. } => {
                if matches!(status_code, Some(429 | 500 | 502 | 503 | 504)) {
                    Some(1)
                } else {
                    None
                }
            }
            LLMError::TimeoutError { timeout_secs, .. } => Some(*timeout_secs),
            _ => None,
        }
    }

    /// Convert to a user-friendly message
    pub fn user_message(&self) -> String {
        match self {
            LLMError::AuthenticationError { message, .. } => {
                format!("Authentication failed: {}. Please check your API key.", message)
            }
            LLMError::RateLimitError { message, retry_after, .. } => {
                if let Some(seconds) = retry_after {
                    format!("Rate limit exceeded. Please try again in {} seconds.", seconds)
                } else {
                    format!("Rate limit exceeded: {}", message)
                }
            }
            LLMError::ModelNotAvailable { model, .. } => {
                format!("Model '{}' is not available. Please try a different model.", model)
            }
            LLMError::InvalidRequestError { message, .. } => {
                format!("Invalid request: {}", message)
            }
            LLMError::HttpError { message, .. } => {
                format!("Network error: {}", message)
            }
            LLMError::TimeoutError { .. } => {
                "The request timed out. Please try again.".to_string()
            }
            LLMError::ProviderError { message, .. } => {
                format!("Provider error: {}", message)
            }
            LLMError::ConfigurationError { message, .. } => {
                format!("Configuration error: {}", message)
            }
            LLMError::SerializationError { message, .. } => {
                format!("Data error: {}", message)
            }
            LLMError::StreamingError { message, .. } => {
                format!("Streaming error: {}", message)
            }
            LLMError::ContentFilterError { reason, .. } => {
                format!("Content was filtered: {}", reason)
            }
            LLMError::UnsupportedError { operation, .. } => {
                format!("Operation '{}' is not supported.", operation)
            }
            LLMError::ContextLengthExceeded { message, .. } => {
                format!("Context too long: {}", message)
            }
            LLMError::ResponseFormatError { message, .. } => {
                format!("Unexpected response: {}", message)
            }
            LLMError::QuotaExceeded { message, .. } => {
                format!("Quota exceeded: {}", message)
            }
        }
    }
}

/// Error category for grouping
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorCategory {
    Authentication,
    RateLimit,
    ModelNotAvailable,
    InvalidRequest,
    Network,
    Provider,
    Configuration,
    Content,
    Unsupported,
    Quota,
    Other,
}

impl From<&LLMError> for ErrorCategory {
    fn from(err: &LLMError) -> Self {
        match err {
            LLMError::AuthenticationError { .. } => Self::Authentication,
            LLMError::RateLimitError { .. } => Self::RateLimit,
            LLMError::ModelNotAvailable { .. } => Self::ModelNotAvailable,
            LLMError::InvalidRequestError { .. } => Self::InvalidRequest,
            LLMError::HttpError { .. } => Self::Network,
            LLMError::TimeoutError { .. } => Self::Network,
            LLMError::ProviderError { .. } => Self::Provider,
            LLMError::ConfigurationError { .. } => Self::Configuration,
            LLMError::ContentFilterError { .. } => Self::Content,
            LLMError::UnsupportedError { .. } => Self::Unsupported,
            LLMError::QuotaExceeded { .. } => Self::Quota,
            _ => Self::Other,
        }
    }
}
