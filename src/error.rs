/*!
 * LLM Provider Errors
 *
 * Comprehensive error types for LLM operations including authentication,
 * rate limiting, model availability, and network issues.
 */

use serde::{Deserialize, Serialize};
use thiserror::Error;

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

    /// The response stream ended before the provider's own terminator.
    ///
    /// Distinct from a generic stream error because the caller's response is
    /// *partial*: acting on it as if complete is the failure mode.
    #[error("Stream interrupted: {message}")]
    StreamInterrupted { message: String },

    /// The provider does not implement the feature the request needs.
    ///
    /// Distinct from a rejected *parameter*: this is a capability the provider
    /// has not got, not a value it disliked. A caller can retry on a different
    /// model; a caller that sees this for a bad parameter value would not.
    #[error("Unsupported feature: {feature} (model {model:?})")]
    UnsupportedFeature { feature: String, model: String },

    /// A parameter the provider does not recognise.
    ///
    /// The name is preserved verbatim: the caller needs to know *which*
    /// parameter, and providers spell their unknowns differently.
    #[error("Unsupported parameter: {parameter}")]
    UnsupportedParameter { parameter: String },

    /// Two individually valid parameters that cannot both be set.
    ///
    /// Rejected locally, before any network I/O, because the provider's own
    /// error for this is a generic 400 that does not say which pair conflicted.
    #[error("Invalid parameter combination: {message}")]
    InvalidParameterCombination { message: String },

    /// The provider understood the parameter and refused this value.
    ///
    /// Not the same as `UnsupportedParameter`: retrying or dropping the
    /// parameter will not help, the caller must change the value.
    #[error("Provider rejected parameter {parameter}: {reason}")]
    ProviderRejectedParameter { parameter: String, reason: String },

    /// The stream did not follow the protocol.
    ///
    /// Kept separate from `SerializationError` (§50): a malformed SSE frame is
    /// a stream-protocol failure, and collapsing it into "could not parse JSON"
    /// loses the fact that the *stream* was wrong, not the document.
    #[error("Stream protocol error: {message}")]
    StreamProtocolError { message: String },

    /// The response completed but never reported usage.
    ///
    /// A billing-relevant state, not a transport failure. The answer may be
    /// perfectly good; what is missing is the accounting.
    #[error("Incomplete usage: {message}")]
    IncompleteUsage { message: String },

    /// A provider failure, with everything needed to diagnose it.
    ///
    /// `request_id` is the provider's correlation handle. Without it a support
    /// conversation about a specific failed call is not possible, which is why
    /// it is preserved rather than logged and dropped.
    // The request id is in `Display`, not only in `user_message`: a log line
    // that carries an error but not its correlation handle cannot be acted on,
    // and `{}` on an error is how most errors reach a log.
    #[error("{provider} error {status}: {message} (request {request_id:?})")]
    ProviderRequest {
        provider: String,
        status: u16,
        message: String,
        body: Option<String>,
        request_id: Option<String>,
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
            // A truncated response is worth retrying: the request was
            // understood, and the failure was in transit.
            LLMError::StreamInterrupted { .. } => true,
            // A protocol violation is the provider's fault and may be
            // transient, but the same violation will recur on retry against
            // the same endpoint, so it is *not* retryable -- retrying turns one
            // bad response into several.
            LLMError::StreamProtocolError { .. } => false,
            // Nothing wrong with the request; there is simply nothing to bill.
            LLMError::IncompleteUsage { .. } => false,
            LLMError::ProviderRequest { status, .. } => {
                matches!(status, 408 | 409 | 429 | 500..=599)
            }
            // The caller asked for something this provider cannot do.
            LLMError::UnsupportedFeature { .. } => false,
            LLMError::UnsupportedParameter { .. } => false,
            LLMError::InvalidParameterCombination { .. } => false,
            LLMError::ProviderRejectedParameter { .. } => false,
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
                format!(
                    "Authentication failed: {}. Please check your API key.",
                    message
                )
            }
            LLMError::RateLimitError {
                message,
                retry_after,
                ..
            } => {
                if let Some(seconds) = retry_after {
                    format!(
                        "Rate limit exceeded. Please try again in {} seconds.",
                        seconds
                    )
                } else {
                    format!("Rate limit exceeded: {}", message)
                }
            }
            LLMError::ModelNotAvailable { model, .. } => {
                format!(
                    "Model '{}' is not available. Please try a different model.",
                    model
                )
            }
            LLMError::UnsupportedFeature { feature, model } => {
                format!(
                    "{model} does not support {feature}. Use a model that does, or drop \
                     the feature."
                )
            }
            LLMError::UnsupportedParameter { parameter } => {
                format!("The provider does not recognise the parameter '{parameter}'.")
            }
            LLMError::InvalidParameterCombination { message } => {
                format!("Conflicting request parameters: {message}")
            }
            LLMError::ProviderRejectedParameter { parameter, reason } => {
                format!("The provider rejected '{parameter}': {reason}")
            }
            LLMError::StreamProtocolError { message } => {
                format!("The response stream was malformed: {message}")
            }
            LLMError::IncompleteUsage { message } => {
                format!(
                    "The response is complete but reported no usage ({message}). \
                     The answer is usable; the cost is not yet known."
                )
            }
            LLMError::ProviderRequest {
                provider,
                status,
                request_id,
                ..
            } => {
                let mut text = format!("{provider} returned HTTP {status}");
                if let Some(id) = request_id {
                    text.push_str(&format!(" (request {id})"));
                }
                text
            }
            LLMError::StreamInterrupted { message } => {
                format!(
                    "The response stream ended early ({}). The answer is incomplete — \
                     do not treat it as a finished response.",
                    message
                )
            }
            LLMError::InvalidRequestError { message, .. } => {
                format!("Invalid request: {}", message)
            }
            LLMError::HttpError { message, .. } => {
                format!("Network error: {}", message)
            }
            LLMError::TimeoutError { .. } => "The request timed out. Please try again.".to_string(),
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

impl LLMError {
    /// Classify a provider's HTTP failure.
    ///
    /// One mapper for every provider, because the classification is a property
    /// of the protocol's error format and not of which provider produced it.
    /// Handlers that each did this inline is how a 400 naming a parameter came
    /// to be reported as `HttpError` and lost the parameter's name.
    ///
    /// The provider's own error body is preferred over our summary, and the
    /// `request_id` header is preserved: without it, a report about one specific
    /// failed call cannot be traced to the provider's logs.
    pub fn from_provider_response(
        provider: &str,
        status: reqwest::StatusCode,
        body: Option<String>,
        request_id: Option<String>,
    ) -> Self {
        let status_code = status.as_u16();
        let raw = body.clone().unwrap_or_default();

        // Pull the provider's own error type out of its JSON body, when it has
        // one. The shape differs per provider, so several are tried rather than
        // one being assumed.
        let extracted = serde_json::from_str::<serde_json::Value>(&raw).ok().and_then(|json| {
            let error = json.get("error")?;
            let kind = error
                .get("type")
                .or_else(|| error.get("code"))
                .and_then(|v| {
                    v.as_str()
                        .map(str::to_string)
                        .or_else(|| v.as_u64().map(|n| n.to_string()))
                });
            let message = error
                .get("message")
                .and_then(|v| v.as_str())
                .map(str::to_string);
            let param = error
                .get("param")
                .or_else(|| error.get("parameter"))
                .and_then(|v| v.as_str())
                .map(str::to_string);
            Some((kind, message, param))
        });
        let (kind, message, param) = match extracted {
            Some((kind, message, param)) => (kind, message, param),
            None => (None, None, None),
        };
        let message = message.unwrap_or_else(|| {
            if raw.trim().is_empty() {
                format!("{provider} returned HTTP {status_code} with no body")
            } else {
                raw.clone()
            }
        });

        match status_code {
            401 | 403 => LLMError::AuthenticationError { message, retry_after: None },
            404 if kind.as_deref() == Some("model_not_found") => {
                LLMError::ModelNotAvailable { model: message, available_models: vec![] }
            }
            404 => LLMError::ModelNotAvailable { model: message, available_models: vec![] },
            413 | 429 if status_code == 429 => LLMError::RateLimitError {
                message,
                retry_after: None,
                limit: None,
            },
            400 | 422 => {
                // A rejected *parameter* is not a rejected request. Reporting it
                // as a generic 400 loses the parameter's name, which is the only
                // actionable part.
                match param {
                    Some(parameter) => LLMError::ProviderRejectedParameter {
                        parameter,
                        reason: message,
                    },
                    None => LLMError::InvalidRequestError {
                        message,
                        // The parameter name, when the provider gave one. Kept
                        // here as well so a caller matching the older variant
                        // still has it.
                        param: None,
                        code: None,
                    },
                }
            }
            _ => LLMError::ProviderRequest {
                provider: provider.to_string(),
                status: status_code,
                message,
                body,
                request_id,
            },
        }
    }
}
