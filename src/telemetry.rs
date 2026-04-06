use crate::error::LLMError;
use crate::types::{ChatCompletionRequest, ChatCompletionResponse, Chunk};
use std::time::{Duration, Instant};
use tracing::{debug, error, span, trace, Level};

pub type TelemetryContext = ();

pub struct TelemetrySpan {
    name: &'static str,
    start: Instant,
    fields: Vec<(&'static str, String)>,
}

impl TelemetrySpan {
    pub fn new(name: &'static str) -> Self {
        TelemetrySpan {
            name,
            start: Instant::now(),
            fields: Vec::new(),
        }
    }

    pub fn with_field(mut self, key: &'static str, value: impl ToString) -> Self {
        self.fields.push((key, value.to_string()));
        self
    }

    pub fn record_request(&self, request: &ChatCompletionRequest) {
        let model = request.model.clone();
        debug!(
            target: "llm",
            provider = "unknown",
            model = model,
            messages = request.messages.len(),
            "LLM request started"
        );
    }

    pub fn record_response(&self, response: &ChatCompletionResponse) {
        let elapsed = self.start.elapsed();
        debug!(
            target: "llm",
            provider = response.model,
            choices = response.choices.len(),
            elapsed_ms = elapsed.as_millis(),
            "LLM response received"
        );
    }

    pub fn record_chunk(&self, chunk: &Chunk) {
        let model = match chunk {
            Chunk::Completion(c) => c.model.clone(),
        };
        trace!(
            target: "llm",
            provider = "unknown",
            model = model,
            "Streaming chunk received"
        );
    }

    pub fn record_error(&self, error: &LLMError) {
        error!(
            target: "llm",
            error_type = error.to_string(),
            "LLM operation failed"
        );
    }
}

#[derive(Debug, Clone)]
pub struct LlmTelemetryAttributes {
    pub provider: String,
    pub model: String,
    pub operation: &'static str,
    pub request_id: Option<String>,
}

impl LlmTelemetryAttributes {
    pub fn new(provider: String, model: String, operation: &'static str) -> Self {
        LlmTelemetryAttributes {
            provider,
            model,
            operation,
            request_id: None,
        }
    }

    pub fn with_request_id(mut self, id: String) -> Self {
        self.request_id = Some(id);
        self
    }
}

pub trait LlmTelemetryRecorder {
    fn record_request_started(&self, attrs: &LlmTelemetryAttributes);
    fn record_request_completed(&self, attrs: &LlmTelemetryAttributes, duration: Duration);
    fn record_error(&self, attrs: &LlmTelemetryAttributes, error: &LLMError);
    fn record_chunk(&self, attrs: &LlmTelemetryAttributes, chunk: &Chunk);
}

#[derive(Default)]
pub struct NoopTelemetryRecorder;

impl LlmTelemetryRecorder for NoopTelemetryRecorder {
    fn record_request_started(&self, _: &LlmTelemetryAttributes) {}
    fn record_request_completed(&self, _: &LlmTelemetryAttributes, _: Duration) {}
    fn record_error(&self, _: &LlmTelemetryAttributes, _: &LLMError) {}
    fn record_chunk(&self, _: &LlmTelemetryAttributes, _: &Chunk) {}
}

pub struct TracingTelemetryRecorder;

impl LlmTelemetryRecorder for TracingTelemetryRecorder {
    fn record_request_started(&self, attrs: &LlmTelemetryAttributes) {
        let span = span!(
            Level::INFO,
            "llm.request",
            provider = attrs.provider.as_str(),
            model = attrs.model.as_str(),
            operation = attrs.operation,
        );
        span.record("request_id", attrs.request_id.clone().unwrap_or_default());
        let _entered = span.enter();
    }

    fn record_request_completed(&self, attrs: &LlmTelemetryAttributes, duration: Duration) {
        tracing::info!(
            target: "llm",
            provider = attrs.provider,
            model = attrs.model,
            operation = attrs.operation,
            duration_ms = duration.as_millis() as u64,
            "LLM request completed"
        );
    }

    fn record_error(&self, attrs: &LlmTelemetryAttributes, error: &LLMError) {
        tracing::error!(
            target: "llm",
            provider = attrs.provider,
            model = attrs.model,
            operation = attrs.operation,
            error = error.to_string(),
            "LLM request failed"
        );
    }

    fn record_chunk(&self, attrs: &LlmTelemetryAttributes, _chunk: &Chunk) {
        tracing::trace!(
            target: "llm",
            provider = attrs.provider,
            model = attrs.model,
            "Streaming chunk"
        );
    }
}
