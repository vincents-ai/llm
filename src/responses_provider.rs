//! The OpenAI Responses API transport.
//!
//! Native, not a translation through Chat Completions -- see the module
//! documentation in [`crate::responses`] for why a translation would have to
//! invent an equivalence for `instructions`, for the typed `output` array, and
//! for provider-operated tools, and why each invented equivalence is a place
//! information cannot survive.

use std::pin::Pin;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::{LLMError, Result};
use crate::responses::{
    Response, ResponseOutputItem, ResponseRequest, ResponseStatus, ResponsesUsage,
};

/// Configuration for [`OpenAIResponsesProvider`].
#[derive(Debug, Clone, Serialize, Deserialize)]
#[derive(Default)]
pub struct OpenAIResponsesConfig {
    pub api_key: String,
    /// Sent on every request, including `/responses`.
    ///
    /// §19: applying an organization to model listing but not to completions
    /// means a configured organization only half works.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub organization_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub project_id: Option<String>,
    pub base_url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_model: Option<String>,
    /// Application attribution, for providers that ask for it (§34).
    ///
    /// Absent on OpenAI, required by OpenRouter's terms. It lives here rather
    /// than in a request field because it names the *application* -- a value
    /// every call site could set differently is the opposite of what attribution
    /// is for.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub referrer: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// Ask for routing metadata (§36).
    #[serde(default)]
    pub metadata: bool,
    /// A stable id for this conversation (§37).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    /// A per-request trace id (§37).
    ///
    /// An identifier only. Nothing here derives one from message content.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trace: Option<String>,
}


/// A client for OpenAI's `/responses` endpoint.
#[derive(Debug, Clone)]
pub struct OpenAIResponsesProvider {
    config: OpenAIResponsesConfig,
    client: reqwest::Client,
}

impl OpenAIResponsesProvider {
    pub fn new(api_key: impl Into<String>) -> Result<Self> {
        Ok(Self {
            config: OpenAIResponsesConfig {
                api_key: api_key.into(),
                ..Default::default()
            },
            client: reqwest::Client::new(),
        })
    }

    pub fn with_config(config: OpenAIResponsesConfig) -> Result<Self> {
        Ok(Self {
            config,
            client: reqwest::Client::new(),
        })
    }

    fn base_url(&self) -> &str {
        self.config
            .base_url
            .as_deref()
            .unwrap_or("https://api.openai.com/v1")
    }

    /// Authentication and routing headers, identical on every call path.
    fn headers(&self, builder: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        let mut builder = builder
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .header("Content-Type", "application/json");
        if let Some(organization) = &self.config.organization_id {
            builder = builder.header("OpenAI-Organization", organization);
        }
        if let Some(project) = &self.config.project_id {
            builder = builder.header("OpenAI-Project", project);
        }
        // Attribution and observability, applied identically on create, stream,
        // retrieve, cancel and delete. A header that only appears on the main
        // path is the same class of defect as an organization header that only
        // appears on `/models`.
        if let Some(referrer) = &self.config.referrer {
            builder = builder.header("HTTP-Referer", referrer);
        }
        if let Some(title) = &self.config.title {
            builder = builder.header("X-OpenRouter-Title", title);
        }
        if self.config.metadata {
            builder = builder.header("X-OpenRouter-Metadata", "enabled");
        }
        if let Some(session_id) = &self.config.session_id {
            builder = builder.header("X-Session-Id", session_id);
        }
        if let Some(trace) = &self.config.trace {
            builder = builder.header("X-OpenRouter-Trace", trace);
        }
        builder
    }

    /// Create a response.
    pub async fn create(&self, mut request: ResponseRequest) -> Result<Response> {
        if request.model.is_empty() {
            request.model = self.config.default_model.clone().ok_or_else(|| {
                LLMError::ConfigurationError {
                    message: "no model configured for OpenAIResponsesProvider".to_string(),
                    field: Some("default_model".to_string()),
                }
            })?;
        }
        let body = request.to_body()?;
        let response = self
            .headers(self.client.post(format!("{}/responses", self.base_url())))
            .json(&body)
            .send()
            .await
            .map_err(|e| LLMError::HttpError {
                message: e.to_string(),
                status_code: None,
                body: None,
            })?;

        let status = response.status();
        if !status.is_success() {
            let request_id = response
                .headers()
                .get("x-request-id")
                .and_then(|v| v.to_str().ok())
                .map(str::to_string);
            let body = response.text().await.ok();
            return Err(LLMError::from_provider_response(
                "OpenAI Responses",
                status,
                body,
                request_id,
            ));
        }

        let response: Response = response.json().await.map_err(|e| {
            LLMError::SerializationError {
                message: e.to_string(),
                context: Some("failed to parse the /responses payload".to_string()),
            }
        })?;
        if let Some(error) = &response.error {
            return Err(LLMError::ProviderRequest {
                provider: "OpenAI Responses".to_string(),
                status: 200,
                message: error
                    .get("message")
                    .and_then(|m| m.as_str())
                    .unwrap_or("the response carried an error")
                    .to_string(),
                body: Some(error.to_string()),
                request_id: Some(response.id.clone()),
            });
        }
        Ok(response)
    }

    /// Create a response and stream its events.
    ///
    /// The stream carries typed items, so the same "index 0 is not text" hazard
    /// applies as for the non-streaming path.
    pub async fn create_stream(
        &self,
        mut request: ResponseRequest,
    ) -> Result<Pin<Box<dyn futures::Stream<Item = Result<ResponseStreamEvent>> + Send>>> {
        if request.model.is_empty() {
            request.model = self.config.default_model.clone().ok_or_else(|| {
                LLMError::ConfigurationError {
                    message: "no model configured for OpenAIResponsesProvider".to_string(),
                    field: Some("default_model".to_string()),
                }
            })?;
        }
        request.validate()?;
        let mut body = request.to_body()?;
        // `stream` is set directly rather than through `to_body`, because
        // streaming is a transport concern on this endpoint rather than a
        // request field a caller sets.
        body["stream"] = Value::Bool(true);

        let response = self
            .headers(self.client.post(format!("{}/responses", self.base_url())))
            .json(&body)
            .send()
            .await
            .map_err(|e| LLMError::HttpError {
                message: e.to_string(),
                status_code: None,
                body: None,
            })?;

        let status = response.status();
        if !status.is_success() {
            let request_id = response
                .headers()
                .get("x-request-id")
                .and_then(|v| v.to_str().ok())
                .map(str::to_string);
            let body = response.text().await.ok();
            return Err(LLMError::from_provider_response(
                "OpenAI Responses",
                status,
                body,
                request_id,
            ));
        }

        let frames = crate::stream::parse_sse_json(response);
        let mapped = futures::StreamExt::map(frames, |frame| {
            frame.map(|frame| match frame {
                crate::stream::SseFrame::Done => ResponseStreamEvent::Done,
                crate::stream::SseFrame::Data(value) => response_event_from_json(&value)
                    .unwrap_or(ResponseStreamEvent::Unknown(value)),
            })
        });
        Ok(Box::pin(mapped))
    }

    /// Retrieve a stored response.
    pub async fn retrieve(&self, id: &str) -> Result<Response> {
        let response = self
            .headers(self.client.get(format!("{}/responses/{}", self.base_url(), id)))
            .send()
            .await
            .map_err(|e| LLMError::HttpError {
                message: e.to_string(),
                status_code: None,
                body: None,
            })?;

        let status = response.status();
        if !status.is_success() {
            let request_id = response
                .headers()
                .get("x-request-id")
                .and_then(|v| v.to_str().ok())
                .map(str::to_string);
            let body = response.text().await.ok();
            return Err(LLMError::from_provider_response(
                "OpenAI Responses",
                status,
                body,
                request_id,
            ));
        }
        response.json().await.map_err(|e| LLMError::SerializationError {
            message: e.to_string(),
            context: Some("failed to parse the /responses payload".to_string()),
        })
    }

    /// Cancel a background response (§56).
    pub async fn cancel(&self, id: &str) -> Result<Response> {
        let response = self
            .headers(
                self.client
                    .post(format!("{}/responses/{}/cancel", self.base_url(), id)),
            )
            .send()
            .await
            .map_err(|e| LLMError::HttpError {
                message: e.to_string(),
                status_code: None,
                body: None,
            })?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.ok();
            return Err(LLMError::from_provider_response(
                "OpenAI Responses",
                status,
                body,
                None,
            ));
        }
        response.json().await.map_err(|e| LLMError::SerializationError {
            message: e.to_string(),
            context: Some("failed to parse the cancel payload".to_string()),
        })
    }

    /// Delete a stored response.
    pub async fn delete(&self, id: &str) -> Result<()> {
        let response = self
            .headers(
                self.client
                    .delete(format!("{}/responses/{}", self.base_url(), id)),
            )
            .send()
            .await
            .map_err(|e| LLMError::HttpError {
                message: e.to_string(),
                status_code: None,
                body: None,
            })?;
        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.ok();
            return Err(LLMError::from_provider_response(
                "OpenAI Responses",
                status,
                body,
                None,
            ));
        }
        Ok(())
    }
}

/// One event of a streamed `/responses` response.
#[derive(Debug, Clone, PartialEq)]
pub enum ResponseStreamEvent {
    /// The response envelope, sent first.
    Created {
        response: Box<Response>,
    },
    /// A completed output item. Typed, because a function call and a message
    /// need different handling and arrive on the same stream.
    OutputItemDone {
        output_index: u32,
        item: ResponseOutputItem,
    },
    /// A text fragment.
    TextDelta {
        text: String,
        output_index: u32,
    },
    /// A reasoning summary fragment.
    ReasoningDelta {
        text: String,
        output_index: u32,
    },
    Usage {
        usage: ResponsesUsage,
    },
    Completed {
        response: Box<Response>,
    },
    Failed {
        message: String,
    },
    /// The provider's terminator.
    Done,
    /// Anything not recognised, kept rather than dropped.
    Unknown(Value),
}

/// Map one decoded SSE frame to an event.
///
/// A pure function over the frame, so the framing (buffering, transport errors,
/// the `[DONE]`-means-clean rule) lives in exactly one place. The Responses
/// stream has a different *vocabulary* from Chat Completions, not different
/// plumbing, and duplicating the plumbing is how it drifts.
pub fn response_event_from_json(value: &Value) -> Option<ResponseStreamEvent> {
    let kind = value["type"].as_str().unwrap_or_default().to_string();
    let output_index = value["output_index"].as_u64().unwrap_or(0) as u32;
    match kind.as_str() {
        "response.created" | "response.in_progress" => {
            serde_json::from_value::<Response>(value["response"].clone())
                .ok()
                .map(|response| ResponseStreamEvent::Created {
                    response: Box::new(response),
                })
        }
        "response.output_item.done" => {
            serde_json::from_value::<ResponseOutputItem>(value["item"].clone())
                .ok()
                .map(|item| ResponseStreamEvent::OutputItemDone { output_index, item })
        }
        "response.output_text.delta" => value["delta"].as_str().map(|text| {
            ResponseStreamEvent::TextDelta {
                text: text.to_string(),
                output_index,
            }
        }),
        "response.reasoning_summary_text.delta" | "response.reasoning_text.delta" => {
            value["delta"].as_str().map(|text| {
                ResponseStreamEvent::ReasoningDelta {
                    text: text.to_string(),
                    output_index,
                }
            })
        }
        "response.completed" => {
            serde_json::from_value::<Response>(value["response"].clone()).ok().map(|response| {
                ResponseStreamEvent::Completed {
                    response: Box::new(response),
                }
            })
        }
        "response.failed" | "response.incomplete" => {
            let message = value["response"]["error"]["message"]
                .as_str()
                .unwrap_or("the response did not complete")
                .to_string();
            Some(ResponseStreamEvent::Failed { message })
        }
        // Anything unrecognised is preserved. A new event type reaching an
        // older build must not vanish.
        _ => Some(ResponseStreamEvent::Unknown(value.clone())),
    }
}

/// A status that a caller can treat as finished.
pub fn is_terminal(status: &ResponseStatus) -> bool {
    status.is_terminal()
}