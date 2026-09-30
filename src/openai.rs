//! OpenAI Provider Implementation
//!
//! This module provides the OpenAI provider implementation using direct HTTP calls.

use async_trait::async_trait;
use reqwest;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::pin::Pin;

use crate::config::ProviderConfig;
use crate::error::{LLMError, Result};
use crate::provider::LLMProvider;
use crate::types::*;
use crate::CostEstimate;
use crate::RateLimitStatus;

/// OpenAI Provider Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIConfig {
    pub api_key: String,
    pub organization_id: Option<String>,
    pub base_url: Option<String>,
    pub default_model: Option<String>,
}

impl Default for OpenAIConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            organization_id: None,
            base_url: Some("https://api.openai.com/v1".to_string()),
            default_model: None,
        }
    }
}

/// OpenAI Provider
#[derive(Debug)]
pub struct OpenAIProvider {
    config: OpenAIConfig,
    client: reqwest::Client,
}

impl OpenAIProvider {
    /// Create a new OpenAI provider
    pub async fn new(api_key: impl Into<String>) -> Result<Self> {
        let client = reqwest::Client::new();

        Ok(Self {
            config: OpenAIConfig {
                api_key: api_key.into(),
                ..Default::default()
            },
            client,
        })
    }

    /// Create with custom configuration
    pub fn with_config(config: OpenAIConfig) -> Result<Self> {
        let client = reqwest::Client::new();

        if config.api_key.is_empty() {
            return Err(LLMError::ConfigurationError {
                message: "OpenAI API key is required".to_string(),
                field: Some("api_key".to_string()),
            });
        }

        Ok(Self { config, client })
    }

    /// Send a streaming request and hand back the raw response.
    ///
    /// One sender for both stream shapes, so a change to authentication, the
    /// body or the headers cannot be applied to one path and missed on the other.
    async fn send_streaming_request(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<(reqwest::Response, String)> {
        let model = if request.model.is_empty() {
            self.config.default_model.clone().ok_or_else(|| {
                LLMError::ConfigurationError {
                    message: "No model configured for OpenAIProvider — set default_model in \
                              OpenAIConfig or pass a model in the request"
                        .to_string(),
                    field: Some("default_model".to_string()),
                }
            })?
        } else {
            request.model.clone()
        };
        let messages: Vec<Value> = request
            .messages
            .iter()
            .map(Self::serialize_message)
            .collect();
        let body = crate::wire::build_chat_request(crate::wire::ChatRequestParts {
            model: &model,
            messages: &messages,
            request: &request,
            stream: true,
            tools: None,
            provider_params: None,
        })?;

        let mut request_builder = self
            .client
            .post(&format!("{}/chat/completions", self.get_base_url()))
            .header("Content-Type", "application/json");
        if let Some(org) = &self.config.organization_id {
            request_builder = request_builder.header("OpenAI-Organization", org);
        }
        let response = request_builder
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
            let body = response.text().await.unwrap_or_default();
            return Err(LLMError::HttpError {
                message: format!("OpenAI API error: {status}"),
                status_code: Some(status.as_u16()),
                body: Some(body),
            });
        }

        Ok((response, model))
    }

    fn get_base_url(&self) -> &str {
        self.config
            .base_url
            .as_deref()
            .unwrap_or("https://api.openai.com/v1")
    }

    /// Serialize a message for the wire.
    ///
    /// Delegates to the protocol-level serializer so this provider and
    /// OpenRouter cannot drift apart -- the two speak the same dialect.
    fn serialize_message(msg: &ChatMessage) -> Value {
        msg.to_openai_json()
    }

    fn parse_assistant_message(msg: &Value) -> ChatMessage {
        let content = msg
            .get("content")
            .and_then(|c| c.as_str())
            .map(|s| s.to_string());

        let tool_calls = msg
            .get("tool_calls")
            .and_then(|tc| tc.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|tc| {
                        let id = tc.get("id").and_then(|v| v.as_str()).map(|s| s.to_string());
                        let func = tc.get("function")?;
                        let name = func.get("name")?.as_str()?.to_string();
                        let arguments = func
                            .get("arguments")
                            .and_then(|a| a.as_str())
                            .unwrap_or("{}")
                            .to_string();
                        Some(ToolCall::Function(FunctionCall::Custom(
                            CustomFunctionCall {
                                id,
                                name,
                                arguments,
                            },
                        )))
                    })
                    .collect::<Vec<_>>()
            });

        let tool_calls = tool_calls.filter(|t| !t.is_empty());

        ChatMessage::Assistant {
            content,
            tool_calls,
            name: None,
        }
    }

    async fn send_chat_request(&self, body: Value) -> Result<Value> {
        let mut request = self
            .client
            .post(&format!("{}/chat/completions", self.get_base_url()))
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .header("Content-Type", "application/json");
        // §19: the organization header must reach *every* OpenAI call, not only
        // the model listing. It was applied to `/models` and silently dropped on
        // every chat request, so a configured organization only worked for
        // discovery.
        if let Some(org) = &self.config.organization_id {
            request = request.header("OpenAI-Organization", org);
        }
        let response = request
            .json(&body)
            .send()
            .await
            .map_err(|e| LLMError::HttpError {
                message: e.to_string(),
                status_code: None,
                body: None,
            })?;

        if !response.status().is_success() {
            let status = response.status();
            let body_text = response.text().await.ok();
            return Err(LLMError::HttpError {
                message: format!("OpenAI API error: {}", status),
                status_code: Some(status.as_u16()),
                body: body_text,
            });
        }

        response
            .json()
            .await
            .map_err(|e| LLMError::SerializationError {
                message: e.to_string(),
                context: Some("Failed to parse OpenAI response".to_string()),
            })
    }

    fn build_response(json: Value, request_model: &str) -> ChatCompletionResponse {
        let id = json["id"].as_str().unwrap_or("unknown").to_string();
        let created = json["created"].as_u64().unwrap_or(0);
        let model = json["model"].as_str().unwrap_or(request_model).to_string();

        let choices: Vec<Choice> = json["choices"]
            .as_array()
            .unwrap_or(&vec![])
            .iter()
            .enumerate()
            .map(|(idx, choice)| {
                let msg = &choice["message"];
                let message = Self::parse_assistant_message(msg);

                Choice {
                    index: idx as u32,
                    message,
                    finish_reason: choice["finish_reason"].as_str().map(|s| s.to_string()),
                    logprobs: None,
                }
            })
            .collect();

        let usage = json["usage"].as_object().map(|u| Usage {
            prompt_tokens: u["prompt_tokens"].as_u64().unwrap_or(0) as u32,
            completion_tokens: u["completion_tokens"].as_u64().unwrap_or(0) as u32,
            total_tokens: u["total_tokens"].as_u64().unwrap_or(0) as u32,
        });

        ChatCompletionResponse {
            id,
            object: "chat.completion".to_string(),
            created,
            model: model.clone(),
            choices,
            usage,
            system_fingerprint: json
                .get("system_fingerprint")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
        }
    }

    /// Fetch models from the live OpenAI API.
    async fn fetch_models_live(&self) -> Result<Vec<crate::types::FullModelInfo>> {
        let mut req = self
            .client
            .get(&format!("{}/models", self.get_base_url()))
            .header("Content-Type", "application/json");

        if !self.config.api_key.is_empty() {
            req = req.header("Authorization", format!("Bearer {}", self.config.api_key));
        }
        if let Some(org) = &self.config.organization_id {
            req = req.header("OpenAI-Organization", org);
        }

        let resp = req
            .send()
            .await
            .map_err(|e| crate::error::LLMError::HttpError {
                message: format!("Failed to fetch OpenAI models: {}", e),
                status_code: None,
                body: None,
            })?;

        if !resp.status().is_success() {
            return Err(crate::error::LLMError::HttpError {
                message: format!("OpenAI /models returned {}", resp.status()),
                status_code: Some(resp.status().as_u16()),
                body: resp.text().await.ok(),
            });
        }

        let json: serde_json::Value =
            resp.json()
                .await
                .map_err(|e| crate::error::LLMError::SerializationError {
                    message: e.to_string(),
                    context: Some("Failed to parse models response".to_string()),
                })?;
        let data =
            json["data"]
                .as_array()
                .ok_or_else(|| crate::error::LLMError::SerializationError {
                    message: "Missing 'data' array".to_string(),
                    context: None,
                })?;

        let mut models = self.known_models_lookup();
        let known_ids: Vec<_> = models.iter().map(|m| m.id.clone()).collect();

        for m in data {
            let id = match m["id"].as_str() {
                Some(id) => id.to_string(),
                None => continue,
            };
            if known_ids.contains(&id) {
                continue;
            }

            let id_lower = id.to_lowercase();
            let has_vision = id_lower.contains("vision") || id_lower.contains("gpt-4o");
            let has_tools = id_lower.contains("gpt-4")
                || id_lower.contains("gpt-3.5")
                || id_lower.contains("o1")
                || id_lower.contains("o3");
            let ctx: u32 = if id_lower.contains("128k") || id_lower.contains("gpt-4o") {
                128000
            } else {
                16384
            };
            let mut strengths = Vec::new();
            if id_lower.contains("o1") || id_lower.contains("o3") {
                strengths.push("reasoning".to_string());
            }
            if has_vision {
                strengths.push("vision".to_string());
            }
            let mut input_mods = vec!["text".to_string()];
            if has_vision {
                input_mods.push("image".to_string());
            }

            let id_clone = id.clone();
            models.push(crate::types::FullModelInfo {
                id,
                name: id_clone,
                provider: "openai".to_string(),
                description: Some("OpenAI model".to_string()),
                context_window: ctx,
                max_output_tokens: ctx,
                capabilities: crate::types::ModelCapabilities {
                    function_calling: has_tools,
                    vision: has_vision,
                    streaming: true,
                    json_mode: has_tools,
                    caching: false,
                    max_tokens: ctx,
                    context_window: ctx,
                    input_modalities: input_mods,
                    output_modalities: vec!["text".to_string()],
                    strengths,
                                        source: crate::types::CapabilitySource::Heuristic,
                    },
                pricing: None,
                created: m["created"].as_u64().unwrap_or(0),
                available: true,
                                supported_parameters: Default::default(),
                });
        }
        Ok(models)
    }

    /// Fallback model descriptors, when the live listing is unavailable.
    ///
    /// Built from the versioned catalog rather than a list literal in the
    /// provider, so a price change is a data change. Only OpenAI's own models
    /// are included: a fallback for a model this provider does not serve would
    /// be exactly the unrelated-price substitution the brief forbids.
    fn known_models_lookup(&self) -> Vec<crate::types::FullModelInfo> {
        crate::pricing::catalog()
            .values()
            .filter(|descriptor| {
                descriptor.id.starts_with("gpt-") || descriptor.id.starts_with("o1")
            })
            .map(|descriptor| crate::types::FullModelInfo {
                id: descriptor.id.to_string(),
                name: descriptor.id.to_string(),
                provider: "openai".to_string(),
                description: None,
                context_window: descriptor.context_window,
                max_output_tokens: descriptor.max_output_tokens,
                capabilities: descriptor.capabilities.clone(),
                pricing: descriptor.pricing.clone(),
                created: 0,
                available: true,
                supported_parameters: Default::default(),
            })
            .collect()
    }
}

#[async_trait]
impl LLMProvider for OpenAIProvider {
    fn name(&self) -> &str {
        "openai"
    }

    fn supported_models(&self) -> Vec<String> {
        // Cannot hardcode — the OpenAI Chat Completions API protocol is used by
        // many providers (LM Studio, LocalAI, vLLM, etc.). Live discovery only.
        vec![]
    }

    fn supports_function_calling(&self, model: &str) -> bool {
        match model {
            "gpt-4" | "gpt-4-turbo" | "gpt-4o" | "gpt-4o-mini" | "gpt-3.5-turbo" | "o1"
            | "o1-mini" | "o1-preview" => true,
            _ => self.supports_model(model),
        }
    }

    /// Fetch models from live API. Falls back to empty list on error.
    async fn list_models(&self) -> Result<Vec<crate::types::FullModelInfo>> {
        match self.fetch_models_live().await {
            Ok(models) if !models.is_empty() => Ok(models),
            _ => Ok(vec![]),
        }
    }

    async fn chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse> {
        let start = std::time::Instant::now();

        // Resolve the model exactly as the streaming and tool-calling paths do.
        // This path previously sent `request.model` raw, so a request that
        // relied on `default_model` sent `"model": ""` and the failure surfaced
        // as an opaque provider error rather than a clear one.
        let model = if request.model.is_empty() {
            self.config.default_model.clone().ok_or_else(|| {
                LLMError::ConfigurationError {
                    message: "No model configured for OpenAIProvider — set default_model in \
                              OpenAIConfig or pass a model in the request"
                        .to_string(),
                    field: Some("default_model".to_string()),
                }
            })?
        } else {
            request.model.clone()
        };

        let messages: Vec<Value> = request
            .messages
            .iter()
            .map(Self::serialize_message)
            .collect();

        // One builder for every path; see `crate::wire`.
        let body = crate::wire::build_chat_request(crate::wire::ChatRequestParts {
            model: &model,
            messages: &messages,
            request: &request,
            stream: false,
            tools: None,
            provider_params: None,
        })?;

        let json = self.send_chat_request(body).await?;

        let model = json["model"].as_str().unwrap_or(&request.model).to_string();
        let response = Self::build_response(json, &request.model);

        let duration = start.elapsed();
        tracing::info!(
            target: "llm",
            provider = "openai",
            model = model,
            duration_ms = duration.as_millis(),
            "LLM request completed"
        );

        Ok(response)
    }

    /// Lossless structured streaming.
    ///
    /// The previous implementation read `choices[0].delta.content` and skipped
    /// everything else. For a response whose deltas are tool calls, that meant
    /// *no* chunks were emitted at all and the caller received a single
    /// synthesized `finish_reason: "stop"` — an agentic loop would see a clean
    /// completion and no tools.
    ///
    /// It also fabricated `id` and `created` on every chunk and appended a final
    /// `stop` event regardless of what the provider actually said.
    async fn chat_completion_stream_events(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<Pin<Box<dyn futures::Stream<Item = Result<ChatStreamEvent>> + Send>>> {
        let (response, _) = self.send_streaming_request(request).await?;
        Ok(crate::stream::parse_sse_stream(response))
    }


    /// Legacy text-only streaming, adapted from the structured events.
    ///
    /// See [`crate::stream::adapt_legacy`]: the legacy `Chunk` shape cannot
    /// represent an indexed tool-call fragment, so a consumer needing parallel
    /// tool calls must use `chat_completion_stream_events`.
    async fn chat_completion_stream(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<Pin<Box<dyn futures::Stream<Item = Result<Chunk>> + Send>>> {
        let (response, _) = self.send_streaming_request(request).await?;
        Ok(crate::stream::adapt_legacy(crate::stream::parse_sse_stream(response)))
    }

    async fn chat_completion_with_functions(
        &self,
        request: ChatCompletionRequest,
        functions: Vec<FunctionDefinition>,
    ) -> Result<ChatCompletionResponse> {
        let start = std::time::Instant::now();

        let messages: Vec<Value> = request
            .messages
            .iter()
            .map(Self::serialize_message)
            .collect();

        let tools: Vec<Value> = functions
            .iter()
            .map(|f| {
                let mut func_obj = json!({
                    "name": f.name,
                });
                if let Some(desc) = &f.description {
                    func_obj["description"] = json!(desc);
                }
                if let Some(params) = &f.parameters {
                    func_obj["parameters"] = params.clone();
                }
                json!({
                    "type": "function",
                    "function": func_obj,
                })
            })
            .collect();

        let model = if request.model.is_empty() {
            self.config.default_model.clone().ok_or_else(|| {
                LLMError::ConfigurationError {
                    message: "No model configured for OpenAIProvider — set default_model in \
                              OpenAIConfig or pass a model in the request"
                        .to_string(),
                    field: Some("default_model".to_string()),
                }
            })?
        } else {
            request.model.clone()
        };

        // The same builder again, now with the serialized tools.
        let body = crate::wire::build_chat_request(crate::wire::ChatRequestParts {
            model: &model,
            messages: &messages,
            request: &request,
            stream: false,
            tools: Some(&tools),
            provider_params: None,
        })?;

        let json = self.send_chat_request(body).await?;

        let model = json["model"].as_str().unwrap_or(&request.model).to_string();
        let response = Self::build_response(json, &request.model);

        let duration = start.elapsed();
        tracing::info!(
            target: "llm",
            provider = "openai",
            model = model,
            duration_ms = duration.as_millis(),
            "LLM function calling request completed"
        );

        Ok(response)
    }

    async fn rate_limit_status(&self) -> Result<RateLimitStatus> {
        Ok(RateLimitStatus {
            remaining: None,
            limit: None,
            tokens_remaining: None,
            tokens_limit: None,
            reset_at: None,
            retry_after: None,
        })
    }

    async fn estimate_cost(&self, request: &ChatCompletionRequest) -> Result<CostEstimate> {
        let input_cost_per_million = match request.model.as_str() {
            m if m.starts_with("gpt-4") => 30.0,
            m if m.starts_with("gpt-4o") => 5.0,
            m if m.starts_with("gpt-4o-mini") => 0.15,
            m if m.starts_with("gpt-3.5-turbo") => 0.5,
            m if m.starts_with("o1") => 15.0,
            m if m.starts_with("o1-mini") => 3.0,
            _ => 30.0,
        };

        let output_cost_per_million = match request.model.as_str() {
            m if m.starts_with("gpt-4") => 60.0,
            m if m.starts_with("gpt-4o") => 15.0,
            m if m.starts_with("gpt-4o-mini") => 0.6,
            m if m.starts_with("gpt-3.5-turbo") => 1.5,
            m if m.starts_with("o1") => 60.0,
            m if m.starts_with("o1-mini") => 12.0,
            _ => 60.0,
        };

        let estimated_prompt_tokens = request
            .messages
            .iter()
            .map(|m| m.content().map(|c| c.len() / 4).unwrap_or(10))
            .sum::<usize>() as u32;

        let estimated_completion_tokens = request.max_tokens.unwrap_or(1000);

        let input_cost = estimated_prompt_tokens as f64 * input_cost_per_million / 1_000_000.0;
        let output_cost =
            estimated_completion_tokens as f64 * output_cost_per_million / 1_000_000.0;

        Ok(CostEstimate {
            input_cost,
            output_cost,
            total_cost: input_cost + output_cost,
            input_tokens: estimated_prompt_tokens,
            output_tokens: estimated_completion_tokens,
            model: request.model.clone(),
            provider: "openai".to_string(),
        })
    }

    fn config(&self) -> &ProviderConfig {
        unimplemented!()
    }
}
