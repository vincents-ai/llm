//! OpenRouter Provider Implementation
//!
//! This module provides the OpenRouter provider implementation with credit tracking,
//! provider ranking/selection, referrer support, and JWT authentication.

use async_trait::async_trait;
use futures::TryStreamExt;
use serde::{Deserialize, Serialize};
use std::pin::Pin;
use serde_json::json;

use crate::types::*;
use crate::error::{LLMError, Result};
use crate::config::ProviderConfig;
use crate::provider::LLMProvider;
use crate::RateLimitStatus;
use crate::CostEstimate;

/// OpenRouter Provider Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenRouterConfig {
    pub api_key: String,
    pub referrer: Option<String>,
    pub default_route: Option<String>,
    pub default_model: Option<String>,
    pub jwt_token: Option<String>,
}

impl Default for OpenRouterConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            referrer: None,
            default_route: None,
            default_model: None,
            jwt_token: None,
        }
    }
}

/// OpenRouter Provider
#[derive(Debug)]
pub struct OpenRouterProvider {
    config: OpenRouterConfig,
    client: reqwest::Client,
}

impl OpenRouterProvider {
    /// Create a new OpenRouter provider
    pub async fn new(api_key: impl Into<String>) -> Result<Self> {
        let client = reqwest::Client::new();
        
        Ok(Self {
            config: OpenRouterConfig {
                api_key: api_key.into(),
                ..Default::default()
            },
            client,
        })
    }

    /// Create with custom configuration
    pub fn with_config(config: OpenRouterConfig) -> Result<Self> {
        if config.api_key.is_empty() && config.jwt_token.as_ref().map_or(true, |s| s.is_empty()) {
            return Err(LLMError::ConfigurationError {
                message: "OpenRouter API key or JWT token is required".to_string(),
                field: Some("api_key".to_string()),
            });
        }

        Ok(Self {
            config,
            client: reqwest::Client::new(),
        })
    }

    /// Set referrer for credit tracking
    pub fn with_referrer(mut self, referrer: impl Into<String>) -> Self {
        self.config.referrer = Some(referrer.into());
        self
    }

    /// Set default route for provider routing
    pub fn with_route(mut self, route: impl Into<String>) -> Self {
        self.config.default_route = Some(route.into());
        self
    }

    fn get_default_model(&self) -> Result<&str> {
        self.config.default_model.as_deref().ok_or_else(|| LLMError::ConfigurationError {
            message: "No model configured for OpenRouterProvider — set default_model in OpenRouterConfig or pass a model in the request".to_string(),
            field: Some("default_model".to_string()),
        })
    }

    fn get_base_url(&self) -> &str {
        "https://openrouter.ai/api/v1"
    }
}

#[async_trait]
impl LLMProvider for OpenRouterProvider {
    fn name(&self) -> &str {
        "openrouter"
    }

    fn supported_models(&self) -> Vec<String> {
        // We no longer hardcode — list_models() hits the live API.
        // Keep a small fallback list for offline / error scenarios.
        vec![
            "openai/gpt-4o".to_string(),
            "openai/gpt-4o-mini".to_string(),
            "anthropic/claude-3.5-sonnet".to_string(),
            "anthropic/claude-3-opus".to_string(),
            "google/gemini-1.5-pro".to_string(),
            "deepseek/deepseek-chat".to_string(),
            "meta-llama/llama-3.1-70b".to_string(),
            "mistralai/mistral-7b".to_string(),
            "cohere/command-r".to_string(),
        ]
    }

    /// Fetch all models from OpenRouter's live `/api/v1/models` endpoint.
    /// No authentication required.
    async fn list_models(&self) -> Result<Vec<FullModelInfo>> {
        let resp = self.client
            .get(&format!("{}/models", self.get_base_url()))
            .header("Content-Type", "application/json")
            .send()
            .await
            .map_err(|e| LLMError::HttpError {
                message: format!("Failed to fetch OpenRouter models: {}", e),
                status_code: None,
                body: None,
            })?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.ok();
            return Err(LLMError::HttpError {
                message: format!("OpenRouter /models returned {}: {}", status, body.clone().unwrap_or_default()),
                status_code: Some(status.as_u16()),
                body,
            });
        }

        let json: serde_json::Value = resp.json().await.map_err(|e| LLMError::SerializationError {
            message: e.to_string(),
            context: Some("Failed to parse OpenRouter models response".to_string()),
        })?;

        let data = json["data"].as_array().ok_or_else(|| LLMError::SerializationError {
            message: "Missing 'data' array in models response".to_string(),
            context: None,
        })?;

        let models: Vec<FullModelInfo> = data.iter().map(|m| {
            let id = m["id"].as_str().unwrap_or("").to_string();
            let name = m["name"].as_str().unwrap_or(&id).to_string();
            let description = m["description"].as_str().map(String::from);
            let context_length = m["context_length"].as_u64().unwrap_or(0) as u32;
            let created = m["created"].as_u64().unwrap_or(0);

            // Parse architecture / modalities
            let arch = &m["architecture"];
            let input_modalities: Vec<String> = arch["input_modalities"]
                .as_array()
                .map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect())
                .unwrap_or_default();
            let output_modalities: Vec<String> = arch["output_modalities"]
                .as_array()
                .map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect())
                .unwrap_or_default();
            let has_vision = input_modalities.iter().any(|m| m == "image");

            // Parse supported parameters → capabilities
            let params: Vec<String> = m["supported_parameters"]
                .as_array()
                .map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect())
                .unwrap_or_default();
            let has_tools = params.iter().any(|p| p == "tools" || p == "tool_choice");
            let has_json = params.iter().any(|p| p == "response_format" || p == "structured_outputs");

            // Parse pricing
            let pricing_obj = &m["pricing"];
            let prompt_price: f64 = pricing_obj["prompt"].as_str()
                .and_then(|s| s.parse().ok())
                .or_else(|| pricing_obj["prompt"].as_f64())
                .unwrap_or(0.0);
            let completion_price: f64 = pricing_obj["completion"].as_str()
                .and_then(|s| s.parse().ok())
                .or_else(|| pricing_obj["completion"].as_f64())
                .unwrap_or(0.0);
            // OpenRouter pricing is per-token, convert to per-1K tokens
            let prompt_per_1k = prompt_price * 1000.0;
            let completion_per_1k = completion_price * 1000.0;
            let is_free = prompt_price == 0.0 && completion_price == 0.0;

            // Max completion tokens
            let max_output = m["top_provider"]["max_completion_tokens"]
                .as_u64()
                .unwrap_or(context_length as u64) as u32;

            // Derive strengths from model id/name heuristics
            let id_lower = id.to_lowercase();
            let mut strengths = Vec::new();
            if id_lower.contains("coder") || id_lower.contains("code") || id_lower.contains("deepseek") {
                strengths.push("coding".to_string());
            }
            if id_lower.contains("reason") || id_lower.contains("o1") || id_lower.contains("o3") || id_lower.contains("think") {
                strengths.push("reasoning".to_string());
            }
            if id_lower.contains("math") {
                strengths.push("math".to_string());
            }
            if id_lower.contains("vision") || has_vision {
                strengths.push("vision".to_string());
            }

            FullModelInfo {
                id,
                name,
                provider: "openrouter".to_string(),
                description,
                context_window: context_length,
                max_output_tokens: max_output,
                capabilities: ModelCapabilities {
                    function_calling: has_tools,
                    vision: has_vision,
                    streaming: true, // OpenRouter supports streaming for all models
                    json_mode: has_json,
                    caching: false,
                    max_tokens: max_output,
                    context_window: context_length,
                    input_modalities,
                    output_modalities,
                    strengths,
                },
                pricing: Some(ModelPricing {
                    prompt_tokens: prompt_per_1k,
                    completion_tokens: completion_per_1k,
                    image_tokens: None,
                    is_free,
                }),
                created,
                available: true,
            }
        }).collect();

        Ok(models)
    }

    async fn chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse> {
        let model = if request.model.is_empty() {
            self.get_default_model()?.to_string()
        } else {
            request.model.clone()
        };

        // Convert messages to OpenRouter format
        let messages: Vec<serde_json::Value> = request.messages
            .iter()
            .map(|msg| {
                let role = match msg.role() {
                    MessageRole::System => "system",
                    MessageRole::User => "user",
                    MessageRole::Assistant => "assistant",
                    MessageRole::Function | MessageRole::Tool => "user",
                };
                
                let content = msg.content()
                    .map(|s| s.to_string())
                    .unwrap_or_default();
                
                json!({
                    "role": role,
                    "content": content
                })
            })
            .collect();

        let mut body = json!({
            "model": model,
            "messages": messages,
        });

        if let Some(obj) = body.as_object_mut() {
            if let Some(max_tokens) = request.max_tokens {
                obj.insert("max_tokens".to_string(), json!(max_tokens));
            }
            if let Some(temp) = request.temperature {
                obj.insert("temperature".to_string(), json!(temp));
            }
            if let Some(top_p) = request.top_p {
                obj.insert("top_p".to_string(), json!(top_p));
            }
            if let Some(n) = request.n {
                obj.insert("n".to_string(), json!(n));
            }
            if !request.stop.is_empty() {
                obj.insert("stop".to_string(), json!(request.stop));
            }
            if let Some(presence) = request.presence_penalty {
                obj.insert("presence_penalty".to_string(), json!(presence));
            }
            if let Some(freq) = request.frequency_penalty {
                obj.insert("frequency_penalty".to_string(), json!(freq));
            }
        }

        let mut request_builder = self.client
            .post(&format!("{}/chat/completions", self.get_base_url()))
            .header("Content-Type", "application/json");

        // Add authentication header
        if let Some(jwt) = &self.config.jwt_token {
            request_builder = request_builder.header("Authorization", format!("Bearer {}", jwt));
        } else {
            request_builder = request_builder.header("Authorization", format!("Bearer {}", self.config.api_key));
        }

        // Add referrer for credit tracking
        if let Some(referrer) = &self.config.referrer {
            request_builder = request_builder.header("HTTP-Referer", referrer);
        }

        // Add route for provider routing
        if let Some(route) = &self.config.default_route {
            request_builder = request_builder.header("OpenRouter-Router", route);
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

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.ok();
            return Err(LLMError::HttpError {
                message: format!("OpenRouter API error: {}", status),
                status_code: Some(status.as_u16()),
                body,
            });
        }

        let json: serde_json::Value = response.json().await
            .map_err(|e| LLMError::SerializationError {
                message: e.to_string(),
                context: Some("Failed to parse OpenRouter response".to_string()),
            })?;

        let id = json["id"].as_str().unwrap_or("unknown").to_string();
        let created = json["created"].as_u64().unwrap_or(0);
        let model = json["model"].as_str().unwrap_or(&model).to_string();

        let choices: Vec<Choice> = json["choices"]
            .as_array()
            .unwrap_or(&vec![])
            .iter()
            .enumerate()
            .map(|(idx, choice)| {
                let msg = &choice["message"];
                let role = msg["role"].as_str().unwrap_or("assistant");
                let content = msg["content"].as_str().unwrap_or("");

                let message = match role {
                    "system" => ChatMessage::system(content),
                    "user" => ChatMessage::user(content),
                    _ => ChatMessage::assistant(content),
                };

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

        Ok(ChatCompletionResponse {
            id,
            object: "chat.completion".to_string(),
            created,
            model,
            choices,
            usage,
            system_fingerprint: json.get("system_fingerprint").and_then(|v| v.as_str()).map(|s| s.to_string()),
        })
    }

    async fn chat_completion_stream(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<Pin<Box<dyn futures::Stream<Item = Result<Chunk>> + Send>>> {
        let model = if request.model.is_empty() {
            self.get_default_model()?.to_string()
        } else {
            request.model.clone()
        };

        let messages: Vec<serde_json::Value> = request.messages
            .iter()
            .map(|msg| {
                let role = match msg.role() {
                    MessageRole::System => "system",
                    MessageRole::User => "user",
                    MessageRole::Assistant => "assistant",
                    MessageRole::Function | MessageRole::Tool => "user",
                };
                
                let content = msg.content()
                    .map(|s| s.to_string())
                    .unwrap_or_default();
                
                serde_json::json!({
                    "role": role,
                    "content": content
                })
            })
            .collect();

        let mut body = serde_json::json!({
            "model": model,
            "messages": messages,
            "stream": true,
        });

        if let Some(obj) = body.as_object_mut() {
            if let Some(max_tokens) = request.max_tokens {
                obj.insert("max_tokens".to_string(), serde_json::json!(max_tokens));
            }
            if let Some(temp) = request.temperature {
                obj.insert("temperature".to_string(), serde_json::json!(temp));
            }
            if let Some(top_p) = request.top_p {
                obj.insert("top_p".to_string(), serde_json::json!(top_p));
            }
        }

        let mut request_builder = self.client
            .post(&format!("{}/chat/completions", self.get_base_url()))
            .header("Content-Type", "application/json");

        if let Some(jwt) = &self.config.jwt_token {
            request_builder = request_builder.header("Authorization", format!("Bearer {}", jwt));
        } else {
            request_builder = request_builder.header("Authorization", format!("Bearer {}", self.config.api_key));
        }

        if let Some(referrer) = &self.config.referrer {
            request_builder = request_builder.header("HTTP-Referer", referrer);
        }

        if let Some(route) = &self.config.default_route {
            request_builder = request_builder.header("OpenRouter-Router", route);
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

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.ok();
            return Err(LLMError::HttpError {
                message: format!("OpenRouter API error: {}", status),
                status_code: Some(status.as_u16()),
                body,
            });
        }

        let model_clone = model.clone();

        let stream = async_stream::stream! {
            let mut buffer = String::new();
            let mut chunk_count = 0;

            let mut stream = response.bytes_stream();
            
            while let Ok(Some(bytes)) = stream.try_next().await {
                let bytes_vec = bytes.to_vec();
                let chunk_str = String::from_utf8_lossy(&bytes_vec);
                buffer.push_str(&chunk_str);
                
                loop {
                    if let Some(pos) = buffer.find("\n") {
                        let line = buffer[..pos].to_string();
                        buffer = buffer[pos + 1..].to_string();
                        
                        if line.starts_with("data: ") {
                            let data = &line[6..];
                            if data == "[DONE]" {
                                continue;
                            }
                            match serde_json::from_str::<serde_json::Value>(data) {
                                Ok(json) => {
                                    let delta = &json["choices"][0]["delta"]["content"];
                                    if delta.is_null() || !delta.is_string() {
                                        continue;
                                    }
                                    
                                    let content = delta.as_str().unwrap_or("");
                                    if content.is_empty() {
                                        continue;
                                    }

                                    let finish_reason = json["choices"][0]["finish_reason"]
                                        .as_str()
                                        .map(|s| s.to_string());

                                    let chunk = Chunk::Completion(ChunkCompletion {
                                        id: format!("or_{}", chrono::Utc::now().timestamp_millis()),
                                        object: "chat.completion.chunk".to_string(),
                                        created: chrono::Utc::now().timestamp() as u64,
                                        model: model_clone.clone(),
                                        choices: vec![ChoiceDelta {
                                            index: chunk_count,
                                            delta: Some(ChatMessage::assistant(content)),
                                            finish_reason,
                                        }],
                                    });

                                    yield Ok(chunk);
                                    chunk_count += 1;
                                }
                                Err(_) => continue,
                            }
                        }
                    } else {
                        break;
                    }
                }
            }

            let final_chunk = Chunk::Completion(ChunkCompletion {
                id: format!("or_{}", chrono::Utc::now().timestamp_millis()),
                object: "chat.completion.chunk".to_string(),
                created: chrono::Utc::now().timestamp() as u64,
                model: model_clone,
                choices: vec![ChoiceDelta {
                    index: chunk_count,
                    delta: None,
                    finish_reason: Some("stop".to_string()),
                }],
            });
            yield Ok(final_chunk);
        };

        Ok(Box::pin(stream))
    }

    async fn chat_completion_with_functions(
        &self,
        request: ChatCompletionRequest,
        functions: Vec<FunctionDefinition>,
    ) -> Result<ChatCompletionResponse> {
        let model = if request.model.is_empty() {
            self.get_default_model()?.to_string()
        } else {
            request.model.clone()
        };

        let tools: Vec<serde_json::Value> = functions.iter().map(|f| {
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
                "function": func_obj
            })
        }).collect();

        let messages: Vec<serde_json::Value> = request.messages
            .iter()
            .map(|msg| {
                match msg {
                    ChatMessage::System { content, name } => {
                        let mut obj = json!({"role": "system", "content": content});
                        if let Some(n) = name { obj["name"] = json!(n); }
                        obj
                    }
                    ChatMessage::User { content, name } => {
                        let mut obj = json!({"role": "user", "content": content});
                        if let Some(n) = name { obj["name"] = json!(n); }
                        obj
                    }
                    ChatMessage::Assistant { content, tool_calls, name } => {
                        let mut obj = json!({"role": "assistant"});
                        if let Some(c) = content {
                            obj["content"] = json!(c);
                        } else {
                            obj["content"] = serde_json::Value::Null;
                        }
                        if let Some(n) = name { obj["name"] = json!(n); }
                        if let Some(tcs) = tool_calls {
                            if !tcs.is_empty() {
                                let or_tcs: Vec<serde_json::Value> = tcs.iter().filter_map(|tc| {
                                    if let ToolCall::Function(FunctionCall::Custom(c)) = tc {
                                        Some(json!({
                                            "id": c.id.as_deref().unwrap_or(""),
                                            "type": "function",
                                            "function": {
                                                "name": c.name,
                                                "arguments": c.arguments,
                                            }
                                        }))
                                    } else { None }
                                }).collect();
                                if !or_tcs.is_empty() { obj["tool_calls"] = json!(or_tcs); }
                            }
                        }
                        obj
                    }
                    ChatMessage::Tool { tool_call_id, content } => {
                        json!({
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "content": content,
                        })
                    }
                    ChatMessage::Function { name, arguments } => {
                        json!({"role": "function", "name": name, "content": arguments})
                    }
                }
            })
            .collect();

        let mut body = json!({
            "model": model,
            "messages": messages,
            "tools": tools,
        });

        if let Some(obj) = body.as_object_mut() {
            if let Some(max_tokens) = request.max_tokens {
                obj.insert("max_tokens".to_string(), json!(max_tokens));
            }
            if let Some(temp) = request.temperature {
                obj.insert("temperature".to_string(), json!(temp));
            }
            if let Some(top_p) = request.top_p {
                obj.insert("top_p".to_string(), json!(top_p));
            }
            for (key, value) in &request.extra_params {
                obj.insert(key.clone(), value.clone());
            }
        }

        let mut request_builder = self.client
            .post(&format!("{}/chat/completions", self.get_base_url()))
            .header("Content-Type", "application/json");

        if let Some(jwt) = &self.config.jwt_token {
            request_builder = request_builder.header("Authorization", format!("Bearer {}", jwt));
        } else {
            request_builder = request_builder.header("Authorization", format!("Bearer {}", self.config.api_key));
        }

        if let Some(referrer) = &self.config.referrer {
            request_builder = request_builder.header("HTTP-Referer", referrer);
        }

        if let Some(route) = &self.config.default_route {
            request_builder = request_builder.header("OpenRouter-Router", route);
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

        if !response.status().is_success() {
            let status = response.status();
            let resp_body = response.text().await.ok();
            return Err(LLMError::HttpError {
                message: format!("OpenRouter API error: {}", status),
                status_code: Some(status.as_u16()),
                body: resp_body,
            });
        }

        let json: serde_json::Value = response.json().await
            .map_err(|e| LLMError::SerializationError {
                message: e.to_string(),
                context: Some("Failed to parse OpenRouter response".to_string()),
            })?;

        let id = json["id"].as_str().unwrap_or("unknown").to_string();
        let created = json["created"].as_u64().unwrap_or(0);
        let resp_model = json["model"].as_str().unwrap_or(&model).to_string();

        let choices: Vec<Choice> = json["choices"]
            .as_array()
            .unwrap_or(&vec![])
            .iter()
            .enumerate()
            .map(|(idx, choice)| {
                let msg = &choice["message"];
                let content = msg.get("content")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());
                let tool_calls = msg.get("tool_calls")
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        arr.iter().filter_map(|tc| {
                            let tc_id = tc.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string();
                            let func = tc.get("function")?;
                            let tc_name = func.get("name").and_then(|v| v.as_str()).unwrap_or("").to_string();
                            let tc_args = func.get("arguments").and_then(|v| v.as_str()).unwrap_or("{}").to_string();
                            Some(ToolCall::Function(FunctionCall::Custom(CustomFunctionCall {
                                id: if tc_id.is_empty() { None } else { Some(tc_id) },
                                name: tc_name,
                                arguments: tc_args,
                            })))
                        }).collect::<Vec<_>>()
                    })
                    .filter(|v| !v.is_empty());

                Choice {
                    index: idx as u32,
                    message: ChatMessage::Assistant {
                        content,
                        tool_calls,
                        name: None,
                    },
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

        Ok(ChatCompletionResponse {
            id,
            object: "chat.completion".to_string(),
            created,
            model: resp_model,
            choices,
            usage,
            system_fingerprint: json.get("system_fingerprint").and_then(|v| v.as_str()).map(|s| s.to_string()),
        })
    }

    fn supports_function_calling(&self, _model: &str) -> bool {
        true
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
        let model = if request.model.is_empty() {
            self.get_default_model()?.to_string()
        } else {
            request.model.clone()
        };

        // OpenRouter pricing varies by model, use conservative estimates
        let input_cost_per_million = match model.as_str() {
            m if m.contains("gpt-4o") => 5.0,
            m if m.contains("gpt-4o-mini") => 0.15,
            m if m.contains("claude-3.5-sonnet") => 3.0,
            m if m.contains("claude-3-opus") => 15.0,
            m if m.contains("gemini") => 1.25,
            m if m.contains("llama") => 0.5,
            m if m.contains("deepseek") => 0.14,
            m if m.contains("mistral") => 0.25,
            _ => 10.0,
        };

        let output_cost_per_million = match model.as_str() {
            m if m.contains("gpt-4o") => 15.0,
            m if m.contains("gpt-4o-mini") => 0.6,
            m if m.contains("claude-3.5-sonnet") => 15.0,
            m if m.contains("claude-3-opus") => 75.0,
            m if m.contains("gemini") => 5.0,
            m if m.contains("llama") => 1.5,
            m if m.contains("deepseek") => 0.28,
            m if m.contains("mistral") => 0.25,
            _ => 30.0,
        };

        let estimated_prompt_tokens = request.messages.iter()
            .map(|m| m.content().map(|c| c.len() / 4).unwrap_or(10))
            .sum::<usize>() as u32;

        let estimated_completion_tokens = request.max_tokens.unwrap_or(1000);

        let input_cost = estimated_prompt_tokens as f64 * input_cost_per_million / 1_000_000.0;
        let output_cost = estimated_completion_tokens as f64 * output_cost_per_million / 1_000_000.0;

        Ok(CostEstimate {
            input_cost,
            output_cost,
            total_cost: input_cost + output_cost,
            input_tokens: estimated_prompt_tokens,
            output_tokens: estimated_completion_tokens,
            model,
            provider: "openrouter".to_string(),
        })
    }

    fn config(&self) -> &ProviderConfig {
        unimplemented!()
    }
}
