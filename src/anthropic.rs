//! Anthropic Claude Provider Implementation
//!
//! This module provides the Anthropic provider implementation using direct HTTP.

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

/// Anthropic Provider Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicConfig {
    pub api_key: String,
    pub default_model: Option<String>,
    /// Base URL for the Anthropic API (or compatible proxy).
    /// Defaults to "https://api.anthropic.com/v1".
    pub base_url: Option<String>,
}

impl Default for AnthropicConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            default_model: None,
            base_url: None,
        }
    }
}

/// Anthropic Provider
#[derive(Debug)]
pub struct AnthropicProvider {
    config: AnthropicConfig,
    client: reqwest::Client,
}

impl AnthropicProvider {
    /// Create a new Anthropic provider
    pub async fn new(api_key: impl Into<String>) -> Result<Self> {
        let client = reqwest::Client::new();
        
        Ok(Self {
            config: AnthropicConfig {
                api_key: api_key.into(),
                ..Default::default()
            },
            client,
        })
    }

    /// Create with custom configuration
    pub fn with_config(config: AnthropicConfig) -> Result<Self> {
        if config.api_key.is_empty() {
            return Err(LLMError::ConfigurationError {
                message: "Anthropic API key is required".to_string(),
                field: Some("api_key".to_string()),
            });
        }

        Ok(Self {
            config,
            client: reqwest::Client::new(),
        })
    }

    fn get_default_model(&self) -> Result<&str> {
        self.config.default_model.as_deref().ok_or_else(|| LLMError::ConfigurationError {
            message: "No model configured for AnthropicProvider — set default_model in AnthropicConfig or pass a model in the request".to_string(),
            field: Some("default_model".to_string()),
        })
    }

    fn get_base_url(&self) -> &str {
        self.config.base_url.as_deref()
            .unwrap_or("https://api.anthropic.com/v1")
    }

    /// Create with a custom endpoint (for Anthropic-compatible proxies such as Z.ai).
    pub async fn with_endpoint(api_key: impl Into<String>, base_url: impl Into<String>) -> Result<Self> {
        Ok(Self {
            config: AnthropicConfig {
                api_key: api_key.into(),
                base_url: Some(base_url.into()),
                ..Default::default()
            },
            client: reqwest::Client::new(),
        })
    }

    fn convert_messages(&self, messages: &[ChatMessage]) -> (Option<String>, Vec<serde_json::Value>) {
        let mut system_prompt: Option<String> = None;
        let mut anthropic_messages: Vec<serde_json::Value> = Vec::new();

        for msg in messages {
            match msg {
                ChatMessage::System { content, .. } => {
                    system_prompt = Some(match system_prompt {
                        Some(existing) => format!("{}\n\n{}", existing, content),
                        None => content.clone(),
                    });
                }
                ChatMessage::User { content, .. } => {
                    let user_msg = json!({
                        "role": "user",
                        "content": content
                    });
                    if let Some(last) = anthropic_messages.last_mut() {
                        if last["role"] == "user" {
                            let existing = last["content"].as_str().unwrap_or("");
                            last["content"] = json!(format!("{}\n\n{}", existing, content));
                            continue;
                        }
                    }
                    anthropic_messages.push(user_msg);
                }
                ChatMessage::Assistant { content, tool_calls, .. } => {
                    if let Some(calls) = tool_calls {
                        if !calls.is_empty() {
                            let mut content_blocks: Vec<serde_json::Value> = Vec::new();
                            if let Some(text) = content {
                                if !text.is_empty() {
                                    content_blocks.push(json!({"type": "text", "text": text}));
                                }
                            }
                            for call in calls {
                                match call {
                                    ToolCall::Function(FunctionCall::Custom(custom)) => {
                                        let input: serde_json::Value = serde_json::from_str(&custom.arguments)
                                            .unwrap_or(json!({}));
                                        content_blocks.push(json!({
                                            "type": "tool_use",
                                            "id": custom.id.as_deref().unwrap_or("unknown"),
                                            "name": custom.name,
                                            "input": input
                                        }));
                                    }
                                    ToolCall::Function(FunctionCall::Code(code)) => {
                                        content_blocks.push(json!({
                                            "type": "tool_use",
                                            "id": "code_interpreter",
                                            "name": "code_interpreter",
                                            "input": json!({"code": code.code})
                                        }));
                                    }
                                    ToolCall::Function(FunctionCall::Retrieval(retrieval)) => {
                                        content_blocks.push(json!({
                                            "type": "tool_use",
                                            "id": "retrieval",
                                            "name": "retrieval",
                                            "input": json!({"query": retrieval.query})
                                        }));
                                    }
                                }
                            }
                            anthropic_messages.push(json!({
                                "role": "assistant",
                                "content": content_blocks
                            }));
                            continue;
                        }
                    }
                    let text = content.as_deref().unwrap_or("");
                    anthropic_messages.push(json!({
                        "role": "assistant",
                        "content": text
                    }));
                }
                ChatMessage::Tool { tool_call_id, content, .. } => {
                    let tool_result = json!({
                        "type": "tool_result",
                        "tool_use_id": tool_call_id,
                        "content": content
                    });
                    if let Some(last) = anthropic_messages.last_mut() {
                        if last["role"] == "user" {
                            if let Some(blocks) = last["content"].as_array_mut() {
                                blocks.push(tool_result);
                                continue;
                            }
                            let existing_content = last["content"].as_str().unwrap_or("");
                            if !existing_content.is_empty() {
                                last["content"] = json!(vec![
                                    json!({"type": "text", "text": existing_content}),
                                    tool_result
                                ]);
                            } else {
                                last["content"] = json!(vec![tool_result]);
                            }
                            continue;
                        }
                    }
                    anthropic_messages.push(json!({
                        "role": "user",
                        "content": vec![tool_result]
                    }));
                }
                ChatMessage::Function { name, arguments } => {
                    let tool_result = json!({
                        "type": "tool_result",
                        "tool_use_id": name,
                        "content": arguments
                    });
                    if let Some(last) = anthropic_messages.last_mut() {
                        if last["role"] == "user" {
                            if let Some(blocks) = last["content"].as_array_mut() {
                                blocks.push(tool_result);
                                continue;
                            }
                            let existing_content = last["content"].as_str().unwrap_or("");
                            if !existing_content.is_empty() {
                                last["content"] = json!(vec![
                                    json!({"type": "text", "text": existing_content}),
                                    tool_result
                                ]);
                            } else {
                                last["content"] = json!(vec![tool_result]);
                            }
                            continue;
                        }
                    }
                    anthropic_messages.push(json!({
                        "role": "user",
                        "content": vec![tool_result]
                    }));
                }
            }
        }

        (system_prompt, anthropic_messages)
    }

    fn convert_response(&self, json: &serde_json::Value, model: &str) -> Result<ChatCompletionResponse> {
        let id = json["id"].as_str().unwrap_or("unknown").to_string();
        let created = json["created"].as_u64().unwrap_or(0);
        let model = json["model"].as_str().unwrap_or(model).to_string();

        let empty_array: Vec<serde_json::Value> = Vec::new();
        let content_blocks = json["content"].as_array().unwrap_or(&empty_array);

        let mut text_parts: Vec<String> = Vec::new();
        let mut tool_calls: Vec<ToolCall> = Vec::new();

        for block in content_blocks {
            let block_type = block["type"].as_str().unwrap_or("");
            match block_type {
                "text" => {
                    if let Some(text) = block["text"].as_str() {
                        text_parts.push(text.to_string());
                    }
                }
                "tool_use" => {
                    let tool_id = block["id"].as_str().unwrap_or("unknown").to_string();
                    let name = block["name"].as_str().unwrap_or("").to_string();
                    let input = &block["input"];
                    let arguments = serde_json::to_string(input).unwrap_or_else(|_| "{}".to_string());
                    tool_calls.push(ToolCall::Function(FunctionCall::Custom(CustomFunctionCall {
                        id: Some(tool_id),
                        name,
                        arguments,
                    })));
                }
                _ => {}
            }
        }

        let finish_reason = match json["stop_reason"].as_str() {
            Some("tool_use") => Some("tool_calls".to_string()),
            Some("end_turn") | Some("stop_sequence") => Some("stop".to_string()),
            Some("max_tokens") => Some("length".to_string()),
            _ => Some("stop".to_string()),
        };

        let message = if !tool_calls.is_empty() {
            ChatMessage::Assistant {
                content: if text_parts.is_empty() { None } else { Some(text_parts.join("")) },
                tool_calls: Some(tool_calls),
                name: None,
            }
        } else {
            ChatMessage::assistant(text_parts.join(""))
        };

        let choice = Choice {
            index: 0,
            message,
            finish_reason,
            logprobs: None,
        };

        let usage = json["usage"].as_object().map(|u| Usage {
            prompt_tokens: u["input_tokens"].as_u64().unwrap_or(0) as u32,
            completion_tokens: u["output_tokens"].as_u64().unwrap_or(0) as u32,
            total_tokens: u["input_tokens"].as_u64().unwrap_or(0) as u32 + u["output_tokens"].as_u64().unwrap_or(0) as u32,
        });

        Ok(ChatCompletionResponse {
            id,
            object: "chat.completion".to_string(),
            created,
            model,
            choices: vec![choice],
            usage,
            system_fingerprint: None,
        })
    }

    async fn send_request(&self, body: serde_json::Value) -> Result<serde_json::Value> {
        let messages_url = format!("{}/messages", self.get_base_url());
        let response = self.client
            .post(&messages_url)
            .header("x-api-key", &self.config.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("Content-Type", "application/json")
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
                message: format!("Anthropic API error: {}", status),
                status_code: Some(status.as_u16()),
                body,
            });
        }

        response.json().await
            .map_err(|e| LLMError::SerializationError {
                message: e.to_string(),
                context: Some("Failed to parse Anthropic response".to_string()),
            })
    }
}

#[async_trait]
impl LLMProvider for AnthropicProvider {
    fn name(&self) -> &str {
        "anthropic"
    }

    fn supported_models(&self) -> Vec<String> {
        vec![
            "claude-3-5-sonnet-20241022".to_string(),
            "claude-3-5-sonnet".to_string(),
            "claude-3-opus-20240229".to_string(),
            "claude-3-haiku-20240307".to_string(),
        ]
    }

    async fn list_models(&self) -> Result<Vec<crate::types::FullModelInfo>> {
        Ok(vec![
            crate::types::FullModelInfo {
                id: "claude-3-5-sonnet-20241022".to_string(),
                name: "Claude 3.5 Sonnet".to_string(),
                provider: "anthropic".to_string(),
                description: Some("Latest Claude model with improved reasoning and code skills".to_string()),
                context_window: 200000,
                max_output_tokens: 4096,
                capabilities: crate::types::ModelCapabilities {
                    function_calling: true,
                    vision: true,
                    streaming: true,
                    json_mode: true,
                    caching: true,
                    max_tokens: 4096,
                    context_window: 200000,
                },
                pricing: Some(crate::types::ModelPricing {
                    prompt_tokens: 0.003,
                    completion_tokens: 0.015,
                    image_tokens: None,
                }),
                created: 1724076800,
                available: true,
            },
            crate::types::FullModelInfo {
                id: "claude-3-opus-20240229".to_string(),
                name: "Claude 3 Opus".to_string(),
                provider: "anthropic".to_string(),
                description: Some("Most capable Claude model for complex reasoning tasks".to_string()),
                context_window: 200000,
                max_output_tokens: 4096,
                capabilities: crate::types::ModelCapabilities {
                    function_calling: true,
                    vision: true,
                    streaming: true,
                    json_mode: true,
                    caching: true,
                    max_tokens: 4096,
                    context_window: 200000,
                },
                pricing: Some(crate::types::ModelPricing {
                    prompt_tokens: 0.015,
                    completion_tokens: 0.075,
                    image_tokens: None,
                }),
                created: 1709251200,
                available: true,
            },
            crate::types::FullModelInfo {
                id: "claude-3-haiku-20240307".to_string(),
                name: "Claude 3 Haiku".to_string(),
                provider: "anthropic".to_string(),
                description: Some("Fast and compact Claude model for simple tasks".to_string()),
                context_window: 200000,
                max_output_tokens: 4096,
                capabilities: crate::types::ModelCapabilities {
                    function_calling: true,
                    vision: true,
                    streaming: true,
                    json_mode: true,
                    caching: false,
                    max_tokens: 4096,
                    context_window: 200000,
                },
                pricing: Some(crate::types::ModelPricing {
                    prompt_tokens: 0.00080,
                    completion_tokens: 0.004,
                    image_tokens: None,
                }),
                created: 1709251200,
                available: true,
            },
        ])
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

        let (system_prompt, messages) = self.convert_messages(&request.messages);

        let mut body = json!({
            "model": model,
            "messages": messages,
            "max_tokens": request.max_tokens.unwrap_or(1024),
        });

        if let Some(sys) = system_prompt {
            body["system"] = json!(sys);
        }

        if let Some(obj) = body.as_object_mut() {
            if let Some(temp) = request.temperature {
                obj.insert("temperature".to_string(), json!(temp));
            }
            if let Some(top_p) = request.top_p {
                obj.insert("top_p".to_string(), json!(top_p));
            }
            if !request.stop.is_empty() {
                obj.insert("stop_sequences".to_string(), json!(request.stop));
            }
        }

        tracing::info!(model = %model, "Sending Anthropic chat completion request");

        let json = self.send_request(body).await?;
        self.convert_response(&json, &model)
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

        let (system_prompt, messages) = self.convert_messages(&request.messages);

        let mut body = serde_json::json!({
            "model": model,
            "messages": messages,
            "max_tokens": request.max_tokens.unwrap_or(1024),
            "stream": true,
        });

        if let Some(sys) = system_prompt {
            body["system"] = json!(sys);
        }

        if let Some(obj) = body.as_object_mut() {
            if let Some(temp) = request.temperature {
                obj.insert("temperature".to_string(), serde_json::json!(temp));
            }
            if let Some(top_p) = request.top_p {
                obj.insert("top_p".to_string(), serde_json::json!(top_p));
            }
        }

        let messages_url = format!("{}/messages", self.get_base_url());
        let response = self.client
            .post(&messages_url)
            .header("x-api-key", &self.config.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("Content-Type", "application/json")
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
                message: format!("Anthropic API error: {}", status),
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
                            if data.is_empty() {
                                continue;
                            }
                            
                            match serde_json::from_str::<serde_json::Value>(data) {
                                Ok(json) => {
                                    let content = json["delta"]["text"].as_str().unwrap_or("");
                                    
                                    if content.is_empty() {
                                        continue;
                                    }

                                    let finish_reason = json["delta"]["stop_reason"]
                                        .as_str()
                                        .map(|s| s.to_string())
                                        .or_else(|| Some("stop".to_string()));

                                    let chunk = Chunk::Completion(ChunkCompletion {
                                        id: format!("msg_{}", chrono::Utc::now().timestamp_millis()),
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
                id: format!("msg_{}", chrono::Utc::now().timestamp_millis()),
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

        let (system_prompt, messages) = self.convert_messages(&request.messages);

        let tools: Vec<serde_json::Value> = functions.iter().map(|f| {
            let schema = f.parameters.as_ref()
                .cloned()
                .unwrap_or(json!({
                    "type": "object",
                    "properties": {}
                }));
            json!({
                "name": f.name,
                "description": f.description,
                "input_schema": schema
            })
        }).collect();

        let mut body = json!({
            "model": model,
            "messages": messages,
            "max_tokens": request.max_tokens.unwrap_or(4096),
            "tools": tools,
            "tool_choice": {"type": "auto"}
        });

        if let Some(sys) = system_prompt {
            body["system"] = json!(sys);
        }

        if let Some(obj) = body.as_object_mut() {
            if let Some(temp) = request.temperature {
                obj.insert("temperature".to_string(), json!(temp));
            }
            if let Some(top_p) = request.top_p {
                obj.insert("top_p".to_string(), json!(top_p));
            }
            if !request.stop.is_empty() {
                obj.insert("stop_sequences".to_string(), json!(request.stop));
            }
        }

        tracing::info!(model = %model, tools_count = tools.len(), "Sending Anthropic chat completion with functions request");

        let json = self.send_request(body).await?;
        self.convert_response(&json, &model)
    }

    fn supports_function_calling(&self, model: &str) -> bool {
        const KNOWN_MODELS: &[&str] = &[
            "claude-3-opus",
            "claude-3-sonnet",
            "claude-3-haiku",
            "claude-3-5-sonnet",
            "claude-3-5-haiku",
            "claude-4",
        ];
        KNOWN_MODELS.iter().any(|m| model.contains(m))
            || model.contains("claude")
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

        let input_cost_per_million = match model.as_str() {
            m if m.contains("opus") => 15.0,
            m if m.contains("sonnet") => 3.0,
            m if m.contains("haiku") => 0.25,
            _ => 3.0,
        };

        let output_cost_per_million = match model.as_str() {
            m if m.contains("opus") => 75.0,
            m if m.contains("sonnet") => 15.0,
            m if m.contains("haiku") => 1.25,
            _ => 15.0,
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
            provider: "anthropic".to_string(),
        })
    }

    fn config(&self) -> &ProviderConfig {
        unimplemented!()
    }
}
