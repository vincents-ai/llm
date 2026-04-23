//! OpenAI Provider Implementation
//!
//! This module provides the OpenAI provider implementation using direct HTTP calls.

use async_trait::async_trait;
use futures::TryStreamExt;
use serde::{Deserialize, Serialize};
use std::pin::Pin;
use reqwest;
use serde_json::{json, Value};

use crate::types::*;
use crate::error::{LLMError, Result};
use crate::config::ProviderConfig;
use crate::provider::LLMProvider;
use crate::RateLimitStatus;
use crate::CostEstimate;

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

    fn get_base_url(&self) -> &str {
        self.config.base_url.as_deref().unwrap_or("https://api.openai.com/v1")
    }

    fn serialize_message(msg: &ChatMessage) -> Value {
        match msg {
            ChatMessage::System { content, name } => {
                let mut obj = json!({
                    "role": "system",
                    "content": content,
                });
                if let Some(name) = name {
                    obj["name"] = json!(name);
                }
                obj
            }
            ChatMessage::User { content, name } => {
                let mut obj = json!({
                    "role": "user",
                    "content": content,
                });
                if let Some(name) = name {
                    obj["name"] = json!(name);
                }
                obj
            }
            ChatMessage::Assistant { content, tool_calls, name } => {
                let mut obj = json!({
                    "role": "assistant",
                });
                if let Some(content) = content {
                    obj["content"] = json!(content);
                } else {
                    obj["content"] = Value::Null;
                }
                if let Some(tool_calls) = tool_calls {
                    let serialized: Vec<Value> = tool_calls.iter().filter_map(|tc| {
                        if let ToolCall::Function(FunctionCall::Custom(custom)) = tc {
                            Some(json!({
                                "id": custom.id.as_deref().unwrap_or(""),
                                "type": "function",
                                "function": {
                                    "name": custom.name,
                                    "arguments": custom.arguments,
                                }
                            }))
                        } else {
                            None
                        }
                    }).collect();
                    if !serialized.is_empty() {
                        obj["tool_calls"] = json!(serialized);
                    }
                }
                if let Some(name) = name {
                    obj["name"] = json!(name);
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
                json!({
                    "role": "function",
                    "name": name,
                    "content": arguments,
                })
            }
        }
    }

    fn parse_assistant_message(msg: &Value) -> ChatMessage {
        let content = msg.get("content")
            .and_then(|c| c.as_str())
            .map(|s| s.to_string());

        let tool_calls = msg.get("tool_calls")
            .and_then(|tc| tc.as_array())
            .map(|arr| {
                arr.iter().filter_map(|tc| {
                    let id = tc.get("id").and_then(|v| v.as_str()).map(|s| s.to_string());
                    let func = tc.get("function")?;
                    let name = func.get("name")?.as_str()?.to_string();
                    let arguments = func.get("arguments")
                        .and_then(|a| a.as_str())
                        .unwrap_or("{}")
                        .to_string();
                    Some(ToolCall::Function(FunctionCall::Custom(CustomFunctionCall {
                        id,
                        name,
                        arguments,
                    })))
                }).collect::<Vec<_>>()
            });

        let tool_calls = tool_calls.filter(|t| !t.is_empty());

        ChatMessage::Assistant {
            content,
            tool_calls,
            name: None,
        }
    }

    async fn send_chat_request(&self, body: Value) -> Result<Value> {
        let response = self.client
            .post(&format!("{}/chat/completions", self.get_base_url()))
            .header("Authorization", format!("Bearer {}", self.config.api_key))
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
            let body_text = response.text().await.ok();
            return Err(LLMError::HttpError {
                message: format!("OpenAI API error: {}", status),
                status_code: Some(status.as_u16()),
                body: body_text,
            });
        }

        response.json().await.map_err(|e| LLMError::SerializationError {
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
            system_fingerprint: json.get("system_fingerprint").and_then(|v| v.as_str()).map(|s| s.to_string()),
        }
    }
}

#[async_trait]
impl LLMProvider for OpenAIProvider {
    fn name(&self) -> &str {
        "openai"
    }

    fn supported_models(&self) -> Vec<String> {
        vec![
            "gpt-4".to_string(),
            "gpt-4-turbo".to_string(),
            "gpt-4o".to_string(),
            "gpt-4o-mini".to_string(),
            "gpt-3.5-turbo".to_string(),
            "o1".to_string(),
            "o1-mini".to_string(),
            "o1-preview".to_string(),
        ]
    }

    fn supports_function_calling(&self, model: &str) -> bool {
        match model {
            "gpt-4" | "gpt-4-turbo" | "gpt-4o" | "gpt-4o-mini"
            | "gpt-3.5-turbo" | "o1" | "o1-mini" | "o1-preview" => true,
            _ => self.supports_model(model),
        }
    }

    async fn list_models(&self) -> Result<Vec<crate::types::FullModelInfo>> {
        Ok(vec![
            crate::types::FullModelInfo {
                id: "gpt-4".to_string(),
                name: "GPT-4".to_string(),
                provider: "openai".to_string(),
                description: Some("Most capable model, optimized for complex tasks".to_string()),
                context_window: 8192,
                max_output_tokens: 8192,
                capabilities: crate::types::ModelCapabilities {
                    function_calling: true,
                    vision: true,
                    streaming: true,
                    json_mode: true,
                    caching: true,
                    max_tokens: 8192,
                    context_window: 8192,
                    input_modalities: vec!["text".to_string()],
                    output_modalities: vec!["text".to_string()],
                    strengths: vec![],
                },
                pricing: Some(crate::types::ModelPricing {
                    prompt_tokens: 0.03,
                    completion_tokens: 0.06,
                    image_tokens: None,
                    is_free: false,
                }),
                created: 1687882411,
                available: true,
            },
            crate::types::FullModelInfo {
                id: "gpt-4-turbo".to_string(),
                name: "GPT-4 Turbo".to_string(),
                provider: "openai".to_string(),
                description: Some("Higher speed at 2x the throughput of GPT-4".to_string()),
                context_window: 128000,
                max_output_tokens: 4096,
                capabilities: crate::types::ModelCapabilities {
                    function_calling: true,
                    vision: true,
                    streaming: true,
                    json_mode: true,
                    caching: true,
                    max_tokens: 4096,
                    context_window: 128000,
                    input_modalities: vec!["text".to_string()],
                    output_modalities: vec!["text".to_string()],
                    strengths: vec![],
                },
                pricing: Some(crate::types::ModelPricing {
                    prompt_tokens: 0.01,
                    completion_tokens: 0.03,
                    image_tokens: None,
                    is_free: false,
                }),
                created: 1694122800,
                available: true,
            },
            crate::types::FullModelInfo {
                id: "gpt-3.5-turbo".to_string(),
                name: "GPT-3.5 Turbo".to_string(),
                provider: "openai".to_string(),
                description: Some("Fast and efficient model for most tasks".to_string()),
                context_window: 4096,
                max_output_tokens: 4096,
                capabilities: crate::types::ModelCapabilities {
                    function_calling: true,
                    vision: false,
                    streaming: true,
                    json_mode: true,
                    caching: false,
                    max_tokens: 4096,
                    context_window: 4096,
                    input_modalities: vec!["text".to_string()],
                    output_modalities: vec!["text".to_string()],
                    strengths: vec![],
                },
                pricing: Some(crate::types::ModelPricing {
                    prompt_tokens: 0.0005,
                    completion_tokens: 0.0015,
                    image_tokens: None,
                    is_free: false,
                }),
                created: 1684758360,
                available: true,
            },
            crate::types::FullModelInfo {
                id: "gpt-4o".to_string(),
                name: "GPT-4 Optimized".to_string(),
                provider: "openai".to_string(),
                description: Some("Optimized version of GPT-4 for faster responses".to_string()),
                context_window: 128000,
                max_output_tokens: 4096,
                capabilities: crate::types::ModelCapabilities {
                    function_calling: true,
                    vision: true,
                    streaming: true,
                    json_mode: true,
                    caching: true,
                    max_tokens: 4096,
                    context_window: 128000,
                    input_modalities: vec!["text".to_string()],
                    output_modalities: vec!["text".to_string()],
                    strengths: vec![],
                },
                pricing: Some(crate::types::ModelPricing {
                    prompt_tokens: 0.005,
                    completion_tokens: 0.015,
                    image_tokens: None,
                    is_free: false,
                }),
                created: 1714073600,
                available: true,
            },
            crate::types::FullModelInfo {
                id: "gpt-4o-mini".to_string(),
                name: "GPT-4 Optimized Mini".to_string(),
                provider: "openai".to_string(),
                description: Some("Lightweight version for efficiency".to_string()),
                context_window: 128000,
                max_output_tokens: 4096,
                capabilities: crate::types::ModelCapabilities {
                    function_calling: true,
                    vision: true,
                    streaming: true,
                    json_mode: true,
                    caching: false,
                    max_tokens: 4096,
                    context_window: 128000,
                    input_modalities: vec!["text".to_string()],
                    output_modalities: vec!["text".to_string()],
                    strengths: vec![],
                },
                pricing: Some(crate::types::ModelPricing {
                    prompt_tokens: 0.00015,
                    completion_tokens: 0.0006,
                    image_tokens: None,
                    is_free: false,
                }),
                created: 1718192000,
                available: true,
            },
        ])
    }

    async fn chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse> {
        let start = std::time::Instant::now();

        let messages: Vec<Value> = request.messages
            .iter()
            .map(Self::serialize_message)
            .collect();
        
        let mut body = json!({
            "model": request.model,
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
            for (key, value) in &request.extra_params {
                obj.insert(key.clone(), value.clone());
            }
        }

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

    async fn chat_completion_stream(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<Pin<Box<dyn futures::Stream<Item = Result<Chunk>> + Send>>> {
        let model = if request.model.is_empty() {
            self.config.default_model.clone().ok_or_else(|| LLMError::ConfigurationError {
                message: "No model configured for OpenAIProvider — set default_model in OpenAIConfig or pass a model in the request".to_string(),
                field: Some("default_model".to_string()),
            })?
        } else {
            request.model.clone()
        };

        let messages: Vec<Value> = request.messages
            .iter()
            .map(Self::serialize_message)
            .collect();
        
        let mut body = json!({
            "model": model,
            "messages": messages,
            "stream": true,
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
        }

        let response = self.client
            .post(&format!("{}/chat/completions", self.get_base_url()))
            .header("Authorization", format!("Bearer {}", self.config.api_key))
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
                message: format!("OpenAI API error: {}", status),
                status_code: Some(status.as_u16()),
                body,
            });
        }

        let model_clone = model.clone();
        let _client = self.client.clone();

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
                                break;
                            }
                            
                            match serde_json::from_str::<Value>(data) {
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
                                        id: format!("cmpl-{}", chrono::Utc::now().timestamp_millis()),
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
                id: format!("cmpl-{}", chrono::Utc::now().timestamp_millis()),
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
        let start = std::time::Instant::now();

        let messages: Vec<Value> = request.messages
            .iter()
            .map(Self::serialize_message)
            .collect();

        let tools: Vec<Value> = functions.iter().map(|f| {
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
        }).collect();

        let mut body = json!({
            "model": request.model,
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
            for (key, value) in &request.extra_params {
                obj.insert(key.clone(), value.clone());
            }
        }

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
            model: request.model.clone(),
            provider: "openai".to_string(),
        })
    }

    fn config(&self) -> &ProviderConfig {
        unimplemented!()
    }
}
