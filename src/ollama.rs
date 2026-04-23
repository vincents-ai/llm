//! Ollama Provider Implementation
//!
//! This module provides the Ollama provider implementation for local models.

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

/// Ollama Provider Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaConfig {
    pub host: String,
    pub default_model: Option<String>,
}

impl Default for OllamaConfig {
    fn default() -> Self {
        Self {
            host: "http://localhost:11434".to_string(),
            default_model: None,
        }
    }
}

/// Ollama Provider
#[derive(Debug)]
pub struct OllamaProvider {
    config: OllamaConfig,
    client: reqwest::Client,
}

fn serialize_message(msg: &ChatMessage) -> serde_json::Value {
    match msg {
        ChatMessage::System { content, .. } => {
            json!({
                "role": "system",
                "content": content
            })
        }
        ChatMessage::User { content, .. } => {
            json!({
                "role": "user",
                "content": content
            })
        }
        ChatMessage::Assistant { content, tool_calls, .. } => {
            let mut obj = json!({
                "role": "assistant",
                "content": content.as_deref().unwrap_or("")
            });
            if let Some(calls) = tool_calls {
                let ollama_calls: Vec<serde_json::Value> = calls
                    .iter()
                    .filter_map(|tc| {
                        if let ToolCall::Function(FunctionCall::Custom(c)) = tc {
                            let arguments: serde_json::Value =
                                serde_json::from_str(&c.arguments).unwrap_or(json!({}));
                            Some(json!({
                                "function": {
                                    "name": c.name,
                                    "arguments": arguments
                                }
                            }))
                        } else {
                            None
                        }
                    })
                    .collect();
                if !ollama_calls.is_empty() {
                    obj["tool_calls"] = json!(ollama_calls);
                }
            }
            obj
        }
        ChatMessage::Tool { content, tool_call_id, .. } => {
            json!({
                "role": "tool",
                "content": content,
                "tool_name": tool_call_id
            })
        }
        ChatMessage::Function { name, arguments } => {
            json!({
                "role": "tool",
                "content": arguments,
                "tool_name": name
            })
        }
    }
}

fn parse_tool_calls(tool_calls: &serde_json::Value) -> Option<Vec<ToolCall>> {
    let arr = tool_calls.as_array()?;
    if arr.is_empty() {
        return None;
    }
    let calls: Vec<ToolCall> = arr
        .iter()
        .filter_map(|tc| {
            let func = tc.get("function")?;
            let name = func.get("name")?.as_str()?.to_string();
            let arguments = func.get("arguments")?;
            let arguments_str = serde_json::to_string(arguments).ok()?;
            Some(ToolCall::Function(FunctionCall::Custom(CustomFunctionCall {
                id: None,
                name,
                arguments: arguments_str,
            })))
        })
        .collect();
    if calls.is_empty() { None } else { Some(calls) }
}

impl OllamaProvider {
    /// Create a new Ollama provider
    pub async fn new(host: impl Into<String>, model: impl Into<String>) -> Result<Self> {
        let client = reqwest::Client::new();
        
        Ok(Self {
            config: OllamaConfig {
                host: host.into(),
                default_model: Some(model.into()),
                ..Default::default()
            },
            client,
        })
    }

    /// Create with custom configuration
    pub fn with_config(config: OllamaConfig) -> Result<Self> {
        if config.host.is_empty() {
            return Err(LLMError::ConfigurationError {
                message: "Ollama host is required".to_string(),
                field: Some("host".to_string()),
            });
        }

        Ok(Self {
            config,
            client: reqwest::Client::new(),
        })
    }

    fn get_base_url(&self) -> &str {
        &self.config.host
    }

    fn get_default_model(&self) -> Result<&str> {
        self.config.default_model.as_deref().ok_or_else(|| LLMError::ConfigurationError {
            message: "No model configured for OllamaProvider — set default_model in OllamaConfig or pass a model in the request".to_string(),
            field: Some("default_model".to_string()),
        })
    }

    /// Check if Ollama server is available
    pub async fn health_check(&self) -> Result<bool> {
        let response = self.client
            .get(&format!("{}/api/version", self.get_base_url()))
            .send()
            .await
            .map_err(|e| LLMError::HttpError {
                message: e.to_string(),
                status_code: None,
                body: None,
            })?;

        Ok(response.status().is_success())
    }

    /// Fetch models from local Ollama instance.
    async fn fetch_models_live(&self) -> Result<Vec<crate::types::FullModelInfo>> {
        let resp = self.client
            .get(&format!("{}/api/tags", self.get_base_url()))
            .send()
            .await
            .map_err(|e| crate::error::LLMError::HttpError {
                message: format!("Failed to fetch Ollama models: {}", e),
                status_code: None,
                body: None,
            })?;

        if !resp.status().is_success() {
            return Err(crate::error::LLMError::HttpError {
                message: format!("Ollama /api/tags returned {}", resp.status()),
                status_code: Some(resp.status().as_u16()),
                body: resp.text().await.ok(),
            });
        }

        let json: serde_json::Value = resp.json().await.map_err(|e| crate::error::LLMError::SerializationError {
            message: e.to_string(),
            context: Some("Failed to parse Ollama models response".to_string()),
        })?;
        let data = json["models"].as_array()
            .ok_or_else(|| crate::error::LLMError::SerializationError {
                message: "Missing 'models' array".to_string(),
                context: None,
            })?;

        let models: Vec<crate::types::FullModelInfo> = data.iter().filter_map(|m| {
            let name = m["name"].as_str()?.to_string();
            let family = m["details"]["family"].as_str().unwrap_or("unknown").to_string();
            let param_size = m["details"]["parameter_size"].as_str().unwrap_or("?").to_string();
            let quant = m["details"]["quantization_level"].as_str().unwrap_or("?").to_string();
            let name_lower = name.to_lowercase();
            let has_tools = family == "llama" || family == "qwen2" || family == "mistral";
            let mut strengths = Vec::new();
            if name_lower.contains("coder") || name_lower.contains("code") { strengths.push("coding".to_string()); }
            if name_lower.contains("reason") || name_lower.contains("deepseek-r") { strengths.push("reasoning".to_string()); }

            Some(crate::types::FullModelInfo {
                id: name.clone(), name: format!("{} ({}, {})", name, param_size, quant),
                provider: "ollama".to_string(), description: Some(format!("Local Ollama ({})", family)),
                context_window: 8192, max_output_tokens: 4096,
                capabilities: crate::types::ModelCapabilities {
                    function_calling: has_tools, vision: false, streaming: true, json_mode: false,
                    caching: false, max_tokens: 4096, context_window: 8192,
                    input_modalities: vec!["text".to_string()], output_modalities: vec!["text".to_string()],
                    strengths,
                },
                pricing: Some(crate::types::ModelPricing {
                    prompt_tokens: 0.0, completion_tokens: 0.0, image_tokens: None, is_free: true,
                }),
                created: 0, available: true,
            })
        }).collect();

        Ok(models)
    }

    fn fallback_models(&self) -> Vec<crate::types::FullModelInfo> {
        vec![]
    }
}

#[async_trait]
impl LLMProvider for OllamaProvider {
    fn name(&self) -> &str {
        "ollama"
    }

    fn supported_models(&self) -> Vec<String> {
        // Cannot hardcode — Ollama serves whatever the user has pulled.
        // Live discovery via /api/tags is the source of truth.
        vec![]
    }

    fn supports_function_calling(&self, model: &str) -> bool {
        let model_lower = model.to_lowercase();
        let known_prefixes = [
            "llama3.1", "llama3.2", "llama3.3", "llama4",
            "mistral", "mistral-large", "mixtral",
            "qwen2.5", "qwen3",
            "deepseek-r1", "deepseek-coder-v2",
            "gemma2", "gemma3",
            "phi4",
            "command-r",
            "codestral",
            "hermes3",
            "dolphin",
        ];
        known_prefixes.iter().any(|p| model_lower.starts_with(p))
    }

    async fn list_models(&self) -> Result<Vec<crate::types::FullModelInfo>> {
        let url = format!("{}/api/tags", self.get_base_url());
        let response = self.client
            .get(&url)
            .send()
            .await
            .map_err(|e| LLMError::HttpError {
                message: format!("Failed to fetch Ollama models: {}", e),
                status_code: None,
                body: None,
            })?;

        if !response.status().is_success() {
            return Err(LLMError::HttpError {
                message: format!("Ollama API returned status: {}", response.status()),
                status_code: Some(response.status().as_u16()),
                body: None,
            });
        }

        #[derive(Debug, Deserialize)]
        struct OllamaModel {
            name: String,
            #[serde(default)]
            size: String,
        }

        #[derive(Debug, Deserialize)]
        struct OllamaTags {
            models: Option<Vec<OllamaModel>>,
        }

        let tags: OllamaTags = response.json().await
            .map_err(|e| LLMError::SerializationError {
                message: format!("Failed to parse Ollama response: {}", e),
                context: Some("list_models".to_string()),
            })?;

        let models = tags.models.unwrap_or_default();
        
        Ok(models
            .into_iter()
            .map(|m| crate::types::FullModelInfo {
                id: m.name.clone(),
                name: m.name.clone(),
                provider: "ollama".to_string(),
                description: Some(format!("Ollama model ({})", m.size)),
                context_window: 4096,
                max_output_tokens: 2048,
                capabilities: crate::types::ModelCapabilities {
                    function_calling: self.supports_function_calling(&m.name),
                    vision: false,
                    streaming: true,
                    json_mode: false,
                    caching: false,
                    max_tokens: 2048,
                    context_window: 4096,
                    input_modalities: vec!["text".to_string()],
                    output_modalities: vec!["text".to_string()],
                    strengths: vec![],
                },
                pricing: Some(crate::types::ModelPricing {
                    prompt_tokens: 0.0,
                    completion_tokens: 0.0,
                    image_tokens: None,
                    is_free: true,
                }),
                created: 0,
                available: true,
            })
            .collect())
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

        let messages: Vec<serde_json::Value> = request.messages
            .iter()
            .map(serialize_message)
            .collect();

        let mut body = json!({
            "model": model,
            "messages": messages,
            "stream": false,
        });

        if let Some(obj) = body.as_object_mut() {
            if let Some(max_tokens) = request.max_tokens {
                obj.insert("num_predict".to_string(), json!(max_tokens));
            }
            if let Some(temp) = request.temperature {
                obj.insert("temperature".to_string(), json!(temp));
            }
            if let Some(top_p) = request.top_p {
                obj.insert("top_p".to_string(), json!(top_p));
            }
        }

        let url = format!("{}/api/chat", self.get_base_url());

        let response = self.client
            .post(&url)
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
                message: format!("Ollama API error: {}", status),
                status_code: Some(status.as_u16()),
                body,
            });
        }

        let json: serde_json::Value = response.json().await
            .map_err(|e| LLMError::SerializationError {
                message: e.to_string(),
                context: Some("Failed to parse Ollama response".to_string()),
            })?;

        let message = &json["message"];

        let content = message["content"]
            .as_str()
            .unwrap_or("");

        let choices = vec![Choice {
            index: 0,
            message: ChatMessage::assistant(content),
            finish_reason: Some("stop".to_string()),
            logprobs: None,
        }];

        // Estimate token usage (Ollama doesn't always return this)
        let estimated_prompt_tokens = request.messages.iter()
            .map(|m| m.content().map(|c| c.len() / 4).unwrap_or(10))
            .sum::<usize>() as u32;

        let estimated_completion_tokens = content.len() / 4;

        let usage = Some(Usage {
            prompt_tokens: estimated_prompt_tokens,
            completion_tokens: estimated_completion_tokens as u32,
            total_tokens: estimated_prompt_tokens + estimated_completion_tokens as u32,
        });

        let id = format!("ollama_{}", chrono::Utc::now().timestamp_millis());

        Ok(ChatCompletionResponse {
            id,
            object: "chat.completion".to_string(),
            created: chrono::Utc::now().timestamp() as u64,
            model,
            choices,
            usage,
            system_fingerprint: None,
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
            .map(serialize_message)
            .collect();

        let mut body = serde_json::json!({
            "model": model,
            "messages": messages,
            "stream": true,
        });

        if let Some(obj) = body.as_object_mut() {
            if let Some(max_tokens) = request.max_tokens {
                obj.insert("num_predict".to_string(), serde_json::json!(max_tokens));
            }
            if let Some(temp) = request.temperature {
                obj.insert("temperature".to_string(), serde_json::json!(temp));
            }
            if let Some(top_p) = request.top_p {
                obj.insert("top_p".to_string(), serde_json::json!(top_p));
            }
        }

        let url = format!("{}/api/chat", self.get_base_url());

        let response = self.client
            .post(&url)
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
                message: format!("Ollama API error: {}", status),
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
                        
                        match serde_json::from_str::<serde_json::Value>(&line) {
                            Ok(json) => {
                                let message = json["message"]["content"].as_str().unwrap_or("");
                                
                                if message.is_empty() {
                                    continue;
                                }

                                let finish_reason = json["done"].as_bool()
                                    .filter(|&d| d)
                                    .map(|_| "stop".to_string());

                                let chunk = Chunk::Completion(ChunkCompletion {
                                    id: format!("ollama_{}", chrono::Utc::now().timestamp_millis()),
                                    object: "chat.completion.chunk".to_string(),
                                    created: chrono::Utc::now().timestamp() as u64,
                                    model: model_clone.clone(),
                                    choices: vec![ChoiceDelta {
                                        index: chunk_count,
                                        delta: Some(ChatMessage::assistant(message)),
                                        finish_reason,
                                    }],
                                });

                                yield Ok(chunk);
                                chunk_count += 1;
                            }
                            Err(_) => continue,
                        }
                    } else {
                        break;
                    }
                }
            }

            let final_chunk = Chunk::Completion(ChunkCompletion {
                id: format!("ollama_{}", chrono::Utc::now().timestamp_millis()),
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

        if !self.supports_function_calling(&model) {
            tracing::info!("Model {} does not support function calling", model);
            return Err(LLMError::UnsupportedError {
                operation: "function_calling".to_string(),
                provider: Some("ollama".to_string()),
                model: Some(model),
            });
        }

        let tools: Vec<serde_json::Value> = functions
            .iter()
            .map(|f| {
                let mut func_obj = json!({
                    "name": f.name
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
            })
            .collect();

        let messages: Vec<serde_json::Value> = request.messages
            .iter()
            .map(serialize_message)
            .collect();

        let mut body = json!({
            "model": model,
            "messages": messages,
            "stream": false,
            "tools": tools,
        });

        if let Some(obj) = body.as_object_mut() {
            if let Some(max_tokens) = request.max_tokens {
                obj.insert("num_predict".to_string(), json!(max_tokens));
            }
            if let Some(temp) = request.temperature {
                obj.insert("temperature".to_string(), json!(temp));
            }
            if let Some(top_p) = request.top_p {
                obj.insert("top_p".to_string(), json!(top_p));
            }
        }

        let url = format!("{}/api/chat", self.get_base_url());

        let response = self.client
            .post(&url)
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
            let resp_body = response.text().await.ok();
            return Err(LLMError::HttpError {
                message: format!("Ollama API error: {}", status),
                status_code: Some(status.as_u16()),
                body: resp_body,
            });
        }

        let resp_json: serde_json::Value = response.json().await
            .map_err(|e| LLMError::SerializationError {
                message: e.to_string(),
                context: Some("Failed to parse Ollama function calling response".to_string()),
            })?;

        let message = &resp_json["message"];

        let content = message["content"]
            .as_str()
            .unwrap_or("")
            .to_string();

        let tool_calls = parse_tool_calls(&message["tool_calls"]);

        let finish_reason = if tool_calls.is_some() {
            "tool_calls"
        } else {
            "stop"
        };

        let estimated_prompt_tokens = request.messages.iter()
            .map(|m| m.content().map(|c| c.len() / 4).unwrap_or(10))
            .sum::<usize>() as u32;

        let estimated_completion_tokens = content.len() / 4;

        let assistant_msg = if let Some(calls) = tool_calls.clone() {
            ChatMessage::Assistant {
                content: if content.is_empty() { None } else { Some(content) },
                tool_calls: Some(calls),
                name: None,
            }
        } else {
            ChatMessage::assistant(&content)
        };

        let usage = Some(Usage {
            prompt_tokens: estimated_prompt_tokens,
            completion_tokens: estimated_completion_tokens as u32,
            total_tokens: estimated_prompt_tokens + estimated_completion_tokens as u32,
        });

        let id = format!("ollama_{}", chrono::Utc::now().timestamp_millis());

        Ok(ChatCompletionResponse {
            id,
            object: "chat.completion".to_string(),
            created: chrono::Utc::now().timestamp() as u64,
            model,
            choices: vec![Choice {
                index: 0,
                message: assistant_msg,
                finish_reason: Some(finish_reason.to_string()),
                logprobs: None,
            }],
            usage,
            system_fingerprint: None,
        })
    }

    async fn rate_limit_status(&self) -> Result<RateLimitStatus> {
        Ok(RateLimitStatus {
            remaining: Some(999999),
            limit: Some(999999),
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

        let estimated_prompt_tokens = request.messages.iter()
            .map(|m| m.content().map(|c| c.len() / 4).unwrap_or(10))
            .sum::<usize>() as u32;

        let estimated_completion_tokens = request.max_tokens.unwrap_or(1000);

        Ok(CostEstimate {
            input_cost: 0.0,
            output_cost: 0.0,
            total_cost: 0.0,
            input_tokens: estimated_prompt_tokens,
            output_tokens: estimated_completion_tokens,
            model,
            provider: "ollama".to_string(),
        })
    }

    fn config(&self) -> &ProviderConfig {
        unimplemented!()
    }
}
