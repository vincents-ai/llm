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
}

impl Default for AnthropicConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            default_model: Some("claude-3-5-sonnet-20241022".to_string()),
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

    fn get_default_model(&self) -> &str {
        self.config.default_model.as_deref()
            .unwrap_or("claude-3-5-sonnet-20241022")
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

    async fn chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse> {
        let model = if request.model.is_empty() {
            self.get_default_model().to_string()
        } else {
            request.model.clone()
        };

        // Convert messages to Anthropic format
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
            "max_tokens": request.max_tokens.unwrap_or(1024),
        });

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

        let response = self.client
            .post("https://api.anthropic.com/v1/messages")
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

        let json: serde_json::Value = response.json().await
            .map_err(|e| LLMError::SerializationError {
                message: e.to_string(),
                context: Some("Failed to parse Anthropic response".to_string()),
            })?;

        let id = json["id"].as_str().unwrap_or("unknown").to_string();
        let created = json["created"].as_u64().unwrap_or(0);
        let model = json["model"].as_str().unwrap_or(&model).to_string();

        let empty_array: Vec<serde_json::Value> = Vec::new();
        let content_blocks = json["content"].as_array().unwrap_or(&empty_array);

        let choices: Vec<Choice> = content_blocks
            .iter()
            .enumerate()
            .filter_map(|(idx, block)| {
                let content = block["text"].as_str()?;
                let block_type = block["type"].as_str()?;
                
                if block_type != "text" {
                    return None;
                }

                Some(Choice {
                    index: idx as u32,
                    message: ChatMessage::assistant(content),
                    finish_reason: Some("stop".to_string()),
                    logprobs: None,
                })
            })
            .collect();

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
            self.get_default_model().to_string()
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
            "max_tokens": request.max_tokens.unwrap_or(1024),
            "stream": true,
        });

        if let Some(obj) = body.as_object_mut() {
            if let Some(temp) = request.temperature {
                obj.insert("temperature".to_string(), serde_json::json!(temp));
            }
            if let Some(top_p) = request.top_p {
                obj.insert("top_p".to_string(), serde_json::json!(top_p));
            }
        }

        let response = self.client
            .post("https://api.anthropic.com/v1/messages")
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
        _request: ChatCompletionRequest,
        _functions: Vec<FunctionDefinition>,
    ) -> Result<ChatCompletionResponse> {
        Err(LLMError::UnsupportedError {
            operation: "function_calling".to_string(),
            provider: Some("anthropic".to_string()),
            model: None,
        })
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
            self.get_default_model().to_string()
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
