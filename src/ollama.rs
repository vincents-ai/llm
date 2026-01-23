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
            default_model: Some("llama3.1:8b".to_string()),
        }
    }
}

/// Ollama Provider
#[derive(Debug)]
pub struct OllamaProvider {
    config: OllamaConfig,
    client: reqwest::Client,
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

    fn get_default_model(&self) -> &str {
        self.config.default_model.as_deref()
            .unwrap_or("llama3.1:8b")
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
}

#[async_trait]
impl LLMProvider for OllamaProvider {
    fn name(&self) -> &str {
        "ollama"
    }

    fn supported_models(&self) -> Vec<String> {
        vec![
            "llama3.1:8b".to_string(),
            "llama3.1:70b".to_string(),
            "llama3.2:1b".to_string(),
            "llama3.2:3b".to_string(),
            "mistral".to_string(),
            "codellama".to_string(),
            "deepseek-coder".to_string(),
            "qwen2.5-coder".to_string(),
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

        // Convert messages to Ollama format
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
        _request: ChatCompletionRequest,
        _functions: Vec<FunctionDefinition>,
    ) -> Result<ChatCompletionResponse> {
        Err(LLMError::UnsupportedError {
            operation: "function_calling".to_string(),
            provider: Some("ollama".to_string()),
            model: None,
        })
    }

    async fn rate_limit_status(&self) -> Result<RateLimitStatus> {
        // Ollama doesn't have rate limits, just return unlimited
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
        // Ollama is local/self-hosted, so cost is zero
        let model = if request.model.is_empty() {
            self.get_default_model().to_string()
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
