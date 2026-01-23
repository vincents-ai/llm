//! Google Gemini Provider Implementation
//!
//! This module provides the Google Gemini provider implementation.

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

/// Gemini Provider Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeminiConfig {
    pub api_key: String,
    pub default_model: Option<String>,
    pub base_url: Option<String>,
}

impl Default for GeminiConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            default_model: Some("gemini-1.5-pro".to_string()),
            base_url: None,
        }
    }
}

/// Gemini Provider
#[derive(Debug)]
pub struct GeminiProvider {
    config: GeminiConfig,
    client: reqwest::Client,
}

impl GeminiProvider {
    /// Create a new Gemini provider
    pub async fn new(api_key: impl Into<String>) -> Result<Self> {
        let client = reqwest::Client::new();
        
        Ok(Self {
            config: GeminiConfig {
                api_key: api_key.into(),
                ..Default::default()
            },
            client,
        })
    }

    /// Create with custom configuration
    pub fn with_config(config: GeminiConfig) -> Result<Self> {
        if config.api_key.is_empty() {
            return Err(LLMError::ConfigurationError {
                message: "Gemini API key is required".to_string(),
                field: Some("api_key".to_string()),
            });
        }

        Ok(Self {
            config,
            client: reqwest::Client::new(),
        })
    }

    fn get_base_url(&self) -> String {
        let model = self.config.default_model.as_deref().unwrap_or("gemini-1.5-pro");
        format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent",
            model
        )
    }

    fn get_default_model(&self) -> &str {
        self.config.default_model.as_deref()
            .unwrap_or("gemini-1.5-pro")
    }
}

#[async_trait]
impl LLMProvider for GeminiProvider {
    fn name(&self) -> &str {
        "gemini"
    }

    fn supported_models(&self) -> Vec<String> {
        vec![
            "gemini-1.5-pro".to_string(),
            "gemini-1.5-flash".to_string(),
            "gemini-1.0-pro".to_string(),
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

        // Convert messages to Gemini format
        let contents: Vec<serde_json::Value> = request.messages
            .iter()
            .map(|msg| {
                let role = match msg.role() {
                    MessageRole::System => "system",
                    MessageRole::User => "user",
                    MessageRole::Assistant => "model",
                    MessageRole::Function | MessageRole::Tool => "user",
                };

                let content = msg.content()
                    .map(|s| s.to_string())
                    .unwrap_or_default();

                json!({
                    "role": role,
                    "parts": [{"text": content}]
                })
            })
            .collect();

        let mut body = json!({
            "contents": contents,
        });

        if let Some(obj) = body.as_object_mut() {
            if let Some(max_tokens) = request.max_tokens {
                obj.insert("maxOutputTokens".to_string(), json!(max_tokens));
            }
            if let Some(temp) = request.temperature {
                obj.insert("temperature".to_string(), json!(temp));
            }
            if let Some(top_p) = request.top_p {
                obj.insert("topP".to_string(), json!(top_p));
            }
        }

        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}",
            model,
            self.config.api_key
        );

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
                message: format!("Gemini API error: {}", status),
                status_code: Some(status.as_u16()),
                body,
            });
        }

        let json: serde_json::Value = response.json().await
            .map_err(|e| LLMError::SerializationError {
                message: e.to_string(),
                context: Some("Failed to parse Gemini response".to_string()),
            })?;

        let empty_array: Vec<serde_json::Value> = Vec::new();
        let candidates = json["candidates"].as_array().unwrap_or(&empty_array);

        let choices: Vec<Choice> = candidates
            .iter()
            .enumerate()
            .map(|(idx, candidate)| {
                let content = candidate["content"]["parts"][0]["text"]
                    .as_str()
                    .unwrap_or("");

                let finish_reason = candidate["finishReason"]
                    .as_str()
                    .map(|s| s.to_string());

                Choice {
                    index: idx as u32,
                    message: ChatMessage::assistant(content),
                    finish_reason,
                    logprobs: None,
                }
            })
            .collect();

        let usage = json.get("usageMetadata").map(|u| {
            let prompt_tokens = u.get("promptTokenCount").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
            let completion_tokens = u.get("candidatesTokenCount").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
            
            Usage {
                prompt_tokens,
                completion_tokens,
                total_tokens: prompt_tokens + completion_tokens,
            }
        });

        let id = format!("gemini_{}", chrono::Utc::now().timestamp_millis());

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

        let contents: Vec<serde_json::Value> = request.messages
            .iter()
            .map(|msg| {
                let role = match msg.role() {
                    MessageRole::System => "system",
                    MessageRole::User => "user",
                    MessageRole::Assistant => "model",
                    MessageRole::Function | MessageRole::Tool => "user",
                };

                let content = msg.content()
                    .map(|s| s.to_string())
                    .unwrap_or_default();

                serde_json::json!({
                    "role": role,
                    "parts": [{"text": content}]
                })
            })
            .collect();

        let mut body = serde_json::json!({
            "contents": contents,
            "stream": true,
        });

        if let Some(obj) = body.as_object_mut() {
            if let Some(max_tokens) = request.max_tokens {
                obj.insert("maxOutputTokens".to_string(), serde_json::json!(max_tokens));
            }
            if let Some(temp) = request.temperature {
                obj.insert("temperature".to_string(), serde_json::json!(temp));
            }
            if let Some(top_p) = request.top_p {
                obj.insert("topP".to_string(), serde_json::json!(top_p));
            }
        }

        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}",
            model,
            self.config.api_key
        );

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
                message: format!("Gemini API error: {}", status),
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
                    if buffer.is_empty() {
                        break;
                    }

                    let empty_vec = vec![];
                    match serde_json::from_str::<serde_json::Value>(&buffer) {
                        Ok(json) => {
                            buffer.clear();
                            
                            let candidates = json["candidates"].as_array().unwrap_or(&empty_vec);
                            
                            for candidate in candidates {
                                let content = candidate["content"]["parts"][0]["text"]
                                    .as_str()
                                    .unwrap_or("");

                                if content.is_empty() {
                                    continue;
                                }

                                let finish_reason = candidate["finishReason"]
                                    .as_str()
                                    .map(|s| s.to_string());

                                let chunk = Chunk::Completion(ChunkCompletion {
                                    id: format!("gemini_{}", chrono::Utc::now().timestamp_millis()),
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
                        }
                        Err(_) => {
                            break;
                        }
                    }
                }
            }

            let final_chunk = Chunk::Completion(ChunkCompletion {
                id: format!("gemini_{}", chrono::Utc::now().timestamp_millis()),
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
            provider: Some("gemini".to_string()),
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
            m if m.contains("flash") => 0.075,
            m if m.contains("pro") => 1.25,
            m if m.contains("1.0") => 0.5,
            _ => 1.25,
        };

        let output_cost_per_million = match model.as_str() {
            m if m.contains("flash") => 0.30,
            m if m.contains("pro") => 5.0,
            m if m.contains("1.0") => 1.5,
            _ => 5.0,
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
            provider: "gemini".to_string(),
        })
    }

    fn config(&self) -> &ProviderConfig {
        unimplemented!()
    }
}
