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
            default_model: None,
            base_url: None,
        }
    }
}

#[derive(Debug)]
pub struct GeminiProvider {
    config: GeminiConfig,
    client: reqwest::Client,
}

fn convert_message(msg: &ChatMessage) -> serde_json::Value {
    match msg {
        ChatMessage::System { content, .. } => {
            json!({
                "role": "system",
                "parts": [{"text": content}]
            })
        }
        ChatMessage::User { content, .. } => {
            json!({
                "role": "user",
                "parts": [{"text": content}]
            })
        }
        ChatMessage::Assistant { content, tool_calls, .. } => {
            let mut parts = Vec::new();
            if let Some(text) = content {
                if !text.is_empty() {
                    parts.push(json!({"text": text}));
                }
            }
            if let Some(calls) = tool_calls {
                for tc in calls {
                    if let ToolCall::Function(FunctionCall::Custom(custom)) = tc {
                        let args: serde_json::Value = serde_json::from_str(&custom.arguments)
                            .unwrap_or(json!({}));
                        parts.push(json!({
                            "functionCall": {
                                "name": custom.name,
                                "args": args,
                            }
                        }));
                    }
                }
            }
            if parts.is_empty() {
                parts.push(json!({"text": ""}));
            }
            json!({
                "role": "model",
                "parts": parts,
            })
        }
        ChatMessage::Tool { tool_call_id, content } => {
            json!({
                "role": "user",
                "parts": [{
                    "functionResponse": {
                        "name": tool_call_id,
                        "response": { "content": content }
                    }
                }]
            })
        }
        ChatMessage::Function { name, arguments } => {
            json!({
                "role": "user",
                "parts": [{
                    "functionResponse": {
                        "name": name,
                        "response": { "content": arguments }
                    }
                }]
            })
        }
    }
}

fn parse_response_parts(parts: &[serde_json::Value]) -> (Option<String>, Vec<ToolCall>) {
    let mut text_parts = Vec::new();
    let mut tool_calls = Vec::new();

    for part in parts {
        if let Some(text) = part.get("text").and_then(|t| t.as_str()) {
            text_parts.push(text.to_string());
        }
        if let Some(fc) = part.get("functionCall") {
            let name = fc.get("name")
                .and_then(|n| n.as_str())
                .unwrap_or("")
                .to_string();
            let args = fc.get("args").cloned().unwrap_or(json!({}));
            let arguments = serde_json::to_string(&args).unwrap_or_else(|_| "{}".to_string());
            tool_calls.push(ToolCall::Function(FunctionCall::Custom(CustomFunctionCall {
                id: None,
                name,
                arguments,
            })));
        }
    }

    let text = if text_parts.is_empty() {
        None
    } else {
        Some(text_parts.join(""))
    };

    (text, tool_calls)
}

fn normalize_finish_reason(raw: Option<&str>, has_tool_calls: bool) -> Option<String> {
    match raw {
        Some("STOP") => {
            if has_tool_calls {
                Some("tool_calls".to_string())
            } else {
                Some("stop".to_string())
            }
        }
        Some("SAFETY") => Some("content_filter".to_string()),
        Some(other) => Some(other.to_lowercase()),
        None => None,
    }
}

impl GeminiProvider {
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

    fn get_base_url(&self) -> Result<String> {
        let model = self.config.default_model.as_deref().ok_or_else(|| LLMError::ConfigurationError {
            message: "No model configured for GeminiProvider — set default_model in GeminiConfig or pass a model in the request".to_string(),
            field: Some("default_model".to_string()),
        })?;
        Ok(format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent",
            model
        ))
    }

    fn get_default_model(&self) -> Result<&str> {
        self.config.default_model.as_deref().ok_or_else(|| LLMError::ConfigurationError {
            message: "No model configured for GeminiProvider — set default_model in GeminiConfig or pass a model in the request".to_string(),
            field: Some("default_model".to_string()),
        })
    }

    async fn send_request(&self, model: &str, body: serde_json::Value) -> Result<serde_json::Value> {
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

        response.json().await
            .map_err(|e| LLMError::SerializationError {
                message: e.to_string(),
                context: Some("Failed to parse Gemini response".to_string()),
            })
    }

    fn build_response(json: serde_json::Value, model: String) -> ChatCompletionResponse {
        let empty_array: Vec<serde_json::Value> = Vec::new();
        let candidates = json["candidates"].as_array().unwrap_or(&empty_array);

        let choices: Vec<Choice> = candidates
            .iter()
            .enumerate()
            .map(|(idx, candidate)| {
                let parts = candidate["content"]["parts"]
                    .as_array()
                    .cloned()
                    .unwrap_or_default();
                let (text, tool_calls) = parse_response_parts(&parts);

                let finish_reason = normalize_finish_reason(
                    candidate["finishReason"].as_str(),
                    !tool_calls.is_empty(),
                );

                let message = if !tool_calls.is_empty() {
                    ChatMessage::Assistant {
                        content: text,
                        tool_calls: Some(tool_calls),
                        name: None,
                    }
                } else {
                    ChatMessage::assistant(text.unwrap_or_default())
                };

                Choice {
                    index: idx as u32,
                    message,
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

        ChatCompletionResponse {
            id,
            object: "chat.completion".to_string(),
            created: chrono::Utc::now().timestamp() as u64,
            model,
            choices,
            usage,
            system_fingerprint: None,
        }
    }

    fn build_body(messages: &[ChatMessage], _model: &str, request: &ChatCompletionRequest) -> serde_json::Value {
        let contents: Vec<serde_json::Value> = messages
            .iter()
            .map(convert_message)
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

        body
    }

    /// Fetch models from Google's Gemini API.
    async fn fetch_models_live(&self) -> Result<Vec<crate::types::FullModelInfo>> {
        let base = self.config.base_url.as_deref()
            .unwrap_or("https://generativelanguage.googleapis.com/v1beta");
        let url = if self.config.api_key.is_empty() {
            format!("{}/models", base)
        } else {
            format!("{}/models?key={}", base, self.config.api_key)
        };

        let resp = self.client.get(&url).send().await.map_err(|e| crate::error::LLMError::HttpError {
            message: format!("Failed to fetch Gemini models: {}", e),
            status_code: None, body: None,
        })?;

        if !resp.status().is_success() {
            return Err(crate::error::LLMError::HttpError {
                message: format!("Gemini /models returned {}", resp.status()),
                status_code: Some(resp.status().as_u16()),
                body: resp.text().await.ok(),
            });
        }

        let json: serde_json::Value = resp.json().await.map_err(|e| crate::error::LLMError::SerializationError {
            message: e.to_string(),
            context: Some("Failed to parse Gemini models".to_string()),
        })?;
        let data = json["models"].as_array()
            .ok_or_else(|| crate::error::LLMError::SerializationError {
                message: "Missing 'models' array".to_string(), context: None,
            })?;

        Ok(data.iter().filter_map(|m| {
            let name = m["name"].as_str()?.to_string();
            let id = name.strip_prefix("models/").unwrap_or(&name).to_string();
            let display_name = m["displayName"].as_str().unwrap_or(&id).to_string();
            let description = m["description"].as_str().map(String::from);
            let input_limit = m["inputTokenLimit"].as_u64().unwrap_or(0) as u32;
            let output_limit = m["outputTokenLimit"].as_u64().unwrap_or(0) as u32;
            let has_vision = id.contains("pro") || id.contains("flash");

            let mut input_mods = vec!["text".to_string()];
            if has_vision { input_mods.push("image".to_string()); }

            Some(crate::types::FullModelInfo {
                id, name: display_name, provider: "gemini".to_string(), description,
                context_window: input_limit, max_output_tokens: output_limit,
                capabilities: crate::types::ModelCapabilities {
                    function_calling: true, vision: has_vision, streaming: true, json_mode: true,
                    caching: false, max_tokens: output_limit, context_window: input_limit,
                    input_modalities: input_mods, output_modalities: vec!["text".to_string()],
                    strengths: vec![],
                },
                pricing: Some(crate::types::ModelPricing {
                    prompt_tokens: 0.0, completion_tokens: 0.0, image_tokens: None, is_free: true,
                }),
                created: 0, available: true,
            })
        }).collect())
    }
}

#[async_trait]
impl LLMProvider for GeminiProvider {
    fn name(&self) -> &str {
        "gemini"
    }

    fn supported_models(&self) -> Vec<String> {
        // Cannot hardcode — Gemini API returns live model list.
        vec![]
    }

    /// Fetch models from live Gemini API with fallback.
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
        let model = if request.model.is_empty() {
            self.get_default_model()?.to_string()
        } else {
            request.model.clone()
        };

        let body = Self::build_body(&request.messages, &model, &request);

        let json = self.send_request(&model, body).await?;
        Ok(Self::build_response(json, model))
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

        let contents: Vec<serde_json::Value> = request.messages
            .iter()
            .map(convert_message)
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
        request: ChatCompletionRequest,
        functions: Vec<FunctionDefinition>,
    ) -> Result<ChatCompletionResponse> {
        let model = if request.model.is_empty() {
            self.get_default_model()?.to_string()
        } else {
            request.model.clone()
        };

        let function_declarations: Vec<serde_json::Value> = functions.iter().map(|f| {
            let mut decl = json!({
                "name": f.name,
            });
            if let Some(desc) = &f.description {
                decl["description"] = json!(desc);
            }
            if let Some(params) = &f.parameters {
                decl["parameters"] = params.clone();
            }
            decl
        }).collect();

        let tools = json!([{ "functionDeclarations": function_declarations }]);
        let tool_config = json!({
            "functionCallingConfig": { "mode": "AUTO" }
        });

        let non_system_messages: Vec<&ChatMessage> = request.messages
            .iter()
            .filter(|m| m.role() != MessageRole::System)
            .collect();

        let system_messages: Vec<&ChatMessage> = request.messages
            .iter()
            .filter(|m| m.role() == MessageRole::System)
            .collect();

        let contents: Vec<serde_json::Value> = non_system_messages
            .iter()
            .map(|m| convert_message(m))
            .collect();

        let mut body = json!({
            "contents": contents,
            "tools": tools,
            "toolConfig": tool_config,
        });

        if !system_messages.is_empty() {
            let system_parts: Vec<serde_json::Value> = system_messages
                .iter()
                .map(|m| {
                    let text = m.content().unwrap_or("");
                    json!({"text": text})
                })
                .collect();
            body["systemInstruction"] = json!({ "parts": system_parts });
        }

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

        let json = self.send_request(&model, body).await?;
        Ok(Self::build_response(json, model))
    }

    fn supports_function_calling(&self, model: &str) -> bool {
        let m = model.to_lowercase();
        m.contains("1.5") || m.contains("2.0") || m.contains("flash") || m.contains("gemini-pro")
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
