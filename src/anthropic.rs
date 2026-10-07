//! Anthropic Claude Provider Implementation
//!
//! This module provides the Anthropic provider implementation using direct HTTP.

use async_trait::async_trait;
use futures::TryStreamExt;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::pin::Pin;

use crate::config::ProviderConfig;
use crate::error::{LLMError, Result};
use crate::provider::LLMProvider;
use crate::types::*;
use crate::CostEstimate;
use crate::RateLimitStatus;

/// Anthropic Provider Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[derive(Default)]
pub struct AnthropicConfig {
    pub api_key: String,
    pub default_model: Option<String>,
    /// Base URL for the Anthropic API (or compatible proxy).
    /// Defaults to "https://api.anthropic.com/v1".
    pub base_url: Option<String>,
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
        self.config
            .base_url
            .as_deref()
            .unwrap_or("https://api.anthropic.com/v1")
    }

    /// Create with a custom endpoint (for Anthropic-compatible proxies such as Z.ai).
    pub async fn with_endpoint(
        api_key: impl Into<String>,
        base_url: impl Into<String>,
    ) -> Result<Self> {
        Ok(Self {
            config: AnthropicConfig {
                api_key: api_key.into(),
                base_url: Some(base_url.into()),
                ..Default::default()
            },
            client: reqwest::Client::new(),
        })
    }

/// Translate a user message into Anthropic content blocks.
///
/// A bare string for text-only content, because that is what this provider has
/// always received and every existing request depends on it. A block list only
/// when there is genuinely more than one part.
///
/// Each block uses Anthropic's own vocabulary. Sending `{"type": "image_url"}`
/// here -- which is what a direct `serde` pass over the shared `ContentPart`
/// produced -- sends the *OpenAI* shape to an Anthropic endpoint, which is
/// worse than refusing: the request looks well-formed in a log and is rejected
/// or misread by the API.
///
/// A part this provider cannot express is [`LLMError::UnsupportedFeature`]
/// rather than a silent omission. Dropping an image yields an answer to a
/// question the model never received the image for.
fn user_blocks(content: &MessageContent, model: &str) -> Result<Value> {
    use crate::types::{ContentPart, MessageContent};
    let MessageContent::Parts(parts) = content else {
        return Ok(Value::String(content.as_text().unwrap_or_default().to_string()));
    };
    let mut blocks = Vec::with_capacity(parts.len());
    for part in parts {
        blocks.push(match part {
            ContentPart::Text { text } => json!({"type": "text", "text": text}),
            ContentPart::ImageUrl { url, .. } => Self::anthropic_image(url, model)?,
            ContentPart::InputAudio { .. } => {
                return Err(LLMError::UnsupportedFeature {
                    feature: "input_audio".to_string(),
                    model: model.to_string(),
                })
            }
            ContentPart::File { .. } => {
                return Err(LLMError::UnsupportedFeature {
                    feature: "file content parts".to_string(),
                    model: model.to_string(),
                })
            }
        });
    }
    // A single text block needs no list: this provider accepts the string, and
    // every text-only request that worked before must keep working.
    if blocks.len() == 1 {
        if let Some(text) = blocks[0].get("text").and_then(|t| t.as_str()) {
            return Ok(Value::String(text.to_string()));
        }
    }
    Ok(Value::Array(blocks))
}

    /// The payload after the first comma of a data URL.
    ///
    /// Needed because a `data:` URL's media-type segment may carry a `;base64`
    /// marker, so the text after the first comma is not always the payload.
    fn rest_data(url: &str) -> &str {
        url.split_once(',').map(|(_, data)| data).unwrap_or("")
    }

    /// One image block, choosing the source shape from the URL.
///
/// A `data:` URL carries its own media type, so it becomes a `base64` source. A
/// bare https URL becomes a `url` source. Anything else -- an `http` URL the
/// provider will not fetch, a relative path -- is refused rather than guessed
/// at, because guessing picks a source type the API rejects.
fn anthropic_image(url: &str, model: &str) -> Result<Value> {
    if let Some(rest) = url.strip_prefix("data:") {
        let (media_type, data) = rest.split_once(',').ok_or_else(|| {
            LLMError::InvalidRequestError {
                message: format!(
                    "data URL has no comma separating media type from data: {url}"
                ),
                param: None,
                code: None,
            }
        })?;
        // `image/jpeg;base64` is not a media type. The `;base64` marker belongs
        // to the URL, not to the type, and sending it back inside
        // `media_type` makes the API reject the block.
        let (media_type, encoded) = match media_type.split_once(';') {
            Some((media_type, marker)) => {
                (media_type.to_string(), marker.eq_ignore_ascii_case("base64"))
            }
            None => (media_type.to_string(), false),
        };
        return Ok(json!({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": if encoded { data } else { Self::rest_data(url) },
            }
        }));
    }
    if url.starts_with("https://") {
        return Ok(json!({
            "type": "image",
            "source": {"type": "url", "url": url}
        }));
    }
    Err(LLMError::UnsupportedFeature {
        feature: format!("an image URL that is neither a data: URL nor https ({url})"),
        model: model.to_string(),
    })
}

    fn convert_messages(
        &self,
        messages: &[ChatMessage],
    ) -> Result<(Option<String>, Vec<serde_json::Value>)> {
        let model = self.config.default_model.clone().unwrap_or_default();
        let mut system_value: Option<String> = None;
        let mut anthropic_messages: Vec<serde_json::Value> = Vec::new();

        for msg in messages {
            match msg {
                ChatMessage::Developer { content, .. }
                | ChatMessage::System { content, .. } => {
                    // Merge prompts. Anthropic has one `system` field and it is
                    // a string, so a developer message lands there too -- the
                    // only place it can go. A multimodal system prompt has no
                    // representation here and is reduced to its text rather
                    // than silently dropped.
                    let text = content.as_text().unwrap_or_default().to_string();
                    system_value = Some(match system_value.take() {
                        Some(existing) => format!("{}\n\n{}", existing, text),
                        None => text,
                    });
                }
                ChatMessage::User { content, .. } => {
                    let blocks = Self::user_blocks(content, &model)?;
                    if let Some(last) = anthropic_messages.last_mut() {
                        if last["role"] == "user" {
                            // Merging two user turns into one is what this
                            // provider expects. The merge has to happen at block
                            // level once either side is multimodal: concatenating
                            // the JSON would produce one malformed block.
                            match (&mut last["content"], blocks) {
                                (Value::String(existing), Value::Array(mut extra)) => {
                                    extra.insert(
                                        0,
                                        json!({"type": "text",
                                               "text": format!("{existing}\n\n")}),
                                    );
                                    last["content"] = Value::Array(extra);
                                }
                                (Value::Array(existing), blocks) => {
                                    match blocks {
                                        Value::String(text) => {
                                            existing.push(json!({"type": "text", "text": text}));
                                        }
                                        Value::Array(more) => existing.extend(more),
                                        _ => {}
                                    }
                                }
                                _ => {}
                            }
                            continue;
                        }
                    }
                    anthropic_messages.push(json!({"role": "user", "content": blocks}));
                }
                ChatMessage::Assistant {
                    content,
                    tool_calls,
                    ..
                } => {
                    if let Some(calls) = tool_calls {
                        if !calls.is_empty() {
                            let mut content_blocks: Vec<serde_json::Value> = Vec::new();
                            if let Some(MessageContent::Text(text)) = content {
                                if !text.is_empty() {
                                    content_blocks.push(json!({"type": "text", "text": text}));
                                }
                            }
                            for call in calls {
                                match call {
                                    ToolCall::Function(FunctionCall::Custom(custom)) => {
                                        let input: serde_json::Value =
                                            serde_json::from_str(&custom.arguments)
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
                    let text = content
                        .as_ref()
                        .and_then(MessageContent::as_text)
                        .unwrap_or("");
                    anthropic_messages.push(json!({
                        "role": "assistant",
                        "content": text
                    }));
                }
                ChatMessage::Tool {
                    tool_call_id,
                    content,
                    ..
                } => {
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

        Ok((system_value, anthropic_messages))
    }

    fn convert_response(
        &self,
        json: &serde_json::Value,
        model: &str,
    ) -> Result<ChatCompletionResponse> {
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
                    let arguments =
                        serde_json::to_string(input).unwrap_or_else(|_| "{}".to_string());
                    tool_calls.push(ToolCall::Function(FunctionCall::Custom(
                        CustomFunctionCall {
                            id: Some(tool_id),
                            name,
                            arguments,
                        },
                    )));
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
                content: if text_parts.is_empty() {
                    None
                } else {
                    Some(MessageContent::Text(text_parts.join("")))
                },
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

        // Anthropic renames the fields *and* nests the cache detail, so this
        // dialect gets its own mapping rather than the OpenAI one. `cache_read`
        // is Anthropic's cached-token count; `cache_creation` is the
        // cache-write count OpenAI does not report at all.
        // `get` throughout, never indexing. Indexing a `Map` with a missing key
        // panics, and every one of these keys is optional: an Anthropic response
        // with no `cache_creation_input_tokens` used to crash response parsing
        // outright. That was live before this change and had no test covering it.
        let usage = json["usage"].as_object().map(|u| {
            let input = u.get("input_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
            let output = u.get("output_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
            Usage {
                prompt_tokens: input,
                completion_tokens: output,
                total_tokens: input + output,
                reasoning_tokens: None,
                cached_tokens: u.get("cache_read_input_tokens").and_then(|v| v.as_u64()).map(|v| v as u32),
                cache_write_tokens: u
                    .get("cache_creation_input_tokens")
                    .and_then(|v| v.as_u64())
                    .map(|v| v as u32),
            }
        });

        Ok(ChatCompletionResponse {
            id,
            object: "chat.completion".to_string(),
            created,
            model,
            choices: vec![choice],
            usage,
            system_fingerprint: None,
            unknown_fields: Default::default(),
        })
    }

    async fn send_request(&self, body: serde_json::Value) -> Result<serde_json::Value> {
        let messages_url = format!("{}/messages", self.get_base_url());
        let response = self
            .client
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

        response
            .json()
            .await
            .map_err(|e| LLMError::SerializationError {
                message: e.to_string(),
                context: Some("Failed to parse Anthropic response".to_string()),
            })
    }

    /// Fetch models from the live Anthropic API.
    async fn fetch_models_live(&self) -> Result<Vec<crate::types::FullModelInfo>> {
        let base = self
            .config
            .base_url
            .as_deref()
            .unwrap_or("https://api.anthropic.com");

        let resp = self
            .client
            .get(format!("{}/v1/models", base))
            .header("Content-Type", "application/json")
            .header("x-api-key", &self.config.api_key)
            .header("anthropic-version", "2023-06-01")
            .send()
            .await
            .map_err(|e| LLMError::HttpError {
                message: format!("Failed to fetch Anthropic models: {}", e),
                status_code: None,
                body: None,
            })?;

        if !resp.status().is_success() {
            return Err(LLMError::HttpError {
                message: format!("Anthropic /v1/models returned {}", resp.status()),
                status_code: Some(resp.status().as_u16()),
                body: resp.text().await.ok(),
            });
        }

        let json: serde_json::Value =
            resp.json()
                .await
                .map_err(|e| LLMError::SerializationError {
                    message: e.to_string(),
                    context: Some("Failed to parse models response".to_string()),
                })?;
        let data = json["data"]
            .as_array()
            .ok_or_else(|| LLMError::SerializationError {
                message: "Missing 'data' array".to_string(),
                context: None,
            })?;

        let mut models: Vec<crate::types::FullModelInfo> = Vec::new();

        for m in data {
            let id = match m["id"].as_str() {
                Some(id) => id.to_string(),
                None => continue,
            };

            let display_name = m["display_name"].as_str().unwrap_or(&id).to_string();

            // Capabilities and the context window come from the catalog when it
            // knows the model, and are otherwise reported as `Unknown` rather
            // than guessed.
            //
            // The previous code derived `has_vision` from
            // `id_lower.contains("claude-3")` and hardcoded `ctx = 200000` for
            // *every* Anthropic model. A substring match is not a capability, and
            // a hardcoded context window is wrong for every model that does not
            // happen to have 200k. §40 is explicit that `Unknown` is better than
            // confidently reporting false data.
            let catalogued = crate::pricing::descriptor_for(&id);
            let ctx = catalogued.map(|d| d.context_window).unwrap_or(0);
            let max_output_tokens = catalogued.map(|d| d.max_output_tokens).unwrap_or(0);
            let capabilities = match catalogued {
                Some(descriptor) => {
                    let mut capabilities = descriptor.capabilities.clone();
                    capabilities.max_tokens = max_output_tokens;
                    capabilities.context_window = ctx;
                    capabilities
                }
                // No catalog entry: report nothing we cannot back. `streaming` is
                // a property of Anthropic's API rather than of the model name,
                // so it is the one flag stated without catalog support.
                None => ModelCapabilities {
                    streaming: true,
                    ..Default::default()
                },
            };

            let mut strengths = vec!["reasoning".to_string(), "coding".to_string()];
            if capabilities.vision {
                strengths.push("vision".to_string());
            }

            // Looked up before `id` is moved into the struct below.
            let pricing = crate::pricing::price_for(&id);

            models.push(crate::types::FullModelInfo {
                id,
                name: display_name,
                provider: "anthropic".to_string(),
                description: None,
                context_window: ctx,
                max_output_tokens,
                capabilities,
                // Anthropic's model list carries no prices, so this is either
                // a dated catalog claim or nothing. Never a guess, and never
                // another model's price.
                pricing,
                created: 0,
                available: true,
                                supported_parameters: Default::default(),
                });
        }
        Ok(models)
    }
}

#[async_trait]
impl LLMProvider for AnthropicProvider {
    fn name(&self) -> &str {
        "anthropic"
    }

    fn supported_models(&self) -> Vec<String> {
        // Cannot hardcode — the Anthropic Messages API protocol is used by
        // many providers (Bedrock, Vertex, proxies). Live discovery only.
        vec![]
    }

    /// Fetch models from live API. Falls back to empty list.
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

        let (system_value, messages) = self.convert_messages(&request.messages)?;

        let mut body = json!({
            "model": model,
            "messages": messages,
            "max_tokens": request.max_tokens.unwrap_or(1024),
        });

        // Handle system field - support caching via cache_system flag
        if let Some(sys) = system_value {
            if request.cache_system {
                // Format as array with cache_control for prompt caching
                body["system"] = json!([{
                    "type": "text",
                    "text": sys,
                    "cache_control": {
                        "type": "ephemeral"
                    }
                }]);
            } else {
                body["system"] = json!(sys);
            }
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

        let (system_value, messages) = self.convert_messages(&request.messages)?;

        let mut body = serde_json::json!({
            "model": model,
            "messages": messages,
            "max_tokens": request.max_tokens.unwrap_or(1024),
            "stream": true,
        });

        // Handle system field - support caching via cache_system flag
        if let Some(sys) = system_value {
            if request.cache_system {
                // Format as array with cache_control for prompt caching
                body["system"] = json!([{
                    "type": "text",
                    "text": sys,
                    "cache_control": {
                        "type": "ephemeral"
                    }
                }]);
            } else {
                body["system"] = json!(sys);
            }
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
        let response = self
            .client
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

                while let Some(pos) = buffer.find("\n") {
                    let line = buffer[..pos].to_string();
                        buffer = buffer[pos + 1..].to_string();

                        if let Some(data) = line.strip_prefix("data: ") {
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

        let (system_value, messages) = self.convert_messages(&request.messages)?;

        let tools: Vec<serde_json::Value> = functions
            .iter()
            .map(|f| {
                let schema = f.parameters.as_ref().cloned().unwrap_or(json!({
                    "type": "object",
                    "properties": {}
                }));
                json!({
                    "name": f.name,
                    "description": f.description,
                    "input_schema": schema
                })
            })
            .collect();

        let mut body = json!({
            "model": model,
            "messages": messages,
            "max_tokens": request.max_tokens.unwrap_or(4096),
            "tools": tools,
            "tool_choice": {"type": "auto"}
        });

        // Handle system field - support caching via cache_system flag
        if let Some(sys) = system_value {
            if request.cache_system {
                // Format as array with cache_control for prompt caching
                body["system"] = json!([{
                    "type": "text",
                    "text": sys,
                    "cache_control": {
                        "type": "ephemeral"
                    }
                }]);
            } else {
                body["system"] = json!(sys);
            }
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
        KNOWN_MODELS.iter().any(|m| model.contains(m)) || model.contains("claude")
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

        // Exact match against the versioned catalog.
        //
        // This was a `match` on `contains(..)`. Two problems: the arms were
        // ordered so that `gemini-1.0-pro` matched `contains("pro")` before it
        // could reach the `1.0` arm, and the `_ =>` arm priced *any*
        // unrecognised model as the middle tier. §41: "Never silently apply an
        // unrelated fallback model price to an unknown model." An unknown model
        // reports zero, which is visible.
        let pricing = crate::pricing::price_for(&model);
        let input_cost_per_million = pricing
            .as_ref()
            .map(|p| p.prompt_tokens * 1_000.0)
            .unwrap_or(0.0);
        let output_cost_per_million = pricing
            .as_ref()
            .map(|p| p.completion_tokens * 1_000.0)
            .unwrap_or(0.0);

        if pricing.is_none() {
            tracing::debug!(
                model = %model,
                provider = "anthropic",
                "no catalog price for this model; reporting a zero estimate rather than \
                 guessing an unrelated model's price"
            );
        }

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
            model,
            provider: "anthropic".to_string(),
        })
    }

    fn config(&self) -> &ProviderConfig {
        unimplemented!()
    }
}
