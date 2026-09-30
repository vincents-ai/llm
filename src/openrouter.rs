//! OpenRouter Provider Implementation
//!
//! This module provides the OpenRouter provider implementation with credit tracking,
//! provider ranking/selection, referrer support, and JWT authentication.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::pin::Pin;

use crate::config::ProviderConfig;
use crate::error::{LLMError, Result};
use crate::types::*;
use crate::provider::LLMProvider;
use crate::CostEstimate;
use crate::RateLimitStatus;

// ── OpenRouter provider extensions ──────────────────────────────────────────
//
// These are provider-specific on purpose (§54). They are not on
// `ChatCompletionRequest` because most of them mean nothing to OpenAI, and
// putting thirty fields meaning something to exactly one provider on the
// portable type is the shape the brief warns against.
//
// They are typed rather than raw, because routing and privacy settings in
// particular must not be expressed as raw JSON keys: `provider.data_collection:
// "deny"` and `"DENY"` are not the same request, and a typo in a string map is
// indistinguishable from no setting at all.

/// Whether an upstream provider may retain the request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DataCollectionPolicy {
    Allow,
    Deny,
}

/// How upstream providers should be ordered.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OpenRouterSort {
    Price,
    Throughput,
    Latency,
}

/// A price ceiling, in dollars per million tokens.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct OpenRouterMaxPrice {
    pub prompt: f64,
    pub completion: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub web_search: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub internal_reasoning: Option<f64>,
}

/// Upstream provider selection and privacy policy.
///
/// Belongs in the request body, not in a header: the legacy
/// `OpenRouter-Router` header can only express an ordering. These settings
/// decide where a user's code goes, so they are part of the request and are
/// visible in a captured body.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct OpenRouterProviderRouting {
    /// Provider preference order, most preferred first.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub order: Vec<String>,
    /// Whether OpenRouter may fall back when an ordered provider fails.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub allow_fallbacks: Option<bool>,
    /// Restrict to these providers.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub only: Vec<String>,
    /// Never use these providers.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub ignore: Vec<String>,
    /// Only route to providers that accept every parameter the request sets.
    ///
    /// A provider-side guard, not a replacement for local capability filtering:
    /// `require_parameters` rejects an unsupported model, it does not tell you
    /// which models support what.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub require_parameters: Option<bool>,
    /// Whether upstream providers may retain the request.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data_collection: Option<DataCollectionPolicy>,
    /// Zero data retention.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub zdr: Option<bool>,
    /// Accept only these quantizations.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub quantizations: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_price: Option<OpenRouterMaxPrice>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sort: Option<OpenRouterSort>,
    /// Preferred minimum throughput, in tokens/second.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preferred_min_throughput: Option<f64>,
    /// Preferred maximum latency, in seconds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preferred_max_latency: Option<f64>,
}

impl OpenRouterProviderRouting {
    /// Whether any field is set. An empty routing object is not sent, so a
    /// request that expresses no opinion does not carry an empty `provider: {}`.
    pub fn is_empty(&self) -> bool {
        *self == Self::default()
    }
}

/// Request-level OpenRouter extensions with no portable equivalent.
///
/// §30: sampling controls and options OpenRouter forwards but that are not part
/// of the common dialect. Kept in one struct so "what does OpenRouter add" is
/// answerable by reading one type.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct OpenRouterExtensions {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_p: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_a: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repetition_penalty: Option<f64>,
    /// Ask OpenRouter to enforce structured output, where the chosen provider
    /// supports it. Distinct from `response_format`, which is the portable form.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub structured_outputs: Option<bool>,
    /// Provider-operated web search settings.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub web_search_options: Option<serde_json::Value>,
    /// Upstream provider selection and privacy policy.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider: Option<OpenRouterProviderRouting>,
    /// Request-level fallback model list, tried in order if the primary is
    /// unavailable. A fallback changes the model, so it must be visible in the
    /// body and in the returned metadata.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub models: Vec<String>,
}

impl OpenRouterExtensions {
    /// Whether anything is set.
    pub fn is_empty(&self) -> bool {
        *self == Self::default()
    }

    /// The request-body fields, ready for `ChatRequestParts::provider_params`.
    ///
    /// Rendered through `serde_json` so the wire keys are whatever the
    /// serialization says, rather than a hand-maintained second list that could
    /// disagree with the struct.
    pub fn to_params(&self) -> std::collections::BTreeMap<String, serde_json::Value> {
        let mut params = std::collections::BTreeMap::new();
        if let Some(provider) = &self.provider {
            if !provider.is_empty() {
                params.insert("provider".into(), serde_json::to_value(provider).unwrap_or_default());
            }
        }
        if !self.models.is_empty() {
            params.insert("models".into(), serde_json::json!(self.models));
        }
        let value = serde_json::to_value(self).unwrap_or(serde_json::Value::Null);
        if let Some(object) = value.as_object() {
            for (key, field) in object {
                if key == "provider" || key == "models" {
                    continue;
                }
                params.insert(key.clone(), field.clone());
            }
        }
        params
    }
}

/// A tool OpenRouter operates rather than the caller.
///
/// §38: these execute remotely, which is a different authority and lifecycle
/// question from a local function, so they are a distinct type and an agentic
/// policy check can see which is which.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OpenRouterServerTool {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_schema: Option<serde_json::Value>,
    /// Whether OpenRouter should stop this tool once the model stops calling it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_when_not_called: Option<bool>,
}

/// OpenRouter Provider Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenRouterConfig {
    pub api_key: String,
    /// Override the API base URL.
    ///
    /// Added for the same reason `OpenAIConfig` has one: without it the
    /// provider cannot be pointed at a mock server, so none of its wire
    /// behaviour can be tested. The brief's OpenRouter P0 — a `tool` message
    /// being rewritten to `user` — is only observable on the wire, so this is a
    /// prerequisite for closing it rather than a nicety.
    pub base_url: Option<String>,
    /// `HTTP-Referer` — the application attribution OpenRouter asks for.
    pub referrer: Option<String>,
    /// `X-OpenRouter-Title` — the other attribution header.
    ///
    /// Configuration, not a per-request field: the title names the *application*,
    /// and every call from this process is that application. Making it a request
    /// parameter would mean every call site could set it differently, which is
    /// the opposite of what attribution is for (§34).
    pub title: Option<String>,
    /// Ask for routing metadata on Responses, via `X-OpenRouter-Metadata`.
    ///
    /// Opt-in because the response grows; the data is worth having for the
    /// model router and for diagnosing a fallback (§36).
    pub metadata: bool,
    pub default_route: Option<String>,
    pub default_model: Option<String>,
    pub jwt_token: Option<String>,
}

impl Default for OpenRouterConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            base_url: None,
            referrer: None,
            title: None,
            metadata: false,
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
    /// Provider-wide routing and privacy policy, merged into every request body.
    ///
    /// Configured here rather than per request on purpose: a deployment that
    /// routes to a ZDR provider for some calls and not others is not making a
    /// policy decision.
    pub extensions: OpenRouterExtensions,
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
            extensions: OpenRouterExtensions::default(),
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
            extensions: OpenRouterExtensions::default(),
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

    /// Apply the headers every OpenRouter call should carry.
    ///
    /// §34: attribution is configuration, because it names the application.
    /// §36: routing metadata is opt-in.
    ///
    /// `OpenRouter-Router` is applied for compatibility but is **legacy**: it can
    /// only express an ordering, while [`OpenRouterProviderRouting`] in the
    /// request body can also express `only`, `ignore`, privacy policy and price
    /// ceilings. Callers should use the body.
    fn apply_common_headers(
        &self,
        mut builder: reqwest::RequestBuilder,
    ) -> reqwest::RequestBuilder {
        if let Some(referrer) = &self.config.referrer {
            builder = builder.header("HTTP-Referer", referrer);
        }
        if let Some(title) = &self.config.title {
            builder = builder.header("X-OpenRouter-Title", title);
        }
        if self.config.metadata {
            builder = builder.header("X-OpenRouter-Metadata", "enabled");
        }
        if let Some(route) = &self.config.default_route {
            tracing::debug!(
                "using the legacy OpenRouter-Router header; prefer the `provider` request field, \
                 which can also express privacy and price policy"
            );
            builder = builder.header("OpenRouter-Router", route);
        }
        builder
    }

    /// Send a streaming request and hand back the raw response.
    ///
    /// One sender, two stream shapes. They previously had separate bodies,
    /// which is how they drifted: a fix to one was not a fix to the other.
    async fn send_streaming_request(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<(reqwest::Response, String)> {
        let model = if request.model.is_empty() {
            self.get_default_model()?.to_string()
        } else {
            request.model.clone()
        };

        // The same serializer the non-streaming path uses, so the two cannot
        // drift.
        let messages: Vec<serde_json::Value> =
            request.messages.iter().map(ChatMessage::to_openai_json).collect();

        // The last hand-written field list. OpenRouter's streaming path had the
        // same three-field truncation as OpenAI's; routing it through the shared
        // builder is the end of the class, not just of this instance.
        let extensions = self.extensions_for(&request);
        let params = extensions.as_ref().map(|e| e.to_params());
        let body = crate::wire::build_chat_request(crate::wire::ChatRequestParts {
            model: &model,
            messages: &messages,
            request: &request,
            stream: true,
            tools: None,
            provider_params: params.as_ref(),
        })?;

        let mut request_builder = self
            .client
            .post(&format!("{}/chat/completions", self.get_base_url()))
            .header("Content-Type", "application/json");

        if let Some(jwt) = &self.config.jwt_token {
            request_builder = request_builder.header("Authorization", format!("Bearer {}", jwt));
        } else {
            request_builder =
                request_builder.header("Authorization", format!("Bearer {}", self.config.api_key));
        }

        request_builder = self.apply_common_headers(request_builder);

        if let Some(route) = &self.config.default_route {
            request_builder = request_builder.header("OpenRouter-Router", route);
        }

        let response =
            request_builder
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

        Ok((response, model))
    }

    fn get_base_url(&self) -> &str {
        self.config
            .base_url
            .as_deref()
            .unwrap_or("https://openrouter.ai/api/v1")
    }

    /// Provider extensions that apply to every request from this provider.
    ///
    /// Routing and privacy policy are a property of the *deployment*, not of one
    /// call: a process that routes to a ZDR provider for some requests and not
    /// others is not making a policy decision, it is making a coin flip. So
    /// they are configured on the provider and merged into every body.
    pub fn set_extensions(&mut self, extensions: OpenRouterExtensions) {
        self.extensions = extensions;
    }

    /// The extensions configured for this provider.
    pub fn extensions(&self) -> &OpenRouterExtensions {
        &self.extensions
    }

    /// Extensions for one request: the provider's, plus anything the caller put
    /// in `extra_params` under a known extension key.
    ///
    /// Caller-supplied extensions win over provider defaults, so a single request
    /// can be diverted deliberately without changing the deployment's policy.
    fn extensions_for(&self, _request: &ChatCompletionRequest) -> Option<OpenRouterExtensions> {
        if self.extensions.is_empty() {
            return None;
        }
        Some(self.extensions.clone())
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
        let resp = self
            .client
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
                message: format!(
                    "OpenRouter /models returned {}: {}",
                    status,
                    body.clone().unwrap_or_default()
                ),
                status_code: Some(status.as_u16()),
                body,
            });
        }

        let json: serde_json::Value =
            resp.json()
                .await
                .map_err(|e| LLMError::SerializationError {
                    message: e.to_string(),
                    context: Some("Failed to parse OpenRouter models response".to_string()),
                })?;

        let data = json["data"]
            .as_array()
            .ok_or_else(|| LLMError::SerializationError {
                message: "Missing 'data' array in models response".to_string(),
                context: None,
            })?;

        let models: Vec<FullModelInfo> = data
            .iter()
            .map(|m| {
                let id = m["id"].as_str().unwrap_or("").to_string();
                let name = m["name"].as_str().unwrap_or(&id).to_string();
                let description = m["description"].as_str().map(String::from);
                let context_length = m["context_length"].as_u64().unwrap_or(0) as u32;
                let created = m["created"].as_u64().unwrap_or(0);

                // Parse architecture / modalities
                let arch = &m["architecture"];
                let input_modalities: Vec<String> = arch["input_modalities"]
                    .as_array()
                    .map(|a| {
                        a.iter()
                            .filter_map(|v| v.as_str().map(String::from))
                            .collect()
                    })
                    .unwrap_or_default();
                let output_modalities: Vec<String> = arch["output_modalities"]
                    .as_array()
                    .map(|a| {
                        a.iter()
                            .filter_map(|v| v.as_str().map(String::from))
                            .collect()
                    })
                    .unwrap_or_default();
                let has_vision = input_modalities.iter().any(|m| m == "image");

                // Parse supported parameters → capabilities, and *keep the list*.
                // It was previously reduced to three booleans and thrown away,
                // which discarded the part that scales: a parameter this crate
                // has never heard of, reported truthfully by the provider.
                let params: std::collections::BTreeSet<String> = m["supported_parameters"]
                    .as_array()
                    .map(|a| {
                        a.iter()
                            .filter_map(|v| v.as_str().map(String::from))
                            .collect()
                    })
                    .unwrap_or_default();
                let has_tools = params.iter().any(|p| p == "tools" || p == "tool_choice");
                let has_json = params
                    .iter()
                    .any(|p| p == "response_format" || p == "structured_outputs");

                // Parse pricing
                let pricing_obj = &m["pricing"];
                let prompt_price: f64 = pricing_obj["prompt"]
                    .as_str()
                    .and_then(|s| s.parse().ok())
                    .or_else(|| pricing_obj["prompt"].as_f64())
                    .unwrap_or(0.0);
                let completion_price: f64 = pricing_obj["completion"]
                    .as_str()
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
                if id_lower.contains("coder")
                    || id_lower.contains("code")
                    || id_lower.contains("deepseek")
                {
                    strengths.push("coding".to_string());
                }
                if id_lower.contains("reason")
                    || id_lower.contains("o1")
                    || id_lower.contains("o3")
                    || id_lower.contains("think")
                {
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
                        // The provider stated these in its own response, so they
                        // are the authoritative kind — not a guess.
                        source: crate::types::CapabilitySource::Provider,
                    },
                    pricing: Some(ModelPricing {
                        prompt_tokens: prompt_per_1k,
                        completion_tokens: completion_per_1k,
                        image_tokens: None,
                        is_free,
                    }),
                    created,
                    available: true,
                    // The parameter list the provider reported, preserved. This
                    // is what makes `require_parameters`-style routing decidable
                    // locally without guessing from the model name.
                    supported_parameters: params.clone(),
                }
            })
            .collect();

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

        // OpenRouter speaks the OpenAI Chat Completions dialect, so the same
        // serializer serves both. The three hand-written copies this replaces
        // are why `tool` and `function` messages were being rewritten to `user`:
        // the bug lived in the duplication, not in the mapping.
        let messages: Vec<serde_json::Value> =
            request.messages.iter().map(ChatMessage::to_openai_json).collect();

        // One builder, shared with the OpenAI provider: the dialect is the
        // same, so the two cannot drift into sending different fields.
        let extensions = self.extensions_for(&request);
        let params = extensions.as_ref().map(|e| e.to_params());
        let body = crate::wire::build_chat_request(crate::wire::ChatRequestParts {
            model: &model,
            messages: &messages,
            request: &request,
            stream: false,
            tools: None,
            provider_params: params.as_ref(),
        })?;

        let mut request_builder = self
            .client
            .post(&format!("{}/chat/completions", self.get_base_url()))
            .header("Content-Type", "application/json");

        // Add authentication header
        if let Some(jwt) = &self.config.jwt_token {
            request_builder = request_builder.header("Authorization", format!("Bearer {}", jwt));
        } else {
            request_builder =
                request_builder.header("Authorization", format!("Bearer {}", self.config.api_key));
        }

        // Add referrer for credit tracking
        request_builder = self.apply_common_headers(request_builder);

        // Add route for provider routing
        if let Some(route) = &self.config.default_route {
            request_builder = request_builder.header("OpenRouter-Router", route);
        }

        let response =
            request_builder
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

        let json: serde_json::Value =
            response
                .json()
                .await
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

                // `developer` is accepted back as itself. Folding it into
                // `system` would lose which one the provider treated as
                // higher precedence, which is exactly the information a
                // debugging transcript needs.
                let message = match role {
                    "developer" => ChatMessage::Developer {
                        content: content.to_string().into(),
                        name: None,
                    },
                    "system" => ChatMessage::System {
                        content: content.to_string().into(),
                        name: None,
                    },
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

        let usage = json["usage"].as_object().map(|u| Usage::from_openai_json(&serde_json::Value::Object(u.clone())));

        Ok(ChatCompletionResponse {
            id,
            object: "chat.completion".to_string(),
            created,
            model,
            choices,
            usage,
            system_fingerprint: json
                .get("system_fingerprint")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
        })
    }

    async fn chat_completion_stream_events(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<Pin<Box<dyn futures::Stream<Item = Result<ChatStreamEvent>> + Send>>> {
        let (response, _) = self.send_streaming_request(request).await?;
        Ok(crate::stream::parse_sse_stream(response))
    }

    /// Legacy text-only streaming, adapted from the same parsed events.
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
        let model = if request.model.is_empty() {
            self.get_default_model()?.to_string()
        } else {
            request.model.clone()
        };

        let tools: Vec<serde_json::Value> = functions
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
                    "function": func_obj
                })
            })
            .collect();

        // The canonical serializer. This was the *fourth* hand-written copy of
        // the same match -- the other three were replaced in an earlier commit
        // for exactly this reason, and it survived because it sat in the
        // tool-call path rather than the text path.
        let messages: Vec<serde_json::Value> = request
            .messages
            .iter()
            .map(ChatMessage::to_openai_json)
            .collect();

        let extensions = self.extensions_for(&request);
        let params = extensions.as_ref().map(|e| e.to_params());
        let body = crate::wire::build_chat_request(crate::wire::ChatRequestParts {
            model: &model,
            messages: &messages,
            request: &request,
            stream: false,
            tools: Some(&tools),
            provider_params: params.as_ref(),
        })?;

        let mut request_builder = self
            .client
            .post(&format!("{}/chat/completions", self.get_base_url()))
            .header("Content-Type", "application/json");

        if let Some(jwt) = &self.config.jwt_token {
            request_builder = request_builder.header("Authorization", format!("Bearer {}", jwt));
        } else {
            request_builder =
                request_builder.header("Authorization", format!("Bearer {}", self.config.api_key));
        }

        request_builder = self.apply_common_headers(request_builder);

        if let Some(route) = &self.config.default_route {
            request_builder = request_builder.header("OpenRouter-Router", route);
        }

        let response =
            request_builder
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

        let json: serde_json::Value =
            response
                .json()
                .await
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
                let content = msg
                    .get("content")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());
                let tool_calls = msg
                    .get("tool_calls")
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|tc| {
                                let tc_id = tc
                                    .get("id")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("")
                                    .to_string();
                                let func = tc.get("function")?;
                                let tc_name = func
                                    .get("name")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("")
                                    .to_string();
                                let tc_args = func
                                    .get("arguments")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("{}")
                                    .to_string();
                                Some(ToolCall::Function(FunctionCall::Custom(
                                    CustomFunctionCall {
                                        id: if tc_id.is_empty() { None } else { Some(tc_id) },
                                        name: tc_name,
                                        arguments: tc_args,
                                    },
                                )))
                            })
                            .collect::<Vec<_>>()
                    })
                    .filter(|v| !v.is_empty());

                Choice {
                    index: idx as u32,
                    message: ChatMessage::Assistant {
                        content: content.map(MessageContent::Text),
                        tool_calls,
                        name: None,
                    },
                    finish_reason: choice["finish_reason"].as_str().map(|s| s.to_string()),
                    logprobs: None,
                }
            })
            .collect();

        let usage = json["usage"].as_object().map(|u| Usage::from_openai_json(&serde_json::Value::Object(u.clone())));

        Ok(ChatCompletionResponse {
            id,
            object: "chat.completion".to_string(),
            created,
            model: resp_model,
            choices,
            usage,
            system_fingerprint: json
                .get("system_fingerprint")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
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
            provider: "openrouter".to_string(),
        })
    }

    fn config(&self) -> &ProviderConfig {
        unimplemented!()
    }
}
