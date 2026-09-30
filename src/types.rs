use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};

/// Sort order for model listings
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SortOrder {
    Ascending,
    Descending,
}

/// Role of a chat message sender
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    System,
    User,
    Assistant,
    Function,
    Tool,
}

/// A chat message in a conversation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "role", rename_all = "lowercase")]
pub enum ChatMessage {
    /// System message that sets the behavior of the assistant
    #[serde(rename = "system")]
    System {
        content: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },

    /// User message providing input
    #[serde(rename = "user")]
    User {
        content: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },

    /// Assistant message (model response)
    #[serde(rename = "assistant")]
    Assistant {
        content: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        tool_calls: Option<Vec<ToolCall>>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },

    /// Function call from the model
    #[serde(rename = "function")]
    Function { name: String, arguments: String },

    /// Tool result from function execution
    #[serde(rename = "tool")]
    Tool {
        tool_call_id: String,
        content: String,
    },
}

impl ChatMessage {
    /// Create a system message (no cache control)
    pub fn system(content: impl Into<String>) -> Self {
        ChatMessage::System {
            content: content.into(),
            name: None,
        }
    }

    /// Create a system message with cache control for prompt caching
    /// Note: cache control is now handled via ChatCompletionRequest::cache_system field
    pub fn system_with_cache(content: impl Into<String>) -> Self {
        ChatMessage::System {
            content: content.into(),
            name: None,
        }
    }

    /// Create a user message
    pub fn user(content: impl Into<String>) -> Self {
        ChatMessage::User {
            content: content.into(),
            name: None,
        }
    }

    /// Create an assistant message
    pub fn assistant(content: impl Into<String>) -> Self {
        ChatMessage::Assistant {
            content: Some(content.into()),
            tool_calls: None,
            name: None,
        }
    }

    /// Get the text content of a message
    pub fn content(&self) -> Option<&str> {
        match self {
            ChatMessage::System { content, .. } => Some(content),
            ChatMessage::User { content, .. } => Some(content),
            ChatMessage::Assistant { content, .. } => content.as_deref(),
            ChatMessage::Function { arguments, .. } => Some(arguments),
            ChatMessage::Tool { content, .. } => Some(content),
        }
    }

    /// Serialize this message into the OpenAI Chat Completions wire shape.
    ///
    /// One canonical serializer for the protocol, used by every provider that
    /// speaks it and by every request path. The duplication this replaces is
    /// what produced the OpenRouter defect: three hand-written copies of the
    /// same match, two of which mapped `tool` and `function` onto `user` and so
    /// silently destroyed native tool-call conversations.
    ///
    /// The shapes preserved, per the brief's required round-trip:
    ///
    /// ```text
    /// system
    /// user
    /// assistant
    /// assistant + tool_calls
    /// tool + tool_call_id
    /// function + name        (legacy spelling of an assistant tool call)
    /// ```
    pub fn to_openai_json(&self) -> serde_json::Value {
        use serde_json::{json, Value};
        match self {
            ChatMessage::System { content, name } => {
                let mut obj = json!({ "role": "system", "content": content });
                if let Some(name) = name {
                    obj["name"] = json!(name);
                }
                obj
            }
            ChatMessage::User { content, name } => {
                let mut obj = json!({ "role": "user", "content": content });
                if let Some(name) = name {
                    obj["name"] = json!(name);
                }
                obj
            }
            ChatMessage::Assistant {
                content,
                tool_calls,
                name,
            } => {
                let mut obj = json!({
                    "role": "assistant",
                    // `null` rather than an omitted key: the API distinguishes
                    // "no content" from "content not supplied" for assistant
                    // turns that carry only tool calls.
                    "content": content.clone().map(Value::String).unwrap_or(Value::Null),
                });
                if let Some(tool_calls) = tool_calls {
                    let serialized: Vec<Value> = tool_calls
                        .iter()
                        .map(crate::types::serialize_tool_call)
                        .collect();
                    if !serialized.is_empty() {
                        obj["tool_calls"] = json!(serialized);
                    }
                }
                if let Some(name) = name {
                    obj["name"] = json!(name);
                }
                obj
            }
            ChatMessage::Function { name, arguments } => {
                // The legacy single-function message. `content` carries the
                // arguments, which is what the wire format means by it.
                json!({
                    "role": "function",
                    "name": name,
                    "content": arguments,
                })
            }
            ChatMessage::Tool {
                tool_call_id,
                content,
            } => {
                // `tool_call_id` is the whole point of a tool message: it is what
                // links the result back to the call. Dropping it, or rewording
                // the role to `user`, breaks the conversation rather than
                // merely changing its shape.
                json!({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": content,
                })
            }
        }
    }

    /// Get the role of the message
    pub fn role(&self) -> MessageRole {
        match self {
            ChatMessage::System { .. } => MessageRole::System,
            ChatMessage::User { .. } => MessageRole::User,
            ChatMessage::Assistant { .. } => MessageRole::Assistant,
            ChatMessage::Function { .. } => MessageRole::Function,
            ChatMessage::Tool { .. } => MessageRole::Tool,
        }
    }
}

/// Cache control for Anthropic prompt caching.
/// When set on a system or user message, the API will cache the prefix.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CacheControl {
    /// Cache type - typically "ephemeral" for Anthropic
    #[serde(default = "default_cache_type")]
    pub r#type: String,
}

fn default_cache_type() -> String {
    "ephemeral".to_string()
}

impl CacheControl {
    /// Create a new cache control with "ephemeral" type
    pub fn ephemeral() -> Self {
        Self {
            r#type: "ephemeral".to_string(),
        }
    }
}

/// A tool call from the model
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ToolCall {
    Function(FunctionCall),
}

/// A function call requested by the model
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum FunctionCall {
    Code(CodeInterpreterCall),
    Retrieval(RetrievalCall),
    Custom(CustomFunctionCall),
}

/// Code interpreter function call
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CodeInterpreterCall {
    pub code: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stdin: Option<String>,
}

/// Retrieval function call
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RetrievalCall {
    pub query: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_results: Option<u32>,
}

/// Custom function call from function definitions
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CustomFunctionCall {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub name: String,
    pub arguments: String,
}

/// Function definition for function calling
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FunctionDefinition {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parameters: Option<serde_json::Value>,
}

/// A tool call from the model
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCallDelta {
    pub index: u32,
    #[serde(rename = "type", default, skip_serializing_if = "Option::is_none")]
    pub tool_type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub function: Option<FunctionCall>,
}

/// Serialize a `ToolCall` into the OpenAI wire shape.
pub fn serialize_tool_call(tool_call: &ToolCall) -> serde_json::Value {
    use serde_json::{json, Value};
    match tool_call {
        ToolCall::Function(FunctionCall::Custom(custom)) => json!({
            "id": custom.id.as_deref().unwrap_or(""),
            "type": "function",
            "function": {
                "name": custom.name,
                "arguments": custom.arguments,
            }
        }),
        // Every other variant is preserved verbatim rather than dropped. A tool
        // call the protocol evolves past should reach the provider as-is, not
        // disappear: a silently omitted call reads to an agent as "the model
        // chose not to call a tool".
        other => serde_json::to_value(other).unwrap_or(Value::Null),
    }
}

/// Streaming choice delta
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChoiceDelta {
    pub index: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub delta: Option<ChatMessage>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

/// Usage statistics for a completion
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Usage {
    #[serde(default)]
    pub prompt_tokens: u32,
    #[serde(default)]
    pub completion_tokens: u32,
    #[serde(default)]
    pub total_tokens: u32,
}

impl Usage {
    /// Create new usage with prompt and completion tokens
    pub fn new(prompt_tokens: u32, completion_tokens: u32) -> Self {
        let total_tokens = prompt_tokens + completion_tokens;
        Self {
            prompt_tokens,
            completion_tokens,
            total_tokens,
        }
    }

    /// Calculate cost based on pricing
    pub fn calculate_cost(&self, input_cost: f64, output_cost: f64) -> f64 {
        (self.prompt_tokens as f64 * input_cost) + (self.completion_tokens as f64 * output_cost)
    }
}

/// A provider string that this crate does not enumerate.
///
/// A validated wrapper rather than an enum with a catch-all variant, because
/// `#[serde(other)]` only works on a unit variant and therefore *discards* the
/// value. A wrapper round-trips whatever the provider sent, so a new `effort`
/// level reaching a build that predates it is forwarded rather than dropped or
/// rejected — which is the whole forward-compatibility requirement.
macro_rules! provider_string {
    ($(#[$meta:meta])* $name:ident, $doc:literal) => {
        $(#[$meta])*
        #[doc = $doc]
        #[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
        #[serde(transparent)]
        pub struct $name(pub String);

        impl $name {
            /// The raw provider value.
            pub fn as_str(&self) -> &str {
                &self.0
            }
        }

        impl std::fmt::Display for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str(&self.0)
            }
        }

        impl From<&str> for $name {
            fn from(value: &str) -> Self {
                Self(value.to_string())
            }
        }
    };
}

provider_string!(
    ReasoningEffort,
    "Reasoning effort. Free-form so a provider that adds a level is not a      breaking change; the known values are the associated constants below."
);

impl ReasoningEffort {
    pub fn none() -> Self {
        "none".into()
    }
    pub fn minimal() -> Self {
        "minimal".into()
    }
    pub fn low() -> Self {
        "low".into()
    }
    pub fn medium() -> Self {
        "medium".into()
    }
    pub fn high() -> Self {
        "high".into()
    }
}

/// How much reasoning detail a model should return.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningSummary {
    Auto,
    Concise,
    Detailed,
}

/// Portable reasoning intent.
///
/// Providers map this onto their own shape — OpenRouter exposes both `reasoning`
/// and `reasoning_effort`, OpenAI's Responses API has a `reasoning` object — so
/// the caller states an intent and the provider decides the wire form. Ordinary
/// reasoning control must not require hand-written JSON.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ReasoningConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effort: Option<ReasoningEffort>,
    /// Cap on reasoning tokens, where the provider exposes one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    /// Ask the provider not to return reasoning content.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exclude: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub summary: Option<ReasoningSummary>,
}

provider_string!(
    Verbosity,
    "Output verbosity, independent of reasoning effort. OpenAI and OpenRouter      expose the two separately, and a new level must not be a breaking      release, so the value is free-form and round-trips."
);

impl Verbosity {
    pub fn low() -> Self {
        "low".into()
    }
    pub fn medium() -> Self {
        "medium".into()
    }
    pub fn high() -> Self {
        "high".into()
    }
}

/// Structured-output request.
///
/// Serialized by hand because the wire shape nests the schema details under a
/// `json_schema` key:
/// `{"type":"json_schema","json_schema":{"name":...,"schema":{...},"strict":true}}`.
/// A derived `#[serde(tag = "type")]` would flatten those fields to the top
/// level, which is a body the provider rejects.
#[derive(Debug, Clone, PartialEq)]
pub enum ResponseFormat {
    /// Free text.
    Text,
    /// Any valid JSON object.
    JsonObject,
    /// JSON matching a schema.
    JsonSchema {
        name: String,
        description: Option<String>,
        schema: serde_json::Value,
        strict: Option<bool>,
    },
}

impl Serialize for ResponseFormat {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeMap;
        use serde_json::Value;

        let mut map = serializer.serialize_map(None)?;
        match self {
            ResponseFormat::Text => {
                map.serialize_entry("type", "text")?;
            }
            ResponseFormat::JsonObject => {
                map.serialize_entry("type", "json_object")?;
            }
            ResponseFormat::JsonSchema {
                name,
                description,
                schema,
                strict,
            } => {
                map.serialize_entry("type", "json_schema")?;
                let mut inner = serde_json::Map::new();
                inner.insert("name".into(), Value::String(name.clone()));
                if let Some(description) = description {
                    inner.insert("description".into(), Value::String(description.clone()));
                }
                inner.insert("schema".into(), schema.clone());
                if let Some(strict) = strict {
                    inner.insert("strict".into(), Value::Bool(*strict));
                }
                map.serialize_entry("json_schema", &Value::Object(inner))?;
            }
        }
        map.end()
    }
}

impl<'de> Deserialize<'de> for ResponseFormat {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        use serde::de::Error;
        let value = serde_json::Value::deserialize(deserializer)?;
        match value.get("type").and_then(|v| v.as_str()) {
            Some("text") => Ok(ResponseFormat::Text),
            Some("json_object") => Ok(ResponseFormat::JsonObject),
            Some("json_schema") => {
                let inner = value
                    .get("json_schema")
                    .ok_or_else(|| D::Error::custom("json_schema format has no 'json_schema' object"))?;
                Ok(ResponseFormat::JsonSchema {
                    name: inner
                        .get("name")
                        .and_then(|v| v.as_str())
                        .ok_or_else(|| D::Error::custom("json_schema requires a name"))?
                        .to_string(),
                    description: inner
                        .get("description")
                        .and_then(|v| v.as_str())
                        .map(str::to_string),
                    schema: inner
                        .get("schema")
                        .cloned()
                        .ok_or_else(|| D::Error::custom("json_schema requires a schema"))?,
                    strict: inner.get("strict").and_then(|v| v.as_bool()),
                })
            }
            other => Err(D::Error::custom(format!(
                "unknown response_format type: {}",
                other.unwrap_or("<missing>")
            ))),
        }
    }
}

/// A tool the caller is offering the model.
///
/// Serialized by hand for the same reason as [`ResponseFormat`]: the wire shape
/// nests a function's details under a `function` key
/// (`{"type":"function","function":{...}}`), and an internally-tagged derived
/// `Serialize` would flatten them beside `type` instead.
#[derive(Debug, Clone, PartialEq)]
pub enum ToolDefinition {
    Function(FunctionDefinitionWire),
    /// A provider-operated tool, which is not a local function and must not be
    /// shaped like one.
    Custom {
        name: String,
        description: Option<String>,
        input_schema: Option<serde_json::Value>,
    },
}

impl Serialize for ToolDefinition {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeMap;

        let mut map = serializer.serialize_map(None)?;
        match self {
            ToolDefinition::Function(function) => {
                map.serialize_entry("type", "function")?;
                map.serialize_entry("function", function)?;
            }
            ToolDefinition::Custom {
                name,
                description,
                input_schema,
            } => {
                map.serialize_entry("type", "custom")?;
                map.serialize_entry("name", name)?;
                if let Some(description) = description {
                    map.serialize_entry("description", description)?;
                }
                if let Some(schema) = input_schema {
                    map.serialize_entry("input_schema", schema)?;
                }
            }
        }
        map.end()
    }
}

impl<'de> Deserialize<'de> for ToolDefinition {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        use serde::de::Error;
        let value = serde_json::Value::deserialize(deserializer)?;
        match value.get("type").and_then(|v| v.as_str()) {
            Some("function") => {
                let inner = value
                    .get("function")
                    .ok_or_else(|| D::Error::custom("function tool has no 'function' object"))?;
                Ok(ToolDefinition::Function(
                    serde_json::from_value(inner.clone()).map_err(D::Error::custom)?,
                ))
            }
            Some("custom") => Ok(ToolDefinition::Custom {
                name: value
                    .get("name")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| D::Error::custom("custom tool requires a name"))?
                    .to_string(),
                description: value
                    .get("description")
                    .and_then(|v| v.as_str())
                    .map(str::to_string),
                input_schema: value.get("input_schema").cloned(),
            }),
            other => Err(D::Error::custom(format!(
                "unknown tool type: {}",
                other.unwrap_or("<missing>")
            ))),
        }
    }
}

/// A function offered to the model, in the wire shape.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FunctionDefinitionWire {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parameters: Option<serde_json::Value>,
    /// Structured-output conformance for the arguments, where the provider
    /// supports it. Without `strict`, a model may return arguments that do not
    /// match the schema, which the caller must then handle.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

/// Which tools the model may call.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolChoice {
    None,
    Auto,
    Required,
    Function { name: String },
}

impl ToolChoice {
    pub fn to_wire(&self) -> serde_json::Value {
        match self {
            ToolChoice::None => serde_json::json!("none"),
            ToolChoice::Auto => serde_json::json!("auto"),
            ToolChoice::Required => serde_json::json!("required"),
            ToolChoice::Function { name } => {
                serde_json::json!({"type": "function", "function": {"name": name}})
            }
        }
    }
}

provider_string!(
    ServiceTier,
    "Service tier, where the provider offers one. Free-form for the same      reason as `Verbosity`."
);

impl ServiceTier {
    pub fn auto() -> Self {
        "auto".into()
    }
    pub fn default_tier() -> Self {
        "default".into()
    }
    pub fn flex() -> Self {
        "flex".into()
    }
    pub fn scale() -> Self {
        "scale".into()
    }
    pub fn priority() -> Self {
        "priority".into()
    }
}

/// A chat completion request
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ChatCompletionRequest {
    /// Model identifier (e.g., "gpt-4", "claude-3-opus")
    pub model: String,

    /// Messages to send to the model
    pub messages: Vec<ChatMessage>,

    /// Maximum number of tokens to generate
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,

    /// Sampling temperature (0.0 - 2.0)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,

    /// Top-p sampling (0.0 - 1.0)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,

    /// Number of completions to generate
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n: Option<u32>,

    /// Whether to stream the response
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,

    /// Stop sequences
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub stop: Vec<String>,

    /// Presence penalty (-2.0 to 2.0)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f64>,

    /// Frequency penalty (-2.0 to 2.0)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f64>,

    /// Logit bias for token selection
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub logit_bias: HashMap<String, i32>,

    /// User identifier for tracking
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,

    /// Additional provider-specific parameters
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub extra_params: HashMap<String, serde_json::Value>,

    /// Enable prompt caching for system messages (Anthropic only)
    #[serde(default)]
    pub cache_system: bool,

    // ── Modern parameters ──────────────────────────────────────────────────
    // Each is optional and omitted from the wire when unset, so a request that
    // does not use them serialises byte-identically to one built before they
    // existed.

    /// Token budget that includes reasoning tokens.
    ///
    /// Reasoning models count reasoning against `max_tokens` and reject it, so
    /// this is not an alias for `max_tokens` and the two are never silently
    /// renamed into each other.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<u32>,

    /// Seed for deterministic sampling, where the provider supports it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,

    /// Request log probabilities.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<bool>,

    /// How many alternatives to return per token.
    ///
    /// Only meaningful with `logprobs`; the builder rejects the combination.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<u8>,

    /// Structured output.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormat>,

    /// Tools offered to the model.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ToolDefinition>>,

    /// Which of the offered tools the model may call.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,

    /// Whether the model may emit more than one tool call per turn.
    ///
    /// Forwarded identically on streaming and non-streaming paths, which is why
    /// it belongs in the one builder rather than in a path-specific list.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,

    /// Reasoning intent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<ReasoningConfig>,

    /// Output verbosity.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub verbosity: Option<Verbosity>,

    /// Service tier, where the provider offers one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<ServiceTier>,

    /// Free-form request metadata.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, String>,
}

impl Default for ChatCompletionRequest {
    fn default() -> Self {
        Self {
            model: String::new(),
            messages: Vec::new(),
            max_tokens: None,
            temperature: None,
            top_p: None,
            n: None,
            stream: None,
            stop: Vec::new(),
            presence_penalty: None,
            frequency_penalty: None,
            logit_bias: HashMap::new(),
            user: None,
            max_completion_tokens: None,
            seed: None,
            logprobs: None,
            top_logprobs: None,
            response_format: None,
            tools: None,
            tool_choice: None,
            parallel_tool_calls: None,
            reasoning: None,
            verbosity: None,
            service_tier: None,
            metadata: BTreeMap::new(),
            extra_params: HashMap::new(),
            cache_system: false,
        }
    }
}

/// A chat completion response
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatCompletionResponse {
    /// Unique identifier for this completion
    pub id: String,

    /// Type of object (should be "chat.completion")
    pub object: String,

    /// Unix timestamp of creation
    pub created: u64,

    /// Model used for completion
    pub model: String,

    /// Completion choices
    pub choices: Vec<Choice>,

    /// Usage statistics
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,

    /// System fingerprint for response
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system_fingerprint: Option<String>,
}

/// A completion choice
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Choice {
    /// Index of this choice
    pub index: u32,

    /// Generated message
    pub message: ChatMessage,

    /// Reason for stopping
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,

    /// Log probabilities (if requested)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<LogProbResponse>,
}

/// Log probability information
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LogProbResponse {
    pub content: Vec<TokenLogProb>,
}

/// Token log probability
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TokenLogProb {
    pub token: String,
    pub logprob: f64,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub top_logprobs: Vec<HashMap<String, f64>>,
}

/// Streaming chunk response
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "object", rename_all = "lowercase")]
pub enum Chunk {
    Completion(ChunkCompletion),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChunkCompletion {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChoiceDelta>,
}

/// Model information (basic)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    pub object: String,
    #[serde(default)]
    pub created: u64,
    #[serde(default)]
    pub owned_by: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub permission: Option<Vec<serde_json::Value>>,
}

/// Comprehensive model information with all details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FullModelInfo {
    /// Unique model identifier
    pub id: String,

    /// Human-readable model name
    #[serde(default)]
    pub name: String,

    /// Provider name
    #[serde(default)]
    pub provider: String,

    /// Model description
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,

    /// Context window size in tokens
    #[serde(default)]
    pub context_window: u32,

    /// Maximum output tokens
    #[serde(default)]
    pub max_output_tokens: u32,

    /// Model capabilities
    #[serde(default)]
    pub capabilities: ModelCapabilities,

    /// Pricing information
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pricing: Option<ModelPricing>,

    /// Release date
    #[serde(default)]
    pub created: u64,

    /// Whether the model is currently available
    #[serde(default = "default_true")]
    pub available: bool,
    /// The parameter names the provider says this model accepts.
    ///
    /// OpenRouter reports this directly and it is the authoritative answer to
    /// "can I send `reasoning_effort` to this model?". It was previously parsed
    /// into three booleans and then thrown away, so the list — the part that
    /// scales to parameters nobody has typed yet — did not survive.
    ///
    /// Empty means *not reported*, not *supports nothing*.
    #[serde(default)]
    pub supported_parameters: std::collections::BTreeSet<String>,
}

fn default_true() -> bool {
    true
}

/// Pricing information for a model
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelPricing {
    pub prompt_tokens: f64,     // Cost per 1K input tokens
    pub completion_tokens: f64, // Cost per 1K output tokens
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image_tokens: Option<f64>,
    /// Whether this model is free (both prompt and completion cost are zero)
    #[serde(default)]
    pub is_free: bool,
}

/// Model capabilities
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelCapabilities {
    pub function_calling: bool,
    pub vision: bool,
    pub streaming: bool,
    pub json_mode: bool,
    pub caching: bool,
    #[serde(default)]
    pub max_tokens: u32,
    #[serde(default)]
    pub context_window: u32,
    /// Input modalities (e.g. "text", "image", "audio", "video")
    #[serde(default)]
    pub input_modalities: Vec<String>,
    /// Output modalities (e.g. "text", "image")
    #[serde(default)]
    pub output_modalities: Vec<String>,
    /// Model strengths (e.g. "coding", "reasoning", "math", "creative")
    #[serde(default)]
    pub strengths: Vec<String>,
    /// Where these capability values came from.
    ///
    /// The booleans above cannot express "we do not know", and a guessed `false`
    /// is indistinguishable from a real "this model has no tool support". Every
    /// field is therefore qualified by how it was obtained, and a caller that
    /// must not act on a guess reads this first.
    #[serde(default)]
    pub source: CapabilitySource,
}

/// How a capability value was determined.
///
/// Ordered by decreasing authority. The brief's rule is that nothing should be
/// reported as an authoritative capability claim on the strength of a model-name
/// substring match, so `Heuristic` is a real, observable state rather than
/// something the parsers quietly imply.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CapabilitySource {
    /// Not determined. The booleans are defaults, not findings.
    Unknown,
    /// Derived from a substring match against the model name. Weak, and
    /// explicitly not authoritative.
    Heuristic,
    /// From a maintained, versioned descriptor or pricing catalog.
    Descriptor,
    /// Stated by the provider in its own response.
    Provider,
}

impl Default for CapabilitySource {
    fn default() -> Self {
        CapabilitySource::Unknown
    }
}

impl CapabilitySource {
    /// Whether the capabilities may be relied on for a routing decision.
    pub fn is_authoritative(&self) -> bool {
        matches!(self, CapabilitySource::Provider | CapabilitySource::Descriptor)
    }
}

impl ModelCapabilities {
    /// Build capabilities from a maintained catalog entry.
    pub fn from_descriptor(values: Self) -> Self {
        Self {
            source: CapabilitySource::Descriptor,
            ..values
        }
    }

    /// Build capabilities inferred from a model name.
    ///
    /// The `source` is the point: a caller can tell this apart from a fact.
    pub fn from_heuristic(values: Self) -> Self {
        Self {
            source: CapabilitySource::Heuristic,
            ..values
        }
    }

    /// Build capabilities the provider itself reported.
    pub fn from_provider(values: Self) -> Self {
        Self {
            source: CapabilitySource::Provider,
            ..values
        }
    }
}

/// A model variant
#[derive(Debug, Clone)]
pub struct ModelVariant {
    pub id: String,
    pub name: String,
    pub pricing: ModelPricing,
    pub capabilities: ModelCapabilities,
}
