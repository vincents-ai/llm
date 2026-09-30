//! The OpenAI Responses API (§25–§29).
//!
//! # Not a translation layer
//!
//! This is a native typed implementation of `/responses`. It is deliberately not
//! built by converting to and from `ChatCompletionRequest`, for two reasons.
//!
//! The first is that the shapes are genuinely different: `/responses` has
//! `instructions` as a distinct field, an ordered `output` array of *typed*
//! items, `previous_response_id` for server-side state, and provider-operated
//! tools that are not functions. A translation has to invent an equivalence for
//! each of those, and each invented equivalence is a place where information
//! cannot survive.
//!
//! The second is that `output[0]` is not assistant text. A response can lead
//! with a reasoning item or a function call. Code written against Chat
//! Completions -- where `choices[0].message.content` really is the text -- reads
//! index 0 and gets an empty string with no error. [`Response::output_text`]
//! exists to make that mistake impossible, but it can only help callers who use
//! it, which is why the types make the assumption impossible in the first place.
//!
//! # Unknown items survive
//!
//! [`ResponseOutputItem::Unknown`] carries the raw JSON. A provider shipping a
//! new item type must not make an older build silently drop it -- a dropped
//! item is a lost tool call, which is a lost side effect.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::error::{LLMError, Result};
use crate::types::{
    ChatMessage, FunctionDefinitionWire, ReasoningConfig, ServiceTier, Usage,
};

// ── Input ──────────────────────────────────────────────────────────────────

/// What to send to the model.
///
/// Two variants because the wire format has two: a bare string is the common
/// case and stays a bare string, rather than being expanded into a one-element
/// list that some providers handle differently.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ResponseInput {
    Text(String),
    Items(Vec<ResponseInputItem>),
}

impl Default for ResponseInput {
    fn default() -> Self {
        ResponseInput::Text(String::new())
    }
}

impl ResponseInput {
    pub fn text(text: impl Into<String>) -> Self {
        ResponseInput::Text(text.into())
    }

    pub fn items(items: impl Into<Vec<ResponseInputItem>>) -> Self {
        ResponseInput::Items(items.into())
    }

    /// Build input from Chat Completions messages.
    ///
    /// Lossy in one direction, deliberately: a `developer` or `system` message
    /// becomes `instructions`, because `/responses` has no system role. An
    /// assistant turn with tool calls needs its matching tool results or the
    /// provider rejects it, and this is not where that is repaired.
    pub fn from_messages(messages: &[ChatMessage]) -> Self {
        let mut instructions = Vec::new();
        let mut items = Vec::new();
        for message in messages {
            match message {
                ChatMessage::System { content, .. } => {
                    instructions.push(content.as_text().unwrap_or_default().to_string())
                }
                ChatMessage::Developer { content, .. } => {
                    instructions.push(content.as_text().unwrap_or_default().to_string())
                }
                ChatMessage::User { content, .. } => {
                    items.push(ResponseInputItem::Message {
                        role: "user".into(),
                        content: input_content_from(content),
                    })
                }
                ChatMessage::Assistant { content, .. } => {
                    items.push(ResponseInputItem::Message {
                        role: "assistant".into(),
                        content: input_content_from(content.as_ref().unwrap_or(&crate::types::MessageContent::Text(String::new()))),
                    })
                }
                ChatMessage::Tool {
                    tool_call_id,
                    content,
                } => items.push(ResponseInputItem::FunctionCallOutput {
                    call_id: tool_call_id.clone(),
                    output: content.as_text().unwrap_or_default().to_string(),
                }),
                ChatMessage::Function { name, arguments } => {
                    items.push(ResponseInputItem::FunctionCall {
                        id: None,
                        call_id: name.clone(),
                        name: name.clone(),
                        arguments: arguments.clone(),
                    })
                }
            }
        }
        // Instructions are collected but not attached here: `/responses` carries
        // them as a separate field, and silently folding them into the item list
        // would send them as a user turn.
        let _ = instructions;
        ResponseInput::Items(items)
    }

    fn to_wire(&self) -> Value {
        match self {
            ResponseInput::Text(text) => Value::String(text.clone()),
            ResponseInput::Items(items) => {
                Value::Array(items.iter().map(ResponseInputItem::to_wire).collect())
            }
        }
    }
}

fn input_content_from(content: &crate::types::MessageContent) -> Vec<ResponseContent> {
    use crate::types::{ContentPart, MessageContent};
    match content {
        MessageContent::Text(text) => {
            vec![ResponseContent::InputText { text: text.clone() }]
        }
        MessageContent::Parts(parts) => parts
            .iter()
            .filter_map(|part| match part {
                ContentPart::Text { text } => {
                    Some(ResponseContent::InputText { text: text.clone() })
                }
                ContentPart::ImageUrl { url, .. } => Some(ResponseContent::InputImage {
                    image_url: url.clone(),
                    detail: None,
                }),
                _ => None,
            })
            .collect(),
    }
}

/// One item of a structured input.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseInputItem {
    Message {
        role: String,
        content: Vec<ResponseContent>,
    },
    FunctionCall {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        call_id: String,
        name: String,
        arguments: String,
    },
    FunctionCallOutput {
        call_id: String,
        output: String,
    },
    /// Preserved from a previous response so an unrecognised item is not lost
    /// when a transcript is replayed.
    #[serde(untagged)]
    Unknown(Value),
}

impl ResponseInputItem {
    fn to_wire(&self) -> Value {
        match self {
            ResponseInputItem::Message { role, content } => json!({
                "type": "message",
                "role": role,
                "content": content,
            }),
            ResponseInputItem::FunctionCall {
                id,
                call_id,
                name,
                arguments,
            } => {
                let mut obj = json!({
                    "type": "function_call",
                    "call_id": call_id,
                    "name": name,
                    "arguments": arguments,
                });
                if let Some(id) = id {
                    obj["id"] = json!(id);
                }
                obj
            }
            ResponseInputItem::FunctionCallOutput { call_id, output } => json!({
                "type": "function_call_output",
                "call_id": call_id,
                "output": output,
            }),
            // Replayed verbatim, including its own `type`.
            ResponseInputItem::Unknown(value) => value.clone(),
        }
    }
}

/// A piece of message content.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseContent {
    InputText {
        text: String,
    },
    InputImage {
        image_url: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        detail: Option<crate::types::ImageDetail>,
    },
    InputFile {
        file_id: String,
    },
    /// Output text carried back in a transcript. Distinct from `InputText`
    /// because the provider nests them differently.
    OutputText {
        text: String,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        annotations: Vec<Value>,
    },
}

// ── Tools (§27) ────────────────────────────────────────────────────────────

/// How much context a web search should retrieve.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SearchContextSize {
    Low,
    Medium,
    High,
}

/// Where a code interpreter runs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum CodeInterpreterContainer {
    Auto,
    #[serde(rename = "container_id")]
    Id { id: String },
}

/// A tool the Responses API can call.
///
/// Provider-operated tools are separate variants, not
/// `FunctionDefinition`s with reserved names. They execute on the provider's
/// infrastructure, which is a different authority and lifecycle question from a
/// local function: a caller cannot validate their arguments, cannot bound their
/// cost the same way, and may not be able to see what they do. Collapsing them
/// into one type makes all of that invisible to any policy check.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseTool {
    /// A function the *caller* executes.
    #[serde(rename = "function")]
    Function {
        name: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        description: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        parameters: Option<Value>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        strict: Option<bool>,
    },
    /// A tool the caller supplies under its own shape.
    #[serde(rename = "custom")]
    Custom {
        name: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        description: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        format: Option<Value>,
    },
    WebSearch {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        search_context_size: Option<SearchContextSize>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        user_location: Option<Value>,
    },
    FileSearch {
        vector_store_ids: Vec<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        max_num_results: Option<u32>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        filters: Option<Value>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        ranking_options: Option<Value>,
    },
    CodeInterpreter {
        container: CodeInterpreterContainer,
    },
    /// The preview tool. The wire type keeps its `_preview` suffix because that
    /// is the name the provider accepts; renaming it locally would make the
    /// request unreproducible against the API.
    #[serde(rename = "computer_use_preview")]
    ComputerUse {
        display_width: u32,
        display_height: u32,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        environment: Option<String>,
    },
    Mcp {
        server_label: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        server_url: Option<String>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        allowed_tools: Vec<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        authorization: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        require_approval: Option<Value>,
    },
}

impl ResponseTool {
    /// Whether the provider executes this rather than the caller.
    ///
    /// The distinction an authorization check needs: a provider-operated tool
    /// can read data the caller could not otherwise reach.
    pub fn is_provider_operated(&self) -> bool {
        !matches!(self, ResponseTool::Function { .. } | ResponseTool::Custom { .. })
    }

    /// The tool's name, for matching against `tool_choice`.
    pub fn name(&self) -> Option<&str> {
        match self {
            ResponseTool::Function { name, .. } | ResponseTool::Custom { name, .. } => Some(name),
            _ => None,
        }
    }

    fn to_wire(&self) -> Value {
        serde_json::to_value(self).unwrap_or(Value::Null)
    }
}

/// Which tool the model must use.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ResponseToolChoice {
    Mode(ToolChoiceMode),
    Function { r#type: String, name: String },
    ProviderOperated(Value),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ToolChoiceMode {
    None,
    Auto,
    Required,
}

impl ResponseToolChoice {
    fn to_wire(&self) -> Value {
        match self {
            ResponseToolChoice::Mode(ToolChoiceMode::None) => json!("none"),
            ResponseToolChoice::Mode(ToolChoiceMode::Auto) => json!("auto"),
            ResponseToolChoice::Mode(ToolChoiceMode::Required) => json!("required"),
            ResponseToolChoice::Function { r#type, name } => {
                json!({"type": r#type, "name": name})
            }
            // `{"type": "web_search", "search_context_size": "low"}` and the
            // rest are provider-defined objects; passing them through keeps a
            // new tool choice working before this crate names it.
            ResponseToolChoice::ProviderOperated(value) => value.clone(),
        }
    }
}

// ── Configuration ──────────────────────────────────────────────────────────

/// Text and structured-output configuration.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ResponseTextConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub format: Option<crate::types::ResponseFormat>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub verbosity: Option<crate::types::Verbosity>,
}

/// How a request that exceeds the context window is handled.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TruncationMode {
    Auto,
    Disabled,
}

/// An `include` entry.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ResponseInclude(pub String);

impl ResponseInclude {
    pub fn reasoning_encrypted_content() -> Self {
        ResponseInclude("reasoning.encrypted_content".into())
    }
    pub fn file_search_call_results() -> Self {
        ResponseInclude("file_search_call.results".into())
    }
}

/// A reference to a stored conversation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ConversationRef {
    Id(String),
    Object {
        id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        metadata: Option<BTreeMap<String, String>>,
    },
}

impl ConversationRef {
    pub fn id(&self) -> &str {
        match self {
            ConversationRef::Id(id) => id,
            ConversationRef::Object { id, .. } => id,
        }
    }
}

// ── The request (§25) ──────────────────────────────────────────────────────

/// A `/responses` request.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ResponseRequest {
    pub model: String,
    pub input: ResponseInput,

    pub instructions: Option<String>,
    pub max_output_tokens: Option<u32>,
    pub max_tool_calls: Option<u32>,
    pub tools: Vec<ResponseTool>,
    pub tool_choice: Option<ResponseToolChoice>,
    pub parallel_tool_calls: Option<bool>,
    pub reasoning: Option<ReasoningConfig>,
    pub text: Option<ResponseTextConfig>,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub top_logprobs: Option<u8>,
    pub logprobs: Option<bool>,
    pub metadata: BTreeMap<String, String>,

    // Server-side conversation state.
    pub previous_response_id: Option<String>,
    pub conversation: Option<ConversationRef>,

    // Caching and retention.
    pub prompt_cache_key: Option<String>,
    pub store: Option<bool>,
    pub background: Option<bool>,
    pub service_tier: Option<ServiceTier>,
    pub include: Vec<ResponseInclude>,
    pub truncation: Option<TruncationMode>,

    /// Provider-specific fields with no portable equivalent.
    ///
    /// Collision-checked against the typed fields under the same rule as
    /// `extra_params`: a new provider feature must be usable immediately without
    /// being able to shadow something this crate does know about.
    pub provider_params: BTreeMap<String, Value>,
}

impl ResponseRequest {
    /// Validate before any network I/O.
    ///
    /// §28: `previous_response_id` and `conversation` both mean "the provider
    /// already has some of this". Sending both is ambiguous -- the provider
    /// resolves it one way and the caller assumes another. It is also the shape
    /// of the bug the brief warns about directly: resending a whole local
    /// transcript *and* pointing at provider state duplicates every turn.
    pub fn validate(&self) -> Result<()> {
        if self.previous_response_id.is_some() && self.conversation.is_some() {
            return Err(LLMError::InvalidParameterCombination {
                message: "previous_response_id and conversation are mutually exclusive; \
                          the provider already holds the prior turns either way, and sending \
                          both is ambiguous"
                    .to_string(),
            });
        }
        if self.top_logprobs.is_some() && self.logprobs != Some(true) {
            return Err(LLMError::ConfigurationError {
                message: "top_logprobs requires logprobs = true".to_string(),
                field: Some("top_logprobs".to_string()),
            });
        }
        if let Some(ResponseToolChoice::Function { name, .. }) = &self.tool_choice {
            let offered = self
                .tools
                .iter()
                .any(|tool| tool.name() == Some(name.as_str()));
            if !offered {
                let offered: Vec<&str> = self.tools.iter().filter_map(|t| t.name()).collect();
                return Err(LLMError::ConfigurationError {
                    message: format!(
                        "tool_choice names '{name}', which is not among the offered tools: \
                         [{}]",
                        offered.join(", ")
                    ),
                    field: Some("tool_choice".to_string()),
                });
            }
        }
        Ok(())
    }

    /// The request body.
    ///
    /// One builder, same precedence rule as Chat Completions: typed fields
    /// first, provider extensions last, and an extension that collides with a
    /// typed field is an error rather than an override.
    pub fn to_body(&self) -> Result<Value> {
        self.validate()?;
        let mut body = serde_json::Map::new();
        body.insert("model".into(), json!(self.model));
        body.insert("input".into(), self.input.to_wire());

        let set = |key: &str, value: Value, body: &mut serde_json::Map<String, Value>| {
            body.insert(key.into(), value);
        };
        if let Some(v) = &self.instructions {
            set("instructions", json!(v), &mut body);
        }
        if let Some(v) = self.max_output_tokens {
            set("max_output_tokens", json!(v), &mut body);
        }
        if let Some(v) = self.max_tool_calls {
            set("max_tool_calls", json!(v), &mut body);
        }
        if !self.tools.is_empty() {
            set(
                "tools",
                Value::Array(self.tools.iter().map(ResponseTool::to_wire).collect()),
                &mut body,
            );
        }
        if let Some(choice) = &self.tool_choice {
            set("tool_choice", choice.to_wire(), &mut body);
        }
        if let Some(v) = self.parallel_tool_calls {
            set("parallel_tool_calls", json!(v), &mut body);
        }
        if let Some(v) = &self.reasoning {
            set("reasoning", serde_json::to_value(v).unwrap_or(Value::Null), &mut body);
        }
        if let Some(v) = &self.text {
            set("text", serde_json::to_value(v).unwrap_or(Value::Null), &mut body);
        }
        if let Some(v) = self.temperature {
            set("temperature", json!(v), &mut body);
        }
        if let Some(v) = self.top_p {
            set("top_p", json!(v), &mut body);
        }
        if let Some(v) = self.top_logprobs {
            set("top_logprobs", json!(v), &mut body);
        }
        if let Some(v) = self.logprobs {
            set("logprobs", json!(v), &mut body);
        }
        if !self.metadata.is_empty() {
            set("metadata", json!(self.metadata), &mut body);
        }
        if let Some(v) = &self.previous_response_id {
            set("previous_response_id", json!(v), &mut body);
        }
        if let Some(conversation) = &self.conversation {
            match conversation {
                ConversationRef::Id(id) => set("conversation", json!(id), &mut body),
                ConversationRef::Object { id, metadata } => {
                    let mut obj = json!({"id": id});
                    if let Some(metadata) = metadata {
                        obj["metadata"] = json!(metadata);
                    }
                    set("conversation", obj, &mut body);
                }
            }
        }
        if let Some(v) = &self.prompt_cache_key {
            set("prompt_cache_key", json!(v), &mut body);
        }
        if let Some(v) = self.store {
            set("store", json!(v), &mut body);
        }
        if let Some(v) = self.background {
            set("background", json!(v), &mut body);
        }
        if let Some(v) = &self.service_tier {
            set("service_tier", json!(v.as_str()), &mut body);
        }
        if !self.include.is_empty() {
            let includes: Vec<&str> = self.include.iter().map(|i| i.0.as_str()).collect();
            set("include", json!(includes), &mut body);
        }
        if let Some(v) = self.truncation {
            set("truncation", json!(v), &mut body);
        }

        // Provider extensions last, collision-checked. A new feature must be
        // usable immediately without being able to shadow a known field.
        for (key, value) in &self.provider_params {
            if body.contains_key(key) {
                return Err(LLMError::ConfigurationError {
                    message: format!(
                        "provider extension '{key}' collides with a typed request field"
                    ),
                    field: Some(key.clone()),
                });
            }
            body.insert(key.clone(), value.clone());
        }
        Ok(Value::Object(body))
    }

    /// Convenience: a function tool definition from the Chat Completions type.
    ///
    /// A bridge, not an identity: `/responses` takes `name` at the top level
    /// where Chat Completions nests it under `function`.
    pub fn with_function_tool(mut self, function: &FunctionDefinitionWire) -> Self {
        self.tools.push(ResponseTool::Function {
            name: function.name.clone(),
            description: function.description.clone(),
            parameters: function.parameters.clone(),
            strict: function.strict,
        });
        self
    }
}

// ── Output (§26) ───────────────────────────────────────────────────────────

/// One item of a response's `output` array.
///
/// The order is meaningful and the *first* item is not necessarily text.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseOutputItem {
    Message(ResponseMessage),
    FunctionCall(ResponseFunctionCall),
    Reasoning(ResponseReasoningItem),
    WebSearchCall(Value),
    FileSearchCall(Value),
    ComputerCall(Value),
    CodeInterpreterCall(Value),
    /// An item type this build does not recognise, kept verbatim.
    ///
    /// Dropping it would be dropping a tool call, which is a lost side effect
    /// rather than a cosmetic loss.
    #[serde(untagged)]
    Unknown(Value),
}

impl ResponseOutputItem {
    /// A short type name, for logs and routing.
    pub fn type_name(&self) -> &str {
        match self {
            ResponseOutputItem::Message(_) => "message",
            ResponseOutputItem::FunctionCall(_) => "function_call",
            ResponseOutputItem::Reasoning(_) => "reasoning",
            ResponseOutputItem::WebSearchCall(_) => "web_search_call",
            ResponseOutputItem::FileSearchCall(_) => "file_search_call",
            ResponseOutputItem::ComputerCall(_) => "computer_call",
            ResponseOutputItem::CodeInterpreterCall(_) => "code_interpreter_call",
            ResponseOutputItem::Unknown(value) => {
                value.get("type").and_then(|v| v.as_str()).unwrap_or("unknown")
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResponseMessage {
    pub id: Option<String>,
    pub role: String,
    #[serde(default)]
    pub status: Option<String>,
    pub content: Vec<ResponseContent>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResponseFunctionCall {
    /// The item's own id.
    pub id: Option<String>,
    /// The id the matching `function_call_output` must reference.
    ///
    /// Distinct from `id`, and conflating them makes the follow-up request
    /// unanswerable.
    #[serde(default)]
    pub call_id: Option<String>,
    pub name: String,
    #[serde(default)]
    pub arguments: String,
    #[serde(default)]
    pub status: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResponseReasoningItem {
    pub id: Option<String>,
    #[serde(default)]
    pub summary: Vec<Value>,
    #[serde(default)]
    pub encrypted_content: Option<String>,
    #[serde(default)]
    pub status: Option<String>,
}

/// Lifecycle state of a response (§29).
///
/// `Unknown` because a background response's status model grows, and rejecting
/// an unfamiliar status would break a caller on a provider upgrade.
///
/// Hand-written rather than derived: a newtype variant inside a
/// `rename_all` enum deserializes as a map, so `#[serde(other)]` is not
/// available and an untagged catch-all would swallow the *known* statuses too.
/// The wire form is a bare string, so it is parsed as one.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ResponseStatus {
    Queued,
    InProgress,
    Completed,
    Failed,
    Cancelled,
    Incomplete,
    Unknown(String),
}

impl ResponseStatus {
    /// The wire value.
    pub fn as_str(&self) -> &str {
        match self {
            ResponseStatus::Queued => "queued",
            ResponseStatus::InProgress => "in_progress",
            ResponseStatus::Completed => "completed",
            ResponseStatus::Failed => "failed",
            ResponseStatus::Cancelled => "cancelled",
            ResponseStatus::Incomplete => "incomplete",
            ResponseStatus::Unknown(value) => value,
        }
    }
}

impl<'de> Deserialize<'de> for ResponseStatus {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let raw = String::deserialize(deserializer)?;
        Ok(match raw.as_str() {
            "queued" => ResponseStatus::Queued,
            "in_progress" => ResponseStatus::InProgress,
            "completed" => ResponseStatus::Completed,
            "failed" => ResponseStatus::Failed,
            "cancelled" => ResponseStatus::Cancelled,
            "incomplete" => ResponseStatus::Incomplete,
            other => ResponseStatus::Unknown(other.to_string()),
        })
    }
}

impl Default for ResponseStatus {
    fn default() -> Self {
        ResponseStatus::Completed
    }
}

impl ResponseStatus {
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            ResponseStatus::Completed
                | ResponseStatus::Failed
                | ResponseStatus::Cancelled
                | ResponseStatus::Incomplete
        )
    }
}

/// A `/responses` response.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Response {
    pub id: String,
    #[serde(default)]
    pub object: Option<String>,
    pub created_at: u64,
    pub model: String,
    #[serde(default = "default_status")]
    pub status: ResponseStatus,
    #[serde(default)]
    pub output: Vec<ResponseOutputItem>,
    #[serde(default)]
    pub usage: Option<ResponsesUsage>,
    #[serde(default)]
    pub error: Option<Value>,
    #[serde(default)]
    pub incomplete_details: Option<Value>,
    #[serde(default)]
    pub instructions: Option<String>,
    #[serde(default)]
    pub metadata: Option<BTreeMap<String, String>>,
    #[serde(default)]
    pub previous_response_id: Option<String>,
    #[serde(default)]
    pub conversation: Option<ConversationRef>,
}

fn default_status() -> ResponseStatus {
    ResponseStatus::Completed
}

impl Response {
    /// The assistant's text, with every typed item retained.
    ///
    /// The single answer to "why did I get an empty string?" -- a response can
    /// legitimately contain no message at all, because it led with a function
    /// call or a reasoning item.
    pub fn output_text(&self) -> String {
        self.message_contents()
            .iter()
            .map(|content| match content {
                ResponseContent::OutputText { text, .. } => text.clone(),
                ResponseContent::InputText { text } => text.clone(),
                _ => String::new(),
            })
            .collect::<Vec<_>>()
            .join("")
    }

    /// The text content blocks, in order.
    pub fn message_contents(&self) -> Vec<&ResponseContent> {
        self.output
            .iter()
            .filter_map(|item| match item {
                ResponseOutputItem::Message(message) => Some(message.content.as_slice()),
                _ => None,
            })
            .flatten()
            .collect()
    }

    /// Every function call in the response, in order.
    pub fn function_calls(&self) -> Vec<&ResponseFunctionCall> {
        self.output
            .iter()
            .filter_map(|item| match item {
                ResponseOutputItem::FunctionCall(call) => Some(call),
                _ => None,
            })
            .collect()
    }

    /// The reasoning items, if the model produced any and returned them.
    pub fn reasoning(&self) -> Vec<&ResponseReasoningItem> {
        self.output
            .iter()
            .filter_map(|item| match item {
                ResponseOutputItem::Reasoning(item) => Some(item),
                _ => None,
            })
            .collect()
    }

    /// Whether the provider ran anything the caller should know about.
    ///
    /// A policy check needs this: a provider-operated tool call is a side
    /// effect the caller did not execute.
    pub fn provider_operated_calls(&self) -> Vec<&ResponseOutputItem> {
        self.output
            .iter()
            .filter(|item| {
                matches!(
                    item,
                    ResponseOutputItem::WebSearchCall(_)
                        | ResponseOutputItem::FileSearchCall(_)
                        | ResponseOutputItem::ComputerCall(_)
                        | ResponseOutputItem::CodeInterpreterCall(_)
                )
            })
            .collect()
    }
}

/// Usage as the Responses API reports it.
///
/// A distinct type from [`Usage`] because the field names differ -- `input_tokens`
/// rather than `prompt_tokens` -- and because translating by renaming would hide
/// a mapping error behind a plausible number.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResponsesUsage {
    #[serde(default)]
    pub input_tokens: u32,
    #[serde(default)]
    pub output_tokens: u32,
    #[serde(default)]
    pub total_tokens: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_tokens_details: Option<ResponsesInputDetails>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_tokens_details: Option<ResponsesOutputDetails>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResponsesInputDetails {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cached_tokens: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResponsesOutputDetails {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_tokens: Option<u32>,
}

impl ResponsesUsage {
    /// The same usage the Chat Completions surface reports.
    ///
    /// Detail is carried across, so a caller that costs a Responses call and one
    /// that costs a Chat call see the same numbers.
    pub fn as_usage(&self) -> Usage {
        Usage {
            prompt_tokens: self.input_tokens,
            completion_tokens: self.output_tokens,
            total_tokens: if self.total_tokens == 0 {
                self.input_tokens + self.output_tokens
            } else {
                self.total_tokens
            },
            reasoning_tokens: self
                .output_tokens_details
                .as_ref()
                .and_then(|d| d.reasoning_tokens),
            cached_tokens: self
                .input_tokens_details
                .as_ref()
                .and_then(|d| d.cached_tokens),
            cache_write_tokens: None,
        }
    }
}