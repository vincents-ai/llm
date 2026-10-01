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
    /// Application-authored instructions. Distinct from `System`; see
    /// [`ChatMessage::Developer`].
    Developer,
    System,
    User,
    Assistant,
    Function,
    Tool,
}

/// How much detail a vision model should spend on an image.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ImageDetail {
    /// Let the provider decide.
    Auto,
    Low,
    High,
}

/// The container format of inline audio.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AudioFormat {
    Wav,
    Mp3,
    Flac,
    Opus,
    Pcm16,
}

/// Serialize content, falling back to `null` when it cannot be expressed.
///
/// `to_openai_json` is infallible because the rest of the crate, including the
/// stored-session format, depends on that. A construction mistake must not
/// panic a decode path, so it degrades to `null` here and is reported by
/// `try_to_openai_json`.
fn content_json_or_null(content: &MessageContent) -> serde_json::Value {
    content.to_openai_json().unwrap_or(serde_json::Value::Null)
}

/// The reserved body key a provider's `x-request-id` is recorded under.
pub const PROVIDER_REQUEST_ID_KEY: &str = "_provider_request_id";

/// One piece of a multimodal message.
///
/// A typed enum rather than a JSON blob because the brief's point is that these
/// are *not* ad-hoc: `detail`, `format` and the `file_id`/`file_data`
/// alternatives are protocol concepts, and a caller assembling them as raw JSON
/// cannot discover them. Keeping them typed also keeps this crate's
/// conformance tests able to assert on them.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    Text {
        text: String,
    },
    ImageUrl {
        url: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        detail: Option<ImageDetail>,
    },
    /// Base64 audio. `data` is the encoded payload without a data-URL prefix.
    InputAudio {
        data: String,
        format: AudioFormat,
    },
    /// A provider-hosted file, by id, or inline, by data.
    File {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        file_id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        file_data: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        filename: Option<String>,
    },
}

/// Message content: plain text, or an ordered list of typed parts.
///
/// `Text` is a distinct variant, not a `Parts` of one, because a text-only
/// message serializes to a bare JSON string. Collapsing the two would change
/// the wire shape of essentially every existing request in the ecosystem for no
/// reason, and `content: "hi"` and `content: [{"type":"text",...}]` are not
/// equally supported across providers.
///
/// The variants are checked at serialization rather than trusted: a `Parts` list
/// that is empty, or that contains a part some role cannot carry, is a caller
/// error worth reporting rather than sending.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

impl Default for MessageContent {
    fn default() -> Self {
        MessageContent::Text(String::new())
    }
}

impl From<String> for MessageContent {
    fn from(value: String) -> Self {
        MessageContent::Text(value)
    }
}

impl From<&String> for MessageContent {
    fn from(value: &String) -> Self {
        MessageContent::Text(value.clone())
    }
}

impl From<&str> for MessageContent {
    fn from(value: &str) -> Self {
        MessageContent::Text(value.to_string())
    }
}

impl From<Vec<ContentPart>> for MessageContent {
    fn from(value: Vec<ContentPart>) -> Self {
        MessageContent::Parts(value)
    }
}

impl MessageContent {
    /// The text, when the content is text.
    ///
    /// `None` for multimodal content rather than a concatenation: joining the
    /// text parts of an image-and-text message into one string is a guess about
    /// what the caller meant, and a lossy one.
    pub fn as_text(&self) -> Option<&str> {
        match self {
            MessageContent::Text(text) => Some(text),
            MessageContent::Parts(_) => None,
        }
    }

    /// The parts, or a single text part when the content is text.
    pub fn parts(&self) -> Vec<ContentPart> {
        match self {
            MessageContent::Text(text) => vec![ContentPart::Text { text: text.clone() }],
            MessageContent::Parts(parts) => parts.clone(),
        }
    }

    /// The number of characters of text in this content.
    ///
    /// For a `Parts` list this sums the text parts and ignores every other
    /// part. It is a size *heuristic* for context-compaction decisions, not a
    /// measure of the content: an image's cost is real but is not expressible
    /// in characters. `Display` gives the same text; this exists because
    /// compaction needs a number.
    pub fn text_char_len(&self) -> usize {
        match self {
            MessageContent::Text(text) => text.chars().count(),
            MessageContent::Parts(parts) => parts
                .iter()
                .map(|part| match part {
                    ContentPart::Text { text } => text.chars().count(),
                    _ => 0,
                })
                .sum(),
        }
    }

    /// The text truncated to at most `max_chars`, if it is text.
    ///
    /// `None` for multimodal content. Truncating a message that contains an
    /// image by character count would silently drop or corrupt the image, so a
    /// caller that compacts context must decide what to do with a
    /// non-truncatable message rather than have this guess.
    pub fn truncated_text(&self, max_chars: usize) -> Option<String> {
        match self {
            MessageContent::Text(text) => Some(text.chars().take(max_chars).collect()),
            MessageContent::Parts(_) => None,
        }
    }

    /// Whether the content carries anything beyond text.
    pub fn is_multimodal(&self) -> bool {
        matches!(self, MessageContent::Parts(parts)
            if parts.iter().any(|part| !matches!(part, ContentPart::Text { .. })))
    }

    /// The canonical wire form.
    ///
    /// A bare string for text, an array for parts, matching what providers
    /// accept. `Err` rather than a lossy fallback for an empty part list, which
    /// is a construction mistake and would otherwise serialize to `"content":
    /// []` and be rejected opaquely by the provider.
    pub fn to_openai_json(&self) -> crate::error::Result<serde_json::Value> {
        use serde_json::Value;
        match self {
            MessageContent::Text(text) => Ok(Value::String(text.clone())),
            MessageContent::Parts(parts) => {
                if parts.is_empty() {
                    return Err(crate::error::LLMError::SerializationError {
                        message: "message content has an empty parts list".to_string(),
                        context: Some(
                            "use MessageContent::Text for a message with no content".to_string(),
                        ),
                    });
                }
                let serialized = parts
                    .iter()
                    .map(serialize_content_part)
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(Value::Array(serialized))
            }
        }
    }
}

impl std::fmt::Display for MessageContent {
    /// The text, for the common case.
    ///
    /// A multimodal message prints its text parts, separated, because `Display`
    /// has nowhere to put an image. It is for logs and diagnostics, not for
    /// building requests -- `to_openai_json` is the wire path.
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MessageContent::Text(text) => formatter.write_str(text),
            MessageContent::Parts(parts) => {
                let text: Vec<&str> = parts
                    .iter()
                    .filter_map(|part| match part {
                        ContentPart::Text { text } => Some(text.as_str()),
                        _ => None,
                    })
                    .collect();
                formatter.write_str(&text.join("\n"))
            }
        }
    }
}

fn serialize_content_part(part: &ContentPart) -> crate::error::Result<serde_json::Value> {
    use serde_json::{json, Value};
    Ok(match part {
        ContentPart::Text { text } => json!({ "type": "text", "text": text }),
        ContentPart::ImageUrl { url, detail } => {
            // The nesting matters: `detail` lives *inside* `image_url`, so a
            // flat `{"type":"image_url","url":...,"detail":...}` is rejected.
            let inner = match detail {
                Some(detail) => json!({ "url": url, "detail": detail }),
                None => json!({ "url": url }),
            };
            json!({ "type": "image_url", "image_url": inner })
        }
        ContentPart::InputAudio { data, format } => {
            let inner = json!({ "data": data, "format": format });
            json!({ "type": "input_audio", "input_audio": inner })
        }
        ContentPart::File {
            file_id,
            file_data,
            filename,
        } => {
            if file_id.is_none() && file_data.is_none() {
                return Err(crate::error::LLMError::SerializationError {
                    message: "a file content part needs either file_id or file_data".to_string(),
                    context: Some(
                        "a part with neither uploads nothing and references nothing".to_string(),
                    ),
                });
            }
            let mut inner = serde_json::Map::new();
            if let Some(file_id) = file_id {
                inner.insert("file_id".into(), json!(file_id));
            }
            if let Some(file_data) = file_data {
                inner.insert("file_data".into(), json!(file_data));
            }
            if let Some(filename) = filename {
                inner.insert("filename".into(), json!(filename));
            }
            json!({ "type": "file", "file": Value::Object(inner) })
        }
    })
}

/// A chat message in a conversation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "role", rename_all = "lowercase")]
pub enum ChatMessage {
    /// Developer message: instructions from the application author.
    ///
    /// A distinct role, not an alias for `System`. On newer OpenAI models the
    /// two are different protocol concepts with different precedence: a
    /// `developer` message is authored code, a `system` message is operator
    /// configuration, and the model resolves a conflict between them in
    /// opposite directions. Folding one into the other would silently pick a
    /// winner, so this variant never serializes as `system`.
    #[serde(rename = "developer")]
    Developer {
        content: MessageContent,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },

    /// System message that sets the behavior of the assistant
    #[serde(rename = "system")]
    System {
        content: MessageContent,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },

    /// User message providing input
    #[serde(rename = "user")]
    User {
        content: MessageContent,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },

    /// Assistant message (model response)
    ///
    /// `content` is `None` rather than empty for a turn that carries only tool
    /// calls: the API distinguishes "no content" from "empty content", and the
    /// distinction changes how the turn is billed and replayed.
    #[serde(rename = "assistant")]
    Assistant {
        content: Option<MessageContent>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        tool_calls: Option<Vec<ToolCall>>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },

    /// Function call from the model
    ///
    /// Retained only for compatibility with conversations recorded against the
    /// legacy single-function message. New tool flows use assistant `tool_calls`
    /// plus `tool` results, which is what the brief asks for and what the
    /// current API expects; this variant is not how new code should say
    /// "the model called a function".
    #[serde(rename = "function")]
    Function { name: String, arguments: String },

    /// Tool result from function execution
    #[serde(rename = "tool")]
    Tool {
        tool_call_id: String,
        content: MessageContent,
    },
}

impl ChatMessage {
    /// Create a developer message.
    ///
    /// Never collapses to `System`. If you want operator configuration, say
    /// [`ChatMessage::system`]; if you want application-authored instructions,
    /// say this. The distinction is the reason the variant exists.
    pub fn developer(content: impl Into<MessageContent>) -> Self {
        ChatMessage::Developer {
            content: content.into(),
            name: None,
        }
    }

    /// Create a system message (no cache control)
    pub fn system(content: impl Into<MessageContent>) -> Self {
        ChatMessage::System {
            content: content.into(),
            name: None,
        }
    }

    /// Create a system message with cache control for prompt caching
    /// Note: cache control is now handled via ChatCompletionRequest::cache_system field
    pub fn system_with_cache(content: impl Into<MessageContent>) -> Self {
        ChatMessage::System {
            content: content.into(),
            name: None,
        }
    }

    /// Create a user message
    pub fn user(content: impl Into<MessageContent>) -> Self {
        ChatMessage::User {
            content: content.into(),
            name: None,
        }
    }

    /// Create a user message from typed content parts.
    pub fn user_parts(parts: impl Into<Vec<ContentPart>>) -> Self {
        ChatMessage::User {
            content: MessageContent::Parts(parts.into()),
            name: None,
        }
    }

    /// Create an assistant message
    pub fn assistant(content: impl Into<MessageContent>) -> Self {
        ChatMessage::Assistant {
            content: Some(content.into()),
            tool_calls: None,
            name: None,
        }
    }

    /// An assistant turn that carries only tool calls.
    ///
    /// `content` is `None`, not empty: see [`ChatMessage::Assistant`].
    pub fn assistant_with_tool_calls(tool_calls: Vec<ToolCall>) -> Self {
        ChatMessage::Assistant {
            content: None,
            tool_calls: Some(tool_calls),
            name: None,
        }
    }

    /// The message content.
    ///
    /// The legacy `function` variant has no `content` field -- the arguments
    /// live under `arguments` -- so it is reported as `None` rather than
    /// fabricating an empty text message that the caller might send onward.
    pub fn message_content(&self) -> Option<&MessageContent> {
        match self {
            ChatMessage::Developer { content, .. }
            | ChatMessage::System { content, .. }
            | ChatMessage::User { content, .. }
            | ChatMessage::Tool { content, .. } => Some(content),
            ChatMessage::Assistant { content, .. } => content.as_ref(),
            ChatMessage::Function { .. } => None,
        }
    }

    /// The text content of a message, when it is text.
    ///
    /// `None` for a multimodal message. Use [`ChatMessage::message_content`] to
    /// get the parts; this returns `None` rather than concatenating the text
    /// parts of an image-and-text message, which would be a guess.
    pub fn content(&self) -> Option<&str> {
        self.message_content().and_then(MessageContent::as_text)
    }

    /// Serialize this message into the OpenAI Chat Completions wire shape.
    ///
    /// Multimodal content is emitted as the canonical wire form; a content
    /// shape the protocol cannot express (an empty parts list) falls back to
    /// `null` rather than panicking, and is reported by
    /// [`ChatMessage::try_to_openai_json`], which callers who care should use.
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
            ChatMessage::Developer { content, name } => {
                // `developer`, always. A provider that does not implement the
                // role will reject it, which is the correct outcome: silently
                // demoting it to `system` would change which instruction wins a
                // conflict.
                let mut obj = json!({ "role": "developer" });
                obj["content"] = content_json_or_null(content);
                if let Some(name) = name {
                    obj["name"] = json!(name);
                }
                obj
            }
            ChatMessage::System { content, name } => {
                let mut obj = json!({ "role": "system" });
                obj["content"] = content_json_or_null(content);
                if let Some(name) = name {
                    obj["name"] = json!(name);
                }
                obj
            }
            ChatMessage::User { content, name } => {
                let mut obj = json!({ "role": "user" });
                obj["content"] = content_json_or_null(content);
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
                let mut obj = json!({ "role": "assistant" });
                // `null` rather than an omitted key: the API distinguishes
                // "no content" from "content not supplied" for assistant
                // turns that carry only tool calls.
                obj["content"] = match content {
                    Some(content) => content_json_or_null(content),
                    None => Value::Null,
                };
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
                let mut obj = json!({ "role": "tool", "tool_call_id": tool_call_id });
                obj["content"] = content_json_or_null(content);
                obj
            }
        }
    }

    /// Get the role of the message
    pub fn role(&self) -> MessageRole {
        match self {
            ChatMessage::Developer { .. } => MessageRole::Developer,
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

    /// Output tokens spent on reasoning, not visible as final text.
    ///
    /// Reported by reasoning models inside
    /// `completion_tokens_details.reasoning_tokens`. It is a *subset* of
    /// `completion_tokens`, not an addition — subtracting it from the total
    /// gives the tokens that reached the caller as text.
    ///
    /// Exposed separately because it is the single largest driver of why a
    /// cheap model is not cheap: a cost optimizer, a model router and a
    /// System-2 escalation analysis all need it, and none of them can recover
    /// it from `total_tokens`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_tokens: Option<u32>,

    /// Input tokens served from the provider's prompt cache.
    ///
    /// Also a subset of `prompt_tokens`. These are billed at a lower rate, so
    /// collapsing all input tokens into one opaque number overstates cost and
    /// makes prompt-cache effectiveness unmeasurable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cached_tokens: Option<u32>,

    /// Input tokens written to the cache this request, where the provider
    /// reports it. Anthropic reports this; OpenAI does not, which is why it is
    /// optional rather than zero.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_write_tokens: Option<u32>,
}

impl Usage {
    /// Input tokens that were *not* served from cache.
    pub fn uncached_prompt_tokens(&self) -> u32 {
        self.prompt_tokens.saturating_sub(self.cached_tokens.unwrap_or(0))
    }

    /// Output tokens that reached the caller as text.
    pub fn visible_completion_tokens(&self) -> u32 {
        self.completion_tokens
            .saturating_sub(self.reasoning_tokens.unwrap_or(0))
    }

    /// Whether the provider reported the fine-grained details.
    ///
    /// `false` means "the provider did not say", which is different from
    /// "the provider said zero". A caller that needs exact cache or reasoning
    /// accounting must check this rather than reading an `Option::None` as 0.
    pub fn has_token_details(&self) -> bool {
        self.reasoning_tokens.is_some() || self.cached_tokens.is_some()
    }
}

impl Usage {
    /// Create new usage with prompt and completion tokens
    pub fn new(prompt_tokens: u32, completion_tokens: u32) -> Self {
        let total_tokens = prompt_tokens + completion_tokens;
        Self {
            prompt_tokens,
            completion_tokens,
            total_tokens,
            reasoning_tokens: None,
            cached_tokens: None,
            cache_write_tokens: None,
        }
    }

    /// Parse the provider's `usage` object.
    ///
    /// One parser for the dialect. The fine-grained tokens live in *nested*
    /// detail objects, and four hand-written copies of this match is how they
    /// got dropped: reading `prompt_tokens` and `completion_tokens` is enough
    /// for a cost calculation to look plausible while being wrong.
    ///
    /// Both `prompt_tokens_details` and `completion_tokens_details` are read
    /// with snake_case keys, which is what the OpenAI Chat Completions API
    /// sends.
    pub fn from_openai_json(value: &serde_json::Value) -> Self {
        let prompt = value["prompt_tokens"].as_u64().unwrap_or(0) as u32;
        let completion = value["completion_tokens"].as_u64().unwrap_or(0) as u32;
        let total = value["total_tokens"].as_u64().unwrap_or(0) as u32;
        Usage {
            prompt_tokens: prompt,
            completion_tokens: completion,
            // Some providers omit `total_tokens`; deriving it is not a guess,
            // it is the definition.
            total_tokens: if total == 0 { prompt + completion } else { total },
            reasoning_tokens: value["completion_tokens_details"]["reasoning_tokens"].as_u64()
                .map(|v| v as u32),
            cached_tokens: value["prompt_tokens_details"]["cached_tokens"]
                .as_u64()
                .map(|v| v as u32),
            cache_write_tokens: value["cache_creation_input_tokens"].as_u64().map(|v| v as u32),
        }
    }

    /// Calculate cost based on pricing.
    ///
    /// Charges for *all* input and output tokens, including cached and
    /// reasoning ones. A cache- or reasoning-aware price needs
    /// [`Usage::uncached_prompt_tokens`] and the reasoning count, because this
    /// method is only correct for providers that price every token the same.
    pub fn calculate_cost(&self, input_cost: f64, output_cost: f64) -> f64 {
        (self.prompt_tokens as f64 * input_cost) + (self.completion_tokens as f64 * output_cost)
    }

    /// Cost with separate rates for cached, uncached and reasoning tokens.
    ///
    /// The rate that applies to reasoning tokens is a judgement call: providers
    /// bill them as output, so that is the default here. A caller that knows
    /// otherwise passes `reasoning_cost_per_1k` explicitly.
    pub fn calculate_detailed_cost(
        &self,
        uncached_input_per_1k: f64,
        cached_input_per_1k: f64,
        output_per_1k: f64,
        reasoning_per_1k: f64,
    ) -> f64 {
        let cached = self.cached_tokens.unwrap_or(0) as f64;
        let uncached = self.uncached_prompt_tokens() as f64;
        let reasoning = self.reasoning_tokens.unwrap_or(0) as f64;
        // Reasoning is a subset of completion, so it is priced separately and
        // removed from the visible completion before that is priced.
        let visible = self.visible_completion_tokens() as f64;
        (uncached * uncached_input_per_1k
            + cached * cached_input_per_1k
            + reasoning * reasoning_per_1k
            + visible * output_per_1k)
            / 1000.0
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

    /// Fields this build does not recognise, kept verbatim.
    ///
    /// §52. `#[serde(flatten)]` is what makes this work: a new provider field
    /// lands here instead of being dropped, so a feature that reaches a build
    /// which predates it is still *visible* to the caller even though this crate
    /// cannot interpret it. Silently discarding it is how a routing decision
    /// ends up made on data the caller believes it received.
    ///
    /// Empty on every response this crate produces, and absent from the
    /// serialized form when empty, so it does not perturb a golden body.
    #[serde(flatten, default, skip_serializing_if = "serde_json::Map::is_empty")]
    pub unknown_fields: serde_json::Map<String, serde_json::Value>,
}

impl ChatCompletionResponse {
    /// Whether an unrecognised field named `key` was present.
    pub fn has_unknown(&self, key: &str) -> bool {
        self.unknown_fields.contains_key(key)
    }

    /// The provider's own correlation id, when it returned one.
    ///
    /// §73: "Provider metadata/request ID survives the wrapper." `id` is the
    /// response's identifier; `x-request-id` is the *provider's* handle for the
    /// call, and it is what a support conversation is indexed by. It only exists
    /// in a response header, so it has to be captured here or not at all.
    pub fn provider_request_id(&self) -> Option<&str> {
        self.unknown_fields
            .get(PROVIDER_REQUEST_ID_KEY)
            .and_then(|value| value.as_str())
    }

    /// An unrecognised field by name.
    pub fn unknown(&self, key: &str) -> Option<&serde_json::Value> {
        self.unknown_fields.get(key)
    }

    /// Collect every key of `body` that this build does not model.
    ///
    /// The `known` list is the point: a provider adding a field is not a
    /// licence to claim *every* unrecognised key, because `id`, `model` and the
    /// rest are mapped by hand elsewhere and would otherwise appear twice --
    /// once interpreted and once raw.
    pub fn collect_unknown(
        body: &serde_json::Value,
        known: &[&str],
    ) -> serde_json::Map<String, serde_json::Value> {
        let mut unknown = serde_json::Map::new();
        let Some(object) = body.as_object() else {
            return unknown;
        };
        for (key, value) in object {
            if !known.contains(&key.as_str()) && !value.is_null() {
                unknown.insert(key.clone(), value.clone());
            }
        }
        unknown
    }
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

/// One alternative token for a position.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TopLogProb {
    pub token: String,
    pub logprob: f64,
    /// The token's UTF-8 bytes, where the provider sends them.
    ///
    /// Not derivable from `token` in general: a provider may return bytes that
    /// do not round-trip to the same string (a byte sequence that is not valid
    /// UTF-8, or a token with leading whitespace the display form normalizes).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bytes: Option<Vec<u8>>,
}

/// A token and its log probability, with the alternatives at that position.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TokenLogProb {
    pub token: String,
    pub logprob: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bytes: Option<Vec<u8>>,
    /// The alternatives, in the provider's order.
    ///
    /// Was `Vec<HashMap<String, f64>>`, which was wrong twice over: two
    /// alternatives with the same token text collapsed into one entry, so the
    /// count -- and therefore the probability mass -- was silently wrong, and
    /// there was nowhere to put `bytes`. A list of records expresses what the
    /// provider actually sent, including duplicates.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub top_logprobs: Vec<TopLogProb>,
}

impl LogProbResponse {
    /// Parse a `choices[].logprobs` object.
    ///
    /// `None` when the provider sent no logprobs -- which is the normal case
    /// unless the request asked for them, and is not an error.
    pub fn from_openai_json(value: &serde_json::Value) -> Option<Self> {
        let content = value.get("content")?.as_array()?;
        Some(Self {
            content: content.iter().filter_map(parse_token).collect(),
        })
    }
}

fn parse_token(entry: &serde_json::Value) -> Option<TokenLogProb> {
    Some(TokenLogProb {
        token: entry.get("token")?.as_str()?.to_string(),
        logprob: entry.get("logprob").and_then(|v| v.as_f64()).unwrap_or(0.0),
        bytes: parse_bytes(entry.get("bytes")),
        top_logprobs: entry
            .get("top_logprobs")
            .and_then(|v| v.as_array())
            .map(|alternatives| alternatives.iter().filter_map(parse_top).collect())
            .unwrap_or_default(),
    })
}

fn parse_top(entry: &serde_json::Value) -> Option<TopLogProb> {
    Some(TopLogProb {
        token: entry.get("token")?.as_str()?.to_string(),
        logprob: entry.get("logprob").and_then(|v| v.as_f64()).unwrap_or(0.0),
        bytes: parse_bytes(entry.get("bytes")),
    })
}

/// A provider's `bytes` array, which is a list of small integers.
fn parse_bytes(value: Option<&serde_json::Value>) -> Option<Vec<u8>> {
    Some(
        value?
            .as_array()?
            .iter()
            .filter_map(|v| v.as_u64().map(|n| n as u8))
            .collect(),
    )
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

/// One provider-neutral event in a chat completion stream.
///
/// The old stream type was a `Chunk` carrying an optional `ChatMessage`, which
/// cannot express a tool-call fragment: `ToolCall` has no index, so two parallel
/// calls interleaved across chunks cannot be assembled — there is nothing to key
/// the fragments by. The result was that the whole tool-call path was silently
/// dropped (the parser read only `delta.content` and skipped everything else).
///
/// These events are what an agentic loop actually needs to reconstruct a
/// response: the provider's own identifiers, per-tool argument fragments in
/// order, refusals, the real finish reason, and usage.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum ChatStreamEvent {
    /// The first chunk of a response, carrying the provider's identity.
    ResponseStart {
        id: String,
        model: String,
        /// The provider's own timestamp, never a locally generated one.
        created: Option<u64>,
        system_fingerprint: Option<String>,
    },
    /// A text fragment.
    TextDelta {
        choice_index: u32,
        text: String,
    },
    /// A tool call has begun: its index, and whatever identity is known.
    ToolCallStart {
        choice_index: u32,
        /// The call's position, which is what fragments are keyed by.
        tool_index: u32,
        id: Option<String>,
        name: Option<String>,
    },
    /// A fragment of a tool call's arguments, to be appended in arrival order.
    ToolCallArgumentsDelta {
        choice_index: u32,
        tool_index: u32,
        delta: String,
    },
    /// The model declined to answer.
    RefusalDelta {
        choice_index: u32,
        delta: String,
    },
    /// A choice is complete, with the provider's own reason.
    ChoiceFinished {
        choice_index: u32,
        reason: Option<String>,
    },
    /// Token accounting, where the provider reports it.
    Usage(Usage),
    /// The provider's stream ended. Not synthesized.
    ResponseFinished,
}

impl ChatStreamEvent {
    /// Adapt a legacy text-only chunk into the event stream.
    ///
    /// Used by the default trait implementation, so a provider that has not
    /// implemented structured streaming still works — degraded to text, which is
    /// what the old stream could express, but no worse.
    pub fn from_chunk(chunk: &Chunk) -> Self {
        match chunk {
            Chunk::Completion(completion) => {
                let mut events = Vec::new();
                for choice in &completion.choices {
                    if let Some(delta) = &choice.delta {
                        if let Some(text) = delta.content() {
                            if !text.is_empty() {
                                events.push(ChatStreamEvent::TextDelta {
                                    choice_index: choice.index,
                                    text: text.to_string(),
                                });
                            }
                        }
                    }
                    if let Some(reason) = &choice.finish_reason {
                        events.push(ChatStreamEvent::ChoiceFinished {
                            choice_index: choice.index,
                            reason: Some(reason.clone()),
                        });
                    }
                }
                // The event stream has no "many" variant, so the first is
                // returned and the rest are dropped; a default adapter cannot
                // do better without inventing an ordering it does not have.
                events
                    .into_iter()
                    .next()
                    .unwrap_or(ChatStreamEvent::ResponseFinished)
            }
        }
    }
}

impl ChatStreamEvent {
    /// Project an event back onto the legacy text-only `Chunk` type.
    ///
    /// The legacy shape has no room for a tool-call index, so a fragment is
    /// carried as a partial `ToolCall` whose arguments are the delta. That is
    /// enough for a consumer that handles one tool at a time; a consumer that
    /// needs interleaved parallel calls must use
    /// [`chat_completion_stream_events`] and [`ToolCallAssembler`].
    pub fn to_legacy_chunk(&self) -> Chunk {
        let (id, created, model, choices) = match self {
            ChatStreamEvent::ResponseStart {
                id,
                model,
                created,
                ..
            } => (
                id.clone(),
                created.unwrap_or(0),
                model.clone(),
                Vec::new(),
            ),
            ChatStreamEvent::TextDelta {
                choice_index,
                text,
            } => (
                String::new(),
                0,
                String::new(),
                vec![ChoiceDelta {
                    index: *choice_index,
                    delta: Some(ChatMessage::Assistant {
                        content: Some(MessageContent::Text(text.clone())),
                        tool_calls: None,
                        name: None,
                    }),
                    finish_reason: None,
                }],
            ),
            ChatStreamEvent::ToolCallStart {
                choice_index,
                id,
                name,
                ..
            } => (
                String::new(),
                0,
                String::new(),
                vec![ChoiceDelta {
                    index: *choice_index,
                    delta: Some(ChatMessage::Assistant {
                        content: None,
                        tool_calls: Some(vec![ToolCall::Function(FunctionCall::Custom(
                            CustomFunctionCall {
                                id: id.clone(),
                                name: name.clone().unwrap_or_default(),
                                arguments: String::new(),
                            },
                        ))]),
                        name: None,
                    }),
                    finish_reason: None,
                }],
            ),
            ChatStreamEvent::ToolCallArgumentsDelta {
                choice_index,
                delta,
                ..
            } => (
                String::new(),
                0,
                String::new(),
                vec![ChoiceDelta {
                    index: *choice_index,
                    delta: Some(ChatMessage::Assistant {
                        content: None,
                        tool_calls: Some(vec![ToolCall::Function(FunctionCall::Custom(
                            CustomFunctionCall {
                                id: None,
                                name: String::new(),
                                arguments: delta.clone(),
                            },
                        ))]),
                        name: None,
                    }),
                    finish_reason: None,
                }],
            ),
            ChatStreamEvent::RefusalDelta {
                choice_index,
                delta,
            } => (
                String::new(),
                0,
                String::new(),
                vec![ChoiceDelta {
                    index: *choice_index,
                    delta: Some(ChatMessage::Assistant {
                        content: Some(MessageContent::Text(delta.clone())),
                        tool_calls: None,
                        name: None,
                    }),
                    finish_reason: None,
                }],
            ),
            ChatStreamEvent::ChoiceFinished {
                choice_index,
                reason,
            } => (
                String::new(),
                0,
                String::new(),
                vec![ChoiceDelta {
                    index: *choice_index,
                    delta: None,
                    finish_reason: reason.clone(),
                }],
            ),
            // Usage and the terminator have nowhere to go in the legacy shape.
            ChatStreamEvent::Usage(_) | ChatStreamEvent::ResponseFinished => {
                (String::new(), 0, String::new(), Vec::new())
            }
        };

        Chunk::Completion(ChunkCompletion {
            id,
            object: "chat.completion.chunk".to_string(),
            created,
            model,
            choices,
        })
    }
}

/// A tool call reassembled from streamed fragments.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AssembledToolCall {
    pub id: Option<String>,
    pub name: Option<String>,
    /// The complete argument JSON, in fragment order.
    pub arguments: String,
}

/// Reassembles streamed tool calls, keyed by tool index.
///
/// Fragments arrive interleaved when the model calls tools in parallel, and
/// `delta.tool_calls[].index` is the only thing that says which call a fragment
/// belongs to. Assembling by arrival order alone produces corrupted JSON, which
/// for an agentic loop means executing the wrong arguments.
#[derive(Debug, Clone, Default)]
pub struct ToolCallAssembler {
    calls: std::collections::BTreeMap<u32, AssembledToolCall>,
    order: Vec<u32>,
}

impl ToolCallAssembler {
    pub fn new() -> Self {
        Self::default()
    }

    fn entry(&mut self, tool_index: u32) -> &mut AssembledToolCall {
        if !self.calls.contains_key(&tool_index) {
            self.order.push(tool_index);
            self.calls.insert(
                tool_index,
                AssembledToolCall {
                    id: None,
                    name: None,
                    arguments: String::new(),
                },
            );
        }
        self.calls.get_mut(&tool_index).expect("just inserted")
    }

    /// Record the start of a tool call.
    pub fn start(&mut self, tool_index: u32, id: Option<String>, name: Option<String>) {
        let call = self.entry(tool_index);
        if id.is_some() {
            call.id = id;
        }
        if name.is_some() {
            call.name = name;
        }
    }

    /// Append an argument fragment.
    pub fn push_arguments(&mut self, tool_index: u32, delta: &str) {
        self.entry(tool_index).arguments.push_str(delta);
    }

    /// The assembled calls, in first-seen order.
    ///
    /// Arrival order, not sorted index: the model usually numbers them the order
    /// it emitted them, and preserving that is what the caller expects.
    pub fn finish(&self) -> Vec<AssembledToolCall> {
        self.order
            .iter()
            .filter_map(|index| self.calls.get(index).cloned())
            .collect()
    }

    /// How many distinct tool calls have been seen.
    pub fn len(&self) -> usize {
        self.calls.len()
    }

    pub fn is_empty(&self) -> bool {
        self.calls.is_empty()
    }
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

/// One capability a model may have.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Capability {
    FunctionCalling,
    Vision,
    Streaming,
    JsonMode,
    Caching,
}

impl Capability {
    /// The field on [`ModelCapabilities`] this capability governs.
    fn flag(self, capabilities: &ModelCapabilities) -> bool {
        match self {
            Capability::FunctionCalling => capabilities.function_calling,
            Capability::Vision => capabilities.vision,
            Capability::Streaming => capabilities.streaming,
            Capability::JsonMode => capabilities.json_mode,
            Capability::Caching => capabilities.caching,
        }
    }
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

    /// The capabilities that have **not** been determined.
    ///
    /// §40: "It is better to represent `Unknown` than confidently report false
    /// capability data." The booleans above cannot say that -- `false` reads as
    /// "this model has no tool support" when the truth is nearly always "we did
    /// not look". A caller that must not act on a guess checks this first.
    ///
    /// One set rather than a tri-state per field: it is additive, so adding it
    /// did not force thirteen struct literals across four providers to be
    /// rewritten, and it composes with `source` for the cases where the whole
    /// set came from one place.
    #[serde(default, skip_serializing_if = "std::collections::BTreeSet::is_empty")]
    pub unknown: std::collections::BTreeSet<Capability>,
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

impl ModelCapabilities {
    /// Every capability undetermined.
    pub fn all_unknown() -> Self {
        Self {
            unknown: [
                Capability::FunctionCalling,
                Capability::Vision,
                Capability::Streaming,
                Capability::JsonMode,
                Capability::Caching,
            ]
            .into_iter()
            .collect(),
            ..Default::default()
        }
    }

    /// Mark one capability as determined.
    pub fn with(mut self, capability: Capability, supported: bool) -> Self {
        self.unknown.remove(&capability);
        match capability {
            Capability::FunctionCalling => self.function_calling = supported,
            Capability::Vision => self.vision = supported,
            Capability::Streaming => self.streaming = supported,
            Capability::JsonMode => self.json_mode = supported,
            Capability::Caching => self.caching = supported,
        }
        self
    }

    /// Whether a capability is known, rather than merely false.
    pub fn is_known(&self, capability: Capability) -> bool {
        !self.unknown.contains(&capability)
    }

    /// A capability's value, or `None` when it was not determined.
    ///
    /// This is the accessor a routing decision should use. `self.vision` reads
    /// as "no vision" when the truth may be "nobody looked".
    pub fn get(&self, capability: Capability) -> Option<bool> {
        self.is_known(capability).then(|| capability.flag(self))
    }

    /// Whether every capability may be relied on.
    pub fn is_authoritative(&self) -> bool {
        self.source.is_authoritative() && self.unknown.is_empty()
    }
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
