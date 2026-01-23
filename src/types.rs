use serde::{Deserialize, Serialize};
use std::collections::HashMap;

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
    Function {
        name: String,
        arguments: String,
    },

    /// Tool result from function execution
    #[serde(rename = "tool")]
    Tool {
        tool_call_id: String,
        content: String,
    },
}

impl ChatMessage {
    /// Create a system message
    pub fn system(content: impl Into<String>) -> Self {
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
            extra_params: HashMap::new(),
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

/// Model information
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

/// Pricing information for a model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPricing {
    pub prompt_tokens: f64,    // Cost per 1K input tokens
    pub completion_tokens: f64, // Cost per 1K output tokens
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image_tokens: Option<f64>,
}

/// Model capabilities
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelCapabilities {
    pub function_calling: bool,
    pub vision: bool,
    pub streaming: bool,
    pub json_mode: bool,
    #[serde(default)]
    pub max_tokens: u32,
    #[serde(default)]
    pub context_window: u32,
}

/// A model variant
#[derive(Debug, Clone)]
pub struct ModelVariant {
    pub id: String,
    pub name: String,
    pub pricing: ModelPricing,
    pub capabilities: ModelCapabilities,
}
