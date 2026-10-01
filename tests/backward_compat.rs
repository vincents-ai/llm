//! §68: backward compatibility.
//!
//! "Keep common existing constructors and fields working during migration."
//! Every item here is a call site that predates the typed-parameter work and
//! must still compile and behave. A compatibility shim that is never exercised
//! is a shim that breaks silently on the next refactor.

use serde_json::json;
use vincents_llm::types::{
    ChatCompletionRequest, ChatMessage, ChatCompletionResponse, Choice, Usage,
    FunctionDefinition, ResponseFormat, ToolChoice,
};
use vincents_llm::{ChatCompletionRequest as Req, LLMProvider};

#[test]
fn the_common_constructors_still_take_bare_strings() {
    // The most-used surface in the codebase. If this needs `.into()`, the
    // migration has leaked onto every call site.
    let _ = ChatMessage::user("hello");
    let _ = ChatMessage::system("be terse");
    let _ = ChatMessage::assistant("sure");
    let _ = ChatMessage::user(String::from("owned"));
    let _ = ChatMessage::user(&String::from("borrowed"));
    let _: Req = ChatCompletionRequest {
        model: "gpt-4o".into(),
        messages: vec![ChatMessage::user("hi")],
        ..Default::default()
    };
}

#[tokio::test]
async fn the_legacy_functions_path_still_works_and_matches_the_typed_one() {
    // `chat_completion_with_functions` is deprecated but present, and produces the
    // same bytes as `request.tools`. A caller mid-migration sees no change.
    use async_trait::async_trait;
    struct Recorder(Box<std::sync::Mutex<Option<serde_json::Value>>>);
    #[async_trait]
    impl LLMProvider for Recorder {
        fn name(&self) -> &str { "recorder" }
        fn supported_models(&self) -> Vec<String> { vec!["m".into()] }
        async fn chat_completion(
            &self,
            request: ChatCompletionRequest,
        ) -> vincents_llm::error::Result<ChatCompletionResponse> {
            *self.0.lock().unwrap() = Some(
                vincents_llm::wire::build_chat_request(vincents_llm::wire::ChatRequestParts {
                    model: &request.model,
                    messages: &request.messages.iter().map(ChatMessage::to_openai_json).collect::<Vec<_>>(),
                    request: &request,
                    stream: false,
                    tools: None,
                    default_service_tier: None,
                    provider_params: None,
                })?,
            );
            Ok(ChatCompletionResponse {
                id: "c".into(), object: "chat.completion".into(), created: 0, model: "m".into(),
                choices: vec![], usage: None, system_fingerprint: None, unknown_fields: Default::default(),
            })
        }
        async fn chat_completion_with_functions(
            &self,
            request: ChatCompletionRequest,
            functions: Vec<FunctionDefinition>,
        ) -> vincents_llm::error::Result<ChatCompletionResponse> {
            let mut request = request;
            let tools = request.tools.get_or_insert_with(Vec::new);
            for f in functions {
                tools.push(vincents_llm::types::ToolDefinition::Function(
                    vincents_llm::types::FunctionDefinitionWire {
                        name: f.name, description: f.description,
                        parameters: f.parameters, strict: None,
                    },
                ));
            }
            self.chat_completion(request).await
        }
        async fn rate_limit_status(&self) -> vincents_llm::error::Result<vincents_llm::RateLimitStatus> {
            unreachable!()
        }
        async fn estimate_cost(&self, _: &ChatCompletionRequest) -> vincents_llm::error::Result<vincents_llm::CostEstimate> {
            unreachable!()
        }
        fn config(&self) -> &vincents_llm::config::ProviderConfig { unreachable!() }
    }

    let recorder = Recorder(Box::new(std::sync::Mutex::new(None)));
    let request = ChatCompletionRequest {
        model: "m".into(),
        messages: vec![ChatMessage::user("hi")],
        ..Default::default()
    };
    let functions = vec![FunctionDefinition {
        name: "f".into(),
        description: Some("d".into()),
        parameters: Some(json!({"type": "object"})),
    }];

    let _legacy = recorder
        .chat_completion_with_functions(request.clone(), functions.clone())
        .await
        .expect("legacy path");
    let legacy_body = recorder.0.lock().unwrap().clone().expect("recorded");

    let _typed = recorder
        .chat_completion(ChatCompletionRequest {
            model: "m".into(),
            messages: vec![ChatMessage::user("hi")],
            tools: Some(vec![vincents_llm::types::ToolDefinition::Function(
                vincents_llm::types::FunctionDefinitionWire {
                    name: "f".into(), description: Some("d".into()),
                    parameters: Some(json!({"type": "object"})), strict: None,
                },
            )]),
            ..Default::default()
        })
        .await
        .expect("typed path");
    let typed_body = recorder.0.lock().unwrap().clone().expect("recorded");

    assert_eq!(legacy_body, typed_body, "a caller mid-migration sees no change");
}

#[test]
fn the_common_response_fields_are_still_readable() {
    let response = ChatCompletionResponse {
        id: "c".into(),
        object: "chat.completion".into(),
        created: 0,
        model: "m".into(),
        choices: vec![Choice {
            index: 0,
            message: ChatMessage::assistant("hi"),
            finish_reason: Some("stop".into()),
            logprobs: None,
        }],
        usage: Some(Usage::new(10, 5)),
        system_fingerprint: None,
        unknown_fields: Default::default(),
    };
    assert_eq!(response.choices[0].message.content(), Some("hi"));
    assert_eq!(response.usage.expect("usage").total_tokens, 15);
}

#[test]
fn responses_format_constructors_still_work() {
    let _ = ResponseFormat::Text;
    let _ = ResponseFormat::JsonObject;
    let _ = ResponseFormat::JsonSchema {
        name: "n".into(),
        description: None,
        schema: json!({"type": "object"}),
        strict: Some(true),
    };
    let _ = ToolChoice::Auto;
    let _ = ToolChoice::Required;
    let _ = ToolChoice::Function { name: "f".into() };
}
