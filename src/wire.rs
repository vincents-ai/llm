//! One request-body builder for the Chat Completions dialect.
//!
//! OpenAI and OpenRouter both speak this dialect, so the wire body is built in
//! exactly one place, once, for every path.
//!
//! The reason is not tidiness. Three hand-written bodies had drifted:
//!
//! ```text
//! non-streaming   model, messages, max_tokens, temperature, top_p, n, stop,
//!                 presence_penalty, frequency_penalty, logit_bias, extra_params
//! streaming       model, messages, stream, max_tokens, temperature, top_p
//! tool-calling    model, messages, tools, max_tokens, temperature, top_p, n,
//!                 stop, presence_penalty, frequency_penalty, logit_bias,
//!                 extra_params
//! ```
//!
//! A caller who set `stop` on a streaming request got no error and no `stop`.
//! The non-streaming body also sent `request.model` raw while the other two sent
//! the resolved default, so an empty `model` became `"model": ""` on exactly
//! one path.
//!
//! With one builder there is no per-path field list left to drift, which is the
//! property the brief asks for: "There should not be separate lists of supported
//! fields for normal completion / stream / tool completion."
//!
//! ## Merge order
//!
//! Fixed and deterministic, never map-insertion order:
//!
//! ```text
//! 1. required protocol fields  (model, messages, stream)
//! 2. typed common fields
//! 3. tools
//! 4. extra_params
//! ```
//!
//! `extra_params` is last so a collision with a typed field is detectable rather
//! than silently overwriting it — see [`build_chat_request`].

use serde_json::{json, Map, Value};

use crate::error::LLMError;
use crate::types::ChatCompletionRequest;

/// The inputs to one request body.
pub struct ChatRequestParts<'a> {
    /// The **resolved** model, after the default has been applied.
    ///
    /// Taking the resolved value rather than the raw request field is what stops
    /// one path sending `"model": ""` while the others send the default.
    pub model: &'a str,
    /// Messages already serialized by the canonical message serializer.
    pub messages: &'a [Value],
    /// The typed request, for its optional fields.
    pub request: &'a ChatCompletionRequest,
    /// Whether this is a streaming request.
    pub stream: bool,
    /// Serialized tool definitions, when the request is tool-bearing.
    pub tools: Option<&'a [Value]>,
}

/// Build a Chat Completions body.
///
/// Fails on a `extra_params` key that collides with a typed field. An arbitrary
/// map must not be able to override a typed setting: that makes a configuration
/// error indistinguishable from a deliberate one, and makes benchmark runs
/// unreproducible.
pub fn build_chat_request(parts: ChatRequestParts<'_>) -> Result<Value, LLMError> {
    let request = parts.request;
    let mut body = Map::new();

    // 1. Required protocol fields.
    body.insert("model".into(), json!(parts.model));
    body.insert("messages".into(), json!(parts.messages));
    if parts.stream {
        body.insert("stream".into(), json!(true));
    }

    // 2. Typed common fields. Every one of them, on every path.
    if let Some(max_tokens) = request.max_tokens {
        body.insert("max_tokens".into(), json!(max_tokens));
    }
    if let Some(temperature) = request.temperature {
        body.insert("temperature".into(), json!(temperature));
    }
    if let Some(top_p) = request.top_p {
        body.insert("top_p".into(), json!(top_p));
    }
    if let Some(n) = request.n {
        body.insert("n".into(), json!(n));
    }
    if !request.stop.is_empty() {
        body.insert("stop".into(), json!(request.stop));
    }
    if let Some(presence) = request.presence_penalty {
        body.insert("presence_penalty".into(), json!(presence));
    }
    if let Some(freq) = request.frequency_penalty {
        body.insert("frequency_penalty".into(), json!(freq));
    }
    if !request.logit_bias.is_empty() {
        body.insert("logit_bias".into(), json!(request.logit_bias));
    }
    if let Some(user) = &request.user {
        body.insert("user".into(), json!(user));
    }

    // 3. Tools, when present.
    if let Some(tools) = parts.tools {
        if !tools.is_empty() {
            body.insert("tools".into(), json!(tools));
        }
    }

    // 4. The escape hatch, last, and only where it cannot shadow a typed field.
    for (key, value) in &request.extra_params {
        if body.contains_key(key) {
            return Err(LLMError::ConfigurationError {
                message: format!(
                    "extra_params key '{key}' collides with a typed request field; remove it \
                     or set the typed field instead"
                ),
                field: Some(key.clone()),
            });
        }
        body.insert(key.clone(), value.clone());
    }

    Ok(Value::Object(body))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ChatMessage;

    fn request() -> ChatCompletionRequest {
        ChatCompletionRequest {
            model: "gpt-4o".into(),
            messages: vec![ChatMessage::user("hi")],
            ..Default::default()
        }
    }

    fn messages() -> Vec<Value> {
        vec![json!({"role": "user", "content": "hi"})]
    }

    fn build(request: &ChatCompletionRequest, stream: bool) -> Result<Value, LLMError> {
        build_chat_request(ChatRequestParts {
            model: "gpt-4o",
            messages: &messages(),
            request,
            stream,
            tools: None,
        })
    }

    #[test]
    fn every_typed_field_reaches_both_paths() {
        let mut request = request();
        request.max_tokens = Some(64);
        request.temperature = Some(0.2);
        request.top_p = Some(0.8);
        request.n = Some(2);
        request.stop = vec!["END".into()];
        request.presence_penalty = Some(0.1);
        request.frequency_penalty = Some(0.2);
        request.logit_bias.insert("50256".into(), -100);
        request.user = Some("user-42".into());

        let non_stream = build(&request, false).expect("non-streaming body");
        let stream = build(&request, true).expect("streaming body");

        // The whole point: identical apart from `stream`.
        let mut a = non_stream.clone();
        let mut b = stream.clone();
        a.as_object_mut().unwrap().remove("stream");
        b.as_object_mut().unwrap().remove("stream");
        assert_eq!(a, b, "the two paths must not drop different fields");

        for field in [
            "max_tokens",
            "temperature",
            "top_p",
            "n",
            "stop",
            "presence_penalty",
            "frequency_penalty",
            "logit_bias",
            "user",
        ] {
            assert!(
                non_stream.get(field).is_some(),
                "{field} missing from the non-streaming body"
            );
            assert!(
                stream.get(field).is_some(),
                "{field} missing from the streaming body"
            );
        }
    }

    #[test]
    fn the_resolved_model_is_what_is_sent() {
        // The old non-streaming body sent `request.model` raw, so an empty model
        // became `"model": ""` on exactly one path.
        let mut request = request();
        request.model = String::new();
        let body = build_chat_request(ChatRequestParts {
            model: "gpt-4o",
            messages: &messages(),
            request: &request,
            stream: false,
            tools: None,
        })
        .expect("body");
        assert_eq!(body["model"], "gpt-4o");
    }

    #[test]
    fn unset_optional_fields_are_omitted_rather_than_null() {
        let body = build(&request(), false).expect("body");
        assert!(body.get("max_tokens").is_none());
        assert!(body.get("n").is_none());
        assert!(body.get("stop").is_none());
        assert!(body.get("stream").is_none());
    }

    #[test]
    fn stream_is_only_present_when_requested() {
        assert_eq!(build(&request(), false).unwrap()["stream"], Value::Null);
        assert_eq!(build(&request(), true).unwrap()["stream"], json!(true));
        assert!(build(&request(), true)
            .unwrap()
            .as_object()
            .unwrap()
            .contains_key("stream"));
        assert!(!build(&request(), false)
            .unwrap()
            .as_object()
            .unwrap()
            .contains_key("stream"));
    }

    #[test]
    fn an_extra_param_cannot_shadow_a_typed_field() {
        // A configuration error, not a silent override: otherwise a benchmark
        // that "set temperature" may not have.
        let mut request = request();
        request.temperature = Some(0.2);
        request
            .extra_params
            .insert("temperature".into(), json!(1.0));

        let error = build(&request, false).expect_err("a collision must fail");
        match error {
            LLMError::ConfigurationError { field, message } => {
                assert_eq!(field.as_deref(), Some("temperature"));
                assert!(message.contains("collides"), "{message}");
            }
            other => panic!("expected a configuration error, got {other:?}"),
        }
    }

    #[test]
    fn a_fresh_extra_param_is_allowed_through() {
        let mut request = request();
        request.extra_params.insert("service_tier".into(), json!("flex"));
        let body = build(&request, false).expect("body");
        assert_eq!(body["service_tier"], "flex");
    }

    #[test]
    fn extra_params_do_not_shadow_required_fields_either() {
        let mut request = request();
        request.extra_params.insert("messages".into(), json!([]));
        let error = build(&request, false).expect_err("must fail");
        assert!(matches!(error, LLMError::ConfigurationError { .. }));
    }

    #[test]
    fn tools_appear_only_when_supplied() {
        let without = build_chat_request(ChatRequestParts {
            model: "m",
            messages: &messages(),
            request: &request(),
            stream: false,
            tools: None,
        })
        .unwrap();
        assert!(without.get("tools").is_none());

        let tools = vec![json!({"type": "function", "function": {"name": "f"}})];
        let with = build_chat_request(ChatRequestParts {
            model: "m",
            messages: &messages(),
            request: &request(),
            stream: false,
            tools: Some(&tools),
        })
        .unwrap();
        assert_eq!(with["tools"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn an_empty_tool_list_is_omitted() {
        // Sending `"tools": []` is a different request from sending no tools.
        let tools: Vec<Value> = Vec::new();
        let body = build_chat_request(ChatRequestParts {
            model: "m",
            messages: &messages(),
            request: &request(),
            stream: false,
            tools: Some(&tools),
        })
        .unwrap();
        assert!(body.get("tools").is_none());
    }
}
