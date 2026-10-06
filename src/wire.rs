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
use crate::types::{ChatCompletionRequest, ToolChoice, ToolDefinition};

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
    /// Serialized tool definitions, when the request is request-bearing.
    pub tools: Option<&'a [Value]>,
    /// A service tier to use when the request does not name one.
    ///
    /// A deployment default rather than a per-request field: the tier is a
    /// billing and latency decision an application makes once, and letting
    /// every call site choose it is how a default quietly stops being one.
    /// A tier the request *does* name wins -- an explicit choice beats a
    /// deployment default.
    pub default_service_tier: Option<&'a str>,

    /// Provider-specific fields that have no portable equivalent.
    ///
    /// Merged after everything else, under the same collision rule as
    /// `extra_params`: a key that is already present is a configuration error
    /// rather than an override. A provider extension must not be able to
    /// shadow a typed field any more than a raw map can — otherwise "one
    /// canonical serializer per protocol" would be true only for the portable
    /// half.
    pub provider_params: Option<&'a std::collections::BTreeMap<String, Value>>,
}

/// Build a Chat Completions body.
///
/// Fails on a `extra_params` key that collides with a typed field. An arbitrary
/// map must not be able to override a typed setting: that makes a configuration
/// error indistinguishable from a deliberate one, and makes benchmark runs
/// unreproducible.
/// Validate a request before any body is assembled or any network I/O happens.
///
/// §51. Every rule here is a conflict that a provider would either reject with
/// an opaque 400 or -- worse -- silently resolve, producing an answer to a
/// question the caller did not ask.
///
/// Ordered so that the most fundamental failure is reported first: validity,
/// then modality, then collisions.
pub fn validate_request(
    request: &crate::types::ChatCompletionRequest,
    parts: &ChatRequestParts<'_>,
) -> crate::error::Result<()> {
    // Cross-field checks that belong before any network I/O. A provider that
    // silently ignores one of these produces a result that looks like an answer
    // to a question nobody asked.

    // §51, added here because the others were not enough: a `stream_options`
    // block on a non-streaming request is either ignored or rejected depending
    // on the provider, and `include_usage` is exactly the setting that decides
    // whether the caller ever learns what the request cost.
    if !parts.stream
        && (request.stream_options.is_some()
            || request.extra_params.contains_key("stream_options"))
    {
        return Err(LLMError::InvalidParameterCombination {
            message: "stream_options was set on a non-streaming request (stream = false); \
                      include_usage only applies to streams"
                .to_string(),
        });
    }
    if request.max_tokens.is_some() && request.max_completion_tokens.is_some() {
        return Err(LLMError::ConfigurationError {
            message: "max_tokens and max_completion_tokens are both set; they are not \
                      interchangeable and no provider documents a precedence"
                .to_string(),
            field: Some("max_completion_tokens".to_string()),
        });
    }
    if request.top_logprobs.is_some() && request.logprobs != Some(true) {
        return Err(LLMError::ConfigurationError {
            message: "top_logprobs requires logprobs = true".to_string(),
            field: Some("top_logprobs".to_string()),
        });
    }
    if let Some(ToolChoice::Function { name }) = &request.tool_choice {
        {
            // Naming a tool that is not offered — including when *no* tools are
            // offered — is a configuration error, not something to forward and
            // let the provider reject after a round trip.
            let offered = request.tools.as_ref().is_some_and(|tools| {
                tools.iter().any(|tool| match tool {
                    ToolDefinition::Function(f) => f.name == *name,
                    ToolDefinition::Custom { name: n, .. } => n == name,
                })
            });
            if !offered {
                // The offered names are in the message, not just the one the
                // caller asked for. "You named get_stock" leaves them to
                // diff it against their own tool list by hand, which is the
                // work this error exists to save.
                let offered_names: Vec<&str> = request
                    .tools
                    .as_ref()
                    .map(|tools| {
                        tools
                            .iter()
                            .map(|tool| match tool {
                                ToolDefinition::Function(f) => f.name.as_str(),
                                ToolDefinition::Custom { name, .. } => name.as_str(),
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                return Err(LLMError::ConfigurationError {
                    message: format!(
                        "tool_choice names '{name}', which is not among the offered tools: \
                         [{}]",
                        offered_names.join(", ")
                    ),
                    field: Some("tool_choice".to_string()),
                });
            }
        }
    }

    Ok(())
}

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
    if let Some(max_completion_tokens) = request.max_completion_tokens {
        body.insert("max_completion_tokens".into(), json!(max_completion_tokens));
    }
    if let Some(seed) = request.seed {
        body.insert("seed".into(), json!(seed));
    }
    if let Some(logprobs) = request.logprobs {
        body.insert("logprobs".into(), json!(logprobs));
    }
    if let Some(top_logprobs) = request.top_logprobs {
        body.insert("top_logprobs".into(), json!(top_logprobs));
    }
    if let Some(response_format) = &request.response_format {
        body.insert("response_format".into(), json!(response_format));
    }
    if let Some(reasoning) = &request.reasoning {
        body.insert("reasoning".into(), json!(reasoning));
    }
    if let Some(verbosity) = &request.verbosity {
        body.insert("verbosity".into(), json!(verbosity.as_str()));
    }
    if let Some(stream_options) = &request.stream_options {
        body.insert(
            "stream_options".into(),
            serde_json::to_value(stream_options).unwrap_or(Value::Null),
        );
    }
    if let Some(store) = request.store {
        body.insert("store".into(), json!(store));
    }
    if let Some(key) = &request.prompt_cache_key {
        body.insert("prompt_cache_key".into(), json!(key));
    }
    if let Some(service_tier) = &request.service_tier {
        body.insert("service_tier".into(), json!(service_tier.as_str()));
    }
    if !request.metadata.is_empty() {
        body.insert("metadata".into(), json!(request.metadata));
    }
    if let Some(parallel_tool_calls) = request.parallel_tool_calls {
        body.insert("parallel_tool_calls".into(), json!(parallel_tool_calls));
    }
    if let Some(tool_choice) = &request.tool_choice {
        body.insert("tool_choice".into(), tool_choice.to_wire());
    }

    // 3. Tools, when present.
    if let Some(tools) = parts.tools {
        if !tools.is_empty() {
            body.insert("tools".into(), json!(tools));
        }
    }
    // Typed `request.tools`, when the path did not supply pre-serialized ones.
    if !body.contains_key("tools") {
        if let Some(tools) = &request.tools {
            if !tools.is_empty() {
                body.insert("tools".into(), json!(tools));
            }
        }
    }

    // All cross-field checks run first, before the body is assembled and before
    // any network I/O. Order is deliberate: validity before collision, because
    // a request that is invalid on its own terms should say so rather than
    // complain that `extra_params` shadows a field the caller never got to use.
    validate_request(request, &parts)?;

    // 3b. The deployment's default service tier, applied only when the request
    // did not choose one.
    if let Some(tier) = parts.default_service_tier {
        if !body.contains_key("service_tier") {
            body.insert("service_tier".into(), json!(tier));
        }
    }

    // 4. Provider-specific extensions, then the raw escape hatch. Both are
    //    collision-checked, and both come last.
    if let Some(extensions) = parts.provider_params {
        for (key, value) in extensions {
            if body.contains_key(key) {
                return Err(LLMError::ConfigurationError {
                    message: format!(
                        "provider extension '{key}' collides with a typed or already-set field"
                    ),
                    field: Some(key.clone()),
                });
            }
            body.insert(key.clone(), value.clone());
        }
    }
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
            default_service_tier: None,
            provider_params: None,
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
            provider_params: None,
    default_service_tier: None,
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
            provider_params: None,
    default_service_tier: None,
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
            provider_params: None,
    default_service_tier: None,
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
            provider_params: None,
    default_service_tier: None,
        })
        .unwrap();
        assert!(body.get("tools").is_none());
    }
}

#[cfg(test)]
mod modern_parameter_tests {
    use super::*;
    use crate::types::{
        ChatMessage, FunctionDefinitionWire, ReasoningConfig, ReasoningEffort, ResponseFormat,
        ToolChoice, ToolDefinition, Verbosity,
    };

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
            default_service_tier: None,
            provider_params: None,
        })
    }

    fn base() -> ChatCompletionRequest {
        ChatCompletionRequest {
            model: "gpt-4o".into(),
            messages: vec![ChatMessage::user("hi")],
            ..Default::default()
        }
    }

    #[test]
    fn unset_modern_parameters_are_omitted() {
        // A request that uses none of them must serialise exactly as it did
        // before the fields existed.
        let body = build(&base(), false).expect("body");
        for field in [
            "max_completion_tokens",
            "seed",
            "logprobs",
            "top_logprobs",
            "response_format",
            "tools",
            "tool_choice",
            "parallel_tool_calls",
            "reasoning",
            "verbosity",
            "service_tier",
            "metadata",
        ] {
            assert!(body.get(field).is_none(), "{field} should be omitted");
        }
    }

    #[test]
    fn every_modern_parameter_reaches_both_paths() {
        let mut request = base();
        request.max_completion_tokens = Some(4096);
        request.seed = Some(7);
        request.logprobs = Some(true);
        request.top_logprobs = Some(5);
        request.response_format = Some(ResponseFormat::JsonObject);
        request.parallel_tool_calls = Some(true);
        request.reasoning = Some(ReasoningConfig {
            effort: Some(ReasoningEffort::high()),
            max_tokens: Some(1024),
            ..Default::default()
        });
        request.verbosity = Some(Verbosity::low());
        request.service_tier = Some("flex".into());

        let non_stream = build(&request, false).expect("non-streaming");
        let stream = build(&request, true).expect("streaming");

        let mut a = non_stream.clone();
        let mut b = stream.clone();
        a.as_object_mut().unwrap().remove("stream");
        b.as_object_mut().unwrap().remove("stream");
        assert_eq!(a, b, "the modern parameters must not differ by path");

        assert_eq!(non_stream["max_completion_tokens"], 4096);
        assert_eq!(non_stream["seed"], 7);
        assert_eq!(non_stream["logprobs"], true);
        assert_eq!(non_stream["top_logprobs"], 5);
        assert_eq!(non_stream["parallel_tool_calls"], true);
        assert_eq!(non_stream["service_tier"], "flex");
        assert_eq!(non_stream["verbosity"], "low");
        assert_eq!(non_stream["reasoning"]["effort"], "high");
    }

    #[test]
    fn max_tokens_and_max_completion_tokens_together_are_rejected() {
        // The brief is explicit: reject unless a provider documents a
        // precedence. They are not interchangeable — a reasoning model counts
        // reasoning tokens against the limit and rejects the request.
        let mut request = base();
        request.max_tokens = Some(100);
        request.max_completion_tokens = Some(100);
        let error = build(&request, false).expect_err("must be rejected");
        match error {
            LLMError::ConfigurationError { field, message } => {
                assert_eq!(field.as_deref(), Some("max_completion_tokens"));
                assert!(message.contains("not"), "{message}");
            }
            other => panic!("expected a configuration error, got {other:?}"),
        }
    }

    #[test]
    fn top_logprobs_without_logprobs_is_rejected() {
        let mut request = base();
        request.top_logprobs = Some(3);
        assert!(build(&request, false).is_err(), "top_logprobs alone is meaningless");

        request.logprobs = Some(false);
        assert!(
            build(&request, false).is_err(),
            "logprobs = false does not make top_logprobs meaningful"
        );

        request.logprobs = Some(true);
        assert!(build(&request, false).is_ok());
    }

    #[test]
    fn a_tool_choice_naming_an_absent_tool_is_rejected() {
        let mut request = base();
        request.tool_choice = Some(ToolChoice::Function {
            name: "get_weather".into(),
        });
        let error = build(&request, false).expect_err("must be rejected");
        assert!(matches!(error, LLMError::ConfigurationError { .. }));

        request.tools = Some(vec![ToolDefinition::Function(FunctionDefinitionWire {
            name: "get_time".into(),
            description: None,
            parameters: None,
            strict: None,
        })]);
        assert!(
            build(&request, false).is_err(),
            "naming a tool that is not offered must still fail"
        );
    }

    #[test]
    fn a_tool_choice_naming_an_offered_tool_is_accepted() {
        let mut request = base();
        request.tools = Some(vec![ToolDefinition::Function(FunctionDefinitionWire {
            name: "get_weather".into(),
            description: Some("Look up the weather".into()),
            parameters: Some(json!({"type": "object", "properties": {}})),
            strict: Some(true),
        })]);
        request.tool_choice = Some(ToolChoice::Function {
            name: "get_weather".into(),
        });
        let body = build(&request, false).expect("body");
        assert_eq!(body["tool_choice"]["function"]["name"], "get_weather");
        assert_eq!(body["tools"][0]["function"]["strict"], true);
    }

    #[test]
    fn tool_choice_variants_serialize_to_the_openai_shapes() {
        let mut request = base();
        request.tool_choice = Some(ToolChoice::Auto);
        assert_eq!(build(&request, false).unwrap()["tool_choice"], "auto");
        request.tool_choice = Some(ToolChoice::None);
        assert_eq!(build(&request, false).unwrap()["tool_choice"], "none");
        request.tool_choice = Some(ToolChoice::Required);
        assert_eq!(build(&request, false).unwrap()["tool_choice"], "required");
    }

    #[test]
    fn a_custom_tool_is_not_shaped_like_a_function() {
        // §27: provider-operated tools have different lifecycle and authority
        // semantics, so they must not masquerade as local functions.
        let mut request = base();
        request.tools = Some(vec![ToolDefinition::Custom {
            name: "openrouter:web_search".into(),
            description: Some("search the web".into()),
            input_schema: Some(json!({"type": "object"})),
        }]);
        let body = build(&request, false).expect("body");
        assert_eq!(body["tools"][0]["type"], "custom");
        assert_eq!(body["tools"][0]["name"], "openrouter:web_search");
    }

    #[test]
    fn a_json_schema_format_carries_its_schema() {
        let mut request = base();
        request.response_format = Some(ResponseFormat::JsonSchema {
            name: "answer".into(),
            description: Some("a structured answer".into()),
            schema: json!({"type": "object", "properties": {"a": {"type": "string"}}}),
            strict: Some(true),
        });
        let body = build(&request, false).expect("body");
        let format = &body["response_format"];
        assert_eq!(format["type"], "json_schema");
        assert_eq!(format["json_schema"]["name"], "answer");
        assert_eq!(format["json_schema"]["strict"], true);
        assert_eq!(
            format["json_schema"]["schema"]["properties"]["a"]["type"],
            "string"
        );
    }

    #[test]
    fn an_unknown_reasoning_effort_round_trips() {
        // The forward-compatibility property: a level this build predates must
        // be forwarded, not dropped and not a parse failure.
        let effort: ReasoningEffort = "ultra_turbo".into();
        let mut request = base();
        request.reasoning = Some(ReasoningConfig {
            effort: Some(effort.clone()),
            ..Default::default()
        });
        let body = build(&request, false).expect("body");
        assert_eq!(body["reasoning"]["effort"], "ultra_turbo");

        let wire = serde_json::to_string(&effort).expect("serializes");
        let back: ReasoningEffort = serde_json::from_str(&wire).expect("deserializes");
        assert_eq!(back, effort);
    }

    #[test]
    fn an_unknown_verbosity_and_service_tier_round_trip() {
        let verbosity = Verbosity::from("terse");
        let tier: crate::types::ServiceTier = "turbo".into();
        let mut request = base();
        request.verbosity = Some(verbosity.clone());
        request.service_tier = Some(tier.clone());
        let body = build(&request, false).expect("body");
        assert_eq!(body["verbosity"], "terse");
        assert_eq!(body["service_tier"], "turbo");
    }

    #[test]
    fn a_typed_field_still_wins_over_extra_params() {
        // Adding a typed field must not open a hole: `temperature` is typed and
        // an `extra_params` entry for it is still a configuration error.
        let mut request = base();
        request.temperature = Some(0.5);
        // Collide with a field that is *set*, so the typed key really is in the
        // body and the collision is with something, not with an absence.
        request
            .extra_params
            .insert("temperature".into(), json!(0.9));
        let error = build(&request, false).expect_err("must be rejected");
        match error {
            LLMError::ConfigurationError { field, .. } => {
                assert_eq!(field.as_deref(), Some("temperature"));
            }
            other => panic!("expected a configuration error, got {other:?}"),
        }
    }
}
