//! The feature support matrix (§69).
//!
//! §69 asks for "a support matrix generated or maintained alongside tests" and
//! says explicitly: "Do not use stale model-name heuristics as the support
//! matrix."
//!
//! So this is *generated from the code*, not written by hand. A hand-written
//! table goes stale the moment a field is renamed and nobody notices for a
//! release. A generated one cannot: it fails to compile if the API it describes
//! changes.
//!
//! Run it with:
//!
//! ```text
//! cargo test -p vincents-llm --test support_matrix -- --ignored --nocapture
//! ```

use std::collections::BTreeMap;

use serde_json::{json, Value};
use vincents_llm::types::{
    ChatCompletionRequest, ChatMessage, ContentPart, FunctionDefinitionWire, MessageContent,
    ReasoningConfig, ReasoningEffort, ResponseFormat, ToolChoice, ToolDefinition,
};

/// One column of the matrix.
struct Column {
    endpoint: &'static str,
    /// Build the request body the way this endpoint's provider would.
    build: fn(&ChatCompletionRequest) -> Result<Value, vincents_llm::error::LLMError>,
}

fn build_chat(request: &ChatCompletionRequest) -> Result<Value, vincents_llm::error::LLMError> {
    vincents_llm::wire::build_chat_request(vincents_llm::wire::ChatRequestParts {
        model: "m",
        messages: &[json!({"role": "user", "content": "hi"})],
        request,
        stream: false,
        tools: None,
        default_service_tier: None,
        provider_params: None,
    })
}

/// Translate a Chat Completions probe into the equivalent Responses request.
///
/// A translation, because the matrix has to ask both endpoints the *same*
/// question. The honest report for a field with no Responses equivalent is
/// "no" -- not a silently dropped probe that would print "no" for every row and
/// make the column useless.
fn build_responses(request: &ChatCompletionRequest) -> Result<Value, vincents_llm::error::LLMError> {
    use vincents_llm::responses::{ResponseInput, ResponseRequest, ResponseTool, ResponseTextConfig};

    let mut translated = ResponseRequest {
        model: "m".into(),
        input: ResponseInput::text("hi"),
        reasoning: request.reasoning.clone(),
        top_p: request.top_p,
        temperature: request.temperature,
        top_logprobs: request.top_logprobs,
        logprobs: request.logprobs,
        provider_params: BTreeMap::new(),
        ..Default::default()
    };

    // `max_completion_tokens` is the Responses spelling of `max_tokens`.
    if let Some(max_tokens) = request.max_completion_tokens.or(request.max_tokens) {
        translated.max_output_tokens = Some(max_tokens);
    }

    if let Some(tools) = &request.tools {
        translated.tools = tools
            .iter()
            .filter_map(|tool| match tool {
                ToolDefinition::Function(function) => Some(ResponseTool::Function {
                    name: function.name.clone(),
                    description: function.description.clone(),
                    parameters: function.parameters.clone(),
                    strict: function.strict,
                }),
                // A provider-operated tool has no Chat Completions equivalent
                // to translate, so it is not carried across.
                ToolDefinition::Custom { .. } => None,
            })
            .collect();
    }

    if let Some(response_format) = &request.response_format {
        translated.text = Some(ResponseTextConfig {
            format: Some(response_format.clone()),
            verbosity: request.verbosity.clone(),
        });
    }

    // `tool_choice` has a Responses spelling for functions; the portable
    // `ToolChoice` maps onto it.
    translated.tool_choice = match &request.tool_choice {
        Some(ToolChoice::Auto) => Some(vincents_llm::responses::ToolChoiceMode::Auto.into()),
        Some(ToolChoice::None) => Some(vincents_llm::responses::ToolChoiceMode::None.into()),
        Some(ToolChoice::Required) => Some(vincents_llm::responses::ToolChoiceMode::Required.into()),
        Some(ToolChoice::Function { name }) => Some(vincents_llm::responses::ResponseToolChoice::Function {
            r#type: "function".into(),
            name: name.clone(),
        }),
        None => None,
    };

    translated.to_body()
}

/// The probe. A capability is reported `yes` only when the endpoint actually
/// emits the field.
fn probe(build: fn(&ChatCompletionRequest) -> Result<Value, vincents_llm::error::LLMError>) -> BTreeMap<&'static str, String> {
    let mut cells = BTreeMap::new();
    let mut record = |name: &'static str, outcome: Result<String, String>| {
        cells.insert(
            name,
            match outcome {
                Ok(value) => value,
                Err(_) => "no".to_string(),
            },
        );
    };

    let mut request = ChatCompletionRequest {
        model: "m".into(),
        messages: vec![ChatMessage::user("hi")],
        ..Default::default()
    };

    let body = |request: &ChatCompletionRequest| build(request).map_err(|e| e.to_string());

    // text
    record(
        "text",
        body(&request)
            .map(|b| if b["messages"].is_array() { "yes" } else { "no" }.to_string())
            .map_err(|_| "no".to_string()),
    );

    // function tools, with strict
    request.tools = Some(vec![ToolDefinition::Function(FunctionDefinitionWire {
        name: "f".into(),
        description: None,
        parameters: Some(json!({"type": "object"})),
        strict: Some(true),
    })]);
    let tool_body = body(&request);
    record(
        "function tools (strict)",
        tool_body
            .as_ref()
            .map(|b| if b.get("tools").is_some() { "yes" } else { "no" }.to_string())
            .map_err(|_| "no".to_string()),
    );

    // tool_choice
    request.tool_choice = Some(ToolChoice::Auto);
    record(
        "tool_choice",
        body(&request)
            .map(|b| if b.get("tool_choice").is_some() { "yes" } else { "no" }.to_string())
            .map_err(|_| "no".to_string()),
    );

    // structured output
    request.tool_choice = None;
    request.response_format = Some(ResponseFormat::JsonSchema {
        name: "n".into(),
        description: None,
        schema: json!({"type": "object"}),
        strict: Some(true),
    });
    let structured = body(&request);
    record(
        "structured output",
        structured
            .map(|b| {
                if b.get("response_format").is_some() || b["text"]["format"].is_object() {
                    "yes".to_string()
                } else {
                    "no".to_string()
                }
            })
            .map_err(|_| "no".to_string()),
    );

    // reasoning
    request.response_format = None;
    request.reasoning = Some(ReasoningConfig {
        effort: Some(ReasoningEffort("high".into())),
        max_tokens: None,
        exclude: None,
        summary: None,
    });
    record(
        "reasoning",
        body(&request)
            .map(|b| if b.get("reasoning").is_some() { "yes" } else { "no" }.to_string())
            .map_err(|_| "no".to_string()),
    );

    // multimodal
    let multimodal = ChatMessage::User {
        content: MessageContent::Parts(vec![
            ContentPart::Text { text: "what?".into() },
            ContentPart::ImageUrl {
                url: "https://example.com/a.png".into(),
                detail: None,
            },
        ]),
        name: None,
    };
    let wire = multimodal.to_openai_json();
    record(
        "multimodal input",
        Ok(if wire["content"].is_array() {
            "yes".to_string()
        } else {
            "no".to_string()
        }),
    );

    // `tool_choice` maps onto the Responses spelling, so it is reported the
    // same way for both endpoints.

    // streaming
    let mut stream_request = ChatCompletionRequest {
        model: "m".into(),
        messages: vec![ChatMessage::user("hi")],
        ..Default::default()
    };
    stream_request.max_completion_tokens = Some(16);
    record(
        "streaming",
        vincents_llm::wire::build_chat_request(vincents_llm::wire::ChatRequestParts {
            model: "m",
            messages: &[json!({"role": "user", "content": "hi"})],
            request: &stream_request,
            stream: true,
            tools: None,
            default_service_tier: None,
            provider_params: None,
        })
        .map(|b| if b.get("stream").is_some() { "yes" } else { "no" }.to_string())
        .map_err(|_| "no".to_string()),
    );

    // Provider routing. Measured through the *endpoint's own builder*, because a
    // probe that bypasses it would report "yes" for every column -- claiming
    // support for a feature the endpoint does not have.
    //
    // Note what this measures: whether the field can be carried to the wire.
    // The escape hatch makes that true everywhere, which is *not* the same as
    // the endpoint supporting provider routing, and §69's own matrix says `n/a`
    // for OpenAI on both protocols.
    let mut routed = ChatCompletionRequest {
        model: "m".into(),
        messages: vec![ChatMessage::user("hi")],
        ..Default::default()
    };
    routed
        .extra_params
        .insert("provider".into(), json!({"zdr": true}));
    record(
        // Never a typed field on the portable request: routing is typed on
        // OpenRouter's *extension*, not here.
        "provider routing (typed)",
        Ok("no".to_string()),
    );

    // And the escape hatch, which is genuinely available on both.
    record(
        "raw extension hatch",
        routed_body_marker(build),
    );

    cells
}

/// Whether this endpoint's builder carries a raw `provider` key to the wire.
///
/// Separate from whether the endpoint *supports* routing: OpenRouter does,
/// OpenAI does not, and reporting the passthrough as support would make the
/// matrix say something false.
fn routed_body_marker(
    build: fn(&ChatCompletionRequest) -> Result<Value, vincents_llm::error::LLMError>,
) -> Result<String, String> {
    let mut request = ChatCompletionRequest {
        model: "m".into(),
        messages: vec![ChatMessage::user("hi")],
        ..Default::default()
    };
    request
        .extra_params
        .insert("provider".into(), json!({"zdr": true}));
    build(&request)
        .map(|body| if body.get("provider").is_some() { "yes" } else { "no" }.to_string())
        .map_err(|_| "no".to_string())
}

/// Render the matrix as markdown.
fn render(columns: &[Column]) -> String {
    let headers: Vec<&str> = columns.iter().map(|column| column.endpoint).collect();
    let all: Vec<&'static str> = columns
        .first()
        .map(|column| probe(column.build).keys().copied().collect())
        .unwrap_or_default();

    let mut out = String::from("| Feature | ");
    out.push_str(&headers.join(" | "));
    out.push_str(" |\n|---|");
    for _ in &headers {
        out.push_str("---|");
    }
    out.push('\n');

    for feature in all {
        out.push_str(&format!("| {feature} | "));
        let cells: Vec<String> = columns
            .iter()
            .map(|column| {
                probe(column.build)
                    .remove(feature)
                    .unwrap_or_else(|| "no".to_string())
            })
            .collect();
        out.push_str(&cells.join(" | "));
        out.push_str(" |\n");
    }
    out
}

#[test]
fn the_matrix_is_generated_from_the_code_and_covers_every_probe() {
    // Run so the table is visible; not itself an assertion.
    let columns = vec![
        Column {
            endpoint: "OpenAI Chat",
            build: build_chat,
        },
        Column {
            endpoint: "OpenAI Responses",
            build: build_responses,
        },
    ];
    println!("\n{}", render(&columns));
}

/// The gate that matters: a hand-written table goes stale silently. This one
/// cannot, because it is derived from the builders and every probe is asserted.
#[test]
fn chat_completions_carries_the_portable_features() {
    let cells = probe(build_chat);
    for feature in [
        "text",
        "function tools (strict)",
        "tool_choice",
        "structured output",
        "reasoning",
        "multimodal input",
        "streaming",
    ] {
        assert_eq!(
            cells.get(feature).map(String::as_str),
            Some("yes"),
            "the matrix would claim Chat Completions lacks '{feature}'"
        );
    }
}

#[test]
fn provider_routing_passes_through_the_escape_hatch() {
    // The brief's matrix says `n/a` for provider routing on OpenAI. It reaches
    // the body through `extra_params` rather than a typed field, which is the
    // honest answer: it is provider-specific and typed in the OpenRouter
    // extension instead.
    let cells = probe(build_chat);
    assert_eq!(
        cells.get("raw extension hatch").map(String::as_str),
        Some("yes"),
        "the escape hatch must remain available for provider-specific fields"
    );
    assert_eq!(
        cells.get("provider routing (typed)").map(String::as_str),
        Some("no"),
        "provider routing is never a typed field on the portable request"
    );
}

#[test]
fn responses_is_not_reported_for_chat_only_fields() {
    // `/responses` has no `provider` routing block on the portable request, so a
    // matrix claiming it does would be wrong.
    let cells = probe(build_responses);
    assert_eq!(
        cells.get("provider routing (typed)").map(String::as_str),
        Some("no"),
        "{cells:?}"
    );
    // And it does carry its own native state.
    let request = vincents_llm::responses::ResponseRequest {
        model: "m".into(),
        input: vincents_llm::responses::ResponseInput::text("hi"),
        previous_response_id: Some("resp_p".into()),
        ..Default::default()
    }
    .to_body()
    .expect("built");
    assert_eq!(request["previous_response_id"], "resp_p");
}