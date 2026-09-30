//! Typed message content and the `developer` role: §44–§51.
//!
//! Two things are asserted that a type-only check cannot establish:
//!
//! 1. A `developer` message is sent as `role: "developer"`, never as
//!    `"system"`. The two roles have different precedence in the protocol, so
//!    an alias would silently pick a winner.
//! 2. Multimodal content nests where the protocol says it nests. `detail` inside
//!    `image_url`, `format` inside `input_audio` — a flat shape compiles fine
//!    and is rejected by the API.

use serde_json::{json, Value};
use vincents_llm::error::LLMError;
use vincents_llm::types::{
    AudioFormat, ChatMessage, ContentPart, ImageDetail, MessageContent, MessageRole,
};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn body_of(msg: &ChatMessage) -> Value {
    msg.to_openai_json()
}

// ── The Developer role (§47) ───────────────────────────────────────────────

#[test]
fn a_developer_message_is_never_sent_as_system() {
    let wire = body_of(&ChatMessage::developer("you are terse"));
    assert_eq!(wire["role"], "developer");
    assert_eq!(wire["content"], "you are terse");
    assert_ne!(
        wire["role"],
        "system",
        "aliasing developer to system silently picks a precedence winner"
    );
}

#[test]
fn developer_and_system_are_distinct_variants() {
    // Not a rename, not a newtype over system: a caller that means one cannot
    // accidentally construct the other.
    let developer = ChatMessage::developer("x");
    let system = ChatMessage::system("x");
    assert_ne!(developer, system);
    assert_eq!(developer.role(), MessageRole::Developer);
    assert_eq!(system.role(), MessageRole::System);
    assert_ne!(
        body_of(&developer)["role"],
        body_of(&system)["role"]
    );
}

#[test]
fn developer_messages_survive_a_json_round_trip() {
    // Stored sessions are reloaded through serde; a `developer` turn that
    // deserializes as `system` would change the conversation on replay.
    let original = ChatMessage::developer("instruction");
    let encoded = serde_json::to_string(&original).expect("encodes");
    assert!(encoded.contains("\"role\":\"developer\""), "{encoded}");
    let decoded: ChatMessage = serde_json::from_str(&encoded).expect("decodes");
    assert_eq!(decoded, original);
}

#[test]
fn every_role_reports_its_own_message_role() {
    // Exhaustiveness at the enum level: a new variant with no `MessageRole`
    // would be a compile error, not a silent default.
    assert_eq!(ChatMessage::system("").role(), MessageRole::System);
    assert_eq!(ChatMessage::user("").role(), MessageRole::User);
    assert_eq!(ChatMessage::assistant("").role(), MessageRole::Assistant);
    assert_eq!(ChatMessage::assistant_with_tool_calls(vec![]).role(), MessageRole::Assistant);
}

// ── Text stays a bare string (§45) ─────────────────────────────────────────

#[test]
fn text_content_serializes_as_a_string_not_an_array() {
    // Collapsing `Text` into a one-element `Parts` would change the wire shape
    // of essentially every request in the ecosystem, and `content: "hi"` and
    // `content: [{...}]` are not equally supported everywhere.
    let wire = body_of(&ChatMessage::user("hello"));
    assert!(wire["content"].is_string(), "{wire}");
    assert_eq!(wire["content"], "hello");
}

#[test]
fn the_convenience_constructors_stay_concise() {
    // The brief requires ordinary call sites to remain concise.
    assert_eq!(body_of(&ChatMessage::user("hi"))["content"], "hi");
    assert_eq!(body_of(&ChatMessage::system("s"))["content"], "s");
    assert_eq!(body_of(&ChatMessage::assistant("a"))["content"], "a");
    assert_eq!(body_of(&ChatMessage::developer("d"))["content"], "d");
}

// ── Multimodal nesting (§46, §49, §50) ────────────────────────────────────

#[test]
fn an_image_url_nests_detail_inside_image_url() {
    let content = MessageContent::Parts(vec![ContentPart::ImageUrl {
        url: "https://example.com/cat.png".into(),
        detail: Some(ImageDetail::High),
    }]);
    let wire = ChatMessage::User { content, name: None }.to_openai_json();

    assert_eq!(wire["role"], "user");
    let part = &wire["content"][0];
    assert_eq!(part["type"], "image_url");
    // The nesting is the part providers reject. A flat
    // `{"type":"image_url","url":...,"detail":...}` is not the protocol.
    assert_eq!(part["image_url"]["url"], "https://example.com/cat.png");
    assert_eq!(part["image_url"]["detail"], "high");
    assert!(
        part.get("url").is_none(),
        "url must not also appear at the top level: {part}"
    );
    assert!(part.get("detail").is_none(), "{part}");
}

#[test]
fn an_image_without_detail_omits_the_key() {
    let content = MessageContent::Parts(vec![ContentPart::ImageUrl {
        url: "https://example.com/cat.png".into(),
        detail: None,
    }]);
    let wire = ChatMessage::User { content, name: None }.to_openai_json();
    assert!(wire["content"][0]["image_url"].get("detail").is_none());
}

#[test]
fn audio_nests_format_and_data_inside_input_audio() {
    let content = MessageContent::Parts(vec![ContentPart::InputAudio {
        data: "UklGRg==".into(),
        format: AudioFormat::Wav,
    }]);
    let wire = ChatMessage::User { content, name: None }.to_openai_json();
    let part = &wire["content"][0];
    assert_eq!(part["type"], "input_audio");
    assert_eq!(part["input_audio"]["data"], "UklGRg==");
    assert_eq!(part["input_audio"]["format"], "wav");
    assert!(part.get("data").is_none(), "{part}");
}

#[test]
fn a_file_part_carries_id_or_data_but_never_neither() {
    let by_id = MessageContent::Parts(vec![ContentPart::File {
        file_id: Some("file-abc".into()),
        file_data: None,
        filename: Some("notes.pdf".into()),
    }]);
    let wire = ChatMessage::User { content: by_id, name: None }.to_openai_json();
    assert_eq!(wire["content"][0]["type"], "file");
    assert_eq!(wire["content"][0]["file"]["file_id"], "file-abc");
    assert_eq!(wire["content"][0]["file"]["filename"], "notes.pdf");

    // Neither uploads nothing and references nothing.
    let neither = MessageContent::Parts(vec![ContentPart::File {
        file_id: None,
        file_data: None,
        filename: Some("x.pdf".into()),
    }]);
    let error = neither
        .to_openai_json()
        .expect_err("a file part with neither id nor data is a construction error");
    assert!(matches!(error, LLMError::SerializationError { .. }), "{error:?}");
}

#[test]
fn mixed_text_and_image_content_preserves_order() {
    let content = MessageContent::Parts(vec![
        ContentPart::Text { text: "what is this?".into() },
        ContentPart::ImageUrl { url: "https://example.com/x.png".into(), detail: None },
    ]);
    let wire = ChatMessage::User { content, name: None }.to_openai_json();
    assert_eq!(wire["content"][0]["type"], "text");
    assert_eq!(wire["content"][0]["text"], "what is this?");
    assert_eq!(wire["content"][1]["type"], "image_url");
    assert!(wire["content"].is_array());
}

// ── Lossless reporting rather than silent loss ─────────────────────────────

#[test]
fn multimodal_content_does_not_report_a_text_string() {
    // Concatenating the text parts of an image-and-text message would be a
    // guess about which part is "the" text.
    let content = MessageContent::Parts(vec![
        ContentPart::Text { text: "describe".into() },
        ContentPart::ImageUrl { url: "https://example.com/x.png".into(), detail: None },
    ]);
    let message = ChatMessage::User { content, name: None };
    assert_eq!(message.content(), None, "not a text-only message");
    assert!(message.message_content().expect("has content").is_multimodal());
}

#[test]
fn an_empty_parts_list_is_an_error_rather_than_empty_content() {
    // `content: []` would be rejected opaquely by the provider.
    let error = MessageContent::Parts(vec![])
        .to_openai_json()
        .expect_err("an empty parts list is a construction mistake");
    assert!(matches!(error, LLMError::SerializationError { .. }), "{error:?}");
}

#[test]
fn assistant_content_is_none_not_empty_for_a_tool_call_turn() {
    // The API distinguishes "no content" from "empty content"; conflating them
    // changes how the turn is billed and replayed.
    let wire = body_of(&ChatMessage::assistant_with_tool_calls(vec![]));
    assert!(wire["content"].is_null(), "{wire}");
    assert!(wire.get("content").is_some(), "the key must be present, not omitted");
}

#[test]
fn a_tool_message_keeps_its_call_id() {
    let wire = body_of(&ChatMessage::Tool {
        tool_call_id: "call_abc".into(),
        content: "12C and sunny".into(),
    });
    assert_eq!(wire["role"], "tool");
    assert_eq!(wire["tool_call_id"], "call_abc");
    assert_eq!(wire["content"], "12C and sunny");
}

// ── End to end ─────────────────────────────────────────────────────────────

#[tokio::test]
async fn a_multimodal_message_reaches_the_wire_through_the_provider() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl-1", "object": "chat.completion", "created": 1,
            "model": "gpt-4o",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "a cat"},
                         "finish_reason": "stop"}]
        })))
        .mount(&server)
        .await;

    let mut config = vincents_llm::openai::OpenAIConfig {
        api_key: "k".into(),
        base_url: Some(server.uri()),
        default_model: Some("gpt-4o".into()),
        ..Default::default()
    };
    config.organization_id = Some("org-1".into());
    let provider = vincents_llm::openai::OpenAIProvider::with_config(config).expect("provider");

    use vincents_llm::provider::LLMProvider;
    provider
        .chat_completion(vincents_llm::types::ChatCompletionRequest {
            model: "gpt-4o".into(),
            messages: vec![
                ChatMessage::developer("answer in one word"),
                ChatMessage::user_parts(vec![
                    ContentPart::Text { text: "what is this?".into() },
                    ContentPart::ImageUrl {
                        url: "https://example.com/cat.png".into(),
                        detail: Some(ImageDetail::Low),
                    },
                ]),
            ],
            ..Default::default()
        })
        .await
        .expect("request succeeds");

    let requests = server.received_requests().await.expect("recorded");
    let body: Value = serde_json::from_slice(&requests[0].body).expect("json");
    assert_eq!(body["messages"][0]["role"], "developer");
    assert_eq!(body["messages"][0]["content"], "answer in one word");
    assert_eq!(body["messages"][1]["content"][0]["type"], "text");
    assert_eq!(body["messages"][1]["content"][1]["image_url"]["detail"], "low");
}

// ═══════════════════════════════════════════════════════════════════════════
// Usage detail and provider metadata (§47, §48, §49)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn reasoning_tokens_are_read_from_the_nested_detail_object() {
    // The detail lives at `completion_tokens_details.reasoning_tokens`. A flat
    // read of `usage.reasoning_tokens` finds nothing, and a cost calculation
    // built on `total_tokens` still looks plausible while being wrong.
    let usage = vincents_llm::types::Usage::from_openai_json(&json!({
        "prompt_tokens": 100,
        "completion_tokens": 500,
        "total_tokens": 600,
        "completion_tokens_details": {"reasoning_tokens": 460}
    }));
    assert_eq!(usage.reasoning_tokens, Some(460));
    assert!(usage.has_token_details());
    // Reasoning is a subset of completion, not an addition.
    assert_eq!(usage.visible_completion_tokens(), 40);
    assert_eq!(usage.total_tokens, 600);
}

#[test]
fn cached_tokens_are_read_from_the_nested_detail_object() {
    let usage = vincents_llm::types::Usage::from_openai_json(&json!({
        "prompt_tokens": 10_000,
        "completion_tokens": 50,
        "total_tokens": 10_050,
        "prompt_tokens_details": {"cached_tokens": 9_000}
    }));
    assert_eq!(usage.cached_tokens, Some(9_000));
    assert_eq!(usage.uncached_prompt_tokens(), 1_000);
}

#[test]
fn absent_token_detail_is_not_zero() {
    // "The provider did not report it" and "the provider reported zero" are
    // different facts, and only one of them is safe to bill on.
    let usage = vincents_llm::types::Usage::from_openai_json(&json!({
        "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15
    }));
    assert_eq!(usage.reasoning_tokens, None);
    assert_eq!(usage.cached_tokens, None);
    assert!(!usage.has_token_details());
    // And the derived helpers behave as "nothing known" rather than inventing it.
    assert_eq!(usage.uncached_prompt_tokens(), 10);
    assert_eq!(usage.visible_completion_tokens(), 5);
}

#[test]
fn detailed_cost_prices_cache_and_reasoning_separately() {
    let usage = vincents_llm::types::Usage::from_openai_json(&json!({
        "prompt_tokens": 10_000, "completion_tokens": 500, "total_tokens": 10_500,
        "prompt_tokens_details": {"cached_tokens": 9_000},
        "completion_tokens_details": {"reasoning_tokens": 400}
    }));
    let cost = usage.calculate_detailed_cost(
        /* uncached input /1k */ 0.003,
        /* cached input   /1k */ 0.0003,
        /* output         /1k */ 0.015,
        /* reasoning      /1k */ 0.015,
    );
    // 1_000 uncached * .003 + 9_000 cached * .0003 + 400 reasoning * .015
    //   + 100 visible * .015, all / 1000
    let expected = (1_000.0 * 0.003 + 9_000.0 * 0.0003 + 400.0 * 0.015 + 100.0 * 0.015) / 1000.0;
    assert!((cost - expected).abs() < 1e-12, "{cost} != {expected}");
}

#[test]
fn a_missing_total_is_derived_rather_than_reported_as_zero() {
    let usage = vincents_llm::types::Usage::from_openai_json(&json!({
        "prompt_tokens": 7, "completion_tokens": 3
    }));
    assert_eq!(usage.total_tokens, 10);
}

#[tokio::test]
async fn provider_metadata_is_captured_without_parsing_headers() {
    // §49: the caller should not have to read HTTP headers to learn which
    // provider actually served the request.
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(json!({
                    "id": "chatcmpl-meta", "object": "chat.completion", "created": 99,
                    "model": "gpt-4o-2024-08-06",
                    "system_fingerprint": "fp_44709d6fcb",
                    "service_tier": "scale",
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": "x"},
                                 "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
                }))
                .insert_header("x-request-id", "req_abc123"),
        )
        .mount(&server)
        .await;

    let provider = vincents_llm::openai::OpenAIProvider::with_config(
        vincents_llm::openai::OpenAIConfig {
            api_key: "k".into(),
            base_url: Some(server.uri()),
            default_model: Some("gpt-4o".into()),
            ..Default::default()
        },
    )
    .expect("provider");

    use vincents_llm::provider::LLMProvider;
    let response = provider
        .chat_completion(vincents_llm::types::ChatCompletionRequest {
            model: "gpt-4o".into(),
            messages: vec![ChatMessage::user("hi")],
            ..Default::default()
        })
        .await
        .expect("request");

    // The returned model is the provider's, not the one requested: a routing
    // layer may have served a snapshot.
    assert_eq!(response.model, "gpt-4o-2024-08-06");
    assert_eq!(response.system_fingerprint.as_deref(), Some("fp_44709d6fcb"));
}
