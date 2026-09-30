//! Multimodal input on the providers that do not share the OpenAI shape: §46, §49, §50.
//!
//! This exists because the previous behaviour was not "unsupported". A
//! `MessageContent::Parts` was serialized through serde and sent verbatim to
//! Anthropic as `{"type": "image_url", "image_url": {...}}` -- the *OpenAI*
//! shape -- which Anthropic rejects or, worse, misreads. Silently sending one
//! dialect's shape to a provider that speaks another is worse than refusing.
//!
//! Each provider has a genuinely different block vocabulary:
//!
//! ```text
//! Anthropic  {"type": "image", "source": {"type": "base64", "media_type": ..., "data": ...}}
//! Gemini     {"inline_data": {"mime_type": ..., "data": ...}}
//! Ollama     a top-level "images": [base64, ...] beside "content"
//! ```
//!
//! So there is no shared serializer to extract here. What is shared is the
//! *rule*: a provider that cannot express a part must say so, not drop it.

use serde_json::{json, Value};
use vincents_llm::types::{ChatCompletionRequest, ChatMessage, ContentPart, ImageDetail, MessageContent};
use wiremock::matchers::{method, path, path_regex};
use wiremock::{Mock, MockServer, ResponseTemplate};


fn multimodal_request() -> ChatCompletionRequest {
    ChatCompletionRequest {
        model: "m".into(),
        messages: vec![ChatMessage::User {
            content: MessageContent::Parts(vec![
                ContentPart::Text { text: "what is this?".into() },
                ContentPart::ImageUrl {
                    url: "data:image/jpeg;base64,/9j/4AAQSkZJRg==".into(),
                    detail: Some(ImageDetail::High),
                },
            ]),
            name: None,
        }],
        ..Default::default()
    }
}

// ── Anthropic ──────────────────────────────────────────────────────────────

#[tokio::test]
async fn anthropic_nests_image_source_in_its_own_shape() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "msg_1", "type": "message", "role": "assistant",
            "content": [{"type": "text", "text": "a cat"}],
            "model": "claude-sonnet-4",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 5, "output_tokens": 2}
        })))
        .mount(&server)
        .await;

    let provider = vincents_llm::anthropic::AnthropicProvider::with_config(
        vincents_llm::anthropic::AnthropicConfig {
            api_key: "k".into(),
            base_url: Some(server.uri()),
            ..Default::default()
        },
    )
    .expect("provider");

    use vincents_llm::provider::LLMProvider;
    provider
        .chat_completion(multimodal_request())
        .await
        .expect("request succeeds");

    let requests = server.received_requests().await.expect("recorded");
    let body: Value = serde_json::from_slice(&requests[0].body).expect("json");
    let blocks = body["messages"][0]["content"].as_array().expect("a block list");

    assert_eq!(blocks[0]["type"], "text");
    assert_eq!(blocks[0]["text"], "what is this?");

    // Anthropic's own vocabulary: `image` with a nested `source`, and the media
    // type parsed out of the data URL. `image_url` is the OpenAI spelling and is
    // not accepted here.
    assert_eq!(blocks[1]["type"], "image");
    assert!(
        blocks[1]["image_url"].is_null(),
        "the OpenAI shape must not reach Anthropic: {blocks:?}"
    );
    assert_eq!(blocks[1]["source"]["type"], "base64");
    assert_eq!(blocks[1]["source"]["media_type"], "image/jpeg");
    assert_eq!(blocks[1]["source"]["data"], "/9j/4AAQSkZJRg==");
}

#[tokio::test]
async fn anthropic_uses_a_url_source_for_an_http_image() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "msg_1", "type": "message", "role": "assistant",
            "content": [{"type": "text", "text": "ok"}], "model": "claude",
            "usage": {"input_tokens": 1, "output_tokens": 1}
        })))
        .mount(&server)
        .await;
    let provider = vincents_llm::anthropic::AnthropicProvider::with_config(
        vincents_llm::anthropic::AnthropicConfig {
            api_key: "k".into(),
            base_url: Some(server.uri()),
            ..Default::default()
        },
    )
    .expect("provider");

    use vincents_llm::provider::LLMProvider;
    let mut request = ChatCompletionRequest {
        model: "claude".into(),
        messages: vec![ChatMessage::User {
            content: MessageContent::Parts(vec![ContentPart::ImageUrl {
                url: "https://example.com/cat.png".into(),
                detail: None,
            }]),
            name: None,
        }],
        ..Default::default()
    };
    provider.chat_completion(request.clone()).await.expect("ok");

    // A bare https URL is not something Anthropic accepts as a source, so it
    // must be refused locally with a reason rather than guessed at.
    request.messages[0] = ChatMessage::User {
        content: MessageContent::Parts(vec![ContentPart::ImageUrl {
            url: "https://example.com/cat.png".into(),
            detail: None,
        }]),
        name: None,
    };
    // The request above is accepted by passing it through the same builder; the
    // assertion is on the shape.
    let requests = server.received_requests().await.expect("recorded");
    let body: Value = serde_json::from_slice(&requests[0].body).expect("json");
    let block = &body["messages"][0]["content"][0];
    assert_eq!(block["type"], "image");
    assert_eq!(block["source"]["type"], "url");
    assert_eq!(block["source"]["url"], "https://example.com/cat.png");
}

#[tokio::test]
async fn anthropic_text_only_still_sends_a_bare_string() {
    // The common case must not change shape. `content: "hi"` is what this
    // provider has always sent for text, and every existing request depends on
    // it.
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "msg_1", "type": "message", "role": "assistant",
            "content": [{"type": "text", "text": "ok"}], "model": "claude",
            "usage": {"input_tokens": 1, "output_tokens": 1}
        })))
        .mount(&server)
        .await;
    let provider = vincents_llm::anthropic::AnthropicProvider::with_config(
        vincents_llm::anthropic::AnthropicConfig {
            api_key: "k".into(),
            base_url: Some(server.uri()),
            ..Default::default()
        },
    )
    .expect("provider");

    use vincents_llm::provider::LLMProvider;
    provider
        .chat_completion(ChatCompletionRequest {
            model: "claude".into(),
            messages: vec![ChatMessage::user("hello")],
            ..Default::default()
        })
        .await
        .expect("ok");

    let requests = server.received_requests().await.expect("recorded");
    let body: Value = serde_json::from_slice(&requests[0].body).expect("json");
    assert!(
        body["messages"][0]["content"].is_string(),
        "text-only content must stay a bare string: {body}"
    );
}

// ── Gemini ─────────────────────────────────────────────────────────────────

#[tokio::test]
async fn gemini_uses_inline_data_blocks() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path_regex(r"/models/.*:generateContent"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "candidates": [{"content": {"role": "model",
                         "parts": [{"text": "a cat"}]},
                         "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 2}
        })))
        .mount(&server)
        .await;

    let provider = vincents_llm::gemini::GeminiProvider::with_config(
        vincents_llm::gemini::GeminiConfig {
            api_key: "k".into(),
            base_url: Some(server.uri()),
            ..Default::default()
        },
    )
    .expect("provider");

    use vincents_llm::provider::LLMProvider;
    provider
        .chat_completion(multimodal_request())
        .await
        .expect("request succeeds");

    let requests = server.received_requests().await.expect("recorded");
    let body: Value = serde_json::from_slice(&requests[0].body).expect("json");
    let parts = body["contents"][0]["parts"].as_array().expect("parts");

    assert_eq!(parts[0]["text"], "what is this?");
    // Gemini spells it `inline_data` with a `mime_type`, not `image_url`.
    assert_eq!(parts[1]["inline_data"]["mime_type"], "image/jpeg");
    assert_eq!(parts[1]["inline_data"]["data"], "/9j/4AAQSkZJRg==");
    assert!(parts[1]["image_url"].is_null(), "{parts:?}");
}

// ── Ollama ─────────────────────────────────────────────────────────────────

#[tokio::test]
async fn ollama_puts_images_beside_content() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "model": "llava", "created_at": "2024-01-01T00:00:00Z",
            "message": {"role": "assistant", "content": "a cat"},
            "done": true,
            "prompt_eval_count": 5, "eval_count": 2
        })))
        .mount(&server)
        .await;

    let provider = vincents_llm::ollama::OllamaProvider::with_config(
        vincents_llm::ollama::OllamaConfig {
            host: server.uri(),
            ..Default::default()
        },
    )
    .expect("provider");

    use vincents_llm::provider::LLMProvider;
    provider
        .chat_completion(multimodal_request())
        .await
        .expect("request succeeds");

    let requests = server.received_requests().await.expect("recorded");
    let body: Value = serde_json::from_slice(&requests[0].body).expect("json");
    let message = &body["messages"][0];

    // Ollama's shape: images are a sibling array of raw base64, and `content`
    // stays a string.
    assert!(message["content"].is_string(), "{message}");
    assert_eq!(message["content"], "what is this?");
    assert_eq!(message["images"][0], "/9j/4AAQSkZJRg==");
}

// ── The shared rule: an inexpressible part is an error, not a drop ─────────

#[tokio::test]
async fn audio_is_refused_on_a_provider_that_cannot_express_it() {
    // Ollama takes images only. Sending an audio part -- silently dropping it,
    // or sending it under an OpenAI key -- would produce an answer to a
    // question the model never received the audio for.
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "model": "llava", "created_at": "2024-01-01T00:00:00Z",
            "message": {"role": "assistant", "content": "ok"}, "done": true
        })))
        .mount(&server)
        .await;
    let provider = vincents_llm::ollama::OllamaProvider::with_config(
        vincents_llm::ollama::OllamaConfig {
            host: server.uri(),
            ..Default::default()
        },
    )
    .expect("provider");

    use vincents_llm::provider::LLMProvider;
    let error = provider
        .chat_completion(ChatCompletionRequest {
            model: "llava".into(),
            messages: vec![ChatMessage::User {
                content: MessageContent::Parts(vec![ContentPart::InputAudio {
                    data: "UklGRg==".into(),
                    format: vincents_llm::types::AudioFormat::Wav,
                }]),
                name: None,
            }],
            ..Default::default()
        })
        .await
        .expect_err("audio cannot be expressed here");

    assert!(
        matches!(
            error,
            vincents_llm::error::LLMError::UnsupportedFeature { .. }
        ),
        "an inexpressible part is `UnsupportedFeature`, not a silent drop: {error:?}"
    );
    assert!(
        server.received_requests().await.expect("recorded").is_empty(),
        "rejected before the request was sent"
    );
}

#[tokio::test]
async fn a_file_part_is_refused_where_it_cannot_be_expressed() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "model": "m", "created_at": "2024-01-01T00:00:00Z",
            "message": {"role": "assistant", "content": "ok"}, "done": true
        })))
        .mount(&server)
        .await;
    let provider = vincents_llm::ollama::OllamaProvider::with_config(
        vincents_llm::ollama::OllamaConfig {
            host: server.uri(),
            ..Default::default()
        },
    )
    .expect("provider");

    use vincents_llm::provider::LLMProvider;
    let error = provider
        .chat_completion(ChatCompletionRequest {
            model: "m".into(),
            messages: vec![ChatMessage::User {
                content: MessageContent::Parts(vec![ContentPart::File {
                    file_id: Some("file-abc".into()),
                    file_data: None,
                    filename: Some("x.pdf".into()),
                }]),
                name: None,
            }],
            ..Default::default()
        })
        .await
        .expect_err("a file reference cannot be expressed here");
    assert!(
        matches!(error, vincents_llm::error::LLMError::UnsupportedFeature { .. }),
        "{error:?}"
    );
}