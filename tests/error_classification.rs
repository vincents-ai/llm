//! Error classification: §50.
//!
//! The property under test is that a failure is reported in a category a caller
//! can *act* on. A 400 naming a parameter is not the same as an auth failure,
//! and a caller that sees both as `HttpError` has to read a string to decide.

use serde_json::json;
use vincents_llm::error::LLMError;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn ok_body() -> serde_json::Value {
    json!({
        "id": "chatcmpl-1", "object": "chat.completion", "created": 1, "model": "gpt-4o",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"},
                     "finish_reason": "stop"}]
    })
}

async fn failing(status: u16, body: serde_json::Value) -> LLMError {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(status)
                .set_body_json(body)
                .insert_header("x-request-id", "req_deadbeef"),
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
    provider
        .chat_completion(vincents_llm::types::ChatCompletionRequest {
            model: "gpt-4o".into(),
            messages: vec![vincents_llm::types::ChatMessage::user("hi")],
            ..Default::default()
        })
        .await
        .expect_err("the request must fail")
}

// ── The categories are distinguishable ─────────────────────────────────────

#[tokio::test]
async fn a_rejected_parameter_names_the_parameter() {
    // The only actionable part of this response is `param`. Reporting it as a
    // generic 400 loses exactly the thing the caller needed.
    let error = failing(
        400,
        json!({"error": {"message": "must be <= 4096", "type": "invalid_request_error",
                         "param": "max_completion_tokens"}}),
    )
    .await;

    match error {
        LLMError::ProviderRejectedParameter { parameter, reason } => {
            assert_eq!(parameter, "max_completion_tokens");
            assert!(reason.contains("4096"));
        }
        other => panic!("expected a rejected *parameter*, got {other:?}"),
    }
}

#[tokio::test]
async fn a_bad_key_is_an_auth_error_not_a_rejected_parameter() {
    let error = failing(
        401,
        json!({"error": {"message": "Incorrect API key provided", "type": "invalid_request_error"}}),
    )
    .await;
    assert!(
        matches!(error, LLMError::AuthenticationError { .. }),
        "{error:?}"
    );
    assert!(!error.is_retryable(), "a bad key does not fix itself");
}

#[tokio::test]
async fn a_missing_model_is_reported_as_such() {
    let error = failing(
        404,
        json!({"error": {"message": "The model `gpt-5` does not exist",
                         "type": "model_not_found", "param": "model"}}),
    )
    .await;
    assert!(matches!(error, LLMError::ModelNotAvailable { .. }), "{error:?}");
    // Retrying the same model will not help; failing over might.
    assert!(!error.is_retryable());
}

#[tokio::test]
async fn a_server_error_is_retryable_and_keeps_its_request_id() {
    // §50: the request id is what makes a report about one specific failed
    // call traceable to the provider's logs.
    let error = failing(
        503,
        json!({"error": {"message": "upstream unavailable", "type": "server_error"}}),
    )
    .await;

    assert!(error.is_retryable(), "a 503 is worth retrying");
    let text = error.to_string();
    assert!(text.contains("OpenAI"), "{text}");
    assert!(text.contains("503"), "{text}");
    assert!(
        text.contains("req_deadbeef"),
        "the provider's correlation handle must survive: {text}"
    );
}

#[tokio::test]
async fn a_non_json_error_body_does_not_lose_the_status() {
    // A proxy's HTML error page is still a provider failure, and it must not
    // become a parse error.
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(502)
                .set_body_string("<html>Bad Gateway</html>")
                .insert_header("x-request-id", "req_proxy"),
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
    let error = provider
        .chat_completion(vincents_llm::types::ChatCompletionRequest {
            model: "gpt-4o".into(),
            messages: vec![vincents_llm::types::ChatMessage::user("hi")],
            ..Default::default()
        })
        .await
        .expect_err("must fail");

    match error {
        LLMError::ProviderRequest { status, body, request_id, .. } => {
            assert_eq!(status, 502);
            assert!(body.expect("the raw body is preserved").contains("Bad Gateway"));
            assert_eq!(request_id.as_deref(), Some("req_proxy"));
        }
        other => panic!("an HTML error page must not become a parse error: {other:?}"),
    }
}

#[tokio::test]
async fn a_successful_response_produces_no_error() {
    // The classifier must not fire on the happy path; this is the check that the
    // status capture inserted at each call site did not change control flow.
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(ok_body()))
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
            messages: vec![vincents_llm::types::ChatMessage::user("hi")],
            ..Default::default()
        })
        .await
        .expect("the request must succeed");
    assert_eq!(response.choices.len(), 1);
}

// ── §50: protocol errors are not downgraded ───────────────────────────────

#[tokio::test]
async fn a_truncated_stream_is_a_stream_error_not_a_parse_error() {
    // §50's explicit prohibition. "Could not deserialize JSON" tells the
    // caller their document is malformed; the stream being cut is a different
    // fact with a different remedy.
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(
            "data: {\"id\":\"g\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"}}]}\n\n",
            "text/event-stream",
        ))
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

    use futures::StreamExt;
    use vincents_llm::provider::LLMProvider;
    let mut stream = provider
        .chat_completion_stream_events(vincents_llm::types::ChatCompletionRequest {
            model: "gpt-4o".into(),
            messages: vec![vincents_llm::types::ChatMessage::user("hi")],
            ..Default::default()
        })
        .await
        .expect("stream opens");

    let mut error = None;
    while let Some(event) = stream.next().await {
        if let Err(e) = event {
            error = Some(e);
        }
    }
    let error = error.expect("a truncated stream must surface an error");
    assert!(
        matches!(error, LLMError::StreamInterrupted { .. }),
        "not a serialization failure: {error:?}"
    );
    assert!(
        !matches!(error, LLMError::SerializationError { .. }),
        "§50 forbids downgrading a protocol error to a parse failure"
    );
}
// ── §52: unknown response fields survive ───────────────────────────────────

#[tokio::test]
async fn an_unrecognised_response_field_is_kept_rather_than_dropped() {
    // The provider sends a field this build predates. A caller routing on it
    // today should still be able to see it.
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl-u", "object": "chat.completion", "created": 1, "model": "gpt-4o",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            "service_tier": "scale",
            "a_field_from_the_future": {"nested": [1, 2, 3]}
        })))
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
            messages: vec![vincents_llm::types::ChatMessage::user("hi")],
            ..Default::default()
        })
        .await
        .expect("ok");

    assert!(response.has_unknown("a_field_from_the_future"));
    assert_eq!(
        response.unknown("a_field_from_the_future").expect("kept"),
        &json!({"nested": [1, 2, 3]}),
        "the value survives whole, not stringified"
    );
    assert_eq!(response.unknown("service_tier"), Some(&json!("scale")));
}

#[tokio::test]
async fn already_modelled_fields_do_not_also_appear_as_unknown() {
    // Otherwise every caller has two copies of `id` and `model`, and it is not
    // obvious which one is authoritative.
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(ok_body()))
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
            messages: vec![vincents_llm::types::ChatMessage::user("hi")],
            ..Default::default()
        })
        .await
        .expect("ok");

    for modelled in ["id", "object", "created", "model", "choices", "usage"] {
        assert!(
            !response.has_unknown(modelled),
            "'{modelled}' is modelled and must not also appear as unknown: {:?}",
            response.unknown_fields
        );
    }
}

#[tokio::test]
async fn a_normal_response_has_no_unknown_fields_and_serializes_unchanged() {
    // The field must not perturb a golden body: `skip_serializing_if` keeps it
    // off the wire when empty.
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(ok_body()))
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
            messages: vec![vincents_llm::types::ChatMessage::user("hi")],
            ..Default::default()
        })
        .await
        .expect("ok");

    assert!(response.unknown_fields.is_empty(), "{:?}", response.unknown_fields);
    let serialized = serde_json::to_value(&response).expect("serializes");
    assert!(serialized.get("unknown_fields").is_none(), "{serialized}");
}

#[tokio::test]
async fn unknown_fields_round_trip_through_a_stored_session() {
    // A session persisted and reloaded must not lose them -- that is the whole
    // point of keeping them.
    let body = json!({
        "id": "chatcmpl-rt", "object": "chat.completion", "created": 1, "model": "gpt-4o",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        "future_field": 42
    });
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
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
            messages: vec![vincents_llm::types::ChatMessage::user("hi")],
            ..Default::default()
        })
        .await
        .expect("ok");

    let stored = serde_json::to_string(&response).expect("stores");
    let reloaded: vincents_llm::types::ChatCompletionResponse =
        serde_json::from_str(&stored).expect("reloads");
    assert_eq!(reloaded.unknown("future_field"), Some(&json!(42)));
}

// ── §73: the provider's request id is observable on success, not only error ──

#[tokio::test]
async fn the_provider_request_id_is_captured_on_a_successful_response() {
    // §73: "Provider metadata/request ID survives the wrapper." `x-request-id` is
    // the provider's handle for the call -- a different namespace from the
    // response `id` -- and a support conversation is indexed by it.
    //
    // It used to be captured on the *error* path only, so a slow or
    // wrong-but-valid response had nothing to quote.
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(ok_body())
                .insert_header("x-request-id", "req_success_1"),
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
            messages: vec![vincents_llm::types::ChatMessage::user("hi")],
            ..Default::default()
        })
        .await
        .expect("ok");

    assert_eq!(
        response.provider_request_id(),
        Some("req_success_1"),
        "the provider's handle for the call"
    );
    // And it is not the response id -- the two are different namespaces.
    assert_ne!(response.provider_request_id(), Some(response.id.as_str()));
}

#[tokio::test]
async fn a_response_without_the_header_reports_none_rather_than_the_response_id() {
    // Backfilling with the response id would make the two indistinguishable and
    // send someone to the provider's logs with an id that is not in them.
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(ok_body()))
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
            messages: vec![vincents_llm::types::ChatMessage::user("hi")],
            ..Default::default()
        })
        .await
        .expect("ok");
    assert_eq!(response.provider_request_id(), None);
    assert!(!response.id.is_empty(), "the response id is still there");
}
