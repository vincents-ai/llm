//! The Chat Completions SSE parser, shared by every provider that speaks the
//! dialect.
//!
//! This exists as one module rather than one per provider because OpenAI and
//! OpenRouter speak the *same* dialect — the differences between them are
//! routing, attribution and a handful of extra body fields, never the wire
//! format of a stream. Two parsers would mean two places to reintroduce the
//! same bug, and the second one is exactly where the bug was.
//!
//! Every rule here is a consequence of a specific way the previous
//! implementation lied to the caller:
//!
//! | Rule | What it prevents |
//! |------|------------------|
//! | Tool-call deltas are events, not skipped | A tool-call response produced *no* events and a synthesized `stop` |
//! | Fragments key on `(choice, tool_index)` | Two interleaved calls assembled in arrival order, producing valid JSON belonging to the wrong call |
//! | `id`/`created` come from the provider | Fabricated `cmpl-{timestamp}` ids and per-chunk regenerated timestamps |
//! | `ResponseFinished` only on `[DONE]` | A truncated connection reported as a clean `stop` |
//! | A transport error yields `Err` | `while let Ok(...)`, which ends the stream and then reports success |
//! | Empty and role-only deltas are skipped, not lost | Nothing — but they must not be mistaken for text |

use std::collections::HashSet;
use std::pin::Pin;

use futures::StreamExt;
use serde_json::Value;

use crate::error::{LLMError, Result};
use crate::types::{ChatStreamEvent, Usage};

/// Parse an SSE byte stream into structured events.
///
/// A malformed *frame* is skipped: one bad event must not end a live stream, and
/// the provider's own `[DONE]` still ends it. A malformed *transport* is not
/// skipped — it is surfaced as an error, because a stream that cannot continue
/// and says nothing is the failure mode this module exists to remove.
pub fn parse_sse_stream(
    response: reqwest::Response,
) -> Pin<Box<dyn futures::Stream<Item = Result<ChatStreamEvent>> + Send>> {
    Box::pin(async_stream::stream! {
        let mut buffer = String::new();
        let mut bytes = response.bytes_stream();
        let mut announced: HashSet<(u32, u32)> = HashSet::new();
        let mut started = false;

        // A transport error must reach the caller, so the read is not written
        // as `while let Ok(..)`. The old form silently stopped the loop and
        // then fell through to the normal-terminator branch, so a dropped
        // connection was reported as a completed response.
        loop {
            let next = match bytes.next().await {
                Some(Ok(chunk)) => chunk,
                Some(Err(error)) => {
                    yield Err(LLMError::HttpError {
                        message: format!("stream transport failed: {error}"),
                        status_code: None,
                        body: None,
                    });
                    return;
                }
                // A clean end of the body without `[DONE]`.
                None => break,
            };

            buffer.push_str(&String::from_utf8_lossy(next.as_ref()));
            loop {
                let Some(position) = buffer.find('\n') else { break };
                let line: String = buffer.drain(..=position).collect();
                let line = line.trim_end();
                let Some(payload) = line.strip_prefix("data: ") else {
                    continue;
                };

                if payload.trim() == "[DONE]" {
                    // The provider's own terminator. `ResponseFinished` is
                    // emitted only here, never at the end of the body.
                    yield Ok(ChatStreamEvent::ResponseFinished);
                    return;
                }

                let Ok(event) = serde_json::from_str::<Value>(payload.trim()) else {
                    continue;
                };

                // The provider's identity, once, before anything derived from
                // it. Emitted ahead of the choices it would otherwise arrive
                // after the text it describes.
                if !started {
                    if let (Some(id), Some(model)) =
                        (event["id"].as_str(), event["model"].as_str())
                    {
                        started = true;
                        yield Ok(ChatStreamEvent::ResponseStart {
                            id: id.to_string(),
                            model: model.to_string(),
                            created: event["created"].as_u64(),
                            system_fingerprint: event["system_fingerprint"]
                                .as_str()
                                .map(str::to_string),
                        });
                    }
                }

                if let Some(usage) = event.get("usage").filter(|u| !u.is_null()) {
                    // The same parser as the non-streaming path, so a stream
                    // and a request report identical token detail.
                    yield Ok(ChatStreamEvent::Usage(Usage::from_openai_json(usage)));
                }

                let Some(choices) = event["choices"].as_array() else {
                    continue;
                };
                for choice in choices {
                    let choice_index = choice["index"].as_u64().unwrap_or(0) as u32;
                    let delta = &choice["delta"];

                    if let Some(text) = delta["content"].as_str() {
                        if !text.is_empty() {
                            yield Ok(ChatStreamEvent::TextDelta {
                                choice_index,
                                text: text.to_string(),
                            });
                        }
                    }
                    if let Some(refusal) = delta["refusal"].as_str() {
                        if !refusal.is_empty() {
                            yield Ok(ChatStreamEvent::RefusalDelta {
                                choice_index,
                                delta: refusal.to_string(),
                            });
                        }
                    }

                    if let Some(tool_calls) = delta["tool_calls"].as_array() {
                        for call in tool_calls {
                            // The index is what makes parallel calls
                            // reconstructable. Without it, two calls
                            // interleaved across frames are one ordered
                            // stream of fragments, and arrival-order assembly
                            // yields syntactically valid JSON for the wrong
                            // call.
                            let tool_index = call["index"].as_u64().unwrap_or(0) as u32;
                            let id = call["id"].as_str().map(str::to_string);
                            let name = call["function"]["name"].as_str().map(str::to_string);
                            let arguments = call["function"]["arguments"]
                                .as_str()
                                .unwrap_or("")
                                .to_string();

                            let key = (choice_index, tool_index);
                            if id.is_some() || name.is_some() || !arguments.is_empty() {
                                if announced.insert(key) {
                                    yield Ok(ChatStreamEvent::ToolCallStart {
                                        choice_index,
                                        tool_index,
                                        id: id.clone(),
                                        name: name.clone(),
                                    });
                                }
                            }
                            // Fragments are still emitted after the start: a
                            // provider may repeat the id, and dropping the
                            // arguments that follow a repeat would corrupt the
                            // call.
                            if !arguments.is_empty() {
                                if !announced.contains(&key) {
                                    announced.insert(key);
                                    yield Ok(ChatStreamEvent::ToolCallStart {
                                        choice_index,
                                        tool_index,
                                        id: id.clone(),
                                        name: name.clone(),
                                    });
                                }
                                yield Ok(ChatStreamEvent::ToolCallArgumentsDelta {
                                    choice_index,
                                    tool_index,
                                    delta: arguments,
                                });
                            }
                        }
                    }

                    if let Some(reason) = choice["finish_reason"].as_str() {
                        if !reason.is_empty() {
                            yield Ok(ChatStreamEvent::ChoiceFinished {
                                choice_index,
                                reason: Some(reason.to_string()),
                            });
                        }
                    }
                }
            }
        }

        // The body ended without `[DONE]`. That is a truncated stream, and it
        // is reported as an error rather than as a normal end -- a caller that
        // treats this as success will act on a partial answer.
        yield Err(LLMError::StreamInterrupted {
            message: "stream ended without the provider's [DONE] terminator".to_string(),
        });
    })
}

/// Adapt structured events to the legacy text-only [`crate::types::Chunk`] shape.
///
/// The legacy shape has no room for a tool-call fragment: `ToolCall` carries no
/// index, so two interleaved calls cannot be told apart and are dropped rather
/// than corrupted. A consumer that needs parallel tool calls must use
/// [`crate::provider::LLMProvider::chat_completion_stream_events`].
///
/// Identity from `ResponseStart` is threaded onto every emitted chunk, because
/// the legacy shape has no way to say "this chunk has no identity of its own" —
/// and an un-stamped chunk is indistinguishable from a fabricated one, which was
/// the original defect.
///
/// Errors pass through unchanged. Truncation in particular must not be smoothed
/// over here, or the adapter would reintroduce the lie it exists to avoid.
pub fn adapt_legacy(
    events: Pin<Box<dyn futures::Stream<Item = Result<ChatStreamEvent>> + Send>>,
) -> Pin<Box<dyn futures::Stream<Item = Result<crate::types::Chunk>> + Send>> {
    use crate::types::{
        ChatMessage, ChoiceDelta, Chunk, ChunkCompletion, CustomFunctionCall, FunctionCall, ToolCall,
    };

    Box::pin(async_stream::stream! {
        let mut events = events;
        let mut id: Option<String> = None;
        let mut model: Option<String> = None;
        let mut created: u64 = 0;
        let mut index: u32 = 0;
        // The response identity, kept separately from the tool-call ids that
        // flow through the same loop.
        let mut chunk_id: Option<String> = None;

        while let Some(event) = events.next().await {
            let event = match event {
                Ok(event) => event,
                Err(error) => {
                    yield Err(error);
                    return;
                }
            };
            match event {
                ChatStreamEvent::ResponseStart {
                    id: response_id,
                    model: response_model,
                    created: response_created,
                    ..
                } => {
                    id = Some(response_id.clone());
                    chunk_id = Some(response_id);
                    model = Some(response_model);
                    created = response_created.unwrap_or(0);
                }
                ChatStreamEvent::TextDelta {
                    choice_index,
                    text,
                } => {
                    index = choice_index;
                    yield Ok(Chunk::Completion(ChunkCompletion {
                        id: id.clone().unwrap_or_default(),
                        object: "chat.completion.chunk".to_string(),
                        created,
                        model: model.clone().unwrap_or_default(),
                        choices: vec![ChoiceDelta {
                            index: choice_index,
                            delta: Some(ChatMessage::assistant(&text)),
                            finish_reason: None,
                        }],
                    }));
                }
                ChatStreamEvent::ChoiceFinished { choice_index, reason } => {
                    // The provider's own reason. A `None` here is kept as
                    // `None`: inventing "stop" is the defect being removed.
                    index = choice_index;
                    if let (Some(id), Some(model)) = (id.clone(), model.clone()) {
                        yield Ok(Chunk::Completion(ChunkCompletion {
                            id,
                            object: "chat.completion.chunk".to_string(),
                            created,
                            model,
                            choices: vec![ChoiceDelta {
                                index: choice_index,
                                delta: None,
                                finish_reason: reason,
                            }],
                        }));
                    }
                }
                ChatStreamEvent::ToolCallStart {
                    choice_index,
                    id,
                    name,
                    ..
                } => {
                    // A start carrying *both* an id and a name identifies a call
                    // completely, so it survives the missing index: the id is
                    // the provider's own unique key for the call. Emitting it is
                    // strictly better than the previous behaviour, which
                    // dropped tool output entirely and left the caller with a
                    // `tool_calls` finish reason and no calls to run.
                    if let (Some(id), Some(name)) = (id, name) {
                        if let (Some(response_id), Some(model)) =
                            (chunk_id.clone(), model.clone())
                        {
                            index = choice_index;
                            // `content: None`, not `""`: a tool-call chunk
                            // has no text, and an empty string reads as "the
                            // model returned an empty answer".
                            let message = ChatMessage::Assistant {
                                content: None,
                                tool_calls: Some(vec![ToolCall::Function(
                                    FunctionCall::Custom(CustomFunctionCall {
                                        id: Some(id.clone()),
                                        name,
                                        // Empty, not fabricated: the arguments
                                        // arrive in the fragments that follow,
                                        // and this shape cannot place them.
                                        arguments: String::new(),
                                    }),
                                )]),
                                name: None,
                            };
                            yield Ok(Chunk::Completion(ChunkCompletion {
                                // The response id, not the tool call's. The two
                                // are different namespaces and conflating them
                                // makes every chunk in a parallel-call response
                                // look like a different conversation.
                                id: response_id,
                                object: "chat.completion.chunk".to_string(),
                                created,
                                model,
                                choices: vec![ChoiceDelta {
                                    index: choice_index,
                                    delta: Some(message),
                                    finish_reason: None,
                                }],
                            }));
                        }
                    }
                }
                // Argument fragments are dropped, and this is the one thing the
                // legacy shape cannot do: `ToolCall` has no index, so two calls
                // interleaved across frames cannot be separated. Emitting them
                // in arrival order would produce syntactically valid JSON
                // belonging to the wrong call — worse than dropping them.
                // `ToolCallAssembler` plus `chat_completion_stream_events` is
                // the way to get them.
                ChatStreamEvent::ToolCallArgumentsDelta { .. } => {}
                ChatStreamEvent::RefusalDelta { .. } => {}
                _ => {}
            }
        }
        let _ = index;
    })
}

/// One frame of an SSE stream.
///
/// `Done` is an explicit variant rather than the stream simply ending: a caller
/// must be able to tell the provider's terminator from a dropped connection,
/// and a stream that just stops looks identical either way. That distinction is
/// the whole reason [`parse_sse_json`] exists separately from a generic
/// "yield JSON lines" helper.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum SseFrame {
    Data(serde_json::Value),
    Done,
}

/// Parse an SSE stream into frames, forwarding each one.
///
/// The shared plumbing for a dialect that has its own event vocabulary: the
/// framing, the buffer handling, the transport-error handling and the
/// "no `[DONE]` means truncated" rule are all identical to the Chat
/// Completions parser, and reimplementing them per dialect is how they drift.
pub(crate) fn parse_sse_json(
    response: reqwest::Response,
) -> Pin<Box<dyn futures::Stream<Item = Result<SseFrame>> + Send>> {
    Box::pin(async_stream::stream! {
        let mut buffer = String::new();
        let mut bytes = response.bytes_stream();

        loop {
            let next = match bytes.next().await {
                Some(Ok(chunk)) => chunk,
                Some(Err(error)) => {
                    yield Err(LLMError::HttpError {
                        message: format!("stream transport failed: {error}"),
                        status_code: None,
                        body: None,
                    });
                    return;
                }
                None => break,
            };

            buffer.push_str(&String::from_utf8_lossy(next.as_ref()));
            while let Some(position) = buffer.find('\n') {
                let line: String = buffer.drain(..=position).collect();
                let line = line.trim_end();
                let Some(payload) = line.strip_prefix("data: ") else {
                    continue;
                };
                if payload.trim() == "[DONE]" {
                    // The provider's own terminator, surfaced rather than
                    // swallowed. Returning here is what ends the stream; the
                    // frame is emitted first so a caller knows it ended cleanly.
                    yield Ok(SseFrame::Done);
                    return;
                }
                // A malformed frame is skipped rather than ending the stream.
                if let Ok(value) = serde_json::from_str::<serde_json::Value>(payload.trim()) {
                    yield Ok(SseFrame::Data(value));
                }
            }
        }

        yield Err(LLMError::StreamInterrupted {
            message: "stream ended without the provider's [DONE] terminator".to_string(),
        });
    })
}
