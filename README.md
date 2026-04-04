# vincents-llm

Multi-provider LLM abstraction for Rust. Unified interface for OpenAI, Anthropic Claude, Google Gemini, Ollama, and OpenRouter — with built-in cost tracking, rate limiting, token counting, provider failover, and telemetry.

## License

This software is proprietary and commercially licensed. All rights reserved by Vincents.ai.
Unauthorized use, copying, modification, or distribution is strictly prohibited.
Contact [Vincents.ai](https://vincents.ai) for licensing enquiries.

---

## Features

- **Unified provider trait** — one interface across all supported providers
- **Five providers out of the box** — OpenAI, Anthropic, Gemini, Ollama, OpenRouter
- **Cost estimation** — per-request input/output cost tracking in USD
- **Rate limiting** — configurable RPM/TPM limits with exponential, linear, and fixed backoff retry strategies
- **Token counting** — accurate token estimates per model
- **Provider manager** — multi-provider coordination, automatic failover, circuit-breaker pattern
- **Streaming** — first-class streaming support via `futures::Stream`
- **Telemetry** — pluggable tracing with `NoopTelemetryRecorder` and `TracingTelemetryRecorder`
- **Feature flags** — compile only the providers you need

---

## Supported Providers

| Provider    | Feature flag   | Environment variable    |
|-------------|----------------|-------------------------|
| OpenAI      | `openai`       | `OPENAI_API_KEY`        |
| Anthropic   | `anthropic`    | `ANTHROPIC_API_KEY`     |
| Google Gemini | `gemini`     | `GEMINI_API_KEY`        |
| Ollama      | `ollama`       | *(local, no key needed)*|
| OpenRouter  | `openrouter`   | `OPENROUTER_API_KEY`    |

All features are enabled by default.

---

## Installation

```toml
[dependencies]
vincents-llm = { git = "https://github.com/vincents-ai/llm.git", branch = "main" }
```

To compile only specific providers:

```toml
[dependencies]
vincents-llm = { git = "https://github.com/vincents-ai/llm.git", branch = "main", default-features = false, features = ["openai", "anthropic"] }
```

---

## Quick Start

### Single provider

```rust
use vincents_llm::{AnthropicProvider, LLMProvider, ChatCompletionRequest, ChatMessage};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let provider = AnthropicProvider::new("your-api-key").await?;

    let request = ChatCompletionRequest {
        model: "claude-3-5-sonnet-20241022".to_string(),
        messages: vec![
            ChatMessage::system("You are a helpful assistant."),
            ChatMessage::user("What is the capital of France?"),
        ],
        max_tokens: Some(256),
        ..Default::default()
    };

    let response = provider.chat_completion(request).await?;
    println!("{}", response.choices[0].message.content().unwrap_or(""));

    Ok(())
}
```

### OpenAI

```rust
use vincents_llm::{OpenAIProvider, LLMProvider, ChatCompletionRequest, ChatMessage};

let provider = OpenAIProvider::new("your-api-key").await?;

let request = ChatCompletionRequest {
    model: "gpt-4o".to_string(),
    messages: vec![ChatMessage::user("Hello!")],
    temperature: Some(0.7),
    ..Default::default()
};

let response = provider.chat_completion(request).await?;
```

### Ollama (local)

```rust
use vincents_llm::{OllamaProvider, LLMProvider, ChatCompletionRequest, ChatMessage};

let provider = OllamaProvider::new("http://localhost:11434", "llama3.2").await?;

let request = ChatCompletionRequest {
    model: "llama3.2".to_string(),
    messages: vec![ChatMessage::user("Explain Rust lifetimes briefly.")],
    ..Default::default()
};

let response = provider.chat_completion(request).await?;
```

---

## Provider Manager (failover + circuit breaker)

`ProviderManager` coordinates multiple providers with automatic failover and circuit-breaking:

```rust
use vincents_llm::{ProviderManager, ProviderManagerConfig, OpenAIProvider, AnthropicProvider};
use std::sync::Arc;

let config = ProviderManagerConfig {
    default_provider: "openai".to_string(),
    failover_enabled: true,
    health_check_interval_sec: 60,
    circuit_breaker_threshold: 5,
    circuit_breaker_timeout_sec: 30,
    ..Default::default()
};

let manager = ProviderManager::new(config);
manager.register_provider(Arc::new(OpenAIProvider::new("openai-key").await?)).await;
manager.register_provider(Arc::new(AnthropicProvider::new("anthropic-key").await?)).await;
```

---

## Cost Tracking

Every provider exposes `estimate_cost` before sending a request, and the response carries `usage` statistics:

```rust
let cost = provider.estimate_cost(&request).await?;
println!("Estimated: ${:.6} ({} input + {} output tokens)",
    cost.total_cost, cost.input_tokens, cost.output_tokens);

let response = provider.chat_completion(request).await?;
if let Some(usage) = response.usage {
    println!("Actual tokens: {} total", usage.total_tokens);
}
```

---

## Rate Limiting

Configure per-provider rate limits via `ProviderConfig`:

```rust
use vincents_llm::config::{ProviderConfig, RateLimitConfig};

let config = ProviderConfig {
    provider_type: "openai".to_string(),
    rate_limits: Some(RateLimitConfig {
        requests_per_minute: 60,
        tokens_per_minute: 90_000,
        requests_per_day: Some(10_000),
    }),
    ..Default::default()
};
```

Retry strategies (`Exponential`, `Linear`, `Fixed`) are available via `RetryConfig`.

---

## Streaming

```rust
use futures::StreamExt;
use vincents_llm::{OpenAIProvider, LLMProvider, ChatCompletionRequest, ChatMessage};

let provider = OpenAIProvider::new("your-api-key").await?;

let request = ChatCompletionRequest {
    model: "gpt-4o".to_string(),
    messages: vec![ChatMessage::user("Write me a haiku.")],
    stream: Some(true),
    ..Default::default()
};

let mut stream = provider.chat_completion_stream(request).await?;
while let Some(chunk) = stream.next().await {
    let chunk = chunk?;
    // process chunk
}
```

---

## Listing Models

```rust
let models = provider.list_models().await?;
for model in models {
    println!("{} — context: {} tokens", model.name, model.context_window);
    if let Some(pricing) = model.pricing {
        println!("  ${}/1K input, ${}/1K output",
            pricing.prompt_tokens, pricing.completion_tokens);
    }
}
```

---

## Telemetry

Plug in the built-in tracing recorder or implement the `LlmTelemetryRecorder` trait:

```rust
use vincents_llm::TracingTelemetryRecorder;

// Uses the tracing crate — wire up your subscriber as usual
let recorder = TracingTelemetryRecorder::new();
```

Use `NoopTelemetryRecorder` in tests or when telemetry is not needed.

---

## Configuration reference

### `LLMConfig`

| Field              | Type                           | Default   | Description                        |
|--------------------|--------------------------------|-----------|------------------------------------|
| `default_provider` | `String`                       | `"openai"`| Provider used when none specified  |
| `providers`        | `HashMap<String, ProviderConfig>` | `{}`   | Named provider configurations      |
| `global`           | `GlobalLLMSettings`            | see below | Global behaviour flags             |

### `GlobalLLMSettings`

| Field                     | Default | Description                             |
|---------------------------|---------|-----------------------------------------|
| `cost_tracking`           | `true`  | Enable cost accumulation                |
| `rate_limiting`           | `true`  | Enable rate limit enforcement           |
| `failover_enabled`        | `true`  | Automatic provider failover             |
| `max_retries`             | `3`     | Retry attempts per request              |
| `default_timeout_secs`    | `60`    | Request timeout                         |
| `streaming_enabled`       | `true`  | Allow streaming responses               |
| `max_concurrent_requests` | `10`    | Concurrency cap                         |

---

## Project structure

```
src/
├── lib.rs              — public re-exports
├── types.rs            — ChatMessage, ChatCompletionRequest/Response, Usage, ModelInfo
├── provider.rs         — LLMProvider trait, ProviderRegistry, ProviderHealth
├── provider_manager.rs — multi-provider coordination + circuit breaker
├── config.rs           — LLMConfig, ProviderConfig, GlobalLLMSettings
├── error.rs            — LLMError enum
├── cost_tracker.rs     — CostEstimate, CostTracker
├── rate_limiter.rs     — RateLimiter, RetryConfig, RetryStrategy
├── token_counter.rs    — TokenCounter, TokenCounterConfig
├── telemetry.rs        — LlmTelemetryRecorder trait + implementations
├── openai.rs           — OpenAI provider
├── anthropic.rs        — Anthropic provider
├── gemini.rs           — Gemini provider
├── ollama.rs           — Ollama provider
└── openrouter.rs       — OpenRouter provider
```

---

## Copyright

Copyright © 2026 Vincents.ai. All rights reserved.
