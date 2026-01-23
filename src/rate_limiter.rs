use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use thiserror::Error;

/// Rate limit status for a provider
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitStatus {
    /// Requests remaining
    pub remaining: Option<u64>,
    /// Total requests allowed
    pub limit: Option<u64>,
    /// Tokens remaining (if available)
    pub tokens_remaining: Option<u64>,
    /// Tokens limit (if available)
    pub tokens_limit: Option<u64>,
    /// Reset timestamp (Unix)
    pub reset_at: Option<u64>,
    /// Retry after seconds (from headers)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retry_after: Option<u64>,
}

/// Retry strategy for rate limiting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RetryStrategy {
    /// Exponential backoff
    Exponential {
        /// Initial delay in milliseconds
        initial_delay_ms: u64,
        /// Maximum delay in milliseconds
        max_delay_ms: u64,
        /// Backoff multiplier
        multiplier: f64,
        /// Maximum number of retries
        max_retries: u32,
    },
    /// Linear backoff
    Linear {
        /// Delay in milliseconds
        delay_ms: u64,
        /// Maximum number of retries
        max_retries: u32,
    },
    /// Fixed delay
    Fixed {
        /// Delay in milliseconds
        delay_ms: u64,
        /// Maximum number of retries
        max_retries: u32,
    },
}

impl Default for RetryStrategy {
    fn default() -> Self {
        RetryStrategy::Exponential {
            initial_delay_ms: 1000,
            max_delay_ms: 60000,
            multiplier: 2.0,
            max_retries: 5,
        }
    }
}

/// Token bucket rate limiter
#[derive(Debug)]
pub struct RateLimiter {
    /// Requests per minute limit
    rpm_limit: AtomicU64,
    /// Current request count
    request_count: AtomicU64,
    /// Last reset time
    last_reset: AtomicU64,
    /// Token bucket state
    token_bucket: AtomicU64,
    /// Max tokens
    max_tokens: AtomicU64,
    /// Refill rate (tokens per second)
    refill_rate: AtomicU64,
}

impl RateLimiter {
    /// Create a new rate limiter
    pub fn new(requests_per_minute: u64) -> Self {
        Self {
            rpm_limit: AtomicU64::new(requests_per_minute),
            request_count: AtomicU64::new(0),
            last_reset: AtomicU64::new(0),
            token_bucket: AtomicU64::new(requests_per_minute),
            max_tokens: AtomicU64::new(requests_per_minute),
            refill_rate: AtomicU64::new(requests_per_minute / 60),
        }
    }

    /// Create with token bucket configuration
    pub fn with_tokens(max_tokens: u64, refill_rate_per_sec: u64) -> Self {
        Self {
            rpm_limit: AtomicU64::new(max_tokens * 60 / refill_rate_per_sec),
            request_count: AtomicU64::new(0),
            last_reset: AtomicU64::new(0),
            token_bucket: AtomicU64::new(max_tokens),
            max_tokens: AtomicU64::new(max_tokens),
            refill_rate: AtomicU64::new(refill_rate_per_sec),
        }
    }

    /// Check if request is allowed
    pub fn try_acquire(&self) -> bool {
        self.refill();
        let tokens = self.token_bucket.load(Ordering::Relaxed);
        tokens > 0 && self.try_consume()
    }

    /// Wait for rate limit availability
    pub async fn acquire(&self) -> std::result::Result<(), RateLimitError> {
        let start = Instant::now();
        let max_wait = Duration::from_secs(60);

        loop {
            self.refill();
            if self.try_consume() {
                return Ok(());
            }

            if start.elapsed() > max_wait {
                return Err(RateLimitError::Timeout);
            }

            let delay = Duration::from_millis(50);
            tokio::time::sleep(delay).await;
        }
    }

    /// Get current status
    pub fn status(&self) -> RateLimitStatus {
        self.refill();
        let tokens = self.token_bucket.load(Ordering::Relaxed);
        let max = self.max_tokens.load(Ordering::Relaxed);

        RateLimitStatus {
            remaining: Some(tokens),
            limit: Some(max),
            tokens_remaining: Some(tokens),
            tokens_limit: Some(max),
            reset_at: None,
            retry_after: None,
        }
    }

    /// Reset the rate limiter
    pub fn reset(&self) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        self.request_count.store(0, Ordering::Relaxed);
        self.last_reset.store(now, Ordering::Relaxed);
        self.token_bucket
            .store(self.max_tokens.load(Ordering::Relaxed), Ordering::Relaxed);
    }

    fn try_consume(&self) -> bool {
        let current = self.token_bucket.load(Ordering::Relaxed);
        if current == 0 {
            return false;
        }
        self.token_bucket
            .store(current.saturating_sub(1), Ordering::Relaxed);
        true
    }

    fn refill(&self) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let last_reset = self.last_reset.load(Ordering::Relaxed);
        let elapsed = now.saturating_sub(last_reset);

        if elapsed == 0 {
            return;
        }

        let refill = elapsed * self.refill_rate.load(Ordering::Relaxed);
        let max = self.max_tokens.load(Ordering::Relaxed);
        let current = self.token_bucket.load(Ordering::Relaxed);
        let new_tokens = std::cmp::min(max, current.saturating_add(refill));

        self.token_bucket.store(new_tokens, Ordering::Relaxed);
    }
}

/// Shared rate limiter reference
pub type SharedRateLimiter = Arc<RateLimiter>;

/// Rate limiter errors
#[derive(Error, Debug)]
pub enum RateLimitError {
    #[error("Rate limit exceeded")]
    Exceeded,
    #[error("Timeout waiting for rate limit")]
    Timeout,
}

/// Retry configuration combining backoff strategy with jitter
#[derive(Debug, Clone)]
pub struct RetryConfig {
    /// The retry strategy to use
    pub strategy: RetryStrategy,
    /// Whether to add random jitter to delays
    pub jitter: bool,
    /// Jitter factor (0.0 to 1.0) - percentage of delay to randomize
    pub jitter_factor: f64,
    /// Maximum total time to spend retrying (in seconds)
    pub max_total_time_sec: u64,
    /// HTTP status codes that should trigger a retry
    pub retry_on_status: &'static [u16],
}

impl Default for RetryConfig {
    fn default() -> Self {
        RetryConfig {
            strategy: RetryStrategy::default(),
            jitter: true,
            jitter_factor: 0.1,
            max_total_time_sec: 300,
            retry_on_status: &[429, 500, 502, 503, 504],
        }
    }
}

/// Retry policy that computes delays and manages retry state
#[derive(Debug)]
pub struct RetryPolicy {
    config: RetryConfig,
    current_attempt: u32,
    last_delay_ms: u64,
    total_elapsed_ms: u64,
}

impl RetryPolicy {
    /// Create a new retry policy
    pub fn new(config: RetryConfig) -> Self {
        RetryPolicy {
            config,
            current_attempt: 0,
            last_delay_ms: 0,
            total_elapsed_ms: 0,
        }

    }

    /// Reset the policy for a new operation
    pub fn reset(&mut self) {
        self.current_attempt = 0;
        self.last_delay_ms = 0;
        self.total_elapsed_ms = 0;
    }

    /// Get the delay for the next retry attempt
    pub fn next_delay(&mut self) -> Option<Duration> {
        if self.current_attempt >= self.config.strategy.max_retries() {
            return None;
        }

        let delay_ms = self.config.strategy.delay_for_attempt(self.current_attempt);

        let jittered = if self.config.jitter {
            let jitter_range = (delay_ms as f64 * self.config.jitter_factor).floor() as u64;
            if jitter_range > 0 {
                let jitter: u64 = rand::random::<u64>() % jitter_range;
                delay_ms.saturating_add(jitter)
            } else {
                delay_ms
            }
        } else {
            delay_ms
        };

        self.last_delay_ms = jittered;
        self.total_elapsed_ms += jittered;

        if self.total_elapsed_ms >= self.config.max_total_time_sec * 1000 {
            return None;
        }

        self.current_attempt += 1;
        Some(Duration::from_millis(jittered))
    }

    /// Get the current attempt number (0-based)
    pub fn attempt(&self) -> u32 {
        self.current_attempt
    }

    /// Check if we should retry based on HTTP status code
    pub fn should_retry_on_status(&self, status: u16) -> bool {
        self.config.retry_on_status.contains(&status)
    }

    /// Get the number of remaining retries
    pub fn remaining_retries(&self) -> u32 {
        self.config.strategy.max_retries().saturating_sub(self.current_attempt)
    }
}

impl RetryStrategy {
    /// Get the maximum number of retries for this strategy
    pub fn max_retries(&self) -> u32 {
        match self {
            RetryStrategy::Exponential { max_retries, .. } => *max_retries,
            RetryStrategy::Linear { max_retries, .. } => *max_retries,
            RetryStrategy::Fixed { max_retries, .. } => *max_retries,
        }
    }

    /// Calculate the delay for a given attempt number
    pub fn delay_for_attempt(&self, attempt: u32) -> u64 {
        match self {
            RetryStrategy::Exponential {
                initial_delay_ms,
                max_delay_ms,
                multiplier,
                ..
            } => {
                let delay = *initial_delay_ms as f64 * multiplier.powf(attempt as f64);
                delay.min(*max_delay_ms as f64) as u64
            }
            RetryStrategy::Linear {
                delay_ms,
                max_retries,
            } => {
                let remaining = max_retries.saturating_sub(attempt);
                delay_ms.saturating_mul(remaining as u64)
            }
            RetryStrategy::Fixed { delay_ms, .. } => *delay_ms,
        }
    }
}

/// Execute an async operation with retry logic
pub async fn with_retry<T, E, F, Fut>(
    mut policy: RetryPolicy,
    mut operation: F,
) -> std::result::Result<T, E>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = std::result::Result<T, E>>,
{
    loop {
        match operation().await {
            Ok(result) => return Ok(result),
            Err(e) => {
                if let Some(delay) = policy.next_delay() {
                    tokio::time::sleep(delay).await;
                } else {
                    return Err(e);
                }
            }
        }
    }
}
