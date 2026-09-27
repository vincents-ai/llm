/*!
 * Circuit Breaker for LLM Provider Calls
 *
 * Per-provider circuit breakers with configurable thresholds, auto-recovery
 * via half-open probing, and comprehensive metrics tracking.
 *
 * # State Machine
 *
 * ```text
 *  Closed ────(failures >= threshold)──→ Open
 *    ↑                                    │
 *    │                                    │ (timeout expires)
 *    │                                    ↓
 *    └──(successes >= success_threshold)── HalfOpen
 *                                         │  │
 *       (any failure) ←───────────────────┘  │
 *       back to Open            probe succeeds│
 * ```
 *
 * # Performance
 *
 * - Counters use `AtomicU64` — no lock contention on the hot path.
 * - Sliding window uses a bounded `VecDeque<Instant>`.
 * - State transitions use `tokio::sync::RwLock` for async safety.
 * - Zero allocation when the breaker is Closed and healthy.
 */

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

// ─── Configuration ──────────────────────────────────────────────────────────

/// Per-provider circuit breaker configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitBreakerConfig {
    /// Number of failures within `window` required to trip the breaker.
    /// Default: 5
    pub failure_threshold: u32,

    /// Number of consecutive successes in HalfOpen required to close the breaker.
    /// Default: 2
    pub success_threshold: u32,

    /// Duration the breaker stays Open before transitioning to HalfOpen.
    /// Default: 30 seconds
    pub timeout: Duration,

    /// Sliding window for counting failures. Only failures within this window
    /// count toward `failure_threshold`.
    /// Default: 60 seconds
    pub window: Duration,

    /// Maximum number of concurrent probe requests allowed in HalfOpen state.
    /// Default: 1
    pub half_open_max_probes: u32,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            success_threshold: 2,
            timeout: Duration::from_secs(30),
            window: Duration::from_secs(60),
            half_open_max_probes: 1,
        }
    }
}

impl CircuitBreakerConfig {
    /// Create a config tuned for aggressive protection (trips fast, recovers slowly).
    pub fn aggressive() -> Self {
        Self {
            failure_threshold: 3,
            success_threshold: 3,
            timeout: Duration::from_secs(60),
            window: Duration::from_secs(30),
            half_open_max_probes: 1,
        }
    }

    /// Create a config tuned for resilience (tolerates more failures before tripping).
    pub fn resilient() -> Self {
        Self {
            failure_threshold: 10,
            success_threshold: 2,
            timeout: Duration::from_secs(15),
            window: Duration::from_secs(120),
            half_open_max_probes: 2,
        }
    }
}

// ─── State ──────────────────────────────────────────────────────────────────

/// Circuit breaker state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CircuitState {
    /// Normal operation — requests flow through.
    Closed,
    /// Blocked — requests are rejected immediately.
    Open,
    /// Probing — limited requests allowed to test recovery.
    HalfOpen,
}

impl std::fmt::Display for CircuitState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CircuitState::Closed => write!(f, "closed"),
            CircuitState::Open => write!(f, "open"),
            CircuitState::HalfOpen => write!(f, "half_open"),
        }
    }
}

/// Internal state machine data, protected by a RwLock.
#[derive(Debug)]
struct BreakerInner {
    state: CircuitState,
    /// Timestamps of failures within the sliding window.
    failure_window: VecDeque<Instant>,
    /// Consecutive successes in HalfOpen (reset on any failure).
    half_open_successes: u32,
    /// Number of active probe requests in HalfOpen.
    half_open_active_probes: u32,
    /// When the breaker transitioned to Open (to calculate timeout).
    opened_at: Option<Instant>,
    /// When the last state change occurred.
    last_state_change: Instant,
    /// Cumulative time spent in Open state (for metrics).
    open_entered_at: Option<Instant>,
}

// ─── Metrics ────────────────────────────────────────────────────────────────

/// Snapshot of circuit breaker metrics for a single provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitBreakerMetrics {
    /// Current state.
    pub state: CircuitState,
    /// Provider name.
    pub provider: String,
    /// Total failure count (all time).
    pub total_failures: u64,
    /// Total success count (all time).
    pub total_successes: u64,
    /// Failures in the current sliding window.
    pub windowed_failures: u32,
    /// Consecutive successes (in HalfOpen).
    pub consecutive_successes: u32,
    /// Requests rejected while Open (all time).
    pub total_rejected: u64,
    /// Times the breaker has tripped to Open (all time).
    pub trip_count: u64,
    /// Time of last failure, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_failure_ago: Option<Duration>,
    /// Time of last success, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_success_ago: Option<Duration>,
    /// Time since last state change.
    pub last_state_change_ago: Duration,
    /// Cumulative time spent in Open state.
    pub total_time_open: Duration,
    /// Current config summary.
    pub config: CircuitBreakerConfigSummary,
}

/// Serializable summary of breaker config (avoids Duration serde issues).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitBreakerConfigSummary {
    pub failure_threshold: u32,
    pub success_threshold: u32,
    pub timeout_secs: u64,
    pub window_secs: u64,
    pub half_open_max_probes: u32,
}

impl From<&CircuitBreakerConfig> for CircuitBreakerConfigSummary {
    fn from(c: &CircuitBreakerConfig) -> Self {
        Self {
            failure_threshold: c.failure_threshold,
            success_threshold: c.success_threshold,
            timeout_secs: c.timeout.as_secs(),
            window_secs: c.window.as_secs(),
            half_open_max_probes: c.half_open_max_probes,
        }
    }
}

// ─── Atomic counters ────────────────────────────────────────────────────────

/// Lock-free counters that are incremented on the hot path without acquiring
/// the inner RwLock.
#[derive(Debug)]
struct AtomicCounters {
    total_failures: AtomicU64,
    total_successes: AtomicU64,
    total_rejected: AtomicU64,
    trip_count: AtomicU64,
}

impl AtomicCounters {
    fn new() -> Self {
        Self {
            total_failures: AtomicU64::new(0),
            total_successes: AtomicU64::new(0),
            total_rejected: AtomicU64::new(0),
            trip_count: AtomicU64::new(0),
        }
    }
}

// ─── Circuit Breaker ────────────────────────────────────────────────────────

/// A per-provider circuit breaker.
///
/// Thread-safe and async-safe. Uses atomic counters for the hot path and
/// a `tokio::sync::RwLock` for state transitions.
pub struct CircuitBreaker {
    provider_name: String,
    config: CircuitBreakerConfig,
    inner: RwLock<BreakerInner>,
    atomics: AtomicCounters,
    last_failure: RwLock<Option<Instant>>,
    last_success: RwLock<Option<Instant>>,
    total_time_open: RwLock<Duration>,
}

impl CircuitBreaker {
    /// Create a new circuit breaker for the given provider.
    pub fn new(provider_name: String, config: CircuitBreakerConfig) -> Self {
        Self {
            provider_name,
            config,
            inner: RwLock::new(BreakerInner {
                state: CircuitState::Closed,
                failure_window: VecDeque::with_capacity(32),
                half_open_successes: 0,
                half_open_active_probes: 0,
                opened_at: None,
                last_state_change: Instant::now(),
                open_entered_at: None,
            }),
            atomics: AtomicCounters::new(),
            last_failure: RwLock::new(None),
            last_success: RwLock::new(None),
            total_time_open: RwLock::new(Duration::ZERO),
        }
    }

    /// Provider name this breaker is guarding.
    pub fn provider_name(&self) -> &str {
        &self.provider_name
    }

    /// Check whether a request is allowed to proceed.
    ///
    /// Returns `Ok(())` if the request should proceed, or `Err(CircuitOpenError)`
    /// if the breaker is Open and blocking requests.
    ///
    /// In HalfOpen state, this atomically checks whether a probe slot is
    /// available and reserves it.
    pub async fn allow_request(&self) -> std::result::Result<(), CircuitOpenError> {
        let mut inner = self.inner.write().await;

        match inner.state {
            CircuitState::Closed => Ok(()),

            CircuitState::Open => {
                // Check if timeout has expired → transition to HalfOpen
                let opened_at = inner.opened_at.unwrap_or_else(|| Instant::now());
                if opened_at.elapsed() >= self.config.timeout {
                    inner.state = CircuitState::HalfOpen;
                    inner.half_open_successes = 0;
                    inner.half_open_active_probes = 0;
                    inner.last_state_change = Instant::now();
                    inner.opened_at = None;

                    info!(
                        provider = %self.provider_name,
                        "Circuit breaker transitioned: Open → HalfOpen"
                    );

                    // Allow this probe request
                    inner.half_open_active_probes += 1;
                    Ok(())
                } else {
                    self.atomics.total_rejected.fetch_add(1, Ordering::Relaxed);
                    Err(CircuitOpenError {
                        provider: self.provider_name.clone(),
                        remaining_timeout: self.config.timeout - opened_at.elapsed(),
                    })
                }
            }

            CircuitState::HalfOpen => {
                if inner.half_open_active_probes < self.config.half_open_max_probes {
                    inner.half_open_active_probes += 1;
                    Ok(())
                } else {
                    self.atomics.total_rejected.fetch_add(1, Ordering::Relaxed);
                    Err(CircuitOpenError {
                        provider: self.provider_name.clone(),
                        remaining_timeout: Duration::ZERO, // in half-open but max probes reached
                    })
                }
            }
        }
    }

    /// Record a successful request.
    ///
    /// In HalfOpen state, consecutive successes eventually close the breaker.
    /// In Closed state, this is a no-op (no counter incremented — we only
    /// track failures to decide when to open).
    pub async fn record_success(&self) {
        self.atomics.total_successes.fetch_add(1, Ordering::Relaxed);
        {
            let mut last = self.last_success.write().await;
            *last = Some(Instant::now());
        }

        let mut inner = self.inner.write().await;

        match inner.state {
            CircuitState::Closed => {
                // No action needed — breaker is healthy
            }
            CircuitState::Open => {
                // Shouldn't happen (requests are rejected), but handle gracefully
                debug!(
                    provider = %self.provider_name,
                    "Success recorded while Open — ignoring"
                );
            }
            CircuitState::HalfOpen => {
                inner.half_open_active_probes = inner.half_open_active_probes.saturating_sub(1);
                inner.half_open_successes += 1;

                if inner.half_open_successes >= self.config.success_threshold {
                    // Accumulate open time before transitioning
                    if let Some(entered) = inner.open_entered_at.take() {
                        let open_duration = entered.elapsed();
                        let mut total = self.total_time_open.write().await;
                        *total += open_duration;
                    }

                    inner.state = CircuitState::Closed;
                    inner.failure_window.clear();
                    inner.half_open_successes = 0;
                    inner.half_open_active_probes = 0;
                    inner.last_state_change = Instant::now();
                    inner.opened_at = None;

                    info!(
                        provider = %self.provider_name,
                        successes = %self.config.success_threshold,
                        "Circuit breaker transitioned: HalfOpen → Closed (recovered)"
                    );
                }
            }
        }
    }

    /// Record a failed request.
    ///
    /// In Closed state, failures accumulate in the sliding window. When the
    /// windowed count exceeds the threshold, the breaker trips to Open.
    /// In HalfOpen, any failure re-opens the breaker.
    pub async fn record_failure(&self) {
        self.atomics.total_failures.fetch_add(1, Ordering::Relaxed);
        {
            let mut last = self.last_failure.write().await;
            *last = Some(Instant::now());
        }

        let mut inner = self.inner.write().await;
        let now = Instant::now();

        match inner.state {
            CircuitState::Closed => {
                // Prune expired entries from the sliding window
                let cutoff = now - self.config.window;
                while inner.failure_window.front().map_or(false, |t| *t < cutoff) {
                    inner.failure_window.pop_front();
                }

                inner.failure_window.push_back(now);
                let windowed_count = inner.failure_window.len() as u32;

                if windowed_count >= self.config.failure_threshold {
                    inner.state = CircuitState::Open;
                    inner.opened_at = Some(now);
                    inner.open_entered_at = Some(now);
                    inner.last_state_change = now;
                    inner.half_open_successes = 0;
                    inner.half_open_active_probes = 0;

                    self.atomics.trip_count.fetch_add(1, Ordering::Relaxed);

                    warn!(
                        provider = %self.provider_name,
                        failures = windowed_count,
                        threshold = self.config.failure_threshold,
                        "Circuit breaker tripped: Closed → Open"
                    );
                }
            }

            CircuitState::Open => {
                // Already open — just update the timestamp to extend the open period
                inner.opened_at = Some(now);
                debug!(
                    provider = %self.provider_name,
                    "Failure recorded while Open — extending timeout"
                );
            }

            CircuitState::HalfOpen => {
                inner.half_open_active_probes = inner.half_open_active_probes.saturating_sub(1);
                // Any failure in HalfOpen re-opens the breaker
                inner.state = CircuitState::Open;
                inner.opened_at = Some(now);
                inner.last_state_change = now;
                inner.half_open_successes = 0;

                self.atomics.trip_count.fetch_add(1, Ordering::Relaxed);

                warn!(
                    provider = %self.provider_name,
                    "Circuit breaker re-opened: HalfOpen → Open (probe failed)"
                );
            }
        }
    }

    /// Get the current state without modifying anything.
    pub async fn state(&self) -> CircuitState {
        let inner = self.inner.read().await;

        match inner.state {
            CircuitState::Open => {
                // Check if we should be in HalfOpen
                if let Some(opened_at) = inner.opened_at {
                    if opened_at.elapsed() >= self.config.timeout {
                        return CircuitState::HalfOpen;
                    }
                }
                CircuitState::Open
            }
            other => other,
        }
    }

    /// Get a snapshot of all metrics for monitoring.
    pub async fn metrics(&self) -> CircuitBreakerMetrics {
        let inner = self.inner.read().await;
        let now = Instant::now();

        // Prune window for accurate count
        let cutoff = now - self.config.window;
        let windowed_failures = inner
            .failure_window
            .iter()
            .filter(|t| **t >= cutoff)
            .count() as u32;

        let current_state = match inner.state {
            CircuitState::Open => {
                if let Some(opened_at) = inner.opened_at {
                    if opened_at.elapsed() >= self.config.timeout {
                        CircuitState::HalfOpen
                    } else {
                        CircuitState::Open
                    }
                } else {
                    CircuitState::Open
                }
            }
            other => other,
        };

        // Calculate total time open (current session + accumulated)
        let total_open = {
            let accumulated = *self.total_time_open.read().await;
            match inner.open_entered_at {
                Some(entered) if inner.state == CircuitState::Open => {
                    accumulated + entered.elapsed()
                }
                _ => accumulated,
            }
        };

        let last_failure_ago = self
            .last_failure
            .read()
            .await
            .map(|t| now.duration_since(t));
        let last_success_ago = self
            .last_success
            .read()
            .await
            .map(|t| now.duration_since(t));

        CircuitBreakerMetrics {
            state: current_state,
            provider: self.provider_name.clone(),
            total_failures: self.atomics.total_failures.load(Ordering::Relaxed),
            total_successes: self.atomics.total_successes.load(Ordering::Relaxed),
            windowed_failures,
            consecutive_successes: inner.half_open_successes,
            total_rejected: self.atomics.total_rejected.load(Ordering::Relaxed),
            trip_count: self.atomics.trip_count.load(Ordering::Relaxed),
            last_failure_ago,
            last_success_ago,
            last_state_change_ago: now.duration_since(inner.last_state_change),
            total_time_open: total_open,
            config: CircuitBreakerConfigSummary::from(&self.config),
        }
    }

    /// Force the breaker to a specific state (for testing or admin override).
    pub async fn force_state(&self, new_state: CircuitState) {
        let mut inner = self.inner.write().await;
        let now = Instant::now();

        // If leaving Open, accumulate the time
        if inner.state == CircuitState::Open && new_state != CircuitState::Open {
            if let Some(entered) = inner.open_entered_at.take() {
                let mut total = self.total_time_open.write().await;
                *total += entered.elapsed();
            }
        }

        info!(
            provider = %self.provider_name,
            from = %inner.state,
            to = %new_state,
            "Circuit breaker state forced"
        );

        inner.state = new_state;
        inner.last_state_change = now;
        inner.failure_window.clear();
        inner.half_open_successes = 0;
        inner.half_open_active_probes = 0;

        if new_state == CircuitState::Open {
            inner.opened_at = Some(now);
            inner.open_entered_at = Some(now);
        } else {
            inner.opened_at = None;
            inner.open_entered_at = None;
        }
    }

    /// Reset all counters and return to Closed state.
    pub async fn reset(&self) {
        self.force_state(CircuitState::Closed).await;
        self.atomics.total_failures.store(0, Ordering::Relaxed);
        self.atomics.total_successes.store(0, Ordering::Relaxed);
        self.atomics.total_rejected.store(0, Ordering::Relaxed);
        self.atomics.trip_count.store(0, Ordering::Relaxed);
        *self.last_failure.write().await = None;
        *self.last_success.write().await = None;
        *self.total_time_open.write().await = Duration::ZERO;

        info!(provider = %self.provider_name, "Circuit breaker fully reset");
    }
}

// ─── Error type ─────────────────────────────────────────────────────────────

/// Error returned when the circuit breaker is Open and rejects a request.
#[derive(Debug, Clone)]
pub struct CircuitOpenError {
    pub provider: String,
    pub remaining_timeout: Duration,
}

impl std::fmt::Display for CircuitOpenError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "circuit breaker open for provider '{}' (retry after {:?})",
            self.provider, self.remaining_timeout
        )
    }
}

impl std::error::Error for CircuitOpenError {}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time;

    #[tokio::test]
    async fn test_closed_state_allows_requests() {
        let breaker =
            CircuitBreaker::new("test-provider".to_string(), CircuitBreakerConfig::default());

        assert_eq!(breaker.state().await, CircuitState::Closed);
        assert!(breaker.allow_request().await.is_ok());
    }

    #[tokio::test]
    async fn test_trips_on_threshold_failures() {
        let config = CircuitBreakerConfig {
            failure_threshold: 3,
            ..Default::default()
        };
        let breaker = CircuitBreaker::new("test".to_string(), config);

        for _ in 0..3 {
            breaker.record_failure().await;
        }

        assert_eq!(breaker.state().await, CircuitState::Open);
        let result = breaker.allow_request().await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_opens_then_recovers_after_timeout() {
        let config = CircuitBreakerConfig {
            failure_threshold: 2,
            timeout: Duration::from_millis(100),
            success_threshold: 1,
            ..Default::default()
        };
        let breaker = CircuitBreaker::new("test".to_string(), config);

        // Trip the breaker
        breaker.record_failure().await;
        breaker.record_failure().await;
        assert_eq!(breaker.state().await, CircuitState::Open);

        // Wait for timeout
        time::sleep(Duration::from_millis(150)).await;

        // Should now be HalfOpen and allow a probe
        assert_eq!(breaker.state().await, CircuitState::HalfOpen);
        assert!(breaker.allow_request().await.is_ok());

        // Successful probe closes the breaker
        breaker.record_success().await;
        assert_eq!(breaker.state().await, CircuitState::Closed);
    }

    #[tokio::test]
    async fn test_half_open_failure_reopens() {
        let config = CircuitBreakerConfig {
            failure_threshold: 2,
            timeout: Duration::from_millis(100),
            success_threshold: 2,
            ..Default::default()
        };
        let breaker = CircuitBreaker::new("test".to_string(), config);

        // Trip
        breaker.record_failure().await;
        breaker.record_failure().await;

        // Wait for half-open
        time::sleep(Duration::from_millis(150)).await;
        let _ = breaker.allow_request().await;

        // Failure in half-open reopens
        breaker.record_failure().await;
        assert_eq!(breaker.state().await, CircuitState::Open);
    }

    #[tokio::test]
    async fn test_sliding_window_expires_old_failures() {
        let config = CircuitBreakerConfig {
            failure_threshold: 3,
            window: Duration::from_millis(200),
            ..Default::default()
        };
        let breaker = CircuitBreaker::new("test".to_string(), config);

        // Two failures
        breaker.record_failure().await;
        breaker.record_failure().await;

        // Wait for window to expire
        time::sleep(Duration::from_millis(250)).await;

        // One more failure — old ones should have expired, so breaker stays Closed
        breaker.record_failure().await;
        assert_eq!(breaker.state().await, CircuitState::Closed);
    }

    #[tokio::test]
    async fn test_metrics_snapshot() {
        let breaker = CircuitBreaker::new("test".to_string(), CircuitBreakerConfig::default());

        breaker.record_success().await;
        breaker.record_failure().await;

        let metrics = breaker.metrics().await;
        assert_eq!(metrics.total_successes, 1);
        assert_eq!(metrics.total_failures, 1);
        assert_eq!(metrics.state, CircuitState::Closed);
        assert_eq!(metrics.provider, "test");
    }

    #[tokio::test]
    async fn test_metrics_rejected_count() {
        let config = CircuitBreakerConfig {
            failure_threshold: 1,
            timeout: Duration::from_secs(300), // long timeout so it stays Open
            ..Default::default()
        };
        let breaker = CircuitBreaker::new("test".to_string(), config);

        breaker.record_failure().await; // trips to Open

        // Try to send 5 requests while Open
        for _ in 0..5 {
            let _ = breaker.allow_request().await;
        }

        let metrics = breaker.metrics().await;
        assert_eq!(metrics.total_rejected, 5);
        assert_eq!(metrics.trip_count, 1);
    }

    #[tokio::test]
    async fn test_force_state() {
        let breaker = CircuitBreaker::new("test".to_string(), CircuitBreakerConfig::default());

        breaker.force_state(CircuitState::Open).await;
        assert_eq!(breaker.state().await, CircuitState::Open);

        breaker.force_state(CircuitState::Closed).await;
        assert_eq!(breaker.state().await, CircuitState::Closed);
    }

    #[tokio::test]
    async fn test_reset_clears_all() {
        let config = CircuitBreakerConfig {
            failure_threshold: 1,
            ..Default::default()
        };
        let breaker = CircuitBreaker::new("test".to_string(), config);

        breaker.record_failure().await;
        assert_eq!(breaker.state().await, CircuitState::Open);

        breaker.reset().await;
        assert_eq!(breaker.state().await, CircuitState::Closed);

        let metrics = breaker.metrics().await;
        assert_eq!(metrics.total_failures, 0);
        assert_eq!(metrics.total_successes, 0);
        assert_eq!(metrics.trip_count, 0);
    }

    #[tokio::test]
    async fn test_multiple_successes_to_close() {
        let config = CircuitBreakerConfig {
            failure_threshold: 1,
            timeout: Duration::from_millis(50),
            success_threshold: 3,
            ..Default::default()
        };
        let breaker = CircuitBreaker::new("test".to_string(), config);

        breaker.record_failure().await; // trips
        time::sleep(Duration::from_millis(100)).await; // wait for half-open

        // Need 3 successes to close
        for i in 0..2 {
            let _ = breaker.allow_request().await;
            breaker.record_success().await;
            assert_eq!(
                breaker.state().await,
                CircuitState::HalfOpen,
                "still half-open after success {}",
                i + 1
            );
        }

        let _ = breaker.allow_request().await;
        breaker.record_success().await;
        assert_eq!(
            breaker.state().await,
            CircuitState::Closed,
            "closed after 3 successes"
        );
    }
}
