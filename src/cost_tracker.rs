/*!
 * Cost Tracking Module
 *
 * Tracks and estimates costs for LLM usage across providers.
 */

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::RwLock;
use std::time::{Duration, Instant};

use crate::types::{Usage, ModelPricing};
use crate::error::Result;

/// Cost estimate for a request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostEstimate {
    /// Input token cost (USD)
    pub input_cost: f64,
    /// Output token cost (USD)
    pub output_cost: f64,
    /// Total cost (USD)
    pub total_cost: f64,
    /// Input tokens
    pub input_tokens: u32,
    /// Output tokens
    pub output_tokens: u32,
    /// Model used
    pub model: String,
    /// Provider name
    pub provider: String,
}

impl CostEstimate {
    /// Create a new cost estimate
    pub fn new(
        input_cost: f64,
        output_cost: f64,
        total_cost: f64,
        input_tokens: u32,
        output_tokens: u32,
        model: impl Into<String>,
        provider: impl Into<String>,
    ) -> Self {
        Self {
            input_cost,
            output_cost,
            total_cost,
            input_tokens,
            output_tokens,
            model: model.into(),
            provider: provider.into(),
        }
    }

    /// Create from usage and pricing
    pub fn from_usage(
        usage: &Usage,
        pricing: &ModelPricing,
        model: impl Into<String>,
        provider: impl Into<String>,
    ) -> Self {
        let input_cost = (usage.prompt_tokens as f64 / 1000.0) * pricing.prompt_tokens;
        let output_cost = (usage.completion_tokens as f64 / 1000.0) * pricing.completion_tokens;
        let total_cost = input_cost + output_cost;

        Self {
            input_cost,
            output_cost,
            total_cost,
            input_tokens: usage.prompt_tokens,
            output_tokens: usage.completion_tokens,
            model: model.into(),
            provider: provider.into(),
        }
    }
}

/// Tracks LLM costs over time
#[derive(Debug, Default)]
pub struct CostTracker {
    total_by_provider: RwLock<HashMap<String, f64>>,
    total_by_model: RwLock<HashMap<String, f64>>,
    tokens_by_provider: RwLock<HashMap<String, u64>>,
    requests_by_provider: RwLock<HashMap<String, u64>>,
    recent_costs: RwLock<Vec<(Instant, f64)>>,
    window_duration: Duration,
}

impl CostTracker {
    /// Create a new cost tracker with specified window duration
    pub fn new(window_duration: Duration) -> Self {
        Self {
            total_by_provider: RwLock::new(HashMap::new()),
            total_by_model: RwLock::new(HashMap::new()),
            tokens_by_provider: RwLock::new(HashMap::new()),
            requests_by_provider: RwLock::new(HashMap::new()),
            recent_costs: RwLock::new(Vec::new()),
            window_duration,
        }
    }

    /// Create a new cost tracker with default window (1 hour)
    pub fn default() -> Self {
        Self::new(Duration::from_secs(3600))
    }

    /// Track cost from a completion
    pub fn track_completion(&self, estimate: &CostEstimate) {
        let mut totals = self.total_by_provider.write().unwrap();
        *totals.entry(estimate.provider.clone()).or_insert(0.0) += estimate.total_cost;

        let mut totals = self.total_by_model.write().unwrap();
        *totals.entry(estimate.model.clone()).or_insert(0.0) += estimate.total_cost;

        let mut tokens = self.tokens_by_provider.write().unwrap();
        let provider_tokens = estimate.input_tokens + estimate.output_tokens;
        *tokens.entry(estimate.provider.clone()).or_insert(0) += provider_tokens as u64;

        let mut requests = self.requests_by_provider.write().unwrap();
        *requests.entry(estimate.provider.clone()).or_insert(0) += 1;

        let mut recent = self.recent_costs.write().unwrap();
        recent.push((Instant::now(), estimate.total_cost));

        self.cleanup_recent();
    }

    /// Track usage directly
    pub fn track_usage(
        &self,
        usage: &Usage,
        pricing: &ModelPricing,
        model: &str,
        provider: &str,
    ) {
        let estimate = CostEstimate::from_usage(usage, pricing, model, provider);
        self.track_completion(&estimate);
    }

    /// Get total cost by provider
    pub fn total_by_provider(&self) -> HashMap<String, f64> {
        self.total_by_provider.read().unwrap().clone()
    }

    /// Get total cost by model
    pub fn total_by_model(&self) -> HashMap<String, f64> {
        self.total_by_model.read().unwrap().clone()
    }

    /// Get total cost across all providers
    pub fn total_cost(&self) -> f64 {
        self.total_by_provider
            .read()
            .unwrap()
            .values()
            .sum()
    }

    /// Get recent cost (within window)
    pub fn recent_cost(&self) -> f64 {
        self.cleanup_recent();
        self.recent_costs
            .read()
            .unwrap()
            .iter()
            .map(|(_, cost)| cost)
            .sum()
    }

    /// Get token usage by provider
    pub fn tokens_by_provider(&self) -> HashMap<String, u64> {
        self.tokens_by_provider.read().unwrap().clone()
    }

    /// Get total tokens
    pub fn total_tokens(&self) -> u64 {
        self.tokens_by_provider
            .read()
            .unwrap()
            .values()
            .sum()
    }

    /// Get request count by provider
    pub fn requests_by_provider(&self) -> HashMap<String, u64> {
        self.requests_by_provider.read().unwrap().clone()
    }

    /// Get total requests
    pub fn total_requests(&self) -> u64 {
        self.requests_by_provider
            .read()
            .unwrap()
            .values()
            .sum()
    }

    /// Get all cost statistics
    pub fn stats(&self) -> CostStats {
        let by_provider = self.total_by_provider();
        let by_model = self.total_by_model();

        let mut provider_costs: Vec<_> = by_provider.iter().collect();
        provider_costs.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());

        let mut model_costs: Vec<_> = by_model.iter().collect();
        model_costs.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());

        CostStats {
            total_cost: self.total_cost(),
            recent_cost: self.recent_cost(),
            total_tokens: self.total_tokens(),
            total_requests: self.total_requests(),
            cost_by_provider: by_provider.clone(),
            cost_by_model: by_model.clone(),
            top_provider: by_provider.iter().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)).map(|(k, v)| (k.clone(), *v)),
            top_model: by_model.iter().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)).map(|(k, v)| (k.clone(), *v)),
            window_duration: self.window_duration,
        }
    }

    /// Export statistics as JSON
    pub fn export_stats(&self) -> std::result::Result<String, serde_json::Error> {
        serde_json::to_string_pretty(&self.stats())
    }

    /// Clean up old entries from recent costs
    fn cleanup_recent(&self) {
        let cutoff = Instant::now() - self.window_duration;
        let mut recent = self.recent_costs.write().unwrap();
        recent.retain(|&(time, _)| time > cutoff);
    }

    /// Reset all tracked data
    pub fn reset(&self) {
        *self.total_by_provider.write().unwrap() = HashMap::new();
        *self.total_by_model.write().unwrap() = HashMap::new();
        *self.tokens_by_provider.write().unwrap() = HashMap::new();
        *self.requests_by_provider.write().unwrap() = HashMap::new();
        *self.recent_costs.write().unwrap() = Vec::new();
    }
}

/// Cost statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostStats {
    pub total_cost: f64,
    pub recent_cost: f64,
    pub total_tokens: u64,
    pub total_requests: u64,
    pub cost_by_provider: HashMap<String, f64>,
    pub cost_by_model: HashMap<String, f64>,
    pub top_provider: Option<(String, f64)>,
    pub top_model: Option<(String, f64)>,
    pub window_duration: Duration,
}
