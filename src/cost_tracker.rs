/*!
 * Cost Tracking Module
 *
 * Tracks and estimates costs for LLM usage across providers.
 * Enhanced with daily usage tracking, pre-request estimation, and budget management.
 */

use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::RwLock;
use std::time::{Duration, Instant};

use crate::types::{ModelPricing, Usage};

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

/// Daily usage statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DailyUsage {
    /// Tokens used on this date
    pub tokens: u64,
    /// Cost incurred on this date
    pub cost: f64,
    /// Requests made on this date
    pub requests: u64,
}

/// Complete usage statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct UsageStats {
    /// Total tokens used
    pub total_tokens: u64,
    /// Total cost incurred
    pub total_cost: f64,
    /// Cost by model
    pub cost_by_model: HashMap<String, f64>,
    /// Tokens by model
    pub tokens_by_model: HashMap<String, u64>,
    /// Usage by date (YYYY-MM-DD)
    pub usage_by_date: HashMap<String, DailyUsage>,
}

/// Budget configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetConfig {
    /// Maximum daily budget (USD)
    pub daily_limit: Option<f64>,
    /// Maximum monthly budget (USD)
    pub monthly_limit: Option<f64>,
    /// Currency for budget
    pub currency: String,
}

/// Internal state for cost tracking
#[derive(Default, Debug)]
struct CostTrackerState {
    total_by_provider: HashMap<String, f64>,
    total_by_model: HashMap<String, f64>,
    tokens_by_provider: HashMap<String, u64>,
    requests_by_provider: HashMap<String, u64>,
    recent_costs: Vec<(Instant, f64)>,
    usage_stats: UsageStats,
}

/// Tracks LLM costs over time
pub struct CostTracker {
    state: RwLock<CostTrackerState>,
    window_duration: Duration,
    budget: RwLock<Option<BudgetConfig>>,
}

impl CostTracker {
    /// Create a new cost tracker with specified window duration
    pub fn new(window_duration: Duration) -> Self {
        Self {
            state: RwLock::new(CostTrackerState::default()),
            window_duration,
            budget: RwLock::new(None),
        }
    }

    /// Create a new cost tracker with default window (1 hour)
    pub fn default() -> Self {
        Self::new(Duration::from_secs(3600))
    }

    /// Create a new cost tracker with budget configuration
    pub fn with_budget(budget: BudgetConfig) -> Self {
        Self {
            budget: RwLock::new(Some(budget)),
            ..Self::default()
        }
    }

    /// Set budget configuration
    pub fn set_budget(&self, budget: BudgetConfig) {
        *self.budget.write().unwrap() = Some(budget);
    }

    /// Get current budget configuration
    pub fn budget(&self) -> Option<BudgetConfig> {
        self.budget.read().unwrap().clone()
    }

    /// Estimate cost before making a request
    pub fn estimate_cost(
        &self,
        model: &str,
        pricing: &ModelPricing,
        input_tokens: u32,
        output_tokens: u32,
    ) -> CostEstimate {
        let input_cost = (input_tokens as f64 / 1000.0) * pricing.prompt_tokens;
        let output_cost = (output_tokens as f64 / 1000.0) * pricing.completion_tokens;
        let total_cost = input_cost + output_cost;

        CostEstimate {
            input_cost,
            output_cost,
            total_cost,
            input_tokens,
            output_tokens,
            model: model.to_string(),
            provider: "unknown".to_string(),
        }
    }

    /// Check if request would exceed budget
    pub fn check_budget(&self, estimated_cost: &CostEstimate) -> Result<(), String> {
        let budget = self.budget.read().unwrap();
        if let Some(ref config) = *budget {
            let today = Utc::now().format("%Y-%m-%d").to_string();
            let state = self.state.read().unwrap();
            
            if let Some(daily) = state.usage_stats.usage_by_date.get(&today) {
                if let Some(daily_limit) = config.daily_limit {
                    if daily.cost + estimated_cost.total_cost > daily_limit {
                        return Err(format!(
                            "Daily budget of {} {} would be exceeded",
                            daily_limit, config.currency
                        ));
                    }
                }
            }
        }
        Ok(())
    }

    /// Track cost from a completion
    pub fn track_completion(&self, estimate: &CostEstimate) {
        let mut state = self.state.write().unwrap();
        
        *state.total_by_provider.entry(estimate.provider.clone()).or_insert(0.0) += estimate.total_cost;
        *state.total_by_model.entry(estimate.model.clone()).or_insert(0.0) += estimate.total_cost;
        
        let provider_tokens = estimate.input_tokens + estimate.output_tokens;
        *state.tokens_by_provider.entry(estimate.provider.clone()).or_insert(0) += provider_tokens as u64;
        *state.requests_by_provider.entry(estimate.provider.clone()).or_insert(0) += 1;
        
        state.recent_costs.push((Instant::now(), estimate.total_cost));
        
        self.record_daily_usage_locked(&mut state, estimate);
    }

    /// Track usage directly with model and provider
    pub fn track_usage(&self, usage: &Usage, pricing: &ModelPricing, model: &str, provider: &str) {
        let estimate = CostEstimate::from_usage(usage, pricing, model, provider);
        self.track_completion(&estimate);
        
        let mut state = self.state.write().unwrap();
        *state.usage_stats.tokens_by_model.entry(model.to_string()).or_insert(0) += usage.total_tokens as u64;
    }

    /// Record daily usage for budget tracking (requires lock)
    fn record_daily_usage_locked(&self, state: &mut CostTrackerState, estimate: &CostEstimate) {
        let today = Utc::now().format("%Y-%m-%d").to_string();
        
        state.usage_stats.total_cost += estimate.total_cost;
        state.usage_stats.total_tokens += estimate.input_tokens as u64 + estimate.output_tokens as u64;
        
        *state.usage_stats.cost_by_model.entry(estimate.model.clone()).or_insert(0.0) += estimate.total_cost;
        
        let daily = state.usage_stats.usage_by_date.entry(today).or_default();
        daily.cost += estimate.total_cost;
        daily.requests += 1;
    }

    /// Get total cost by provider
    pub fn total_by_provider(&self) -> HashMap<String, f64> {
        self.state.read().unwrap().total_by_provider.clone()
    }

    /// Get total cost by model
    pub fn total_by_model(&self) -> HashMap<String, f64> {
        self.state.read().unwrap().total_by_model.clone()
    }

    /// Get total cost across all providers
    pub fn total_cost(&self) -> f64 {
        self.state.read().unwrap().total_by_provider.values().sum()
    }

    /// Get recent cost (within window)
    pub fn recent_cost(&self) -> f64 {
        let mut state = self.state.write().unwrap();
        let cutoff = Instant::now() - self.window_duration;
        state.recent_costs.retain(|&(time, _)| time > cutoff);
        state.recent_costs.iter().map(|(_, cost)| *cost).sum()
    }

    /// Get token usage by provider
    pub fn tokens_by_provider(&self) -> HashMap<String, u64> {
        self.state.read().unwrap().tokens_by_provider.clone()
    }

    /// Get total tokens
    pub fn total_tokens(&self) -> u64 {
        self.state.read().unwrap().tokens_by_provider.values().sum()
    }

    /// Get request count by provider
    pub fn requests_by_provider(&self) -> HashMap<String, u64> {
        self.state.read().unwrap().requests_by_provider.clone()
    }

    /// Get total requests
    pub fn total_requests(&self) -> u64 {
        self.state.read().unwrap().requests_by_provider.values().sum()
    }

    /// Get complete usage statistics
    pub fn get_usage_stats(&self) -> UsageStats {
        self.state.read().unwrap().usage_stats.clone()
    }

    /// Get usage for a specific date range
    pub fn get_usage_in_range(&self, start_date: &str, end_date: &str) -> HashMap<String, DailyUsage> {
        let state = self.state.read().unwrap();
        state
            .usage_stats
            .usage_by_date
            .iter()
            .filter(|(date, _)| date.as_str() >= start_date && date.as_str() <= end_date)
            .map(|(date, usage)| (date.clone(), usage.clone()))
            .collect()
    }

    /// Get all cost statistics
    pub fn stats(&self) -> CostStats {
        let state = self.state.read().unwrap();
        let by_provider = state.total_by_provider.clone();
        let by_model = state.total_by_model.clone();

        let mut provider_costs: Vec<_> = by_provider.iter().collect();
        provider_costs.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());

        let mut model_costs: Vec<_> = by_model.iter().collect();
        model_costs.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());

        CostStats {
            total_cost: by_provider.values().sum(),
            recent_cost: state.recent_costs.iter().map(|(_, c)| *c).sum(),
            total_tokens: state.tokens_by_provider.values().sum(),
            total_requests: state.requests_by_provider.values().sum(),
            cost_by_provider: by_provider.clone(),
            cost_by_model: by_model.clone(),
            top_provider: by_provider
                .iter()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(k, v)| (k.clone(), *v)),
            top_model: by_model
                .iter()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(k, v)| (k.clone(), *v)),
            window_duration: self.window_duration,
        }
    }

    /// Export statistics as JSON
    pub fn export_stats(&self) -> std::result::Result<String, serde_json::Error> {
        serde_json::to_string_pretty(&self.stats())
    }

    /// Export usage stats as JSON
    pub fn export_usage_stats(&self) -> std::result::Result<String, serde_json::Error> {
        serde_json::to_string_pretty(&self.get_usage_stats())
    }

    /// Reset all tracked data
    pub fn reset(&self) {
        *self.state.write().unwrap() = CostTrackerState::default();
    }

    /// Reset usage statistics only
    pub fn reset_stats(&self) {
        self.state.write().unwrap().usage_stats = UsageStats::default();
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

#[cfg(test)]
mod tests {
    use super::*;

    fn test_pricing() -> ModelPricing {
        ModelPricing {
            prompt_tokens: 0.03,
            completion_tokens: 0.06,
            image_tokens: None,
            is_free: false,
        }
    }

    #[test]
    fn test_cost_estimation() {
        let tracker = CostTracker::default();
        let estimate = tracker.estimate_cost("gpt-4", &test_pricing(), 1000, 500);
        
        assert!((estimate.input_cost - 0.03).abs() < 0.001);
        assert!((estimate.output_cost - 0.03).abs() < 0.001);
        assert!((estimate.total_cost - 0.06).abs() < 0.001);
    }

    #[test]
    fn test_usage_tracking() {
        let tracker = CostTracker::default();
        let usage = Usage::new(100, 50);
        
        tracker.track_usage(&usage, &test_pricing(), "gpt-4", "openai");
        
        assert_eq!(tracker.total_tokens(), 150);
        assert!(tracker.total_cost() > 0.0);
        
        let stats = tracker.get_usage_stats();
        assert_eq!(stats.total_tokens, 150);
    }

    #[test]
    fn test_budget_check() {
        let budget = BudgetConfig {
            daily_limit: Some(1.0),
            monthly_limit: None,
            currency: "USD".to_string(),
        };
        
        let tracker = CostTracker::with_budget(budget);
        
        let estimate = CostEstimate {
            input_cost: 0.5,
            output_cost: 0.3,
            total_cost: 0.8,
            input_tokens: 1000,
            output_tokens: 500,
            model: "gpt-4".to_string(),
            provider: "openai".to_string(),
        };
        
        assert!(tracker.check_budget(&estimate).is_ok());
        tracker.track_completion(&estimate);
        
        let new_estimate = CostEstimate {
            total_cost: 0.3,
            ..estimate.clone()
        };
        assert!(tracker.check_budget(&new_estimate).is_err());
    }

    #[test]
    fn test_daily_usage_tracking() {
        let tracker = CostTracker::default();
        
        let estimate = CostEstimate {
            input_cost: 0.1,
            output_cost: 0.2,
            total_cost: 0.3,
            input_tokens: 100,
            output_tokens: 50,
            model: "gpt-4".to_string(),
            provider: "openai".to_string(),
        };
        
        tracker.track_completion(&estimate);
        
        let stats = tracker.get_usage_stats();
        let today = Utc::now().format("%Y-%m-%d").to_string();
        assert!(stats.usage_by_date.contains_key(&today));
    }
}