use std::collections::{HashMap, VecDeque};
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum TokenCountingError {
    #[error("Model not supported: {model}")]
    UnsupportedModel { model: String },
    #[error("Encoding error: {message}")]
    EncodingError { message: String },
    #[error("Cache error: {message}")]
    CacheError { message: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenCount {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

impl TokenCount {
    pub fn new(prompt_tokens: u32, completion_tokens: u32) -> Self {
        Self {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TokenCounterConfig {
    pub default_model: String,
    pub cache_size: usize,
    pub enable_cache: bool,
    pub tokens_per_character: f64,
}

impl Default for TokenCounterConfig {
    fn default() -> Self {
        Self {
            default_model: "gpt-4".to_string(),
            cache_size: 1000,
            enable_cache: true,
            tokens_per_character: 0.25,
        }
    }
}

#[derive(Debug)]
pub struct TokenCounter {
    config: TokenCounterConfig,
    cache: HashMap<String, u32>,
    /// VecDeque gives O(1) pop_front, replacing the previous Vec which had
    /// O(n) remove(0) on every cache eviction.
    cache_order: VecDeque<String>,
}

impl TokenCounter {
    pub fn new(config: Option<TokenCounterConfig>) -> Self {
        Self {
            config: config.unwrap_or_default(),
            cache: HashMap::new(),
            cache_order: VecDeque::new(),
        }
    }

    pub fn count_tokens(&mut self, text: &str, _model: &str) -> Result<u32, TokenCountingError> {
        let cache_key = text.to_string();

        if self.config.enable_cache {
            if let Some(&cached) = self.cache.get(&cache_key) {
                return Ok(cached);
            }
        }

        let count = Self::approximate_token_count(text);

        if self.config.enable_cache {
            self.add_to_cache(cache_key, count);
        }

        Ok(count)
    }

    pub fn count_messages(&mut self, messages: &[crate::types::ChatMessage], model: &str) -> Result<u32, TokenCountingError> {
        let mut total = 0;

        for message in messages {
            let content = message.content().unwrap_or("");
            total += self.count_tokens(content, model)?;
            
            let role_str = match message.role() {
                crate::types::MessageRole::System => "system",
                crate::types::MessageRole::User => "user",
                crate::types::MessageRole::Assistant => "assistant",
                crate::types::MessageRole::Function => "function",
                crate::types::MessageRole::Tool => "tool",
            };

            let role_token_count = self.count_tokens(role_str, model)?;
            total += role_token_count;

            total += 4;
        }

        total += 3;

        Ok(total)
    }

    pub fn estimate_tokens_for_request(
        &mut self,
        prompt: &str,
        max_tokens_to_generate: u32,
        model: &str,
    ) -> Result<TokenCount, TokenCountingError> {
        let prompt_tokens = self.count_tokens(prompt, model)?;

        Ok(TokenCount::new(prompt_tokens, max_tokens_to_generate))
    }

    fn approximate_token_count(text: &str) -> u32 {
        let char_count = text.chars().count();
        let tokens = (char_count as f64 * 0.25).ceil() as u32;
        tokens.max(1)
    }

    fn add_to_cache(&mut self, key: String, value: u32) {
        if self.cache_order.len() >= self.config.cache_size {
            // O(1) pop from the front of the deque
            if let Some(oldest) = self.cache_order.pop_front() {
                self.cache.remove(&oldest);
            }
        }

        self.cache.insert(key.clone(), value);
        self.cache_order.push_back(key);
    }

    pub fn clear_cache(&mut self) {
        self.cache.clear();
        self.cache_order.clear();
    }

    pub fn cache_size(&self) -> usize {
        self.cache.len()
    }

    pub fn with_tiktoken(&mut self, _bpe_data: &[u8]) -> Result<(), TokenCountingError> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_counter_basic() {
        let mut counter = TokenCounter::new(None);
        let text = "Hello, world! This is a test.";
        let tokens = counter.count_tokens(text, "gpt-4");
        assert!(tokens.is_ok());
        assert!(tokens.unwrap() > 0);
    }

    #[test]
    fn test_token_counter_cache() {
        let mut counter = TokenCounter::new(None);
        let text = "Test text for caching";
        
        let first = counter.count_tokens(text, "gpt-4").unwrap();
        let second = counter.count_tokens(text, "gpt-4").unwrap();
        
        assert_eq!(first, second);
        assert_eq!(counter.cache_size(), 1);
    }

    #[test]
    fn test_estimate_tokens() {
        let mut counter = TokenCounter::new(None);
        let result = counter.estimate_tokens_for_request("Hello", 100, "gpt-4");
        assert!(result.is_ok());
        let tokens = result.unwrap();
        assert!(tokens.prompt_tokens > 0);
        assert_eq!(tokens.completion_tokens, 100);
    }

    #[test]
    fn test_count_messages() {
        let mut counter = TokenCounter::new(None);
        let messages = vec![
            crate::types::ChatMessage::system("You are a helpful assistant."),
            crate::types::ChatMessage::user("Hello!"),
        ];
        let result = counter.count_messages(&messages, "gpt-4");
        assert!(result.is_ok());
        assert!(result.unwrap() > 0);
    }
}
