//! Pricing and capability descriptors, as data.
//!
//! The brief's rule (§41) is that a price must never be *guessed*: "Never
//! silently apply an unrelated fallback model price to an unknown model." A
//! wrong price is worse than a missing one, because the wrapper bills against it.
//!
//! So the table is:
//!
//! * **versioned** — `CATALOG_VERSION`, so a stale catalog is detectable rather
//!   than silently wrong;
//! * **exact-match only** — a model not in the table has no price, and there is
//!   no "nearest model" fallback, because `gpt-4o-mini` is not `gpt-4o` and
//!   billing one as the other is a financial bug, not a rounding error;
//! * **separate from protocol code** — a price change is a data change, not a
//!   code change.
//!
//! Live provider data always wins: this is consulted only when a provider does
//! not report pricing itself, which today is Anthropic (whose model list carries
//! no prices) and any provider whose listing was unavailable.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use crate::types::{CapabilitySource, ModelCapabilities, ModelPricing};

/// Version of the table below.
///
/// Bump when entries change. Anything that persists a price should record this
/// alongside it, so a historical cost can be re-derived against the catalog that
/// produced it.
pub const CATALOG_VERSION: &str = "2026-09-29";

/// One catalog entry: capability descriptors plus pricing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Descriptor {
    pub id: &'static str,
    pub context_window: u32,
    pub max_output_tokens: u32,
    pub capabilities: ModelCapabilities,
    /// `None` means "this model exists and we do not know what it costs".
    pub pricing: Option<ModelPricing>,
}

const fn price(prompt: f64, completion: f64) -> Option<ModelPricing> {
    Some(ModelPricing {
        prompt_tokens: prompt,
        completion_tokens: completion,
        image_tokens: None,
        is_free: false,
    })
}

fn free() -> Option<ModelPricing> {
    Some(ModelPricing {
        prompt_tokens: 0.0,
        completion_tokens: 0.0,
        image_tokens: None,
        is_free: true,
    })
}

/// A text+vision model with the given limits.
///
/// The limits are parameters rather than zeros on purpose: an entry whose
/// capability block disagrees with its own descriptor is an entry that has to be
/// patched at every read site, and the patch is what eventually gets forgotten.
/// `every_catalog_entry_is_internally_consistent` exists to keep that honest.
fn text_vision(tools: bool, ctx: u32, max_output: u32) -> ModelCapabilities {
    ModelCapabilities {
        function_calling: tools,
        vision: true,
        streaming: true,
        json_mode: true,
        caching: true,
        max_tokens: max_output,
        context_window: ctx,
        input_modalities: vec!["text".into(), "image".into()],
        output_modalities: vec!["text".into()],
        strengths: vec![],
        source: CapabilitySource::Descriptor,
    }
}

/// Build the maintained catalog.
///
/// Deliberately small and dated rather than exhaustive and stale: the point is
/// that an entry is a dated claim that can be checked, not a guess.
///
/// A `LazyLock` rather than a `static`, because the capability blocks own
/// `String`s and so cannot be built in a `static` initializer.
fn catalog_entries() -> Vec<Descriptor> {
    vec![
    Descriptor {
        id: "gpt-4o",
        context_window: 128_000,
        max_output_tokens: 16_384,
        capabilities: text_vision(true, 128_000, 16_384),
        pricing: price(0.005, 0.015),
    },
    Descriptor {
        id: "gpt-4o-mini",
        context_window: 128_000,
        max_output_tokens: 16_384,
        capabilities: ModelCapabilities {
            caching: false,
            ..text_vision(true, 128_000, 16_384)
        },
        pricing: price(0.00015, 0.0006),
    },
    Descriptor {
        id: "gpt-4",
        context_window: 8_192,
        max_output_tokens: 8_192,
        capabilities: ModelCapabilities {
            function_calling: true,
            vision: false,
            streaming: true,
            json_mode: true,
            caching: false,
            max_tokens: 8_192,
            context_window: 8_192,
            input_modalities: vec!["text".into()],
            output_modalities: vec!["text".into()],
            strengths: vec![],
            source: CapabilitySource::Descriptor,
        },
        pricing: price(0.03, 0.06),
    },
    Descriptor {
        id: "gpt-4-turbo",
        context_window: 128_000,
        max_output_tokens: 4_096,
        capabilities: ModelCapabilities {
            function_calling: true,
            vision: true,
            streaming: true,
            json_mode: true,
            caching: false,
            max_tokens: 4_096,
            context_window: 128_000,
            input_modalities: vec!["text".into(), "image".into()],
            output_modalities: vec!["text".into()],
            strengths: vec![],
            source: CapabilitySource::Descriptor,
        },
        pricing: price(0.01, 0.03),
    },
    Descriptor {
        id: "gpt-3.5-turbo",
        context_window: 4_096,
        max_output_tokens: 4_096,
        capabilities: ModelCapabilities {
            function_calling: true,
            vision: false,
            streaming: true,
            json_mode: true,
            caching: false,
            max_tokens: 4_096,
            context_window: 4_096,
            input_modalities: vec!["text".into()],
            output_modalities: vec!["text".into()],
            strengths: vec![],
            source: CapabilitySource::Descriptor,
        },
        pricing: price(0.0005, 0.0015),
    },
    Descriptor {
        id: "claude-3-5-sonnet-20241022",
        context_window: 200_000,
        max_output_tokens: 8_192,
        capabilities: text_vision(true, 200_000, 8_192),
        pricing: price(0.003, 0.015),
    },
    Descriptor {
        id: "claude-3-opus-20240229",
        context_window: 200_000,
        max_output_tokens: 4_096,
        capabilities: text_vision(true, 200_000, 4_096),
        pricing: price(0.015, 0.075),
    },
    Descriptor {
        id: "claude-3-haiku-20240307",
        context_window: 200_000,
        max_output_tokens: 4_096,
        capabilities: ModelCapabilities {
            caching: false,
            ..text_vision(true, 200_000, 4_096)
        },
        pricing: price(0.0008, 0.004),
    },
    Descriptor {
        id: "claude-3-haiku-20240307-latest",
        context_window: 200_000,
        max_output_tokens: 4_096,
        capabilities: ModelCapabilities {
            caching: false,
            ..text_vision(true, 200_000, 4_096)
        },
        pricing: price(0.0008, 0.004),
    },
    ]
}

/// The maintained catalog.
static CATALOG: std::sync::LazyLock<Vec<Descriptor>> = std::sync::LazyLock::new(catalog_entries);

/// Look up a descriptor by exact model id.
///
/// No normalisation, no prefix matching, no fallback. `gpt-4` and
/// `gpt-4-turbo` are different models with different prices, and a lookup that
/// treated one as the other would bill a customer incorrectly.
pub fn descriptor_for(model: &str) -> Option<&'static Descriptor> {
    CATALOG.iter().find(|d| d.id == model)
}

/// The catalog as an owned map, for callers that want to merge with live data.
pub fn catalog() -> BTreeMap<&'static str, &'static Descriptor> {
    CATALOG.iter().map(|d| (d.id, d)).collect()
}

/// The price for a model, or `None` when the catalog does not know it.
///
/// `None` is the correct answer for an unknown model. It is not an invitation to
/// substitute a similar-looking one.
pub fn price_for(model: &str) -> Option<ModelPricing> {
    descriptor_for(model).and_then(|d| d.pricing.clone())
}

/// Context window for a model, or `None` when unknown.
pub fn context_window_for(model: &str) -> Option<u32> {
    descriptor_for(model).map(|d| d.context_window)
}

/// Whether the catalog claims support for a named capability.
///
/// Deliberately conservative: a model absent from the catalog supports nothing we
/// can vouch for, so this returns `false` rather than guessing. Callers that need
/// to distinguish "no" from "unknown" should check
/// [`descriptor_for`] first.
pub fn supports(model: &str, capability: &str) -> bool {
    let Some(descriptor) = descriptor_for(model) else {
        return false;
    };
    match capability {
        "tools" | "tool_choice" | "function_calling" => descriptor.capabilities.function_calling,
        "vision" => descriptor.capabilities.vision,
        "streaming" => descriptor.capabilities.streaming,
        "response_format" | "structured_outputs" | "json_mode" => {
            descriptor.capabilities.json_mode
        }
        "caching" => descriptor.capabilities.caching,
        _ => false,
    }
}

/// Prices for locally-run models, which cost tokens nothing.
pub fn local_model_price(_model: &str) -> Option<ModelPricing> {
    free()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_known_model_has_a_price() {
        let price = price_for("gpt-4o").expect("a catalogued model has a price");
        assert_eq!(price.prompt_tokens, 0.005);
        assert_eq!(price.completion_tokens, 0.015);
    }

    #[test]
    fn an_unknown_model_has_no_price() {
        assert_eq!(price_for("gpt-9-ultra"), None);
        assert_eq!(price_for(""), None);
    }

    /// The rule the brief states most forcefully: never apply an unrelated
    /// model's price to an unknown one.
    #[test]
    fn a_near_miss_is_not_a_match() {
        // Every one of these is a plausible typo or variant of a catalogued
        // model. None of them may inherit that model's price.
        for near_miss in [
            "gpt-4o-2024-08-06",
            "gpt-4o-mini-2024-07-18",
            "gpt-4-32k",
            "gpt-4-turbo-preview",
            "gpt-3.5-turbo-16k",
            "claude-3-5-sonnet-latest",
            "claude-3-opus-20240229-v1",
            "CLAUDE-3-OPUS-20240229",
        ] {
            assert_eq!(
                price_for(near_miss),
                None,
                "{near_miss} must not inherit a similar model's price"
            );
        }
    }

    #[test]
    fn sibling_models_keep_their_own_prices() {
        // The case that makes a prefix match dangerous: gpt-4 and gpt-4-turbo
        // share a prefix and differ by 30x on completion tokens.
        let gpt4 = price_for("gpt-4").expect("catalogued");
        let turbo = price_for("gpt-4-turbo").expect("catalogued");
        assert_ne!(gpt4.prompt_tokens, turbo.prompt_tokens);
        assert_ne!(gpt4.completion_tokens, turbo.completion_tokens);
    }

    #[test]
    fn context_windows_come_from_the_catalog_not_a_guess() {
        assert_eq!(context_window_for("gpt-4"), Some(8_192));
        assert_eq!(context_window_for("gpt-4-turbo"), Some(128_000));
        assert_eq!(context_window_for("mystery-model"), None);
    }

    #[test]
    fn an_unknown_model_supports_nothing_we_can_vouch_for() {
        assert!(!supports("mystery-model", "tools"));
        assert!(!supports("mystery-model", "vision"));
        assert!(supports("gpt-4o", "tools"));
        assert!(supports("gpt-4o", "vision"));
        assert!(!supports("gpt-4", "vision"));
    }

    #[test]
    fn an_unrecognised_capability_name_is_not_assumed_present() {
        assert!(!supports("gpt-4o", "telepathy"));
    }

    #[test]
    fn every_catalog_entry_is_internally_consistent() {
        for descriptor in CATALOG.iter() {
            let p = descriptor.pricing.as_ref();
            assert!(
                p.map(|p| p.prompt_tokens >= 0.0 && p.completion_tokens >= 0.0)
                    .unwrap_or(true),
                "{} has a negative price",
                descriptor.id
            );
            assert!(
                descriptor.capabilities.source.is_authoritative(),
                "{} is catalog data, so it must not be marked as a guess",
                descriptor.id
            );
            assert_eq!(
                descriptor.capabilities.context_window, descriptor.context_window,
                "{}: the capability block and the descriptor disagree on context window",
                descriptor.id
            );
        }
    }

    #[test]
    fn the_catalog_is_versioned() {
        // A catalog with no version cannot be detected as stale.
        assert!(!CATALOG_VERSION.is_empty());
    }

    #[test]
    fn local_models_are_free() {
        assert!(local_model_price("llama3.1:8b").expect("free").is_free);
    }
}
