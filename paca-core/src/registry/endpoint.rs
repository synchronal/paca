use std::env;
use std::sync::OnceLock;

const DEFAULT_ENDPOINT: &str = "https://huggingface.co";

/// Returns the HuggingFace model endpoint, resolved from `MODEL_ENDPOINT`
/// or `HF_ENDPOINT` with the default falling back to `huggingface.co`.
///
/// Cached after the first call; env changes within a single process are
/// not picked up.
pub fn model_endpoint() -> &'static str {
    static ENDPOINT: OnceLock<String> = OnceLock::new();
    ENDPOINT.get_or_init(resolve_endpoint)
}

fn resolve_endpoint() -> String {
    env::var("MODEL_ENDPOINT")
        .or_else(|_| env::var("HF_ENDPOINT"))
        .unwrap_or_else(|_| DEFAULT_ENDPOINT.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_endpoint_returns_default_when_no_env_vars_set() {
        temp_env::with_vars_unset(["HF_ENDPOINT", "MODEL_ENDPOINT"], || {
            assert_eq!(resolve_endpoint(), "https://huggingface.co");
        });
    }

    #[test]
    fn resolve_endpoint_returns_hf_endpoint_when_set() {
        temp_env::with_vars(
            [
                ("HF_ENDPOINT", Some("https://custom-hf.example.com")),
                ("MODEL_ENDPOINT", None),
            ],
            || {
                assert_eq!(resolve_endpoint(), "https://custom-hf.example.com");
            },
        );
    }

    #[test]
    fn resolve_endpoint_returns_model_endpoint_when_set() {
        temp_env::with_vars(
            [
                ("HF_ENDPOINT", None),
                ("MODEL_ENDPOINT", Some("https://custom-model.example.com")),
            ],
            || {
                assert_eq!(resolve_endpoint(), "https://custom-model.example.com");
            },
        );
    }

    #[test]
    fn resolve_endpoint_prefers_model_endpoint_over_hf_endpoint() {
        temp_env::with_vars(
            [
                ("HF_ENDPOINT", Some("https://hf.example.com")),
                ("MODEL_ENDPOINT", Some("https://model.example.com")),
            ],
            || {
                assert_eq!(resolve_endpoint(), "https://model.example.com");
            },
        );
    }
}
