use std::env;

/// Default HuggingFace endpoint URL
const DEFAULT_ENDPOINT: &str = "https://huggingface.co";

/// Gets the model endpoint from environment variables
/// Prefers MODEL_ENDPOINT over HF_ENDPOINT, falls back to default
pub fn get_model_endpoint() -> String {
    env::var("MODEL_ENDPOINT")
        .or_else(|_| env::var("HF_ENDPOINT"))
        .unwrap_or_else(|_| DEFAULT_ENDPOINT.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn returns_default_endpoint_when_no_env_vars_set() {
        temp_env::with_vars_unset(["HF_ENDPOINT", "MODEL_ENDPOINT"], || {
            let endpoint = get_model_endpoint();
            assert_eq!(endpoint, "https://huggingface.co");
        });
    }

    #[test]
    fn returns_hf_endpoint_when_set() {
        temp_env::with_vars(
            [
                ("HF_ENDPOINT", Some("https://custom-hf.example.com")),
                ("MODEL_ENDPOINT", None),
            ],
            || {
                let endpoint = get_model_endpoint();
                assert_eq!(endpoint, "https://custom-hf.example.com");
            },
        );
    }

    #[test]
    fn returns_model_endpoint_when_set() {
        temp_env::with_vars(
            [
                ("HF_ENDPOINT", None),
                ("MODEL_ENDPOINT", Some("https://custom-model.example.com")),
            ],
            || {
                let endpoint = get_model_endpoint();
                assert_eq!(endpoint, "https://custom-model.example.com");
            },
        );
    }

    #[test]
    fn model_endpoint_takes_precedence_over_hf_endpoint() {
        temp_env::with_vars(
            [
                ("HF_ENDPOINT", Some("https://hf.example.com")),
                ("MODEL_ENDPOINT", Some("https://model.example.com")),
            ],
            || {
                let endpoint = get_model_endpoint();
                assert_eq!(endpoint, "https://model.example.com");
            },
        );
    }
}
