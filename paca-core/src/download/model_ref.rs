use std::fmt;
use std::str::FromStr;

use crate::error::ModelRefError;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelRef {
    pub model: String,
    pub owner: String,
    pub tag: String,
}

impl FromStr for ModelRef {
    type Err = ModelRefError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let (repo, tag) = s.split_once(':').ok_or(ModelRefError::MissingTag)?;
        let (owner, model) = repo.split_once('/').ok_or(ModelRefError::MissingOwner)?;

        Ok(Self {
            model: model.to_string(),
            owner: owner.to_string(),
            tag: tag.to_string(),
        })
    }
}

impl fmt::Display for ModelRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}/{}:{}", self.owner, self.model, self.tag)
    }
}

impl ModelRef {
    pub fn repo(&self) -> String {
        format!("{}/{}", self.owner, self.model)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_valid_model_ref() {
        let model_ref: ModelRef = "unsloth/GLM-4.7-Flash-GGUF:Q2_K_XL".parse().unwrap();
        assert_eq!(model_ref.owner, "unsloth");
        assert_eq!(model_ref.model, "GLM-4.7-Flash-GGUF");
        assert_eq!(model_ref.tag, "Q2_K_XL");
    }

    #[test]
    fn returns_repo() {
        let model_ref: ModelRef = "unsloth/GLM-4.7-Flash-GGUF:Q2_K_XL".parse().unwrap();
        assert_eq!(model_ref.repo(), "unsloth/GLM-4.7-Flash-GGUF");
    }

    #[test]
    fn displays_full_reference() {
        let model_ref: ModelRef = "unsloth/GLM-4.7-Flash-GGUF:Q2_K_XL".parse().unwrap();
        assert_eq!(model_ref.to_string(), "unsloth/GLM-4.7-Flash-GGUF:Q2_K_XL");
    }

    #[test]
    fn errors_when_tag_missing() {
        let result = "unsloth/GLM-4.7-Flash-GGUF".parse::<ModelRef>();
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ModelRefError::MissingTag));
    }

    #[test]
    fn errors_when_owner_missing() {
        let result = "GLM-4.7-Flash-GGUF:Q2_K_XL".parse::<ModelRef>();
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ModelRefError::MissingOwner));
    }
}
