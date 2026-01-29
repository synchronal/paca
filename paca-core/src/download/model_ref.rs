use crate::error::ModelRefError;

#[derive(Debug)]
pub struct ModelRef {
    pub model: String,
    pub owner: String,
    pub tag: String,
}

impl ModelRef {
    pub fn parse(s: &str) -> Result<Self, ModelRefError> {
        let (repo, tag) = s.split_once(':').ok_or(ModelRefError::MissingTag)?;
        let (owner, model) = repo.split_once('/').ok_or(ModelRefError::MissingOwner)?;

        Ok(Self {
            model: model.to_string(),
            owner: owner.to_string(),
            tag: tag.to_string(),
        })
    }

    pub fn repo(&self) -> String {
        format!("{}/{}", self.owner, self.model)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_valid_model_ref() {
        let model_ref = ModelRef::parse("unsloth/GLM-4.7-Flash-GGUF:Q2_K_XL").unwrap();
        assert_eq!(model_ref.owner, "unsloth");
        assert_eq!(model_ref.model, "GLM-4.7-Flash-GGUF");
        assert_eq!(model_ref.tag, "Q2_K_XL");
    }

    #[test]
    fn returns_repo() {
        let model_ref = ModelRef::parse("unsloth/GLM-4.7-Flash-GGUF:Q2_K_XL").unwrap();
        assert_eq!(model_ref.repo(), "unsloth/GLM-4.7-Flash-GGUF");
    }

    #[test]
    fn errors_when_tag_missing() {
        let result = ModelRef::parse("unsloth/GLM-4.7-Flash-GGUF");
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ModelRefError::MissingTag));
    }

    #[test]
    fn errors_when_owner_missing() {
        let result = ModelRef::parse("GLM-4.7-Flash-GGUF:Q2_K_XL");
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ModelRefError::MissingOwner));
    }
}
