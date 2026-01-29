use reqwest::blocking::Client;
use serde::Deserialize;

use crate::error::DownloadError;

use super::endpoint::get_model_endpoint;
use super::model_ref::ModelRef;

#[derive(Debug, Deserialize)]
struct ManifestResponse {
    #[serde(rename = "ggufFile")]
    gguf_file: Option<GgufFileInfo>,
}

#[derive(Debug, Deserialize)]
struct GgufFileInfo {
    rfilename: String,
    size: u64,
}

#[derive(Debug)]
pub struct Manifest {
    pub gguf_file: String,
    pub raw_json: String,
    pub size: u64,
}

pub fn fetch_manifest(model_ref: &ModelRef) -> Result<Manifest, DownloadError> {
    let endpoint = get_model_endpoint();
    let url = format!(
        "{}/v2/{}/manifests/{}",
        endpoint,
        model_ref.repo(),
        model_ref.tag
    );

    let client = Client::new();
    let response = client
        .get(&url)
        .header("User-Agent", "llama-cpp")
        .send()?
        .error_for_status()?;

    let raw_json = response.text()?;
    let manifest_response: ManifestResponse = serde_json::from_str(&raw_json)?;

    let gguf_file_info = manifest_response
        .gguf_file
        .ok_or(DownloadError::NoGgufFile)?;

    Ok(Manifest {
        gguf_file: gguf_file_info.rfilename,
        raw_json,
        size: gguf_file_info.size,
    })
}

pub fn manifest_filename(model_ref: &ModelRef) -> String {
    format!(
        "manifest={}={}={}.json",
        model_ref.owner, model_ref.model, model_ref.tag
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_struct_holds_required_fields() {
        let manifest = Manifest {
            gguf_file: "model.gguf".to_string(),
            raw_json: "{}".to_string(),
            size: 1024,
        };
        assert_eq!(manifest.gguf_file, "model.gguf");
        assert_eq!(manifest.raw_json, "{}");
        assert_eq!(manifest.size, 1024);
    }

    #[test]
    fn manifest_filename_formats_correctly() {
        let model_ref = ModelRef::parse("unsloth/GLM-4.7-Flash-GGUF:BF16").unwrap();
        let filename = manifest_filename(&model_ref);
        assert_eq!(filename, "manifest=unsloth=GLM-4.7-Flash-GGUF=BF16.json");
    }
}
