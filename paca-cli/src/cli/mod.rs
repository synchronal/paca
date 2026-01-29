#![cfg_attr(feature = "strict", deny(warnings))]

use clap::{Args, Parser, Subcommand};

#[derive(Parser, Debug)]
#[command(name = "paca")]
#[command(author, version, about = "Helpers for interacting with llama.cpp", long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Debug, PartialEq, Subcommand)]
pub enum Commands {
    /// Download a model from HuggingFace
    Download(ModelArgs),
    /// Print version information
    Version,
}

#[derive(Args, Debug, PartialEq)]
pub struct ModelArgs {
    /// Model identifier (e.g., unsloth/GLM-4.7-Flash-GGUF:Q2_K_XL)
    pub model: String,
}
