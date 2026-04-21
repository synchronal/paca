#![cfg_attr(feature = "strict", deny(warnings))]

use clap::{Args, Parser, Subcommand};
use std::path::PathBuf;

/// Command-line interface for the llama.cpp model downloader
#[derive(Parser, Debug)]
#[command(name = "paca")]
#[command(author, version, about = "Helpers for interacting with llama.cpp", long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Debug, PartialEq, Subcommand)]
pub enum Commands {
    /// Remove stale files from the cache
    Clean(CommonArgs),
    /// Download a model from HuggingFace
    #[command(aliases = ["dl"])]
    Download(ModelArgs),
    /// List all downloaded models
    #[command(aliases = ["ls"])]
    List(CommonArgs),
    /// Check which downloaded models have outdated files
    #[command(aliases = ["o"])]
    Outdated(CommonArgs),
    /// Remove a downloaded model or tag
    #[command(aliases = ["rm"])]
    Remove(RemoveArgs),
    /// Print version information
    Version,
}

#[derive(Args, Debug, PartialEq)]
pub struct CommonArgs {
    /// Override the HuggingFace Hub cache directory
    #[arg(long)]
    pub hub_dir: Option<PathBuf>,
}

#[derive(Args, Debug, PartialEq)]
pub struct ModelArgs {
    /// Override the HuggingFace Hub cache directory
    #[arg(long)]
    pub hub_dir: Option<PathBuf>,

    /// Model identifier (e.g., unsloth/GLM-4.7-Flash-GGUF:Q2_K_XL)
    pub model: String,
}

#[derive(Args, Debug, PartialEq)]
pub struct RemoveArgs {
    /// Override the HuggingFace Hub cache directory
    #[arg(long)]
    pub hub_dir: Option<PathBuf>,

    /// Model repository or tag (e.g., unsloth/GLM-4.7-Flash-GGUF or unsloth/GLM-4.7-Flash-GGUF:Q2_K_XL)
    pub target: String,
}
