#![cfg_attr(feature = "strict", deny(warnings))]

use clap::{Parser, Subcommand};

#[derive(Parser, Debug)]
#[command(name = "paca")]
#[command(author, version, about = "Helpers for interacting with llama.cpp", long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Print version information
    Version,
}
