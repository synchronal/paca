use anyhow::Result;
use clap::Parser;
use paca_cli::cli::Cli;

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    paca_cli::run(cli).await
}
