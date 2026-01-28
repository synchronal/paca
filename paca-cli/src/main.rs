use anyhow::Result;
use clap::Parser;
use paca_cli::cli::Cli;

fn main() -> Result<()> {
    let cli = Cli::parse();
    paca_cli::run(cli)
}
