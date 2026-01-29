#![cfg_attr(feature = "strict", deny(warnings))]

pub mod cli;

use cli::Cli;

pub fn run(cli: Cli) -> anyhow::Result<()> {
    match cli.command {
        cli::Commands::Download(args) => {
            let paths = paca_core::download::download_model(&args.model)?;
            for path in &paths {
                println!("{}", path.display());
            }
        }
        cli::Commands::Version => {
            println!("paca {}", env!("CARGO_PKG_VERSION"));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use clap::Parser;

    use crate::cli::ModelArgs;

    use super::*;

    #[test]
    fn cli_parses_help() {
        let result = Cli::try_parse_from(["paca", "--help"]);
        assert!(result.is_err()); // --help returns an error with status 0
        let err = result.unwrap_err();
        assert_eq!(err.kind(), clap::error::ErrorKind::DisplayHelp);
    }

    #[test]
    fn cli_parses_version_subcommand() {
        let result = Cli::try_parse_from(["paca", "version"]);
        assert!(result.is_ok());
        let cli = result.unwrap();
        assert!(matches!(cli.command, cli::Commands::Version));
    }

    #[test]
    fn cli_parses_download_subcommand() {
        let result = Cli::try_parse_from(["paca", "download", "owner/model:tag"]);
        assert!(result.is_ok());
        let cli = result.unwrap();
        assert_eq!(
            cli.command,
            cli::Commands::Download(ModelArgs {
                model: String::from("owner/model:tag")
            })
        );
    }

    #[test]
    fn cli_download_requires_model_argument() {
        let result = Cli::try_parse_from(["paca", "download"]);
        assert!(result.is_err());
    }
}
