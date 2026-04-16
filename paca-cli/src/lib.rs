#![cfg_attr(feature = "strict", deny(warnings))]

pub mod cli;
use cli::Cli;

/// Executes the command-line interface logic
pub async fn run(cli: Cli) -> anyhow::Result<()> {
    match cli.command {
        cli::Commands::Clean(args) => {
            let result = paca_core::cache::clean::clean_cache(args.hub_dir)?;
            if result.removed_files.is_empty() {
                println!("Cache is clean.");
            } else {
                for file in &result.removed_files {
                    println!("{}", file.path.display());
                }
            }
        }
        cli::Commands::Download(args) => {
            let paths = paca_core::download::download_model(&args.model, args.hub_dir).await?;
            for path in &paths {
                println!("{}", path.display());
            }
        }
        cli::Commands::Version => {
            println!("paca {}", env!("CARGO_PKG_VERSION"));
        }
        cli::Commands::List(args) => {
            let models = paca_core::cache::list_models(args.hub_dir)?;
            if models.is_empty() {
                println!("No downloaded models found.");
            } else {
                for model in &models {
                    println!("{}", model.model_ref);
                }
            }
        }
        cli::Commands::Outdated(args) => {
            let outdated = paca_core::cache::check_outdated_models(args.hub_dir).await?;
            if outdated.is_empty() {
                println!("All downloaded models are up to date.");
            } else {
                for model in &outdated {
                    println!("{}  {}", model.model_ref, model.filename);
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::ModelArgs;
    use clap::Parser;
    use std::path::PathBuf;

    #[test]
    fn cli_parses_clean_subcommand() {
        let result = Cli::try_parse_from(["paca", "clean"]);
        assert!(result.is_ok());
        let cli = result.unwrap();
        assert_eq!(
            cli.command,
            cli::Commands::Clean(cli::CommonArgs { hub_dir: None })
        );
    }

    #[test]
    fn cli_parses_clean_with_hub_dir() {
        let result = Cli::try_parse_from(["paca", "clean", "--hub-dir", "/tmp/models"]);
        assert!(result.is_ok());
        let cli = result.unwrap();
        assert_eq!(
            cli.command,
            cli::Commands::Clean(cli::CommonArgs {
                hub_dir: Some(PathBuf::from("/tmp/models")),
            })
        );
    }

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
                hub_dir: None,
                model: String::from("owner/model:tag"),
            })
        );
    }

    #[test]
    fn cli_parses_download_with_hub_dir() {
        let result = Cli::try_parse_from([
            "paca",
            "download",
            "--hub-dir",
            "/tmp/models",
            "owner/model:tag",
        ]);
        assert!(result.is_ok());
        let cli = result.unwrap();
        assert_eq!(
            cli.command,
            cli::Commands::Download(ModelArgs {
                hub_dir: Some(PathBuf::from("/tmp/models")),
                model: String::from("owner/model:tag"),
            })
        );
    }

    #[test]
    fn cli_parses_download_requires_model_argument() {
        let result = Cli::try_parse_from(["paca", "download"]);
        assert!(result.is_err());
    }

    #[test]
    fn cli_parses_outdated_subcommand() {
        let result = Cli::try_parse_from(["paca", "outdated"]);
        assert!(result.is_ok());
        let cli = result.unwrap();
        assert!(matches!(cli.command, cli::Commands::Outdated(_)));
    }

    #[test]
    fn cli_parses_outdated_with_hub_dir() {
        let result = Cli::try_parse_from(["paca", "outdated", "--hub-dir", "/tmp/models"]);
        assert!(result.is_ok());
        let cli = result.unwrap();
        assert_eq!(
            cli.command,
            cli::Commands::Outdated(cli::CommonArgs {
                hub_dir: Some(PathBuf::from("/tmp/models")),
            })
        );
    }
}
