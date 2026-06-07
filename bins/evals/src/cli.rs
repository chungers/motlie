use std::path::PathBuf;

use anyhow::{bail, Context, Result};

use crate::scenario;

pub fn run(args: impl IntoIterator<Item = String>) -> Result<()> {
    let args = args.into_iter().skip(1).collect::<Vec<_>>();

    match args.as_slice() {
        [] => {
            print_usage();
            Ok(())
        }
        [single] if single == "--help" || single == "-h" => {
            print_usage();
            Ok(())
        }
        [command, subject] if command == "list" && subject == "scenarios" => {
            list_scenarios(default_eval_root())
        }
        [command, subject, flag, root]
            if command == "list" && subject == "scenarios" && flag == "--root" =>
        {
            list_scenarios(PathBuf::from(root))
        }
        [command, subject] if command == "list" && subject == "bundles" => {
            bail!("bundle listing is planned for the next evals CLI slice")
        }
        [command, ..] if command == "run" || command == "matrix" || command == "report" => {
            bail!("`evals {command}` is planned after the embeddings exemplar is reviewed")
        }
        _ => {
            print_usage();
            bail!("unknown evals command")
        }
    }
}

fn list_scenarios(root: PathBuf) -> Result<()> {
    let scenarios = scenario::list_scenarios(&root)
        .with_context(|| format!("failed to list scenarios under `{}`", root.display()))?;
    for scenario in scenarios {
        println!("{}", scenario.id);
    }
    Ok(())
}

fn default_eval_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
        .join("evals")
}

fn print_usage() {
    println!("usage:");
    println!("  evals list scenarios [--root PATH]");
    println!("  evals list bundles");
    println!("  evals run --bundle <bundle_id> --scenario <scenario_id>");
    println!("  evals matrix");
    println!("  evals report --input <jsonl> --format markdown");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_eval_root_points_to_repo_evals_dir() {
        let root = default_eval_root();
        assert!(root.ends_with("evals"));
    }

    #[test]
    fn help_returns_ok() {
        run(["evals".to_owned(), "--help".to_owned()]).unwrap();
    }

    #[test]
    fn unknown_command_returns_error() {
        let error =
            run(["evals".to_owned(), "nope".to_owned()]).expect_err("unknown command should fail");
        assert!(error.to_string().contains("unknown evals command"));
    }
}
