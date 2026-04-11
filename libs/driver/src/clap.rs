use clap::{Arg, ArgAction, Command};

use crate::completion::CompletionCandidate;

#[derive(Debug, Clone)]
pub struct CompletionContextOwned {
    pub command_path: Vec<String>,
    pub arg_id: Option<String>,
    pub prefix: String,
    pub static_candidates: Vec<CompletionCandidate>,
}

pub fn root(name: &'static str) -> Command {
    Command::new(name)
}

pub fn analyze_completion(root: &Command, line: &str, cursor: usize) -> CompletionContextOwned {
    let partial = line.get(..cursor).unwrap_or(line);
    let trailing_whitespace = partial
        .chars()
        .last()
        .map(char::is_whitespace)
        .unwrap_or(true);

    let mut tokens = partial
        .split_whitespace()
        .map(ToOwned::to_owned)
        .collect::<Vec<_>>();

    let prefix = if trailing_whitespace {
        String::new()
    } else {
        tokens.pop().unwrap_or_default()
    };

    if tokens.first().map(|t| t == root.get_name()).unwrap_or(false) {
        let _ = tokens.remove(0);
    }

    let mut command_path = Vec::new();
    let mut current = root;
    let mut idx = 0usize;

    while idx < tokens.len() {
        if let Some(subcommand) = current
            .get_subcommands()
            .find(|candidate| candidate.get_name() == tokens[idx])
        {
            command_path.push(tokens[idx].clone());
            current = subcommand;
            idx += 1;
        } else {
            break;
        }
    }

    let remainder = &tokens[idx..];
    let expecting_value_for = option_value_target(current, remainder);
    let expecting_option_value = expecting_value_for.is_some();
    let positional_index = positional_index(current, remainder, expecting_option_value);

    let arg_id = if prefix.starts_with('-') {
        None
    } else if let Some(option_id) = expecting_value_for {
        Some(option_id)
    } else {
        positional_args(current)
            .get(positional_index)
            .map(|arg| arg.get_id().to_string())
    };

    let static_candidates = static_candidates(current, &prefix, expecting_option_value);

    CompletionContextOwned {
        command_path,
        arg_id,
        prefix,
        static_candidates,
    }
}

pub fn render_help(root: &Command, command_path: &[String]) -> anyhow::Result<String> {
    let mut current = root.clone();

    for segment in command_path {
        let next = current
            .get_subcommands()
            .find(|candidate| candidate.get_name() == segment)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("unknown help topic '{}'", segment))?;
        current = next;
    }

    let mut out = Vec::new();
    current.write_long_help(&mut out)?;
    Ok(String::from_utf8(out)?)
}

fn static_candidates(
    current: &Command,
    prefix: &str,
    expecting_option_value: bool,
) -> Vec<CompletionCandidate> {
    let mut out = Vec::new();

    if prefix.starts_with('-') {
        for arg in current.get_arguments() {
            if let Some(long) = arg.get_long() {
                let candidate = format!("--{long}");
                if candidate.starts_with(prefix) {
                    out.push(CompletionCandidate::new(candidate));
                }
            }
            if let Some(short) = arg.get_short() {
                let candidate = format!("-{short}");
                if candidate.starts_with(prefix) {
                    out.push(CompletionCandidate::new(candidate));
                }
            }
        }
        return out;
    }

    if !expecting_option_value {
        for subcommand in current.get_subcommands() {
            let name = subcommand.get_name();
            if name.starts_with(prefix) {
                out.push(CompletionCandidate::new(name));
            }
        }
    }

    out
}

fn option_value_target(current: &Command, tokens: &[String]) -> Option<String> {
    let mut expecting: Option<String> = None;

    for token in tokens {
        if expecting.take().is_some() {
            continue;
        }

        if let Some(long) = token.strip_prefix("--") {
            if let Some(arg) = current.get_arguments().find(|arg| arg.get_long() == Some(long)) {
                if arg_takes_value(arg) {
                    expecting = Some(arg.get_id().to_string());
                }
            }
            continue;
        }

        if token.starts_with('-') && token.len() == 2 {
            let short = token.chars().nth(1).unwrap_or_default();
            if let Some(arg) = current.get_arguments().find(|arg| arg.get_short() == Some(short)) {
                if arg_takes_value(arg) {
                    expecting = Some(arg.get_id().to_string());
                }
            }
            continue;
        }
    }

    expecting
}

fn positional_index(current: &Command, tokens: &[String], expecting_option_value: bool) -> usize {
    if expecting_option_value {
        return 0;
    }

    let mut count = 0usize;
    let mut skip_option_value = false;

    for token in tokens {
        if skip_option_value {
            skip_option_value = false;
            continue;
        }

        if let Some(long) = token.strip_prefix("--") {
            if let Some(arg) = current.get_arguments().find(|arg| arg.get_long() == Some(long)) {
                skip_option_value = arg_takes_value(arg);
                continue;
            }
        }

        if token.starts_with('-') && token.len() == 2 {
            let short = token.chars().nth(1).unwrap_or_default();
            if let Some(arg) = current.get_arguments().find(|arg| arg.get_short() == Some(short)) {
                skip_option_value = arg_takes_value(arg);
                continue;
            }
        }

        count += 1;
    }

    count
}

fn positional_args(current: &Command) -> Vec<&Arg> {
    current
        .get_arguments()
        .filter(|arg| arg.get_long().is_none() && arg.get_short().is_none())
        .collect()
}

fn arg_takes_value(arg: &Arg) -> bool {
    matches!(arg.get_action(), ArgAction::Set | ArgAction::Append)
}
