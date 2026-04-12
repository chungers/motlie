use clap::{Arg, ArgAction, Command};

use crate::completion::CompletionCandidate;
use crate::error::{DriverError, DriverResult};

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

    if tokens
        .first()
        .map(|t| t == root.get_name())
        .unwrap_or(false)
    {
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

pub fn render_help(root: &Command, command_path: &[String]) -> DriverResult<String> {
    let mut current = root.clone();

    for segment in command_path {
        let next = current
            .get_subcommands()
            .find(|candidate| candidate.get_name() == segment)
            .cloned()
            .ok_or_else(|| DriverError::UnknownHelpTopic(segment.clone()))?;
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
    let mut parsing_options = true;

    for token in tokens {
        if expecting.take().is_some() {
            continue;
        }

        if !parsing_options {
            continue;
        }

        match analyze_option_token(current, token) {
            OptionToken::Terminator => parsing_options = false,
            OptionToken::ConsumesNextValue(arg_id) => expecting = Some(arg_id),
            OptionToken::ConsumesInlineValue | OptionToken::NoValue => {}
            OptionToken::NotOption => {}
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
    let mut parsing_options = true;

    for token in tokens {
        if skip_option_value {
            skip_option_value = false;
            continue;
        }

        if !parsing_options {
            count += 1;
            continue;
        }

        match analyze_option_token(current, token) {
            OptionToken::Terminator => parsing_options = false,
            OptionToken::ConsumesNextValue(_) => {
                skip_option_value = true;
            }
            OptionToken::ConsumesInlineValue | OptionToken::NoValue => {}
            OptionToken::NotOption => count += 1,
        }
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

enum OptionToken {
    NotOption,
    Terminator,
    NoValue,
    ConsumesNextValue(String),
    ConsumesInlineValue,
}

fn analyze_option_token(current: &Command, token: &str) -> OptionToken {
    if token == "--" {
        return OptionToken::Terminator;
    }

    if let Some(long) = token.strip_prefix("--") {
        let (name, inline_value) = long
            .split_once('=')
            .map_or((long, None), |(name, value)| (name, Some(value)));
        if let Some(arg) = current
            .get_arguments()
            .find(|arg| arg.get_long() == Some(name))
        {
            if arg_takes_value(arg) {
                if inline_value.is_some() {
                    return OptionToken::ConsumesInlineValue;
                }
                return OptionToken::ConsumesNextValue(arg.get_id().to_string());
            }
            return OptionToken::NoValue;
        }
        return OptionToken::NotOption;
    }

    if token.starts_with('-') && !token.starts_with("--") && token.len() > 1 {
        for (offset, short) in token[1..].char_indices() {
            let Some(arg) = current
                .get_arguments()
                .find(|arg| arg.get_short() == Some(short))
            else {
                return OptionToken::NotOption;
            };

            if arg_takes_value(arg) {
                let inline_start = 1 + offset + short.len_utf8();
                if inline_start < token.len() {
                    return OptionToken::ConsumesInlineValue;
                }
                return OptionToken::ConsumesNextValue(arg.get_id().to_string());
            }
        }

        return OptionToken::NoValue;
    }

    OptionToken::NotOption
}

#[cfg(test)]
mod tests {
    use clap::{Arg, ArgAction, Command};

    use super::{analyze_completion, render_help};

    fn root() -> Command {
        Command::new("demo")
            .subcommand(
                Command::new("stream")
                    .arg(Arg::new("target").required(true))
                    .arg(Arg::new("mode").long("mode").action(ArgAction::Set))
                    .arg(Arg::new("pattern").long("pattern").action(ArgAction::Set)),
            )
            .subcommand(Command::new("capture").arg(Arg::new("lines").required(true)))
    }

    #[test]
    fn analyze_completion_tracks_command_path_and_positional_arg() {
        let completion = analyze_completion(&root(), "stream dem", "stream dem".len());
        assert_eq!(completion.command_path, vec!["stream".to_string()]);
        assert_eq!(completion.arg_id.as_deref(), Some("target"));
        assert_eq!(completion.prefix, "dem");
    }

    #[test]
    fn analyze_completion_suggests_option_flags() {
        let completion = analyze_completion(&root(), "stream demo --mo", "stream demo --mo".len());
        let values = completion
            .static_candidates
            .into_iter()
            .map(|candidate| candidate.value)
            .collect::<Vec<_>>();

        assert!(values.contains(&"--mode".to_string()));
    }

    #[test]
    fn analyze_completion_marks_option_value_target() {
        let completion =
            analyze_completion(&root(), "stream demo --mode ", "stream demo --mode ".len());
        assert_eq!(completion.arg_id.as_deref(), Some("mode"));
        assert_eq!(completion.prefix, "");
    }

    #[test]
    fn render_help_walks_subcommand_tree() {
        let rendered = render_help(&root(), &[String::from("stream")]).expect("stream help");
        assert!(rendered.contains("stream"));
        assert!(rendered.contains("--mode"));
    }

    #[test]
    fn analyze_completion_handles_long_flag_with_inline_value() {
        let completion = analyze_completion(
            &root(),
            "stream demo --mode=tail next",
            "stream demo --mode=tail next".len(),
        );

        assert_eq!(completion.command_path, vec!["stream".to_string()]);
        assert_eq!(completion.arg_id, None);
        assert_eq!(completion.prefix, "next");
    }

    #[test]
    fn analyze_completion_handles_combined_short_flags() {
        let root = Command::new("demo").subcommand(
            Command::new("run")
                .arg(Arg::new("all").short('a').action(ArgAction::SetTrue))
                .arg(Arg::new("binary").short('b').action(ArgAction::SetTrue))
                .arg(Arg::new("target").required(true)),
        );

        let completion = analyze_completion(&root, "run -ab ta", "run -ab ta".len());
        assert_eq!(completion.command_path, vec!["run".to_string()]);
        assert_eq!(completion.arg_id.as_deref(), Some("target"));
        assert_eq!(completion.prefix, "ta");
    }
}
