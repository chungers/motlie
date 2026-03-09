use anyhow::{anyhow, Result};

/// Special keys recognized by tmux send-keys (without -l).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpecialKey {
    Enter,
    Tab,
    Escape,
    Up,
    Down,
    Left,
    Right,
    CtrlC,
    CtrlD,
    CtrlZ,
    CtrlL,
    Space,
    BSpace,
    /// Arbitrary tmux key name.
    Raw(String),
}

impl SpecialKey {
    /// The tmux key name string.
    pub fn tmux_name(&self) -> &str {
        match self {
            SpecialKey::Enter => "Enter",
            SpecialKey::Tab => "Tab",
            SpecialKey::Escape => "Escape",
            SpecialKey::Up => "Up",
            SpecialKey::Down => "Down",
            SpecialKey::Left => "Left",
            SpecialKey::Right => "Right",
            SpecialKey::CtrlC => "C-c",
            SpecialKey::CtrlD => "C-d",
            SpecialKey::CtrlZ => "C-z",
            SpecialKey::CtrlL => "C-l",
            SpecialKey::Space => "Space",
            SpecialKey::BSpace => "BSpace",
            SpecialKey::Raw(s) => s.as_str(),
        }
    }

    /// Parse a key name from inside `{...}` braces.
    fn parse(name: &str) -> Result<Self> {
        match name {
            "Enter" => Ok(SpecialKey::Enter),
            "Tab" => Ok(SpecialKey::Tab),
            "Escape" | "Esc" => Ok(SpecialKey::Escape),
            "Up" => Ok(SpecialKey::Up),
            "Down" => Ok(SpecialKey::Down),
            "Left" => Ok(SpecialKey::Left),
            "Right" => Ok(SpecialKey::Right),
            "C-c" => Ok(SpecialKey::CtrlC),
            "C-d" => Ok(SpecialKey::CtrlD),
            "C-z" => Ok(SpecialKey::CtrlZ),
            "C-l" => Ok(SpecialKey::CtrlL),
            "Space" => Ok(SpecialKey::Space),
            "BSpace" => Ok(SpecialKey::BSpace),
            other => {
                if other.is_empty() {
                    Err(anyhow!("empty key name in braces"))
                } else if !is_valid_tmux_key_name(other) {
                    Err(anyhow!(
                        "invalid tmux key name '{}': contains shell-dangerous character (spaces, semicolons, backticks, $, |, &, etc. are rejected)",
                        other
                    ))
                } else {
                    Ok(SpecialKey::Raw(other.to_string()))
                }
            }
        }
    }
}

/// Validate that a raw tmux key name contains only safe characters.
/// Tmux key names include alphanumeric chars, C-/M-/S- modifier prefixes,
/// and punctuation forms (e.g., "F1", "C-a", "M-S-Left", "C-\\", "IC").
/// We reject characters that are dangerous in shell contexts:
/// spaces, semicolons, backticks, $, |, &, >, <, newlines, null bytes.
/// Since `control::send_keys` also shell-escapes all key names, this
/// validation is defense-in-depth.
fn is_valid_tmux_key_name(name: &str) -> bool {
    const SHELL_DANGEROUS: &[char] = &[
        ' ', '\t', '\n', '\r', '\0', ';', '`', '$', '|', '&', '>', '<', '(', ')', '{',
        '}', '[', ']', '!', '#', '~', '"', '\'',
    ];
    !name.is_empty() && !name.contains(SHELL_DANGEROUS)
}

/// Internal segment: literal text or special key.
#[derive(Debug, Clone, PartialEq, Eq)]
enum KeySegment {
    /// Literal text sent via `send-keys -l`.
    Literal(String),
    /// Special key sent via `send-keys` (without -l).
    Special(SpecialKey),
}

/// A sequence of key inputs to send to a tmux pane.
///
/// Supports inline escapes: `"hello{Enter}"` → Literal("hello") + Special(Enter).
#[derive(Debug, Clone)]
pub struct KeySequence {
    segments: Vec<KeySegment>,
}

impl KeySequence {
    /// Parse from a human-friendly string with `{...}` inline escapes.
    ///
    /// Examples:
    /// - `"hello{Enter}"` → Literal("hello") + Special(Enter)
    /// - `"{C-c}"` → Special(CtrlC)
    /// - `"plain text"` → Literal("plain text")
    pub fn parse(input: &str) -> Result<Self> {
        let mut segments = Vec::new();
        let mut literal_buf = String::new();
        let mut chars = input.chars().peekable();

        while let Some(ch) = chars.next() {
            if ch == '{' {
                // Collect key name until '}'
                let mut key_name = String::new();
                let mut found_close = false;
                for ch2 in chars.by_ref() {
                    if ch2 == '}' {
                        found_close = true;
                        break;
                    }
                    key_name.push(ch2);
                }
                if !found_close {
                    return Err(anyhow!("unclosed '{{' in key sequence"));
                }
                // Flush literal buffer
                if !literal_buf.is_empty() {
                    segments.push(KeySegment::Literal(literal_buf.clone()));
                    literal_buf.clear();
                }
                segments.push(KeySegment::Special(SpecialKey::parse(&key_name)?));
            } else {
                literal_buf.push(ch);
            }
        }

        if !literal_buf.is_empty() {
            segments.push(KeySegment::Literal(literal_buf));
        }

        Ok(KeySequence { segments })
    }

    /// Create a sequence with a single literal segment.
    pub fn literal(text: &str) -> Self {
        KeySequence {
            segments: vec![KeySegment::Literal(text.to_string())],
        }
    }

    /// Append literal text.
    pub fn then_literal(mut self, text: &str) -> Self {
        self.segments
            .push(KeySegment::Literal(text.to_string()));
        self
    }

    /// Append a special key.
    pub fn then_key(mut self, key: SpecialKey) -> Self {
        self.segments.push(KeySegment::Special(key));
        self
    }

    /// Append Enter key.
    pub fn then_enter(self) -> Self {
        self.then_key(SpecialKey::Enter)
    }

    /// Render to tmux send-keys commands. Each command is a complete
    /// shell command string (without the tmux prefix — caller adds that).
    ///
    /// Adjacent literal segments are merged. Each transition between
    /// literal and special produces a separate command.
    pub fn to_tmux_args(&self, target: &str) -> Vec<Vec<String>> {
        let mut commands: Vec<Vec<String>> = Vec::new();

        for segment in &self.segments {
            match segment {
                KeySegment::Literal(text) => {
                    commands.push(vec![
                        "send-keys".to_string(),
                        "-l".to_string(),
                        "-t".to_string(),
                        target.to_string(),
                        text.clone(),
                    ]);
                }
                KeySegment::Special(key) => {
                    commands.push(vec![
                        "send-keys".to_string(),
                        "-t".to_string(),
                        target.to_string(),
                        key.tmux_name().to_string(),
                    ]);
                }
            }
        }

        commands
    }

    /// Check if the sequence is empty.
    pub fn is_empty(&self) -> bool {
        self.segments.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_literal_only() {
        let ks = KeySequence::parse("hello world").unwrap();
        assert_eq!(ks.segments.len(), 1);
        assert_eq!(ks.segments[0], KeySegment::Literal("hello world".to_string()));
    }

    #[test]
    fn parse_special_only() {
        let ks = KeySequence::parse("{Enter}").unwrap();
        assert_eq!(ks.segments.len(), 1);
        assert_eq!(ks.segments[0], KeySegment::Special(SpecialKey::Enter));
    }

    #[test]
    fn parse_mixed() {
        let ks = KeySequence::parse("hello{Enter}").unwrap();
        assert_eq!(ks.segments.len(), 2);
        assert_eq!(ks.segments[0], KeySegment::Literal("hello".to_string()));
        assert_eq!(ks.segments[1], KeySegment::Special(SpecialKey::Enter));
    }

    #[test]
    fn parse_multiple_specials() {
        let ks = KeySequence::parse("{C-c}{Enter}").unwrap();
        assert_eq!(ks.segments.len(), 2);
        assert_eq!(ks.segments[0], KeySegment::Special(SpecialKey::CtrlC));
        assert_eq!(ks.segments[1], KeySegment::Special(SpecialKey::Enter));
    }

    #[test]
    fn parse_complex_sequence() {
        let ks = KeySequence::parse("yes{Enter}no{C-c}").unwrap();
        assert_eq!(ks.segments.len(), 4);
        assert_eq!(ks.segments[0], KeySegment::Literal("yes".to_string()));
        assert_eq!(ks.segments[1], KeySegment::Special(SpecialKey::Enter));
        assert_eq!(ks.segments[2], KeySegment::Literal("no".to_string()));
        assert_eq!(ks.segments[3], KeySegment::Special(SpecialKey::CtrlC));
    }

    #[test]
    fn parse_empty_input() {
        let ks = KeySequence::parse("").unwrap();
        assert!(ks.is_empty());
    }

    #[test]
    fn parse_unclosed_brace() {
        assert!(KeySequence::parse("hello{Enter").is_err());
    }

    #[test]
    fn parse_empty_braces() {
        assert!(KeySequence::parse("{}").is_err());
    }

    #[test]
    fn parse_raw_key() {
        let ks = KeySequence::parse("{F1}").unwrap();
        assert_eq!(ks.segments[0], KeySegment::Special(SpecialKey::Raw("F1".to_string())));
    }

    #[test]
    fn parse_raw_key_rejects_shell_metacharacters() {
        assert!(KeySequence::parse("{; rm -rf /}").is_err());
        assert!(KeySequence::parse("{$(whoami)}").is_err());
        assert!(KeySequence::parse("{`id`}").is_err());
        assert!(KeySequence::parse("{key name}").is_err()); // space not allowed
    }

    #[test]
    fn parse_raw_key_allows_valid_names() {
        assert!(KeySequence::parse("{F12}").is_ok());
        assert!(KeySequence::parse("{KP0}").is_ok());
        assert!(KeySequence::parse("{M-a}").is_ok());
        assert!(KeySequence::parse("{C-M-Left}").is_ok());
    }

    #[test]
    fn builder_api() {
        let ks = KeySequence::literal("echo hello")
            .then_enter()
            .then_key(SpecialKey::CtrlC);
        assert_eq!(ks.segments.len(), 3);
    }

    #[test]
    fn to_tmux_args_literal() {
        let ks = KeySequence::literal("hello");
        let cmds = ks.to_tmux_args("build:0.1");
        assert_eq!(cmds.len(), 1);
        assert_eq!(cmds[0], vec!["send-keys", "-l", "-t", "build:0.1", "hello"]);
    }

    #[test]
    fn to_tmux_args_special() {
        let ks = KeySequence::parse("{Enter}").unwrap();
        let cmds = ks.to_tmux_args("build:0.1");
        assert_eq!(cmds.len(), 1);
        assert_eq!(cmds[0], vec!["send-keys", "-t", "build:0.1", "Enter"]);
    }

    #[test]
    fn to_tmux_args_mixed() {
        let ks = KeySequence::parse("ls -la{Enter}").unwrap();
        let cmds = ks.to_tmux_args("test:0.0");
        assert_eq!(cmds.len(), 2);
        assert_eq!(cmds[0], vec!["send-keys", "-l", "-t", "test:0.0", "ls -la"]);
        assert_eq!(cmds[1], vec!["send-keys", "-t", "test:0.0", "Enter"]);
    }

    #[test]
    fn all_special_keys() {
        let keys = vec![
            ("{Enter}", "Enter"),
            ("{Tab}", "Tab"),
            ("{Escape}", "Escape"),
            ("{Esc}", "Escape"),
            ("{Up}", "Up"),
            ("{Down}", "Down"),
            ("{Left}", "Left"),
            ("{Right}", "Right"),
            ("{C-c}", "C-c"),
            ("{C-d}", "C-d"),
            ("{C-z}", "C-z"),
            ("{C-l}", "C-l"),
            ("{Space}", "Space"),
            ("{BSpace}", "BSpace"),
        ];
        for (input, expected_name) in keys {
            let ks = KeySequence::parse(input).unwrap();
            let cmds = ks.to_tmux_args("t");
            assert_eq!(cmds[0][3], expected_name, "failed for input: {}", input);
        }
    }
}
