#[derive(Debug, Clone, Copy)]
pub struct CompletionRequest<'a> {
    pub command_path: &'a [&'a str],
    pub arg_id: Option<&'a str>,
    pub prefix: &'a str,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct CompletionCandidate {
    pub value: String,
    pub help: Option<String>,
}

impl CompletionCandidate {
    pub fn new(value: impl Into<String>) -> Self {
        Self {
            value: value.into(),
            help: None,
        }
    }
}

pub fn dedup_sorted(mut values: Vec<CompletionCandidate>) -> Vec<CompletionCandidate> {
    values.sort();
    values.dedup();
    values
}
