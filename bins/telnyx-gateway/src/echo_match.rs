use std::collections::HashSet;

use crate::operator::state::speech_echo_signature;
use crate::quality::EchoSuppressionQualityConfig;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct AssistantEchoMatch {
    pub(crate) token_coverage_percent: u64,
    pub(crate) longest_token_run: usize,
}

pub(crate) fn match_assistant_echo_signature(
    config: &EchoSuppressionQualityConfig,
    transcript_text: &str,
    assistant_echo_signature: &str,
) -> Option<AssistantEchoMatch> {
    if !config.enabled || assistant_echo_signature.is_empty() {
        return None;
    }

    let candidate = speech_echo_signature(transcript_text);
    if candidate.chars().count() < config.min_text_chars {
        return None;
    }
    if assistant_echo_signature.contains(&candidate) {
        return Some(AssistantEchoMatch {
            token_coverage_percent: 100,
            longest_token_run: candidate.split_whitespace().count(),
        });
    }

    let candidate_tokens = candidate.split_whitespace().collect::<Vec<_>>();
    let assistant_tokens = assistant_echo_signature
        .split_whitespace()
        .collect::<Vec<_>>();
    if candidate_tokens.len() < 2 || assistant_tokens.len() < 2 {
        return None;
    }

    let assistant_token_set = assistant_tokens.iter().copied().collect::<HashSet<_>>();
    let matching_tokens = candidate_tokens
        .iter()
        .filter(|token| assistant_token_set.contains(**token))
        .count();
    let token_coverage_percent = ((matching_tokens * 100) / candidate_tokens.len().max(1)) as u64;
    let longest_token_run = longest_common_token_run(&candidate_tokens, &assistant_tokens);
    let is_short_echo = candidate_tokens.len() < config.long_min_tokens
        && matching_tokens >= 2
        && token_coverage_percent >= config.short_token_coverage_percent
        && longest_token_run >= config.short_longest_token_run;
    let is_long_echo = candidate_tokens.len() >= config.long_min_tokens
        && matching_tokens >= config.long_longest_token_run
        && longest_token_run >= config.long_longest_token_run
        && (token_coverage_percent >= config.long_token_coverage_percent
            || has_strong_contiguous_echo_run(config, longest_token_run));
    (is_short_echo || is_long_echo).then_some(AssistantEchoMatch {
        token_coverage_percent,
        longest_token_run,
    })
}

fn has_strong_contiguous_echo_run(
    config: &EchoSuppressionQualityConfig,
    longest_token_run: usize,
) -> bool {
    let run_only_threshold = config
        .long_longest_token_run
        .saturating_mul(2)
        .max(config.long_longest_token_run);
    longest_token_run >= run_only_threshold
}

fn longest_common_token_run(left: &[&str], right: &[&str]) -> usize {
    let mut previous = vec![0usize; right.len() + 1];
    let mut longest = 0usize;
    for left_token in left {
        let mut current = vec![0usize; right.len() + 1];
        for (index, right_token) in right.iter().enumerate() {
            if left_token == right_token {
                current[index + 1] = previous[index] + 1;
                longest = longest.max(current[index + 1]);
            }
        }
        previous = current;
    }
    longest
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matcher_normalizes_candidate_like_stored_signature() {
        let config = EchoSuppressionQualityConfig::default();
        let assistant = speech_echo_signature("Don't repeat this after it's canceled");
        let result = match_assistant_echo_signature(
            &config,
            "dont repeat this after its canceled",
            &assistant,
        );

        assert!(result.is_some());
    }

    #[test]
    fn matcher_catches_long_contiguous_echo_run_with_asr_noise() {
        let config = EchoSuppressionQualityConfig::default();
        let assistant = speech_echo_signature(
            "please repeat this replacement sentence clearly after the interruption",
        );
        let result = match_assistant_echo_signature(
            &config,
            "He's been paint this replacement sentence clearly after the interrup",
            &assistant,
        );

        assert!(result.is_some());
    }
}
