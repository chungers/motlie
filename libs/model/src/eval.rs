//! Lightweight eval-facing contracts that belong next to the model API.
//!
//! Substantial harness tooling should live in `motlie-model-eval`.

/// High-level evaluation tracks used to group bundle assessment suites.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum EvalTrack {
    Chat,
    Classification,
    Embeddings,
    Reasoning,
    Summarization,
}

/// Stable identifier for a single evaluation case.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct EvalCaseId(String);

impl EvalCaseId {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Lightweight metadata for a single evaluation case.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EvalCase {
    pub id: EvalCaseId,
    pub track: EvalTrack,
    pub prompt: String,
}

/// Structured result surface that higher-level harness tooling can aggregate.
#[derive(Clone, Debug, PartialEq)]
pub struct EvalResult {
    pub case_id: EvalCaseId,
    pub score: f32,
    pub passed: bool,
    pub notes: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eval_case_ids_are_stable_strings() {
        let case_id = EvalCaseId::new("embeddings-basic-001");

        assert_eq!(case_id.as_str(), "embeddings-basic-001");
    }

    #[test]
    fn eval_result_construction_is_straightforward() {
        let case = EvalCase {
            id: EvalCaseId::new("embeddings-basic-001"),
            track: EvalTrack::Embeddings,
            prompt: "motlie model bundle".into(),
        };

        let result = EvalResult {
            case_id: case.id.clone(),
            score: 1.0,
            passed: true,
            notes: vec!["vector count matched input count".into()],
        };

        assert_eq!(result.case_id, case.id);
        assert!(result.passed);
    }
}
