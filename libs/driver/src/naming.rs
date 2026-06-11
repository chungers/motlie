use crate::error::{DriverError, DriverResult};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QualifiedName<'a> {
    pub scope: Option<&'a str>,
    pub value: &'a str,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedName {
    pub scope: String,
    pub value: String,
}

pub fn parse_qualified_name(raw: &str) -> QualifiedName<'_> {
    match raw.split_once('/') {
        Some((scope, value)) => QualifiedName {
            scope: Some(scope),
            value,
        },
        None => QualifiedName {
            scope: None,
            value: raw,
        },
    }
}

impl ResolvedName {
    pub fn new(scope: impl Into<String>, value: impl Into<String>) -> Self {
        Self {
            scope: scope.into(),
            value: value.into(),
        }
    }
}

pub trait ResolveName<K> {
    type Resolved;

    fn resolve_name(&self, kind: K, raw: &str) -> DriverResult<Self::Resolved>;
}

pub fn validate_qualified_name(raw: &str) -> DriverResult<QualifiedName<'_>> {
    let qualified = parse_qualified_name(raw);
    if qualified.value.is_empty() || matches!(qualified.scope, Some("")) {
        return Err(DriverError::malformed_qualified_name(raw));
    }
    Ok(qualified)
}

#[cfg(test)]
mod tests {
    use super::{parse_qualified_name, validate_qualified_name, QualifiedName};
    use crate::error::DriverError;

    #[test]
    fn parse_qualified_name_handles_bare_values() {
        assert_eq!(
            parse_qualified_name("demo:0.1"),
            QualifiedName {
                scope: None,
                value: "demo:0.1",
            }
        );
    }

    #[test]
    fn parse_qualified_name_splits_scope_and_value() {
        assert_eq!(
            parse_qualified_name("alpha/demo:0.1"),
            QualifiedName {
                scope: Some("alpha"),
                value: "demo:0.1",
            }
        );
    }

    #[test]
    fn validate_qualified_name_rejects_empty_segments() {
        let err = validate_qualified_name("alpha/").expect_err("empty value rejected");
        assert!(matches!(err, DriverError::MalformedQualifiedName { .. }));

        let err = validate_qualified_name("/demo").expect_err("empty scope rejected");
        assert!(matches!(err, DriverError::MalformedQualifiedName { .. }));
    }
}
