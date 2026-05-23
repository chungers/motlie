use serde_json::{json, Value};

pub fn ok(op: &str) -> Value {
    json!({ "type": "ok", "op": op })
}

pub fn error(kind: &str, message: impl Into<String>) -> Value {
    json!({
        "type": "error",
        "kind": kind,
        "message": message.into(),
    })
}

pub fn error_with_field(
    kind: &str,
    message: impl Into<String>,
    field: &str,
    value: Value,
) -> Value {
    let mut record = error(kind, message);
    if let Some(obj) = record.as_object_mut() {
        obj.insert(field.to_string(), value);
    }
    record
}

pub fn print_records(records: &[Value]) -> anyhow::Result<()> {
    for record in records {
        println!("{}", serde_json::to_string(record)?);
    }
    Ok(())
}
