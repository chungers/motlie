use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ResultRecord {
    pub schema_version: u32,
    pub run_id: String,
    pub bundle_id: String,
    pub scenario: String,
    pub capability: String,
    pub profile: String,
    pub acceptance: Acceptance,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Acceptance {
    pub behavior_status: AcceptanceStatus,
    pub performance_status: AcceptanceStatus,
    pub resource_status: AcceptanceStatus,
    pub overall_status: AcceptanceStatus,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AcceptanceStatus {
    Pass,
    Fail,
    Blocked,
    NotMeasured,
    NotApplicable,
}
