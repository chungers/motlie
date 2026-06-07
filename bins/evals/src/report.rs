use crate::result::{AcceptanceStatus, ResultRecord};

pub fn summarize_status(records: &[ResultRecord]) -> AcceptanceStatus {
    if records
        .iter()
        .any(|record| record.acceptance.overall_status == AcceptanceStatus::Fail)
    {
        return AcceptanceStatus::Fail;
    }

    if records
        .iter()
        .any(|record| record.acceptance.overall_status == AcceptanceStatus::Blocked)
    {
        return AcceptanceStatus::Blocked;
    }

    if records
        .iter()
        .all(|record| record.acceptance.overall_status == AcceptanceStatus::Pass)
    {
        AcceptanceStatus::Pass
    } else {
        AcceptanceStatus::NotMeasured
    }
}
