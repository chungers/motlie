use anyhow::Result;

use crate::result::ResultRecord;

pub trait ScenarioRunner {
    fn run(&self) -> Result<ResultRecord>;
}
