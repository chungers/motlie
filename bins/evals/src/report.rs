use std::fs::OpenOptions;
use std::io::{self, Write};
use std::path::PathBuf;

use anyhow::{Context, Result};

use crate::result::{AcceptanceStatus, ResultRecord};

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum OutputSink {
    Stdout,
    JsonlFile(PathBuf),
}

impl OutputSink {
    pub fn emit(&self, record: &ResultRecord) -> Result<()> {
        match self {
            Self::Stdout => {
                let mut stdout = io::stdout().lock();
                serde_json::to_writer(&mut stdout, record)?;
                writeln!(stdout)?;
                Ok(())
            }
            Self::JsonlFile(path) => {
                if let Some(parent) = path.parent() {
                    std::fs::create_dir_all(parent).with_context(|| {
                        format!("failed to create report directory `{}`", parent.display())
                    })?;
                }
                let mut file = OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(path)
                    .with_context(|| format!("failed to open `{}`", path.display()))?;
                serde_json::to_writer(&mut file, record)?;
                writeln!(file)?;
                Ok(())
            }
        }
    }
}

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
