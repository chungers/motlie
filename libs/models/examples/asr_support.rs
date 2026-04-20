use std::path::PathBuf;

use anyhow::Result;

use crate::audio_support::{DynWavReader, WavInput, open_wav_reader};

pub struct AsrInput {
    pub source: WavInput,
    pub reader: DynWavReader,
}

pub fn open_asr_input(wav_path: Option<PathBuf>) -> Result<AsrInput> {
    let (source, reader) = open_wav_reader(wav_path.as_deref())?;
    Ok(AsrInput { source, reader })
}

pub fn describe_input(source: &WavInput) -> String {
    match source {
        WavInput::File(path) => path.display().to_string(),
        WavInput::Stdin => "<stdin>".into(),
    }
}

pub fn log_status(message: &str) {
    eprintln!("{message}");
}
