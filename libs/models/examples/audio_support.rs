use std::fs::File;
use std::io::{self, BufReader, Read};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

pub type DynWavReader = hound::WavReader<BufReader<WavReaderInput>>;

pub enum WavReaderInput {
    File(File),
    Stdin(std::io::Stdin),
}

impl Read for WavReaderInput {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        match self {
            Self::File(file) => file.read(buf),
            Self::Stdin(stdin) => stdin.read(buf),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum WavInput {
    File(PathBuf),
    Stdin,
}

pub fn open_wav_reader(path: Option<&Path>) -> Result<(WavInput, DynWavReader)> {
    match path {
        Some(path) => {
            let file = File::open(path)
                .with_context(|| format!("failed to open wav file `{}`", path.display()))?;
            let input = WavInput::File(path.to_path_buf());
            let reader = hound::WavReader::new(BufReader::new(WavReaderInput::File(file)))
                .with_context(|| format!("failed to parse wav file `{}`", path.display()))?;
            Ok((input, reader))
        }
        None => {
            let reader =
                hound::WavReader::new(BufReader::new(WavReaderInput::Stdin(std::io::stdin())))
                    .context("failed to parse wav stream from stdin")?;
            Ok((WavInput::Stdin, reader))
        }
    }
}
