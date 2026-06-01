use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use anyhow::Context;
use chrono::Utc;
use motlie_voice::wav::StreamingWavWriter;
use serde_json::json;

use super::MediaFormat;

pub(super) struct MediaCapture {
    dir: PathBuf,
    raw_jsonl: BufWriter<File>,
    transcript_jsonl: BufWriter<File>,
    decoded_wav: StreamingWavWriter<File, i16>,
    asr_wav: StreamingWavWriter<File, i16>,
}

impl MediaCapture {
    pub(super) fn start(
        root: &Path,
        gateway_call_id: &str,
        stream_id: &str,
        format: &MediaFormat,
    ) -> anyhow::Result<Self> {
        let dir = root
            .join(sanitize_path_component(gateway_call_id))
            .join(sanitize_path_component(stream_id));
        fs::create_dir_all(&dir)
            .with_context(|| format!("create media capture dir {}", dir.display()))?;

        let raw_jsonl = BufWriter::new(create_file(&dir, "telnyx-media.jsonl")?);
        let transcript_jsonl = BufWriter::new(create_file(&dir, "transcripts.jsonl")?);
        let decoded_wav = StreamingWavWriter::<_, i16>::new(
            create_file(&dir, "decoded-inbound.wav")?,
            format.sample_rate_hz,
            format.channels,
        )
        .context("create decoded inbound WAV capture")?;
        let asr_wav =
            StreamingWavWriter::<_, i16>::new(create_file(&dir, "asr-input-16khz.wav")?, 16_000, 1)
                .context("create ASR input WAV capture")?;

        fs::write(
            dir.join("manifest.json"),
            serde_json::to_vec_pretty(&json!({
                "created_at": Utc::now(),
                "gateway_call_id": gateway_call_id,
                "stream_id": stream_id,
                "observed_codec": format.encoding,
                "observed_sample_rate_hz": format.sample_rate_hz,
                "observed_channels": format.channels,
                "files": {
                    "telnyx_media_jsonl": "telnyx-media.jsonl",
                    "decoded_inbound_wav": "decoded-inbound.wav",
                    "asr_input_wav": "asr-input-16khz.wav",
                    "transcripts_jsonl": "transcripts.jsonl"
                }
            }))?,
        )
        .with_context(|| format!("write media capture manifest {}", dir.display()))?;

        Ok(Self {
            dir,
            raw_jsonl,
            transcript_jsonl,
            decoded_wav,
            asr_wav,
        })
    }

    pub(super) fn dir(&self) -> &Path {
        &self.dir
    }

    pub(super) fn record_raw_event(&mut self, raw: &str) -> anyhow::Result<()> {
        self.raw_jsonl.write_all(raw.as_bytes())?;
        self.raw_jsonl.write_all(b"\n")?;
        self.raw_jsonl.flush()?;
        Ok(())
    }

    pub(super) fn record_decoded_samples(&mut self, samples: &[i16]) -> anyhow::Result<()> {
        self.decoded_wav.write_chunk(samples)?;
        Ok(())
    }

    pub(super) fn record_asr_samples(&mut self, samples: &[i16]) -> anyhow::Result<()> {
        self.asr_wav.write_chunk(samples)?;
        Ok(())
    }

    pub(super) fn record_transcript(
        &mut self,
        kind: &str,
        text: &str,
        suppressed: bool,
    ) -> anyhow::Result<()> {
        serde_json::to_writer(
            &mut self.transcript_jsonl,
            &json!({
                "at": Utc::now(),
                "kind": kind,
                "suppressed": suppressed,
                "text": text,
            }),
        )?;
        self.transcript_jsonl.write_all(b"\n")?;
        self.transcript_jsonl.flush()?;
        Ok(())
    }

    pub(super) fn finalize(mut self) -> anyhow::Result<PathBuf> {
        self.raw_jsonl.flush()?;
        self.transcript_jsonl.flush()?;
        self.decoded_wav.finalize()?;
        self.asr_wav.finalize()?;
        Ok(self.dir)
    }
}

fn create_file(dir: &Path, file_name: &str) -> anyhow::Result<File> {
    let path = dir.join(file_name);
    File::create(&path).with_context(|| format!("create media capture file {}", path.display()))
}

fn sanitize_path_component(value: &str) -> String {
    value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.') {
                ch
            } else {
                '_'
            }
        })
        .collect()
}
