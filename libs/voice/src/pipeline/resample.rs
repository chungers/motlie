use crate::{Result, VoiceError};

pub trait Resampler {
    fn resample_f32(
        &self,
        samples: &[f32],
        input_rate_hz: u32,
        output_rate_hz: u32,
    ) -> Result<Vec<f32>>;
}

#[derive(Clone, Copy, Debug, Default)]
pub struct LinearInterpolator;

impl Resampler for LinearInterpolator {
    fn resample_f32(
        &self,
        samples: &[f32],
        input_rate_hz: u32,
        output_rate_hz: u32,
    ) -> Result<Vec<f32>> {
        if input_rate_hz == 0 || output_rate_hz == 0 {
            return Err(VoiceError::InvalidSampleRate {
                input_rate_hz,
                output_rate_hz,
            });
        }
        if samples.is_empty() || input_rate_hz == output_rate_hz {
            return Ok(samples.to_vec());
        }

        let ratio = input_rate_hz as f64 / output_rate_hz as f64;
        let out_len = ((samples.len() as f64) * output_rate_hz as f64 / input_rate_hz as f64)
            .ceil()
            .clamp(0.0, usize::MAX as f64) as usize;
        let max_index = samples.len().saturating_sub(1);
        let mut output = Vec::with_capacity(out_len.max(1));

        for out_idx in 0..out_len {
            let src_pos = out_idx as f64 * ratio;
            let left_idx = src_pos.floor() as usize;
            let right_idx = (left_idx + 1).min(max_index);
            let frac = (src_pos - left_idx as f64) as f32;
            let left = samples[left_idx.min(max_index)];
            let right = samples[right_idx];
            output.push(left + (right - left) * frac);
        }

        Ok(output)
    }
}
