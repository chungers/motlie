use crate::pipeline::convert::{f32_to_i16_clamped, i16_to_f32};
use crate::{Result, VoiceError};

pub trait Resampler {
    fn resample_f32(
        &self,
        samples: &[f32],
        input_rate_hz: u32,
        output_rate_hz: u32,
    ) -> Result<Vec<f32>>;
}

pub fn resample_i16_mono<R: Resampler>(
    resampler: &R,
    samples: &[i16],
    input_rate_hz: u32,
    output_rate_hz: u32,
) -> Result<Vec<i16>> {
    let input = i16_to_f32(samples);
    let output = resampler.resample_f32(&input, input_rate_hz, output_rate_hz)?;
    Ok(f32_to_i16_clamped(&output))
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

#[derive(Clone, Copy, Debug)]
pub struct WindowedSincResampler {
    radius: usize,
}

impl Default for WindowedSincResampler {
    fn default() -> Self {
        Self { radius: 16 }
    }
}

impl WindowedSincResampler {
    pub fn new(radius: usize) -> Self {
        Self {
            radius: radius.max(4),
        }
    }
}

impl Resampler for WindowedSincResampler {
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
        let cutoff = (output_rate_hz as f64 / input_rate_hz as f64).min(1.0);
        let radius = self.radius as isize;
        let last = samples.len().saturating_sub(1) as isize;
        let mut output = Vec::with_capacity(out_len.max(1));

        for out_idx in 0..out_len {
            let center = out_idx as f64 * ratio;
            let left = center.floor() as isize - radius + 1;
            let right = center.floor() as isize + radius;
            let mut value = 0.0;
            let mut weight_sum = 0.0;

            for idx in left..=right {
                let clamped = idx.clamp(0, last) as usize;
                let distance = center - idx as f64;
                let window = hann_window(distance, radius as f64);
                let weight = cutoff * sinc(cutoff * distance) * window;
                value += samples[clamped] as f64 * weight;
                weight_sum += weight;
            }

            if weight_sum.abs() > f64::EPSILON {
                output.push((value / weight_sum) as f32);
            } else {
                output.push(0.0);
            }
        }

        Ok(output)
    }
}

fn sinc(x: f64) -> f64 {
    if x.abs() < 1e-8 {
        1.0
    } else {
        let pix = std::f64::consts::PI * x;
        pix.sin() / pix
    }
}

fn hann_window(distance: f64, radius: f64) -> f64 {
    let normalized = (distance.abs() / radius).min(1.0);
    0.5 + 0.5 * (std::f64::consts::PI * normalized).cos()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sinc_resampler_changes_frame_count() {
        let input = vec![0.0; 800];
        let output = WindowedSincResampler::default()
            .resample_f32(&input, 8_000, 16_000)
            .expect("resample should succeed");
        assert!((output.len() as isize - 1600).abs() <= 1);
    }

    #[test]
    fn i16_wrapper_resamples_through_f32_path() {
        let input = vec![0_i16; 800];
        let output = resample_i16_mono(&WindowedSincResampler::default(), &input, 8_000, 16_000)
            .expect("resample should succeed");
        assert!((output.len() as isize - 1600).abs() <= 1);
    }
}
