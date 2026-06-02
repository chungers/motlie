pub fn encode_pcmu(samples: &[i16]) -> Vec<u8> {
    samples.iter().copied().map(encode_ulaw_sample).collect()
}

pub fn decode_pcmu(payload: &[u8]) -> Vec<i16> {
    payload.iter().copied().map(decode_ulaw_sample).collect()
}

pub fn decode_pcma(payload: &[u8]) -> Vec<i16> {
    payload.iter().copied().map(decode_alaw_sample).collect()
}

fn decode_ulaw_sample(byte: u8) -> i16 {
    let value = !byte;
    let sign = value & 0x80;
    let exponent = (value >> 4) & 0x07;
    let mantissa = value & 0x0f;
    let sample = (((mantissa as i32) << 3) + 0x84) << exponent;
    let sample = sample - 0x84;
    if sign != 0 {
        -(sample as i16)
    } else {
        sample as i16
    }
}

fn encode_ulaw_sample(sample: i16) -> u8 {
    const BIAS: i32 = 0x84;
    const CLIP: i32 = 32_635;

    let mut linear = i32::from(sample);
    let mask = if linear < 0 {
        linear = (-linear).min(CLIP);
        0x7f
    } else {
        linear = linear.min(CLIP);
        0xff
    };
    linear += BIAS;

    let mut segment = 0;
    let mut threshold = 0x100;
    while segment < 8 && linear >= threshold {
        segment += 1;
        threshold <<= 1;
    }

    if segment >= 8 {
        return 0x7f ^ mask;
    }

    let quantization = (linear >> (segment + 3)) & 0x0f;
    ((segment << 4) | quantization) as u8 ^ mask
}

fn decode_alaw_sample(byte: u8) -> i16 {
    let value = byte ^ 0x55;
    let sign = value & 0x80;
    let exponent = (value >> 4) & 0x07;
    let mantissa = value & 0x0f;
    let sample = if exponent == 0 {
        ((mantissa as i32) << 4) + 8
    } else {
        (((mantissa as i32) << 4) + 0x108) << (exponent - 1)
    };
    if sign != 0 {
        sample as i16
    } else {
        -(sample as i16)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pcmu_silence_decodes_near_zero() {
        assert!(decode_pcmu(&[0xff])[0].abs() <= 4);
        assert!(decode_pcmu(&[0x7f])[0].abs() <= 4);
    }

    #[test]
    fn pcmu_encoder_round_trips_common_levels() {
        let samples = [-30_000, -1_000, 0, 1_000, 30_000];
        let decoded = decode_pcmu(&encode_pcmu(&samples));

        assert_eq!(decoded[2], 0);
        assert!(decoded[0] < -28_000);
        assert!(decoded[1] < -900);
        assert!(decoded[3] > 900);
        assert!(decoded[4] > 28_000);
    }

    #[test]
    fn pcma_silence_decodes_near_zero() {
        assert!(decode_pcma(&[0xd5])[0].abs() <= 16);
        assert!(decode_pcma(&[0x55])[0].abs() <= 16);
    }
}
