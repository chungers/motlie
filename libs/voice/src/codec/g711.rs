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
    fn pcma_silence_decodes_near_zero() {
        assert!(decode_pcma(&[0xd5])[0].abs() <= 16);
        assert!(decode_pcma(&[0x55])[0].abs() <= 16);
    }
}
