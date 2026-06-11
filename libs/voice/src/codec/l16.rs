use crate::{Result, VoiceError};

pub fn decode_l16_be(payload: &[u8]) -> Result<Vec<i16>> {
    if !payload.len().is_multiple_of(2) {
        return Err(VoiceError::MisalignedEncodedPayload {
            payload_len: payload.len(),
            sample_width: 2,
        });
    }

    Ok(payload
        .chunks_exact(2)
        .map(|chunk| i16::from_be_bytes([chunk[0], chunk[1]]))
        .collect())
}

pub fn decode_l16_le(payload: &[u8]) -> Result<Vec<i16>> {
    if !payload.len().is_multiple_of(2) {
        return Err(VoiceError::MisalignedEncodedPayload {
            payload_len: payload.len(),
            sample_width: 2,
        });
    }

    Ok(payload
        .chunks_exact(2)
        .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]))
        .collect())
}

pub fn encode_l16_be(samples: &[i16]) -> Vec<u8> {
    samples
        .iter()
        .flat_map(|sample| sample.to_be_bytes())
        .collect()
}

pub fn encode_l16_le(samples: &[i16]) -> Vec<u8> {
    samples
        .iter()
        .flat_map(|sample| sample.to_le_bytes())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn l16_round_trips_big_endian_samples() {
        let samples = vec![i16::MIN, -1, 0, 1, i16::MAX];
        let encoded = encode_l16_be(&samples);
        let decoded = decode_l16_be(&encoded).expect("valid l16 payload should decode");
        assert_eq!(decoded, samples);
    }

    #[test]
    fn l16_round_trips_little_endian_samples() {
        let samples = vec![i16::MIN, -1, 0, 1, i16::MAX];
        let encoded = encode_l16_le(&samples);
        let decoded = decode_l16_le(&encoded).expect("valid l16 payload should decode");
        assert_eq!(decoded, samples);
    }

    #[test]
    fn l16_little_endian_differs_from_big_endian_for_telnyx_capture_shape() {
        let encoded = [0x26, 0x03, 0x10, 0x02, 0x07, 0x01, 0x34, 0xff];
        let little = decode_l16_le(&encoded).expect("valid l16 payload should decode");
        let big = decode_l16_be(&encoded).expect("valid l16 payload should decode");

        assert_eq!(little, vec![806, 528, 263, -204]);
        assert_eq!(big, vec![9731, 4098, 1793, 13567]);
    }

    #[test]
    fn l16_rejects_odd_byte_count() {
        let err = decode_l16_be(&[0]).expect_err("odd byte count should fail");
        assert!(matches!(
            err,
            VoiceError::MisalignedEncodedPayload {
                payload_len: 1,
                sample_width: 2
            }
        ));

        let err = decode_l16_le(&[0]).expect_err("odd byte count should fail");
        assert!(matches!(
            err,
            VoiceError::MisalignedEncodedPayload {
                payload_len: 1,
                sample_width: 2
            }
        ));
    }
}
