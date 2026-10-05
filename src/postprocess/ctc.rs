use crate::error::{OcrError, Result};
use crate::tensor::Tensor;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

pub struct CtcDecoder {
    vocab: Vec<String>,
}

impl CtcDecoder {
    pub fn from_file<P: AsRef<Path>>(dict_path: P) -> Result<Self> {
        let file = File::open(dict_path.as_ref())
            .map_err(|e| OcrError::ModelError(format!("Failed to open dict file: {}", e)))?;
        let reader = BufReader::new(file);

        let mut vocab = Vec::new();
        // Index 0: CTC blank
        vocab.push("blank".to_string());

        for line in reader.lines() {
            let line = line.map_err(|e| OcrError::ModelError(format!("Dict read error: {}", e)))?;
            vocab.push(line.trim_end_matches(&['\r', '\n'][..]).to_string());
        }

        // Add trailing space character
        vocab.push(" ".to_string());

        Ok(Self { vocab })
    }

    pub fn from_vocab(vocab: Vec<String>) -> Self {
        Self { vocab }
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    pub fn decode(&self, preds: &Tensor) -> Result<(String, f32)> {
        let shape = preds.shape();
        if shape.len() != 3 {
            return Err(OcrError::OperatorError {
                op: "CtcDecode".into(),
                msg: format!(
                    "Expected 3D tensor [batch, time, vocab], got shape {:?}",
                    shape
                ),
            });
        }

        let batch = shape[0];
        let seq_len = shape[1];
        let vocab_size = shape[2];

        // The character dictionary must belong to this model. PP-OCR recognition heads
        // emit exactly `blank + dict entries + trailing space` classes, so a mismatch means
        // the wrong dict file was passed. Silently dropping out-of-range classes (as the
        // loop below used to) turns that configuration error into mysteriously truncated
        // text, so fail loudly instead.
        if vocab_size != self.vocab.len() {
            return Err(OcrError::OperatorError {
                op: "CtcDecode".into(),
                msg: format!(
                    "dictionary/model mismatch: model outputs {} classes but the dictionary \
                     has {} entries (expected blank + {} characters + trailing space)",
                    vocab_size,
                    self.vocab.len(),
                    self.vocab.len().saturating_sub(2)
                ),
            });
        }

        let data = preds.as_slice();
        if batch == 0 || seq_len == 0 {
            return Ok((String::new(), 0.0));
        }

        let expected_len = batch
            .checked_mul(seq_len)
            .and_then(|n| n.checked_mul(vocab_size))
            .ok_or_else(|| OcrError::OperatorError {
                op: "CtcDecode".into(),
                msg: format!("shape {:?} overflows", shape),
            })?;
        if data.len() < expected_len {
            return Err(OcrError::OperatorError {
                op: "CtcDecode".into(),
                msg: format!(
                    "tensor holds {} elements but shape {:?} requires {}",
                    data.len(),
                    shape,
                    expected_len
                ),
            });
        }

        let batch_results = self.decode_batch(preds)?;
        Ok(batch_results.into_iter().next().unwrap_or_default())
    }

    /// Decode all items in a batched 3D tensor [batch, time, vocab].
    pub fn decode_batch(&self, preds: &Tensor) -> Result<Vec<(String, f32)>> {
        let shape = preds.shape();
        if shape.len() != 3 {
            return Err(OcrError::OperatorError {
                op: "CtcDecode".into(),
                msg: format!(
                    "Expected 3D tensor [batch, time, vocab], got shape {:?}",
                    shape
                ),
            });
        }

        let batch = shape[0];
        let seq_len = shape[1];
        let vocab_size = shape[2];

        if vocab_size != self.vocab.len() {
            return Err(OcrError::OperatorError {
                op: "CtcDecode".into(),
                msg: format!(
                    "dictionary/model mismatch: model outputs {} classes but the dictionary \
                     has {} entries",
                    vocab_size,
                    self.vocab.len()
                ),
            });
        }

        if batch == 0 || seq_len == 0 {
            return Ok(Vec::new());
        }

        let data = preds.as_slice();
        let expected_len = batch * seq_len * vocab_size;
        if data.len() < expected_len {
            return Err(OcrError::OperatorError {
                op: "CtcDecode".into(),
                msg: format!("tensor holds {} elements, expected {}", data.len(), expected_len),
            });
        }

        let mut results = Vec::with_capacity(batch);
        for b in 0..batch {
            let b_offset = b * seq_len * vocab_size;
            let mut text = String::new();
            let mut scores = Vec::new();
            let mut last_idx = 0usize;

            for t in 0..seq_len {
                let step_offset = b_offset + t * vocab_size;
                let slice = &data[step_offset..step_offset + vocab_size];

                let (max_idx, max_val) = find_argmax(slice);

                if max_idx != 0 && max_idx != last_idx {
                    text.push_str(&self.vocab[max_idx]);
                    scores.push(max_val);
                }

                last_idx = max_idx;
            }

            let avg_score = if scores.is_empty() {
                0.0
            } else {
                scores.iter().sum::<f32>() / (scores.len() as f32)
            };

            results.push((text, avg_score));
        }

        Ok(results)
    }
}

#[inline]
fn find_argmax(slice: &[f32]) -> (usize, f32) {
    if slice.is_empty() {
        return (0, f32::NEG_INFINITY);
    }
    #[cfg(target_arch = "x86_64")]
    if crate::ops::avx2_fma_enabled() && slice.len() >= 16 {
        return unsafe { find_argmax_avx2(slice) };
    }
    find_argmax_scalar(slice)
}

fn find_argmax_scalar(slice: &[f32]) -> (usize, f32) {
    let mut max_idx = 0usize;
    let mut max_val = f32::NEG_INFINITY;
    for (idx, &v) in slice.iter().enumerate() {
        if v > max_val {
            max_val = v;
            max_idx = idx;
        }
    }
    (max_idx, max_val)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn find_argmax_avx2(slice: &[f32]) -> (usize, f32) {
    use std::arch::x86_64::*;
    let n = slice.len();
    let ptr = slice.as_ptr();

    unsafe {
        let mut v_max = _mm256_set1_ps(f32::NEG_INFINITY);
        let mut v_idx = _mm256_setzero_si256();
        let mut cur_indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
        let v_step = _mm256_set1_epi32(8);
        let mut i = 0;

        while i + 8 <= n {
            let v = _mm256_loadu_ps(ptr.add(i));
            let cmp = _mm256_cmp_ps(v, v_max, _CMP_GT_OQ);
            v_max = _mm256_blendv_ps(v_max, v, cmp);
            v_idx = _mm256_castps_si256(_mm256_blendv_ps(
                _mm256_castsi256_ps(v_idx),
                _mm256_castsi256_ps(cur_indices),
                cmp,
            ));
            cur_indices = _mm256_add_epi32(cur_indices, v_step);
            i += 8;
        }

        let mut max_buf = [0.0f32; 8];
        let mut idx_buf = [0i32; 8];
        _mm256_storeu_ps(max_buf.as_mut_ptr(), v_max);
        _mm256_storeu_si256(idx_buf.as_mut_ptr() as *mut __m256i, v_idx);

        let mut best_val = max_buf[0];
        let mut best_idx = idx_buf[0] as usize;
        for lane in 1..8 {
            let v = max_buf[lane];
            let idx = idx_buf[lane] as usize;
            if v > best_val {
                best_val = v;
                best_idx = idx;
            } else if v == best_val && idx < best_idx {
                best_idx = idx;
            }
        }

        while i < n {
            let val = *ptr.add(i);
            if val > best_val {
                best_val = val;
                best_idx = i;
            }
            i += 1;
        }

        (best_idx, best_val)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn one_hot(seq_len: usize, vocab: usize, picks: &[(usize, usize)]) -> Tensor {
        let mut data = vec![0.0f32; seq_len * vocab];
        for &(t, c) in picks {
            data[t * vocab + c] = 1.0;
        }
        Tensor::from_vec(&[1, seq_len, vocab], data).expect("valid tensor")
    }

    #[test]
    fn decode_collapses_repeats_and_blank() {
        let dec = CtcDecoder::from_vocab(vec![
            "blank".to_string(),
            "a".to_string(),
            "b".to_string(),
            " ".to_string(),
        ]);
        // frames: a, a (repeat), blank, b
        let t = one_hot(4, 4, &[(0, 1), (1, 1), (2, 0), (3, 2)]);
        let (text, score) = dec.decode(&t).expect("decode ok");
        assert_eq!(text, "ab");
        assert!((score - 1.0).abs() < 1e-6);
    }

    #[test]
    fn decode_rejects_dictionary_mismatch() {
        // Dictionary of 3 entries vs a model head emitting 4 classes.
        let dec =
            CtcDecoder::from_vocab(vec!["blank".to_string(), "a".to_string(), " ".to_string()]);
        let t = one_hot(2, 4, &[(0, 1)]);
        let err = dec
            .decode(&t)
            .expect_err("a dictionary/model mismatch must be an error, not silent truncation");
        let msg = format!("{err:?}");
        assert!(msg.contains("mismatch"), "unexpected message: {msg}");
    }

    #[test]
    fn decode_rejects_wrong_rank() {
        let dec = CtcDecoder::from_vocab(vec!["blank".to_string(), " ".to_string()]);
        let t = Tensor::from_vec(&[2, 2], vec![0.0, 1.0, 1.0, 0.0]).expect("valid tensor");
        assert!(dec.decode(&t).is_err());
    }

    #[test]
    fn test_find_argmax_avx2_matches_scalar() {
        let mut data = vec![0.01f32; 18710];
        data[42] = 0.95;
        data[1000] = 0.88;
        let (idx, val) = find_argmax(&data);
        let (scalar_idx, scalar_val) = find_argmax_scalar(&data);
        assert_eq!(idx, 42);
        assert_eq!(idx, scalar_idx);
        assert_eq!(val, scalar_val);

        // Max at end
        let mut data2 = vec![-1.0f32; 35];
        data2[34] = 10.0;
        let (idx2, val2) = find_argmax(&data2);
        let (s_idx2, s_val2) = find_argmax_scalar(&data2);
        assert_eq!(idx2, 34);
        assert_eq!(idx2, s_idx2);
        assert_eq!(val2, s_val2);
    }

    #[test]
    fn decode_handles_empty_batch_gracefully() {
        let dec = CtcDecoder::from_vocab(vec!["blank".to_string(), "a".to_string(), " ".to_string()]);
        let empty_batch = Tensor::from_vec(&[0, 5, 3], vec![]).expect("valid 0-batch tensor");
        let (text, score) = dec.decode(&empty_batch).expect("decode ok for empty batch");
        assert_eq!(text, "");
        assert_eq!(score, 0.0);
    }

    #[test]
    fn decode_batch_decodes_multiple_lines() {
        let dec = CtcDecoder::from_vocab(vec![
            "blank".to_string(),
            "a".to_string(),
            "b".to_string(),
            " ".to_string(),
        ]);
        // Batch 2, seq_len 2, vocab 4
        // Batch 0: a, b -> "ab"
        // Batch 1: b, blank -> "b"
        let mut data = vec![0.0f32; 2 * 2 * 4];
        // Batch 0, t0 = 'a' (idx 1)
        data[1] = 1.0;
        // Batch 0, t1 = 'b' (idx 2)
        data[4 + 2] = 1.0;
        // Batch 1, t0 = 'b' (idx 2)
        data[8 + 2] = 1.0;
        // Batch 1, t1 = 'blank' (idx 0)
        data[8 + 4] = 1.0;

        let t = Tensor::from_vec(&[2, 2, 4], data).expect("valid tensor");
        let res = dec.decode_batch(&t).expect("decode batch ok");
        assert_eq!(res.len(), 2);
        assert_eq!(res[0].0, "ab");
        assert_eq!(res[1].0, "b");
    }
}
