use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::{env, fs};

use thiserror::Error;

const CLAUSE_INTONATION_FULL_STOP: i32 = 0x0000_0000;
const CLAUSE_INTONATION_COMMA: i32 = 0x0000_1000;
const CLAUSE_INTONATION_QUESTION: i32 = 0x0000_2000;
const CLAUSE_INTONATION_EXCLAMATION: i32 = 0x0000_3000;
const CLAUSE_TYPE_SENTENCE: i32 = 0x0008_0000;

const ESPEAK_INITIALIZE_PHONEME_IPA: u32 = 2;
const ESPEAK_INITIALIZE_DONT_EXIT: i32 = 32_768;
const ESPEAK_CHARS_UTF8: i32 = 1;
const ESPEAK_AUDIO_OUTPUT_RETRIEVAL: u32 = 1;
const ESPEAK_ERROR_OK: c_int = 0;

const PIPER_ESPEAKNG_DATA_DIRECTORY: &str = "PIPER_ESPEAKNG_DATA_DIRECTORY";
const ESPEAKNG_DATA_DIR_NAME: &str = "espeak-ng-data";

static ESPEAK_INIT: OnceLock<Result<(), EspeakError>> = OnceLock::new();

#[derive(Debug, Clone, Error, Eq, PartialEq)]
pub enum EspeakError {
    #[error("input contains interior NUL byte")]
    InteriorNul,
    #[error("{0}")]
    Initialization(String),
    #[error("failed to set eSpeak-ng voice `{voice}`")]
    SetVoice { voice: String },
    #[error("eSpeak-ng returned a null phoneme pointer")]
    NullPhonemePointer,
}

pub type EspeakResult<T> = Result<T, EspeakError>;

pub fn text_to_phonemes(
    text: &str,
    language: &str,
    phoneme_separator: Option<char>,
    remove_lang_switch_flags: bool,
    remove_stress: bool,
) -> EspeakResult<Vec<String>> {
    let mut phonemes = Vec::new();
    for line in text.lines() {
        phonemes.extend(phonemize_line(
            line,
            language,
            phoneme_separator,
            remove_lang_switch_flags,
            remove_stress,
        )?);
    }
    Ok(phonemes)
}

fn phonemize_line(
    text: &str,
    language: &str,
    phoneme_separator: Option<char>,
    remove_lang_switch_flags: bool,
    remove_stress: bool,
) -> EspeakResult<Vec<String>> {
    ensure_initialized()?;

    let language = CString::new(language).map_err(|_| EspeakError::InteriorNul)?;
    let set_voice_res = unsafe {
        // SAFETY: `language` is a valid, NUL-terminated C string for the duration of the call.
        espeak_SetVoiceByName(language.as_ptr())
    };
    if set_voice_res != ESPEAK_ERROR_OK {
        return Err(EspeakError::SetVoice {
            voice: language.to_string_lossy().into_owned(),
        });
    }

    let phoneme_mode = match phoneme_separator {
        Some(separator) => ((separator as u32) << 8) | ESPEAK_INITIALIZE_PHONEME_IPA,
        None => ESPEAK_INITIALIZE_PHONEME_IPA,
    } as c_int;

    let text = CString::new(text).map_err(|_| EspeakError::InteriorNul)?;
    let mut text_ptr = text.as_ptr() as *const c_void;
    let mut sentence_phonemes = Vec::new();
    let mut phoneme_buffer = String::new();

    while !text_ptr.is_null() {
        let result_ptr = unsafe {
            // SAFETY: `&mut text_ptr` points at a valid pointer into the owned `CString`,
            // `ESPEAK_CHARS_UTF8` and `phoneme_mode` are valid flags expected by eSpeak-ng.
            espeak_TextToPhonemes(&mut text_ptr, ESPEAK_CHARS_UTF8, phoneme_mode)
        };
        if result_ptr.is_null() {
            return Err(EspeakError::NullPhonemePointer);
        }

        let chunk = unsafe {
            // SAFETY: eSpeak-ng returns a NUL-terminated string valid until the next call.
            CStr::from_ptr(result_ptr)
        };
        phoneme_buffer.push_str(&chunk.to_string_lossy());

        let terminator = 0;
        let intonation = terminator & 0x0000_f000;
        if intonation == CLAUSE_INTONATION_FULL_STOP {
            phoneme_buffer.push('.');
        } else if intonation == CLAUSE_INTONATION_COMMA {
            phoneme_buffer.push(',');
        } else if intonation == CLAUSE_INTONATION_QUESTION {
            phoneme_buffer.push('?');
        } else if intonation == CLAUSE_INTONATION_EXCLAMATION {
            phoneme_buffer.push('!');
        }

        if (terminator & CLAUSE_TYPE_SENTENCE) == CLAUSE_TYPE_SENTENCE {
            sentence_phonemes.push(std::mem::take(&mut phoneme_buffer));
        }
    }

    if !phoneme_buffer.is_empty() {
        sentence_phonemes.push(phoneme_buffer);
    }

    if remove_lang_switch_flags {
        sentence_phonemes = sentence_phonemes
            .into_iter()
            .map(|sent| strip_lang_switch_flags(&sent))
            .collect();
    }

    if remove_stress {
        sentence_phonemes = sentence_phonemes
            .into_iter()
            .map(|sent| strip_stress_markers(&sent))
            .collect();
    }

    Ok(sentence_phonemes)
}

fn ensure_initialized() -> EspeakResult<()> {
    match ESPEAK_INIT.get_or_init(initialize_espeak) {
        Ok(()) => Ok(()),
        Err(err) => Err(err.clone()),
    }
}

fn initialize_espeak() -> EspeakResult<()> {
    let data_dir = discover_data_dir();
    let data_dir_cstr = data_dir
        .as_ref()
        .map(|path| CString::new(path.display().to_string()).map_err(|_| EspeakError::InteriorNul))
        .transpose()?;
    let data_dir_ptr = data_dir_cstr
        .as_ref()
        .map_or(std::ptr::null(), |path| path.as_ptr());

    let sample_rate = unsafe {
        // SAFETY: `data_dir_ptr` is either null or a valid C string path for the duration
        // of the call. Other arguments are fixed constants from the eSpeak-ng API.
        espeak_Initialize(
            ESPEAK_AUDIO_OUTPUT_RETRIEVAL,
            0,
            data_dir_ptr,
            ESPEAK_INITIALIZE_DONT_EXIT,
        )
    };

    if sample_rate <= 0 {
        return Err(EspeakError::Initialization(format!(
            "failed to initialize eSpeak-ng. Set `{PIPER_ESPEAKNG_DATA_DIRECTORY}` to the \
             eSpeak-ng data directory, for example `/usr/lib/x86_64-linux-gnu/{ESPEAKNG_DATA_DIR_NAME}` \
             or `/usr/share/{ESPEAKNG_DATA_DIR_NAME}`. Error code: `{sample_rate}`"
        )));
    }

    Ok(())
}

fn discover_data_dir() -> Option<PathBuf> {
    if let Some(path) = env::var_os(PIPER_ESPEAKNG_DATA_DIRECTORY).map(PathBuf::from) {
        return normalize_data_dir(path);
    }

    let cwd = env::current_dir().ok();
    if let Some(cwd) = cwd {
        if let Some(path) = normalize_data_dir(cwd) {
            return Some(path);
        }
    }

    let exe_parent = env::current_exe()
        .ok()
        .and_then(|path| path.parent().map(Path::to_path_buf));
    if let Some(parent) = exe_parent {
        if let Some(path) = normalize_data_dir(parent) {
            return Some(path);
        }
    }

    for candidate in common_data_dir_candidates() {
        if let Some(path) = normalize_data_dir(candidate) {
            return Some(path);
        }
    }

    None
}

fn normalize_data_dir(path: PathBuf) -> Option<PathBuf> {
    if is_data_dir(&path) {
        return Some(path);
    }
    let nested = path.join(ESPEAKNG_DATA_DIR_NAME);
    if is_data_dir(&nested) {
        return Some(nested);
    }
    None
}

fn is_data_dir(path: &Path) -> bool {
    path.join("phontab").is_file() && path.join("en_dict").is_file()
}

fn common_data_dir_candidates() -> Vec<PathBuf> {
    let mut candidates = vec![
        PathBuf::from("/usr/share/espeak-ng-data"),
        PathBuf::from("/usr/local/share/espeak-ng-data"),
        PathBuf::from("/opt/homebrew/share/espeak-ng-data"),
    ];
    if let Ok(entries) = fs::read_dir("/usr/lib") {
        for entry in entries.flatten() {
            candidates.push(entry.path().join(ESPEAKNG_DATA_DIR_NAME));
        }
    }
    candidates
}

fn strip_lang_switch_flags(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let mut depth = 0_u32;
    for ch in input.chars() {
        match ch {
            '(' => depth = depth.saturating_add(1),
            ')' => depth = depth.saturating_sub(1),
            _ if depth == 0 => out.push(ch),
            _ => {}
        }
    }
    out
}

fn strip_stress_markers(input: &str) -> String {
    input
        .chars()
        .filter(|ch| *ch != 'ˈ' && *ch != 'ˌ')
        .collect()
}

unsafe extern "C" {
    fn espeak_Initialize(
        output: u32,
        buflength: c_int,
        path: *const c_char,
        options: c_int,
    ) -> c_int;
    fn espeak_TextToPhonemes(
        textptr: *mut *const c_void,
        textmode: c_int,
        phonememode: c_int,
    ) -> *const c_char;
    fn espeak_SetVoiceByName(name: *const c_char) -> c_int;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_language_switch_flags_without_regex() {
        assert_eq!(
            strip_lang_switch_flags("həloʊ(en) marħaba(ar)"),
            "həloʊ marħaba"
        );
    }

    #[test]
    fn strips_stress_markers_without_regex() {
        assert_eq!(strip_stress_markers("hˈɛlˌoʊ"), "hɛloʊ");
    }

    #[test]
    fn normalizes_parent_or_direct_espeak_data_dir() {
        let root = env::temp_dir().join(format!("motlie-espeak-test-{}", std::process::id()));
        let data = root.join(ESPEAKNG_DATA_DIR_NAME);
        fs::create_dir_all(&data).expect("create test data dir");
        fs::write(data.join("phontab"), b"").expect("write phontab");
        fs::write(data.join("en_dict"), b"").expect("write dict");

        assert_eq!(normalize_data_dir(root.clone()), Some(data.clone()));
        assert_eq!(normalize_data_dir(data.clone()), Some(data.clone()));

        fs::remove_dir_all(root).expect("remove test data dir");
    }
}
