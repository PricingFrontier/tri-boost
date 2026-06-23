#![no_main]

use libfuzzer_sys::fuzz_target;
use tri_boost_core::{decode_doc, decode_doc_json, Model};

fuzz_target!(|data: &[u8]| {
    let _ = decode_doc(data);
    if let Ok(s) = std::str::from_utf8(data) {
        let _ = decode_doc_json(s);
        let _ = Model::from_json(s);
    }
});
