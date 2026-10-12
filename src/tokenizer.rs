//! Stable byte-level tokenizer contract shared by the Rust data tools and C++ model.
//! IDs 0..=255 represent UTF-8 bytes; 256 is BOS and 257 is EOS.

pub const BYTE_VOCAB_SIZE: usize = 258;
pub const BOS_ID: u16 = 256;
pub const EOS_ID: u16 = 257;

pub fn encode(text: &str, add_boundaries: bool) -> Vec<u16> {
    let mut ids = Vec::with_capacity(text.len() + usize::from(add_boundaries) * 2);
    if add_boundaries { ids.push(BOS_ID); }
    ids.extend(text.as_bytes().iter().map(|byte| *byte as u16));
    if add_boundaries { ids.push(EOS_ID); }
    ids
}

pub fn decode(ids: &[u16]) -> String {
    let bytes: Vec<u8> = ids.iter().filter_map(|id| {
        if *id <= 255 { Some(*id as u8) } else { None }
    }).collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn round_trips_utf8_bytes() {
        let text = "Clodex — Rust/C++ 世界";
        assert_eq!(decode(&encode(text, false)), text);
    }
    #[test]
    fn boundary_ids_do_not_collide_with_bytes() {
        assert_eq!(encode("A", true), vec![BOS_ID, 65, EOS_ID]);
        assert_eq!(BYTE_VOCAB_SIZE, 258);
    }
}
