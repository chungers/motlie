pub(crate) fn char_to_byte_index(input: &str, cursor: usize) -> usize {
    input
        .char_indices()
        .nth(cursor)
        .map(|(index, _)| index)
        .unwrap_or(input.len())
}
