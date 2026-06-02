#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MediaTrack {
    Inbound,
    Outbound,
    Both,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CallDirection {
    Inbound,
    Outbound,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CallAction {
    Answer,
    Reject,
    Hangup,
    Transfer { destination: String },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DtmfDigit {
    Digit(u8),
    Star,
    Pound,
}
