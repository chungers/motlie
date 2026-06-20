//! Compatibility alias: canonical Telnyx text-call protocol definitions live in `motlie_agent::voice::telnyx::text` (#460).
//! Gateway-internal imports may continue using this module path.

pub use motlie_agent::voice::telnyx::text::{
    AcceptCallResponse, AgentTextFrame, CallConnectedPayload, CallOfferPayload, CallerSpeechState,
    DebugTextStreamFrame, GatewayTextFrame, PlaybackFinishedStatus, TextCallAggregationPolicy,
    TextCallDirection, TextCallInfo, TextCallMetadata, TextStreamDescriptor,
    TEXT_CALL_CONTENT_TYPE, TEXT_CALL_DEBUG_EXTENSION, TEXT_CALL_EARLY_TURNS_EXTENSION,
    TEXT_CALL_PARTIALS_EXTENSION, TEXT_CALL_PROTOCOL,
};
