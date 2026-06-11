use crate::operator::state::{InboundSubscription, SharedState};

pub async fn ordered_subscribers_for_phone(
    state: &SharedState,
    phone_number: Option<&str>,
) -> Vec<InboundSubscription> {
    let Some(phone_number) = phone_number else {
        return Vec::new();
    };
    state
        .read()
        .await
        .inbound_subscriptions_for_phone(phone_number)
}
