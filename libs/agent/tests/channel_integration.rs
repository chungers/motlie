use std::time::{Duration, SystemTime, UNIX_EPOCH};

use motlie_agent::{
    ChannelConfig, ChannelManager, DeliveryEvent, ManagedMessage, MessageId, MessageSource,
    QuietGuardPolicy, ResolvedSession, SendOptions, SessionKey, SubmitPolicy, UiProfile,
};
use motlie_tmux::transport::MockTransport;
use motlie_tmux::{HostHandle, TargetSpec, TransportKind};

fn submit_policy() -> SubmitPolicy {
    SubmitPolicy {
        prompt_submit: true,
        settle: Duration::ZERO,
        retries: 0,
        retry_delay: Duration::ZERO,
        require_verification: false,
    }
}

#[tokio::test(start_paused = true)]
async fn concurrent_senders_defer_by_stable_session_id_then_dedup_and_coalesce() {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let old = now.saturating_sub(20);
    let recent_client = format!("200 50 renamed $1 {now} 0 /dev/ttys001\n");
    let old_client = format!("200 50 renamed $1 {old} 0 /dev/ttys001\n");
    let mock = MockTransport::new()
        .with_response("list-sessions", "__MOTLIE_S__ old $1 100 0 1  200\n")
        .with_response("list-clients", &recent_client)
        .with_response("list-clients", &old_client)
        .with_response("send-keys", "")
        .with_response("send-keys", "");
    let command_log = mock.command_log();
    let host = HostHandle::new(TransportKind::Mock(mock), None);
    let target = host
        .target(&TargetSpec::session_id("$1").unwrap())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(target.session_name(), "old");
    assert_eq!(target.session_id(), Some("$1"));

    let mut config = ChannelConfig::default();
    config.input_quiet_for = Duration::from_secs(10);
    config.coalesce_window = Duration::from_millis(50);
    config.default_ui_profile = UiProfile::Generic;
    let manager = ChannelManager::new(config);
    let resolved = ResolvedSession::new(
        SessionKey::from_target("local", "local", &target),
        host,
        target,
    );
    let first = manager.get_or_bind(resolved.clone()).unwrap();
    let second = manager.get_or_bind(resolved.clone()).unwrap();
    let third = manager.get_or_bind(resolved).unwrap();
    let mut events = manager.subscribe();
    let options = SendOptions {
        submit: submit_policy(),
        timeout: Duration::from_secs(30),
    };

    let first_task = tokio::spawn(async move {
        first
            .send(
                ManagedMessage::new(MessageSource::human("sender.one"), "same directive"),
                options,
            )
            .await
    });
    let second_task = tokio::spawn(async move {
        second
            .send(
                ManagedMessage::new(MessageSource::human("sender.two"), "same directive"),
                options,
            )
            .await
    });
    let third_task = tokio::spawn(async move {
        third
            .enqueue(
                ManagedMessage::new(MessageSource::broadcast("sender.three"), "other directive"),
                motlie_agent::EnqueueOptions {
                    submit: submit_policy(),
                    quiet_guard: QuietGuardPolicy::Default,
                },
            )
            .await
    });

    tokio::time::advance(Duration::from_millis(50)).await;
    let mut deferred = None;
    for _ in 0..8 {
        match events.recv().await.unwrap() {
            DeliveryEvent::Deferred { message_ids, .. } => {
                deferred = Some(message_ids);
                break;
            }
            DeliveryEvent::Submitted { message_ids, .. } => {
                panic!("submitted before no-barge-in deferred: {message_ids:?}");
            }
            _ => {}
        }
    }
    assert_eq!(deferred, Some(vec![MessageId(1), MessageId(2)]));
    assert!(!command_log
        .lock()
        .unwrap()
        .iter()
        .any(|command| command.contains("send-keys")));

    let third_delivery = third_task.await.unwrap().unwrap();
    assert_eq!(third_delivery.message_id, MessageId(2));

    tokio::time::advance(Duration::from_secs(10)).await;
    tokio::time::advance(Duration::from_millis(50)).await;
    let first_outcome = first_task.await.unwrap().unwrap();
    let second_outcome = second_task.await.unwrap().unwrap();
    assert_eq!(first_outcome.message_id(), MessageId(1));
    assert_eq!(second_outcome.message_id(), MessageId(1));

    let mut saw_coalesced = false;
    let mut saw_submitted = false;
    while !saw_submitted {
        match events.recv().await.unwrap() {
            DeliveryEvent::Coalesced {
                message_ids,
                segment_count,
                ..
            } => {
                assert_eq!(message_ids, vec![MessageId(1), MessageId(2)]);
                assert_eq!(segment_count, 2);
                saw_coalesced = true;
            }
            DeliveryEvent::Submitted { message_ids, .. } => {
                assert_eq!(message_ids, vec![MessageId(1), MessageId(2)]);
                saw_submitted = true;
            }
            _ => {}
        }
    }
    assert!(saw_coalesced);

    let rendered_commands = command_log.lock().unwrap().join("\n");
    assert!(rendered_commands.contains("sender.one, sender.two"));
    assert!(rendered_commands.contains("same directive"));
    assert!(rendered_commands.contains("sender.three"));
    assert!(rendered_commands.contains("other directive"));
}
