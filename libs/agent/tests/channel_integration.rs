use std::time::{Duration, SystemTime, UNIX_EPOCH};

use motlie_agent::{
    ChannelConfig, ChannelManager, DeliveryEvent, EnqueueOptions, ManagedMessage, MessageId,
    MessageSource, QuietGuardPolicy, ResolvedSession, SendOptions, SessionKey, SubmitPolicy,
    UiProfile,
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

fn typing_only_policy() -> SubmitPolicy {
    SubmitPolicy {
        prompt_submit: false,
        settle: Duration::ZERO,
        retries: 0,
        retry_delay: Duration::ZERO,
        require_verification: false,
    }
}

#[tokio::test(start_paused = true)]
async fn coalesce_max_wait_forces_drain_under_sustained_input() {
    let mock = MockTransport::new()
        .with_response(
            "list-sessions",
            "__MOTLIE_S__ coalesce-cap $1 100 0 1  200\n",
        )
        .with_response("send-keys", "");
    let command_log = mock.command_log();
    let host = HostHandle::new(TransportKind::Mock(mock), None);
    let target = host
        .target(&TargetSpec::session("coalesce-cap"))
        .await
        .unwrap()
        .unwrap();

    let mut config = ChannelConfig::default();
    config.input_quiet_for = Duration::ZERO;
    config.coalesce_window = Duration::from_millis(50);
    config.coalesce_max_wait = Duration::from_millis(120);
    config.default_ui_profile = UiProfile::Generic;
    let manager = ChannelManager::new(config);
    let channel = manager
        .get_or_bind(ResolvedSession::new(
            SessionKey::from_target("local", "local", &target),
            host,
            target,
        ))
        .unwrap();
    let mut events = manager.subscribe();
    let options = EnqueueOptions {
        submit: typing_only_policy(),
        quiet_guard: QuietGuardPolicy::Default,
    };

    channel
        .enqueue(
            ManagedMessage::new(MessageSource::broadcast("sender.one"), "first"),
            options,
        )
        .await
        .unwrap();
    tokio::task::yield_now().await;

    for (source, body) in [
        ("sender.two", "second"),
        ("sender.three", "third"),
        ("sender.four", "fourth"),
    ] {
        tokio::time::advance(Duration::from_millis(30)).await;
        channel
            .enqueue(
                ManagedMessage::new(MessageSource::broadcast(source), body),
                options,
            )
            .await
            .unwrap();
    }

    assert!(!command_log
        .lock()
        .unwrap()
        .iter()
        .any(|command| command.contains("send-keys")));

    tokio::time::advance(Duration::from_millis(31)).await;

    let mut saw_coalesced = false;
    let mut saw_submitted = false;
    for _ in 0..10 {
        match events.recv().await.unwrap() {
            DeliveryEvent::Coalesced {
                message_ids,
                segment_count,
                ..
            } => {
                assert_eq!(
                    message_ids,
                    vec![MessageId(1), MessageId(2), MessageId(3), MessageId(4)]
                );
                assert_eq!(segment_count, 4);
                saw_coalesced = true;
            }
            DeliveryEvent::Submitted { message_ids, .. } => {
                assert_eq!(
                    message_ids,
                    vec![MessageId(1), MessageId(2), MessageId(3), MessageId(4)]
                );
                saw_submitted = true;
                break;
            }
            _ => {}
        }
    }
    assert!(saw_coalesced);
    assert!(saw_submitted);

    let commands = command_log.lock().unwrap();
    let send_keys = commands
        .iter()
        .filter(|command| command.contains("send-keys"))
        .collect::<Vec<_>>();
    assert_eq!(send_keys.len(), 1);
    let rendered = send_keys[0];
    assert!(rendered.contains("first"));
    assert!(rendered.contains("second"));
    assert!(rendered.contains("third"));
    assert!(rendered.contains("fourth"));
}

#[tokio::test(start_paused = true)]
async fn writable_activity_during_coalesce_redefers_before_payload_delivery() {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let old = now.saturating_sub(20);
    let old_client = format!("200 50 final-guard $1 {old} 0 /dev/pts/48\n");
    let recent_client = format!("200 50 final-guard $1 {now} 0 /dev/pts/48\n");
    let mock = MockTransport::new()
        .with_response(
            "list-sessions",
            "__MOTLIE_S__ final-guard $1 100 0 1  200\n",
        )
        .with_response("list-clients", &old_client)
        .with_response("list-clients", &recent_client)
        .with_response("list-clients", &old_client)
        .with_response("list-clients", &old_client)
        .with_response("send-keys", "");
    let command_log = mock.command_log();
    let host = HostHandle::new(TransportKind::Mock(mock), None);
    let target = host
        .target(&TargetSpec::session("final-guard"))
        .await
        .unwrap()
        .unwrap();

    let mut config = ChannelConfig::default();
    config.input_quiet_for = Duration::from_secs(10);
    config.coalesce_window = Duration::from_millis(50);
    config.coalesce_max_wait = Duration::from_millis(150);
    config.default_ui_profile = UiProfile::Generic;
    let manager = ChannelManager::new(config);
    let channel = manager
        .get_or_bind(ResolvedSession::new(
            SessionKey::from_target("local", "local", &target),
            host,
            target,
        ))
        .unwrap();
    let mut events = manager.subscribe();

    let send_channel = channel.clone();
    let send_task = tokio::spawn(async move {
        send_channel
            .send(
                ManagedMessage::new(MessageSource::human("mstream.send"), "do not interleave"),
                SendOptions {
                    submit: typing_only_policy(),
                    quiet_guard: QuietGuardPolicy::Default,
                    timeout: Duration::from_secs(30),
                },
            )
            .await
    });

    tokio::task::yield_now().await;
    tokio::time::advance(Duration::from_millis(50)).await;

    let mut deferred = None;
    for _ in 0..6 {
        match events.recv().await.unwrap() {
            DeliveryEvent::Deferred { message_ids, .. } => {
                deferred = Some(message_ids);
                break;
            }
            DeliveryEvent::Submitted { message_ids, .. } => {
                panic!("submitted instead of re-deferring after coalesce: {message_ids:?}");
            }
            _ => {}
        }
    }
    assert_eq!(deferred, Some(vec![MessageId(1)]));
    assert!(!command_log
        .lock()
        .unwrap()
        .iter()
        .any(|command| command.contains("send-keys")));

    tokio::time::advance(Duration::from_secs(10)).await;
    tokio::time::advance(Duration::from_millis(50)).await;
    let outcome = send_task.await.unwrap().unwrap();
    assert_eq!(outcome.message_id(), MessageId(1));

    let commands = command_log.lock().unwrap();
    let send_keys = commands
        .iter()
        .filter(|command| command.contains("send-keys"))
        .collect::<Vec<_>>();
    assert_eq!(send_keys.len(), 1);
    assert!(send_keys[0].contains("do not interleave"));
}

#[tokio::test(start_paused = true)]
async fn typing_only_send_defers_when_client_session_id_is_missing_but_name_matches() {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let old = now.saturating_sub(20);
    let recent_client = format!("200 50 smoke-codex  {now} 0 /dev/pts/48\n");
    let old_client = format!("200 50 smoke-codex  {old} 0 /dev/pts/48\n");
    let mock = MockTransport::new()
        .with_response(
            "list-sessions",
            "__MOTLIE_S__ smoke-codex $1 100 0 1  200\n",
        )
        .with_response("list-clients", &recent_client)
        .with_response("list-clients", &recent_client)
        .with_response("list-clients", &old_client)
        .with_response("list-clients", &old_client)
        .with_response("send-keys", "");
    let command_log = mock.command_log();
    let host = HostHandle::new(TransportKind::Mock(mock), None);
    let target = host
        .target(&TargetSpec::session_id("$1").unwrap())
        .await
        .unwrap()
        .unwrap();

    let mut config = ChannelConfig::default();
    config.input_quiet_for = Duration::from_secs(10);
    config.coalesce_window = Duration::from_millis(50);
    let manager = ChannelManager::new(config);
    let channel = manager
        .get_or_bind(ResolvedSession::new(
            SessionKey::from_target("local", "local", &target),
            host,
            target,
        ))
        .unwrap();
    let mut events = manager.subscribe();

    let send_channel = channel.clone();
    let send_task = tokio::spawn(async move {
        send_channel
            .send(
                ManagedMessage::new(MessageSource::human("mstream.send"), "typing only"),
                SendOptions {
                    submit: SubmitPolicy {
                        prompt_submit: false,
                        settle: Duration::ZERO,
                        retries: 0,
                        retry_delay: Duration::ZERO,
                        require_verification: false,
                    },
                    quiet_guard: QuietGuardPolicy::Default,
                    timeout: Duration::from_secs(30),
                },
            )
            .await
    });

    tokio::time::advance(Duration::from_millis(50)).await;
    let mut deferred = None;
    for _ in 0..6 {
        match events.recv().await.unwrap() {
            DeliveryEvent::Deferred { message_ids, .. } => {
                deferred = Some(message_ids);
                break;
            }
            DeliveryEvent::Submitted { message_ids, .. } => {
                panic!(
                    "typing-only payload submitted before quiet guard deferred: {message_ids:?}"
                );
            }
            _ => {}
        }
    }
    assert_eq!(deferred, Some(vec![MessageId(1)]));
    assert!(!command_log
        .lock()
        .unwrap()
        .iter()
        .any(|command| command.contains("send-keys")));

    tokio::time::advance(Duration::from_secs(10)).await;
    tokio::time::advance(Duration::from_millis(50)).await;
    let outcome = send_task.await.unwrap().unwrap();
    assert_eq!(outcome.message_id(), MessageId(1));

    let commands = command_log.lock().unwrap();
    let send_keys = commands
        .iter()
        .filter(|command| command.contains("send-keys"))
        .collect::<Vec<_>>();
    assert_eq!(send_keys.len(), 1);
    assert!(send_keys[0].contains("typing only"));
    assert!(!send_keys[0].contains("Enter"));
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
        quiet_guard: QuietGuardPolicy::Default,
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
