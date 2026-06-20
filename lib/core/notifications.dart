import 'package:flutter_local_notifications/flutter_local_notifications.dart';
import '../domain/enums/dealing_label.dart';

final FlutterLocalNotificationsPlugin notifications =
    FlutterLocalNotificationsPlugin();

Future<void> initNotifications() async {
  const android = AndroidInitializationSettings('@mipmap/ic_launcher');
  const ios = DarwinInitializationSettings(
    requestAlertPermission: true,
    requestBadgePermission: true,
    requestSoundPermission: true,
  );
  await notifications.initialize(
    settings: const InitializationSettings(android: android, iOS: ios),
  );
  try {
    final androidImpl = notifications
        .resolvePlatformSpecificImplementation<
          AndroidFlutterLocalNotificationsPlugin
        >();
    if (androidImpl != null) {
      final dyn = androidImpl as dynamic;
      try {
        await dyn.requestPermission();
      } catch (_) {
        try {
          await dyn.requestPermissions();
        } catch (_) {
          try {
            await dyn.requestNotificationsPermission();
          } catch (_) {}
        }
      }
    }
  } catch (_) {}
}

Future<void> sendAnomalyNotification(DealingLabel label) async {
  const androidDetails = AndroidNotificationDetails(
    'tcg_anomaly_channel',
    'TCG Anomaly Alerts',
    channelDescription: 'Alerts when a dealing anomaly is detected',
    importance: Importance.high,
    priority: Priority.high,
    playSound: true,
    enableVibration: true,
    color: Color(0xFFFF6B6B),
    icon: '@mipmap/ic_launcher',
  );
  const iosDetails = DarwinNotificationDetails(
    presentAlert: true,
    presentBadge: true,
    presentSound: true,
  );
  await notifications.show(
    id: DateTime.now().millisecondsSinceEpoch ~/ 1000,
    title: '⚠️ ${label.title} detected',
    body: label.subtitle,
    notificationDetails: const NotificationDetails(
      android: androidDetails,
      iOS: iosDetails,
    ),
  );
}
