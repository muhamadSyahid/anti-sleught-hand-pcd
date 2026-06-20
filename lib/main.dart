import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import 'core/notifications.dart';
import 'data/hive_service.dart';
import 'presentation/app.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await initNotifications();
  await HiveService.init();
  await SystemChrome.setPreferredOrientations([
    DeviceOrientation.portraitUp,
    DeviceOrientation.portraitDown,
    DeviceOrientation.landscapeLeft,
    DeviceOrientation.landscapeRight,
  ]);
  runApp(const AntiSleughtHandApp());
}
