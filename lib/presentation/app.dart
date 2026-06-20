import 'package:flutter/material.dart';
import 'screens/landing_screen.dart';

class AntiSleughtHandApp extends StatelessWidget {
  const AntiSleughtHandApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Anti Sleught Hand TCG',
      theme: ThemeData(
        brightness: Brightness.dark,
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF2A9D8F),
          brightness: Brightness.dark,
        ),
        scaffoldBackgroundColor: const Color(0xFF07111D),
        useMaterial3: true,
      ),
      home: const LandingScreen(),
    );
  }
}
