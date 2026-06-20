import 'package:flutter/material.dart';
import 'stream_screen.dart';
import 'history_screen.dart';

class LandingScreen extends StatelessWidget {
  const LandingScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Container(
        decoration: const BoxDecoration(
          gradient: LinearGradient(
            colors: [Color(0xFF07111D), Color(0xFF0F2236), Color(0xFF133B48)],
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
          ),
        ),
        child: SafeArea(
          child: Center(
            child: ConstrainedBox(
              constraints: const BoxConstraints(maxWidth: 520),
              child: Padding(
                padding: const EdgeInsets.all(24),
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    const Icon(
                      Icons.visibility,
                      size: 72,
                      color: Color(0xFF56E39F),
                    ),
                    const SizedBox(height: 20),
                    Text(
                      'Anti Sleught Hand TCG',
                      textAlign: TextAlign.center,
                      style: Theme.of(context)
                          .textTheme
                          .headlineMedium
                          ?.copyWith(
                            fontWeight: FontWeight.w800,
                            letterSpacing: 0.4,
                          ),
                    ),
                    const SizedBox(height: 12),
                    Text(
                      'Live camera monitoring with bounding boxes, hand skeleton overlay, and anomaly notifications.',
                      textAlign: TextAlign.center,
                      style: Theme.of(context)
                          .textTheme
                          .bodyLarge
                          ?.copyWith(color: Colors.white70, height: 1.4),
                    ),
                    const SizedBox(height: 36),
                    FilledButton(
                      style: FilledButton.styleFrom(
                        minimumSize: const Size.fromHeight(60),
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(18),
                        ),
                      ),
                      onPressed: () => Navigator.of(context).push(
                        MaterialPageRoute<void>(
                          builder: (_) => const StreamScreen(),
                        ),
                      ),
                      child: const Text('Start monitoring'),
                    ),
                    const SizedBox(height: 12),
                    OutlinedButton.icon(
                      style: OutlinedButton.styleFrom(
                        minimumSize: const Size.fromHeight(48),
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(14),
                        ),
                        side: const BorderSide(color: Colors.white24),
                      ),
                      onPressed: () => Navigator.of(context).push(
                        MaterialPageRoute<void>(
                          builder: (_) => const HistoryScreen(),
                        ),
                      ),
                      icon: const Icon(Icons.history_rounded, size: 18),
                      label: const Text('History'),
                    ),
                    const SizedBox(height: 16),
                    Text(
                      'Bounding boxes and hand skeleton render live. A push notification fires on every anomaly detected. Works in both portrait and landscape.',
                      textAlign: TextAlign.center,
                      style: Theme.of(context)
                          .textTheme
                          .bodySmall
                          ?.copyWith(color: Colors.white54),
                    ),
                  ],
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }
}
