import 'dart:io';

import 'package:anti_sleught_hand_tcg/domain/enums/dealing_label.dart';
import 'package:flutter/material.dart';

import '../../data/hive_service.dart';
import '../../domain/models/detection_capture.dart';

class HistoryScreen extends StatefulWidget {
  const HistoryScreen({super.key});

  @override
  State<HistoryScreen> createState() => _HistoryScreenState();
}

class _HistoryScreenState extends State<HistoryScreen> {
  List<DetectionCapture> _captures = [];

  @override
  void initState() {
    super.initState();
    _load();
  }

  void _load() {
    setState(() {
      _captures = HiveService.getCaptures();
    });
  }

  Future<void> _delete(int index) async {
    final c = _captures[index];
    try {
      await File(c.rawImagePath).delete();
    } catch (_) {}
    if (c.processedImagePath != null) {
      try {
        await File(c.processedImagePath!).delete();
      } catch (_) {}
    }
    await HiveService.deleteCapture(c);
    _load();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(
          _captures.isEmpty ? 'History' : 'History (${_captures.length})',
        ),
      ),
      body: _captures.isEmpty
          ? const Center(
              child: Text(
                'No captures in history.',
                style: TextStyle(color: Colors.white54),
              ),
            )
          : ListView.separated(
              padding: const EdgeInsets.all(14),
              itemCount: _captures.length,
              separatorBuilder: (_, __) => const SizedBox(height: 12),
              itemBuilder: (context, i) {
                final c = _captures[i];
                return Stack(
                  children: [
                    Container(
                      decoration: BoxDecoration(
                        color: const Color(0xFF10233A),
                        borderRadius: BorderRadius.circular(18),
                        border: Border.all(color: Colors.white10),
                      ),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          ClipRRect(
                            borderRadius: const BorderRadius.vertical(
                              top: Radius.circular(18),
                            ),
                            child: Image.file(
                              File(c.rawImagePath),
                              height: 200,
                              width: double.infinity,
                              fit: BoxFit.cover,
                            ),
                          ),
                          if (c.processedImagePath != null)
                            Padding(
                              padding: const EdgeInsets.fromLTRB(12, 12, 12, 0),
                              child: ClipRRect(
                                borderRadius: BorderRadius.circular(12),
                                child: Image.file(
                                  File(c.processedImagePath!),
                                  height: 160,
                                  width: double.infinity,
                                  fit: BoxFit.cover,
                                ),
                              ),
                            ),
                          Padding(
                            padding: const EdgeInsets.all(14),
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Row(
                                  children: [
                                    Container(
                                      padding: const EdgeInsets.symmetric(
                                        horizontal: 10,
                                        vertical: 5,
                                      ),
                                      decoration: BoxDecoration(
                                        color: c.label.color.withOpacity(0.15),
                                        borderRadius: BorderRadius.circular(999),
                                        border: Border.all(
                                          color: c.label.color.withOpacity(0.4),
                                        ),
                                      ),
                                      child: Text(
                                        c.label.title,
                                        style: TextStyle(
                                          color: c.label.color,
                                          fontWeight: FontWeight.w700,
                                          fontSize: 12,
                                        ),
                                      ),
                                    ),
                                    const SizedBox(width: 10),
                                    Expanded(
                                      child: Text(
                                        c.timestamp.toLocal().toString(),
                                        style: const TextStyle(
                                          color: Colors.white38,
                                          fontSize: 11,
                                        ),
                                      ),
                                    ),
                                  ],
                                ),
                                const SizedBox(height: 8),
                                Text(
                                  'Confidence: ${(c.confidence * 100).toStringAsFixed(1)}%',
                                  style: const TextStyle(
                                    color: Colors.white70,
                                    fontSize: 12,
                                  ),
                                ),
                                const SizedBox(height: 4),
                                Text(
                                  c.label.subtitle,
                                  style: const TextStyle(
                                    color: Colors.white54,
                                    fontSize: 12,
                                  ),
                                ),
                              ],
                            ),
                          ),
                        ],
                      ),
                    ),
                    Positioned(
                      bottom: 14,
                      right: 14,
                      child: GestureDetector(
                        onTap: () => _delete(i),
                        child: Container(
                          width: 36,
                          height: 36,
                          decoration: BoxDecoration(
                            color: const Color(0xFFFF6B6B).withOpacity(0.9),
                            shape: BoxShape.circle,
                          ),
                          child: const Icon(
                            Icons.delete_outline_rounded,
                            size: 18,
                            color: Colors.white,
                          ),
                        ),
                      ),
                    ),
                  ],
                );
              },
            ),
    );
  }
}
