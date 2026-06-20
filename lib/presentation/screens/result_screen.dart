import 'dart:io';

import 'package:anti_sleught_hand_tcg/domain/enums/dealing_label.dart';
import 'package:flutter/material.dart';
import 'package:path_provider/path_provider.dart';
import 'package:share_plus/share_plus.dart';

import '../../domain/models/detection_capture.dart';

class ResultScreen extends StatelessWidget {
  const ResultScreen({super.key, required this.captures});
  final List<DetectionCapture> captures;

  Future<void> _share(BuildContext context) async {
    if (captures.isEmpty) return;
    await Share.shareXFiles(
      captures.map((c) => XFile(c.rawImagePath)).toList(),
      text: 'Anti Sleught Hand TCG — anomaly captures',
    );
  }

  Future<void> _export(BuildContext context) async {
    final storage = await getApplicationDocumentsDirectory();
    final file = File(
      '${storage.path}${Platform.pathSeparator}anomaly_report.csv',
    );
    final buf = StringBuffer(
      'timestamp,label,confidence,notes,raw_path,processed_path\n',
    );
    for (final c in captures) {
      buf.writeln(
        '${c.timestamp.toIso8601String()},${c.label.title},'
        '${c.confidence.toStringAsFixed(2)},"${c.notes.join(' | ')}",'
        '${c.rawImagePath},${c.processedImagePath ?? ''}',
      );
    }
    await file.writeAsString(buf.toString());
    if (context.mounted) {
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(SnackBar(content: Text('Report saved to ${file.path}')));
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(
          captures.isEmpty
              ? 'No anomalies'
              : '${captures.length} anomaly frame(s)',
        ),
        actions: [
          IconButton(
            onPressed: () => _export(context),
            icon: const Icon(Icons.download_outlined),
            tooltip: 'Export CSV',
          ),
          IconButton(
            onPressed: () => _share(context),
            icon: const Icon(Icons.ios_share_rounded),
            tooltip: 'Share',
          ),
        ],
      ),
      body: captures.isEmpty
          ? const Center(
              child: Padding(
                padding: EdgeInsets.all(24),
                child: Text(
                  'No anomaly frames captured during this session.',
                  textAlign: TextAlign.center,
                ),
              ),
            )
          : ListView.separated(
              padding: const EdgeInsets.all(14),
              itemCount: captures.length,
              separatorBuilder: (_, __) => const SizedBox(height: 12),
              itemBuilder: (context, i) {
                final c = captures[i];
                return Container(
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
                            const SizedBox(height: 10),
                            Wrap(
                              spacing: 6,
                              runSpacing: 6,
                              children: c.notes
                                  .map(
                                    (n) => Chip(
                                      label: Text(
                                        n,
                                        style: const TextStyle(fontSize: 10),
                                      ),
                                      padding: EdgeInsets.zero,
                                      visualDensity: VisualDensity.compact,
                                    ),
                                  )
                                  .toList(),
                            ),
                          ],
                        ),
                      ),
                    ],
                  ),
                );
              },
            ),
    );
  }
}
