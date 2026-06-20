import '../enums/dealing_label.dart';

class DetectionCapture {
  DetectionCapture({
    required this.label,
    required this.timestamp,
    required this.rawImagePath,
    required this.processedImagePath,
    required this.confidence,
    required this.notes,
  });
  final DealingLabel label;
  final DateTime timestamp;
  final String rawImagePath;
  final String? processedImagePath;
  final double confidence;
  final List<String> notes;
}
