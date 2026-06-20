import 'package:hive/hive.dart';
import 'package:path_provider/path_provider.dart';
import '../domain/enums/dealing_label.dart';
import '../domain/models/detection_capture.dart';

class HiveService {
  static const _boxName = 'captures';
  static Box? _box;

  static Future<void> init() async {
    final dir = await getApplicationDocumentsDirectory();
    Hive.init(dir.path);
    _box = await Hive.openBox(_boxName);
  }

  static Box get _b {
    if (_box == null) throw StateError('Hive not initialised');
    return _box!;
  }

  static Future<void> addCapture(DetectionCapture capture) async {
    try {
      final list = _getList();
      list.add(_toMap(capture));
      await _b.put('list', list);
    } catch (_) {}
  }

  static List<DetectionCapture> getCaptures() {
    return _getList().map(_fromMap).toList();
  }

  static Future<void> deleteCapture(DetectionCapture capture) async {
    try {
      final list = _getList();
      list.removeWhere(
        (m) =>
            m['timestamp'] == capture.timestamp.millisecondsSinceEpoch &&
            m['rawImagePath'] == capture.rawImagePath,
      );
      await _b.put('list', list);
    } catch (_) {}
  }

  static List<Map<String, dynamic>> _getList() {
    final raw = _b.get('list', defaultValue: []) as List;
    return raw.map((e) => Map<String, dynamic>.from(e as Map)).toList();
  }

  static Map<String, dynamic> _toMap(DetectionCapture c) => {
    'label': c.label.index,
    'timestamp': c.timestamp.millisecondsSinceEpoch,
    'rawImagePath': c.rawImagePath,
    'processedImagePath': c.processedImagePath,
    'confidence': c.confidence,
    'notes': c.notes,
  };

  static DetectionCapture _fromMap(Map<String, dynamic> m) => DetectionCapture(
    label: DealingLabel.values[m['label'] as int],
    timestamp: DateTime.fromMillisecondsSinceEpoch(m['timestamp'] as int),
    rawImagePath: m['rawImagePath'] as String,
    processedImagePath: m['processedImagePath'] as String?,
    confidence: (m['confidence'] as num).toDouble(),
    notes: List<String>.from(m['notes'] as List),
  );
}
