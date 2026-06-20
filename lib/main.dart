import 'dart:async';
import 'dart:io';
import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_local_notifications/flutter_local_notifications.dart';
import 'package:hive/hive.dart';
import 'package:image/image.dart' as img;
import 'package:opencv_dart/opencv.dart' as cv;
import 'package:path_provider/path_provider.dart';
import 'package:share_plus/share_plus.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

// ─────────────────────────────────────────────────────────────
// Notifications
// ─────────────────────────────────────────────────────────────

final FlutterLocalNotificationsPlugin _notifications =
    FlutterLocalNotificationsPlugin();

Future<void> initNotifications() async {
  const android = AndroidInitializationSettings('@mipmap/ic_launcher');
  const ios = DarwinInitializationSettings(
    requestAlertPermission: true,
    requestBadgePermission: true,
    requestSoundPermission: true,
  );
  await _notifications.initialize(
    settings: const InitializationSettings(android: android, iOS: ios),
  );
  try {
    final androidImpl = _notifications
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
  await _notifications.show(
    id: DateTime.now().millisecondsSinceEpoch ~/ 1000,
    title: '⚠️ ${label.title} detected',
    body: label.subtitle,
    notificationDetails: const NotificationDetails(
      android: androidDetails,
      iOS: iosDetails,
    ),
  );
}

// ─────────────────────────────────────────────────────────────
// Entry point
// ─────────────────────────────────────────────────────────────

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await initNotifications();
  await _HiveService.init();
  runApp(const AntiSleughtHandApp());
}

// ─────────────────────────────────────────────────────────────
// Data models
// ─────────────────────────────────────────────────────────────

enum DealingLabel { normal, secondDealing, bottomDealing, unknown }

extension DealingLabelX on DealingLabel {
  String get title => switch (this) {
    DealingLabel.normal => 'Normal',
    DealingLabel.secondDealing => 'Second dealing',
    DealingLabel.bottomDealing => 'Bottom dealing',
    DealingLabel.unknown => 'Unknown',
  };

  String get subtitle => switch (this) {
    DealingLabel.normal => 'Top card taken as expected',
    DealingLabel.secondDealing => 'Second card from top taken',
    DealingLabel.bottomDealing => 'Bottom card taken from deck',
    DealingLabel.unknown => 'Waiting for detection…',
  };

  Color get color => switch (this) {
    DealingLabel.normal => const Color(0xFF56E39F),
    DealingLabel.secondDealing => const Color(0xFFFFC857),
    DealingLabel.bottomDealing => const Color(0xFFFF6B6B),
    DealingLabel.unknown => const Color(0xFF9FB3C8),
  };

  bool get isAnomaly =>
      this == DealingLabel.secondDealing || this == DealingLabel.bottomDealing;
}

class VisionDetection {
  VisionDetection({
    required this.label,
    required this.confidence,
    required this.box,
  });
  final String label;
  final double confidence;
  final Rect box; // always in model-input-pixel space (e.g. 640×640)
}

/// 21 MediaPipe hand landmark points, each normalised 0–1
class HandLandmarks {
  HandLandmarks(this.points);
  final List<Offset> points; // length == 21, values in [0,1]
}

class AnalysisResult {
  AnalysisResult({
    required this.dealingLabel,
    required this.confidence,
    required this.notes,
    required this.detections,
    required this.handLandmarks,
    required this.processedImagePath,
    required this.modelInputW,
    required this.modelInputH,
  });
  final DealingLabel dealingLabel;
  final double confidence;
  final List<String> notes;
  final List<VisionDetection> detections;
  final HandLandmarks? handLandmarks;
  final String? processedImagePath;
  // The pixel dimensions of the space the YOLO boxes live in.
  // The overlay painter needs these to scale boxes to screen coords.
  final double modelInputW;
  final double modelInputH;
}

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

// ─────────────────────────────────────────────────────────────
// Hive persistence service
// ─────────────────────────────────────────────────────────────

class _HiveService {
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
    return List<Map<String, dynamic>>.from(_b.get('list', defaultValue: []));
  }

  static Map<String, dynamic> _toMap(DetectionCapture c) => {
        'label': c.label.index,
        'timestamp': c.timestamp.millisecondsSinceEpoch,
        'rawImagePath': c.rawImagePath,
        'processedImagePath': c.processedImagePath,
        'confidence': c.confidence,
        'notes': c.notes,
      };

  static DetectionCapture _fromMap(Map<String, dynamic> m) =>
      DetectionCapture(
        label: DealingLabel.values[m['label'] as int],
        timestamp:
            DateTime.fromMillisecondsSinceEpoch(m['timestamp'] as int),
        rawImagePath: m['rawImagePath'] as String,
        processedImagePath: m['processedImagePath'] as String?,
        confidence: (m['confidence'] as num).toDouble(),
        notes: List<String>.from(m['notes'] as List),
    );
  }

class _PlaneData {
  const _PlaneData({required this.bytes, required this.stride});
  final Uint8List bytes;
  final int stride;
}

// ─────────────────────────────────────────────────────────────
// MediaPipe hand skeleton connections
// ─────────────────────────────────────────────────────────────

const _handConnections = [
  [0, 1],
  [1, 2],
  [2, 3],
  [3, 4],
  [0, 5],
  [5, 6],
  [6, 7],
  [7, 8],
  [0, 9],
  [9, 10],
  [10, 11],
  [11, 12],
  [0, 13],
  [13, 14],
  [14, 15],
  [15, 16],
  [0, 17],
  [17, 18],
  [18, 19],
  [19, 20],
  [5, 9],
  [9, 13],
  [13, 17],
];

// ─────────────────────────────────────────────────────────────
// Vision pipeline
// ─────────────────────────────────────────────────────────────

class VisionPipeline {
  Interpreter? _yoloInterpreter;
  Interpreter? _poseInterpreter;
  List<String> _labels = const ['hand', 'card', 'deck'];

  // Model input dimensions — populated after interpreter is loaded
  int _yoloW = 640;
  int _yoloH = 640;

  String modelStatus = 'Not initialised';

  /// Analyse a decoded [img.Image] directly (no file I/O).
  Future<AnalysisResult> analyzeImage(img.Image image) async {
    final notes = <String>[];
    final detections = await _runYoloOnImage(image, notes);
    HandLandmarks? landmarks;
    if (detections.isNotEmpty) {
      landmarks = await _runHandLandmarkOnImage(image, notes);
    } else {
      notes.add('No detections — skipping hand landmark');
      landmarks = null;
    }
    final label = _classifyDeal(detections, landmarks, notes);
    final confidence = detections.isEmpty
        ? 0.0
        : detections.map((d) => d.confidence).reduce((a, b) => a > b ? a : b);
    return AnalysisResult(
      dealingLabel: label,
      confidence: confidence,
      notes: notes,
      detections: detections,
      handLandmarks: landmarks,
      processedImagePath: null,
      modelInputW: _yoloW.toDouble(),
      modelInputH: _yoloH.toDouble(),
    );
  }

  /// Analyse a [CameraImage] from the live camera stream.
  Future<AnalysisResult> analyzeCameraImage(CameraImage cameraImage) async {
    final image = _cameraImageToImage(cameraImage);
    if (image == null) {
      return AnalysisResult(
        dealingLabel: DealingLabel.unknown,
        confidence: 0.0,
        notes: ['Failed to decode camera frame'],
        detections: [],
        handLandmarks: null,
        processedImagePath: null,
        modelInputW: _yoloW.toDouble(),
        modelInputH: _yoloH.toDouble(),
      );
    }
    return analyzeImage(image);
  }

  /// Converts a [CameraImage] (YUV420 or BGRA8888) to [img.Image].
  static img.Image? _cameraImageToImage(CameraImage image) {
    final w = image.width;
    final h = image.height;
    final result = img.Image(width: w, height: h);
    final planes = image.planes;

    if (image.format.group == ImageFormatGroup.bgra8888) {
      _fillFromBgra(result, planes[0].bytes, planes[0].bytesPerRow);
      return result;
    }

    if (image.format.group == ImageFormatGroup.yuv420) {
      final pd = [
        _PlaneData(bytes: planes[0].bytes, stride: planes[0].bytesPerRow),
        _PlaneData(bytes: planes[1].bytes, stride: planes[1].bytesPerRow),
        if (planes.length > 2)
          _PlaneData(bytes: planes[2].bytes, stride: planes[2].bytesPerRow),
      ];
      _fillFromYuv(result, pd, w, h);
      return result;
    }

    return null;
  }

  /// Same conversion as [_cameraImageToImage] but from our safe copy.
  static img.Image _yuvCopyToImage(_CameraImageCopy copy) {
    final w = copy.width;
    final h = copy.height;
    final result = img.Image(width: w, height: h);

    if (copy.format == ImageFormatGroup.bgra8888) {
      _fillFromBgra(result, copy.planes[0], w * 4);
      return result;
    }

    final planes = [
      _PlaneData(bytes: copy.planes[0], stride: w),
      _PlaneData(bytes: copy.planes[1], stride: w >> 1),
      if (copy.planes.length > 2)
        _PlaneData(bytes: copy.planes[2], stride: w >> 1),
    ];
    _fillFromYuv(result, planes, w, h);
    return result;
  }

  static void _fillFromBgra(img.Image dst, Uint8List bytes, int stride) {
    final w = dst.width;
    final h = dst.height;
    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
        final i = y * stride + x * 4;
        dst.setPixelRgba(x, y, bytes[i + 2], bytes[i + 1], bytes[i], 255);
      }
    }
  }

  static void _fillFromYuv(
    img.Image dst,
    List<_PlaneData> planes,
    int w,
    int h,
  ) {
    final yStride = planes[0].stride;
    final uStride = planes[1].stride;
    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
        final yVal = planes[0].bytes[y * yStride + x];
        final uvx = x >> 1;
        final uvy = y >> 1;
        final uvi = uvy * uStride + uvx;
        final uVal = planes[1].bytes[uvi] - 128;
        final vVal = planes[2].bytes[uvi] - 128;
        dst.setPixelRgba(
          x, y,
          (yVal + 1.402 * vVal).clamp(0, 255).round(),
          (yVal - 0.344 * uVal - 0.714 * vVal).clamp(0, 255).round(),
          (yVal + 1.772 * uVal).clamp(0, 255).round(),
          255,
        );
      }
    }
  }
  String poseStatus = 'Not initialised';

  Future<void> initialize() async {
    // Labels
    try {
      final text = await rootBundle.loadString('assets/models/labels.txt');
      final parsed = text
          .split(RegExp(r'\r?\n'))
          .map((l) => l.trim())
          .where((l) => l.isNotEmpty)
          .toList();
      if (parsed.isNotEmpty) _labels = parsed;
    } catch (_) {
      _labels = const ['card', 'deck', 'hand'];
    }

    // YOLOv8 TFLite with hardware acceleration
    _yoloInterpreter = await _loadInterpreter(
      assetPath: 'assets/models/yolov8_tcg.tflite',
      onLoaded: (interp) {
        final shape = interp.getInputTensor(0).shape;
        _yoloH = shape[1];
        _yoloW = shape[2];
        modelStatus = 'YOLOv8 loaded ($_yoloW×$_yoloH)';
      },
      onError: (e) => modelStatus = 'YOLOv8 missing: $e',
    );

    // MediaPipe hand landmark with hardware acceleration
    _poseInterpreter = await _loadInterpreter(
      assetPath: 'assets/models/hand_landmark.tflite',
      onLoaded: (_) => poseStatus = 'Hand pose loaded',
      onError: (e) => poseStatus = 'Hand pose missing: $e',
    );
  }

  Future<Interpreter?> _loadInterpreter({
    required String assetPath,
    required void Function(Interpreter) onLoaded,
    required void Function(Object) onError,
  }) async {
    // Attempt 1: GPU delegate + XNNPack + multi-thread
    try {
      final opts = InterpreterOptions()..threads = 4;
      try {
        opts.addDelegate(GpuDelegateV2());
      } catch (_) {}
      try {
        opts.addDelegate(XNNPackDelegate());
      } catch (_) {}
      final interp = await Interpreter.fromAsset(assetPath, options: opts);
      onLoaded(interp);
      return interp;
    } catch (_) {}

    // Attempt 2: XNNPack + multi-thread (no GPU)
    try {
      final opts = InterpreterOptions()..threads = 4;
      try {
        opts.addDelegate(XNNPackDelegate());
      } catch (_) {}
      final interp = await Interpreter.fromAsset(assetPath, options: opts);
      onLoaded(interp);
      return interp;
    } catch (_) {}

    // Attempt 3: multi-thread CPU only
    try {
      final opts = InterpreterOptions()..threads = 4;
      final interp = await Interpreter.fromAsset(assetPath, options: opts);
      onLoaded(interp);
      return interp;
    } catch (_) {}

    // Attempt 4: plain (no options)
    try {
      final interp = await Interpreter.fromAsset(assetPath);
      onLoaded(interp);
      return interp;
    } catch (e) {
      onError(e);
      return null;
    }
  }

  Future<AnalysisResult> analyzeImagePath(String imagePath) async {
    final notes = <String>[];

    // Run YOLO first (cheaper than running all stages). Only run the
    // hand-landmark model if YOLO found relevant objects to reduce work.
    final detections = await _runYolo(imagePath, notes);
    HandLandmarks? landmarks;
    if (detections.isNotEmpty) {
      landmarks = await _runHandLandmark(imagePath, notes);
    } else {
      notes.add('No detections — skipping hand landmark');
      landmarks = null;
    }

    final label = _classifyDeal(detections, landmarks, notes);

    // Only create the OpenCV processed image when an anomaly is detected
    // (we previously did this every frame which caused heavy IO).
    String? processedPath;
    if (label.isAnomaly) {
      processedPath = await _buildOpenCvMask(imagePath, notes);
    } else {
      processedPath = null;
    }

    final confidence = detections.isEmpty
        ? 0.0
        : detections.map((d) => d.confidence).reduce((a, b) => a > b ? a : b);

    return AnalysisResult(
      dealingLabel: label,
      confidence: confidence,
      notes: notes,
      detections: detections,
      handLandmarks: landmarks,
      processedImagePath: processedPath,
      modelInputW: _yoloW.toDouble(),
      modelInputH: _yoloH.toDouble(),
    );
  }

  // ── OpenCV threshold mask ────────────────────────────────

  Future<String?> _buildOpenCvMask(String imagePath, List<String> notes) async {
    try {
      final bytes = await File(imagePath).readAsBytes();
      final src = cv.imdecode(bytes, cv.IMREAD_COLOR);
      final gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY);
      final (_, thresh) = cv.threshold(gray, 120, 255, cv.THRESH_BINARY);
      final encoded = cv.imencode('.png', thresh).$2;
      src.dispose();
      gray.dispose();
      thresh.dispose();

      final dir = await getTemporaryDirectory();
      final path =
          '${dir.path}${Platform.pathSeparator}cv_${DateTime.now().millisecondsSinceEpoch}.png';
      await File(path).writeAsBytes(encoded, flush: true);
      notes.add('OpenCV mask OK');
      return path;
    } catch (e) {
      notes.add('OpenCV failed: $e');
      return null;
    }
  }

  // ── YOLOv8 TFLite inference ──────────────────────────────
  //
  // YOLOv8 exported to TFLite with default settings produces one of two
  // output tensor shapes:
  //
  //   [1, num_boxes, 4 + num_classes]   ← "boxes first"  (nms=False)
  //   [1, 4+num_classes, num_boxes]     ← "features first" (transposed)
  //
  // Each row is:  [cx, cy, w, h, class0_score, class1_score, ...]
  // There is NO separate objectness column in YOLOv8 — that was YOLOv5.
  // Coordinates are in model-input pixels (0..inputW, 0..inputH) or
  // normalised (0..1) depending on export options; we handle both.

  Future<List<VisionDetection>> _runYolo(
    String imagePath,
    List<String> notes,
  ) async {
    try {
      final imageBytes = await File(imagePath).readAsBytes();
      final decoded = img.decodeImage(imageBytes);
      if (decoded == null) return const [];
      return _runYoloOnImage(decoded, notes);
    } catch (e, st) {
      notes.add('YOLO error: $e');
      debugPrint('YOLO error: $e\n$st');
      return const [];
    }
  }

  Future<List<VisionDetection>> _runYoloOnImage(
    img.Image image,
    List<String> notes,
  ) async {
    final interp = _yoloInterpreter;
    if (interp == null) {
      notes.add('YOLO skipped — no model');
      return const [];
    }

    try {
      // Resize to model input size
      final resized = img.copyResize(
        image,
        width: _yoloW,
        height: _yoloH,
        interpolation: img.Interpolation.linear,
      );

      // Build [1, H, W, 3] float32 input, values 0–1
      final input = List.generate(
        1,
        (_) => List.generate(
          _yoloH,
          (y) => List.generate(_yoloW, (x) {
            final p = resized.getPixel(x, y);
            return [p.r / 255.0, p.g / 255.0, p.b / 255.0];
          }),
        ),
      );

      final outShape = interp.getOutputTensor(0).shape;
      notes.add('YOLO out shape: $outShape');

      if (outShape.length != 3) {
        notes.add('Unsupported output shape');
        return const [];
      }

      // Determine layout
      // [1, N, F] where F = 4+classes  →  boxes first
      // [1, F, N] where F = 4+classes  →  features first (transposed)
      final dim1 = outShape[1];
      final dim2 = outShape[2];
      final numClasses = _labels.length;
      final expectedFeats = 4 + numClasses;

      // If dim2 == expectedFeats → boxes first
      // If dim1 == expectedFeats → features first
      // Fallback: whichever dim is smaller is the feature dim
      final bool boxesFirst;
      if (dim2 == expectedFeats) {
        boxesFirst = true;
      } else if (dim1 == expectedFeats) {
        boxesFirst = false;
      } else {
        // Fallback heuristic: more boxes than features
        boxesFirst = dim1 > dim2;
      }

      final boxCount = boxesFirst ? dim1 : dim2;
      final featCount = boxesFirst ? dim2 : dim1;

      // Allocate output buffer
      final output = boxesFirst
          ? [
              List.generate(
                boxCount,
                (_) => List<double>.filled(featCount, 0.0),
              ),
            ]
          : [
              List.generate(
                featCount,
                (_) => List<double>.filled(boxCount, 0.0),
              ),
            ];

      interp.run(input, output);

      final detections = <VisionDetection>[];
      const confThreshold = 0.25;

      for (var i = 0; i < boxCount; i++) {
        // Extract the feature vector for box i
        final List<double> row;
        if (boxesFirst) {
          row = output[0][i];
        } else {
          row = List.generate(
            featCount,
            (fi) => (output[0][fi][i] as num).toDouble(),
          );
        }

        if (row.length < 4 + numClasses) continue;

        // YOLOv8 layout: [cx, cy, w, h, c0, c1, ..., cN]
        // No objectness score — class scores ARE the confidence
        double bestScore = 0.0;
        int bestIdx = 0;
        for (var ci = 0; ci < numClasses; ci++) {
          final score = row[4 + ci];
          if (score > bestScore) {
            bestScore = score;
            bestIdx = ci;
          }
        }

        if (bestScore < confThreshold) continue;

        // Coordinates: determine if normalised (0..1) or pixel (0..W/H)
        final cx = row[0];
        final cy = row[1];
        final bw = row[2];
        final bh = row[3];

        final bool isNorm =
            cx <= 1.01 && cy <= 1.01 && bw <= 1.01 && bh <= 1.01;
        final double px = isNorm ? cx * _yoloW : cx;
        final double py = isNorm ? cy * _yoloH : cy;
        final double pw = isNorm ? bw * _yoloW : bw;
        final double ph = isNorm ? bh * _yoloH : bh;

        // Convert centre-format to LTWH
        final box = Rect.fromLTWH(px - pw / 2, py - ph / 2, pw, ph);

        final label = bestIdx < _labels.length ? _labels[bestIdx] : 'unknown';
        detections.add(
          VisionDetection(label: label, confidence: bestScore, box: box),
        );
      }

      // Simple greedy NMS per class to remove duplicates
      final filtered = _nms(detections, iouThreshold: 0.45);
      notes.add(
        'YOLO detections: ${filtered.length} (raw: ${detections.length})',
      );
      return filtered;
    } catch (e, st) {
      notes.add('YOLO error: $e');
      debugPrint('YOLO error: $e\n$st');
      return const [];
    }
  }

  /// Greedy NMS — keeps the highest-confidence box per class when boxes overlap
  List<VisionDetection> _nms(
    List<VisionDetection> dets, {
    double iouThreshold = 0.45,
  }) {
    final byLabel = <String, List<VisionDetection>>{};
    for (final d in dets) {
      byLabel.putIfAbsent(d.label, () => []).add(d);
    }

    final result = <VisionDetection>[];
    for (final group in byLabel.values) {
      group.sort((a, b) => b.confidence.compareTo(a.confidence));
      final kept = <VisionDetection>[];
      for (final d in group) {
        bool suppressed = false;
        for (final k in kept) {
          if (_iou(d.box, k.box) > iouThreshold) {
            suppressed = true;
            break;
          }
        }
        if (!suppressed) kept.add(d);
      }
      result.addAll(kept);
    }
    return result;
  }

  double _iou(Rect a, Rect b) {
    final inter = a.intersect(b);
    if (inter.isEmpty) return 0.0;
    final interArea = inter.width * inter.height;
    final unionArea = a.width * a.height + b.width * b.height - interArea;
    return unionArea <= 0 ? 0.0 : interArea / unionArea;
  }

  // ── MediaPipe hand landmark inference ───────────────────
  // Input:  [1, 224, 224, 3]  float32  values 0–1
  // Output: [1, 63]           float32  x,y,z per landmark, values 0–1

  Future<HandLandmarks?> _runHandLandmark(
    String imagePath,
    List<String> notes,
  ) async {
    try {
      final imageBytes = await File(imagePath).readAsBytes();
      final decoded = img.decodeImage(imageBytes);
      if (decoded == null) return null;
      return _runHandLandmarkOnImage(decoded, notes);
    } catch (e) {
      notes.add('Hand landmark error: $e');
      return null;
    }
  }

  Future<HandLandmarks?> _runHandLandmarkOnImage(
    img.Image image,
    List<String> notes,
  ) async {
    final interp = _poseInterpreter;
    if (interp == null) return null;

    try {
      const sz = 224;
      final resized = img.copyResize(image, width: sz, height: sz);

      final input = List.generate(
        1,
        (_) => List.generate(
          sz,
          (y) => List.generate(sz, (x) {
            final p = resized.getPixel(x, y);
            return [p.r / 255.0, p.g / 255.0, p.b / 255.0];
          }),
        ),
      );

      final output = [List<double>.filled(63, 0.0)];
      interp.run(input, output);

      final raw = output[0];
      final points = List.generate(
        21,
        (i) => Offset(raw[i * 3], raw[i * 3 + 1]),
      );
      notes.add('Hand landmarks: 21 pts');
      return HandLandmarks(points);
    } catch (e) {
      notes.add('Hand landmark error: $e');
      return null;
    }
  }

  // ── Dealing classification ────────────────────────────────
  //
  // Strategy: find the deck bounding box, find the card being dealt,
  // then compute where on the deck the card originates.
  //
  // All coordinates are in YOLO model-input space (0..W × 0..H).
  // Hand landmarks are normalised 0–1, so we multiply by _yoloW/_yoloH
  // to bring them into the same space before any geometry.

  DealingLabel _classifyDeal(
    List<VisionDetection> detections,
    HandLandmarks? landmarks,
    List<String> notes,
  ) {
    final cards = detections.where((d) => d.label == 'card').toList();
    final decks = detections.where((d) => d.label == 'deck').toList();
    final hands = detections.where((d) => d.label == 'hand').toList();

    notes.add(
      'Labels found: ${detections.map((d) => d.label).toSet().join(', ')}',
    );

    // ── Need at least a card + deck to classify ─────────────
    if (cards.isEmpty || decks.isEmpty) {
      // If we can see a hand but no deck/card, stay unknown
      notes.add('Need card + deck — returning unknown');
      return DealingLabel.unknown;
    }

    // Pick highest-confidence deck and card
    final deck = decks.reduce((a, b) => a.confidence >= b.confidence ? a : b);
    final card = cards.reduce((a, b) => a.confidence >= b.confidence ? a : b);

    final deckTop = deck.box.top;
    final deckBottom = deck.box.bottom;
    final deckHeight = deckBottom - deckTop;

    if (deckHeight < 1) {
      notes.add('Deck box too small');
      return DealingLabel.unknown;
    }

    // Where on the deck (vertically) does the card's centre sit?
    // 0.0 = top of deck, 1.0 = bottom of deck
    final cardCenterY = card.box.center.dy;
    final relPos = (cardCenterY - deckTop) / deckHeight;

    notes.add('Card rel pos in deck: ${relPos.toStringAsFixed(3)}');

    // ── Refine using hand landmark wrist position ────────────
    // Convert landmarks from [0,1] to YOLO pixel space so we can
    // compare them with the YOLO bounding boxes directly.
    if (landmarks != null) {
      // Landmark 0 = wrist, landmark 9 = middle-finger MCP (palm base)
      final wristY = landmarks.points[0].dy * _yoloH;
      final palmBaseY = landmarks.points[9].dy * _yoloH;
      // Thumb tip (4) and index tip (8) — fingertips doing the dealing
      final thumbTipY = landmarks.points[4].dy * _yoloH;
      final indexTipY = landmarks.points[8].dy * _yoloH;
      final gripY = (thumbTipY + indexTipY) / 2; // where fingers contact card

      // gripY relative to deck height
      final gripRelPos = (gripY - deckTop) / deckHeight;
      notes.add('Grip rel pos: ${gripRelPos.toStringAsFixed(3)}');
      notes.add(
        'Wrist Y: ${wristY.toStringAsFixed(1)}, palmBase Y: ${palmBaseY.toStringAsFixed(1)}',
      );

      // Use the average of card position and grip position for robustness
      final combinedPos = (relPos + gripRelPos) / 2;
      notes.add('Combined pos: ${combinedPos.toStringAsFixed(3)}');

      return _labelFromRelPos(combinedPos, notes);
    }

    // ── Fallback: hand bounding box ──────────────────────────
    if (hands.isNotEmpty) {
      final hand = hands.reduce((a, b) => a.confidence >= b.confidence ? a : b);
      final handCenterY = hand.box.center.dy;
      final handRelPos = (handCenterY - deckTop) / deckHeight;
      notes.add('Hand rel pos: ${handRelPos.toStringAsFixed(3)}');

      // Combine card position with hand position
      final combinedPos = (relPos * 0.6 + handRelPos * 0.4);
      return _labelFromRelPos(combinedPos, notes);
    }

    // ── Last resort: card position only ─────────────────────
    return _labelFromRelPos(relPos, notes);
  }

  DealingLabel _labelFromRelPos(double pos, List<String> notes) {
    // pos < 0 means card is above the deck entirely (taking from top) → normal
    // pos 0.0–0.30  → normal deal (top ~30% of deck)
    // pos 0.30–0.70 → second deal (middle portion)
    // pos > 0.70    → bottom deal (bottom 30%)
    if (pos < 0.30) {
      notes.add('→ NORMAL (top of deck)');
      return DealingLabel.normal;
    } else if (pos < 0.70) {
      notes.add('→ SECOND DEALING');
      return DealingLabel.secondDealing;
    } else {
      notes.add('→ BOTTOM DEALING');
      return DealingLabel.bottomDealing;
    }
  }
}

// ─────────────────────────────────────────────────────────────
// Overlay painter
// ─────────────────────────────────────────────────────────────

class DetectionOverlayPainter extends CustomPainter {
  DetectionOverlayPainter({
    required this.detections,
    required this.handLandmarks,
    required this.modelInputW,
    required this.modelInputH,
    required this.dealingLabel,
  });

  final List<VisionDetection> detections;
  final HandLandmarks? handLandmarks;
  final double modelInputW;
  final double modelInputH;
  final DealingLabel dealingLabel;

  static const _labelColors = {
    'hand': Color(0xFF56E39F),
    'card': Color(0xFF5BC8FF),
    'deck': Color(0xFFFFC857),
  };

  @override
  void paint(Canvas canvas, Size size) {
    if (modelInputW <= 0 || modelInputH <= 0) return;

    final scaleX = size.width / modelInputW;
    final scaleY = size.height / modelInputH;

    for (final det in detections) {
      final color = _labelColors[det.label] ?? Colors.white;

      final scaled = Rect.fromLTWH(
        det.box.left * scaleX,
        det.box.top * scaleY,
        det.box.width * scaleX,
        det.box.height * scaleY,
      );

      // Box outline
      canvas.drawRRect(
        RRect.fromRectAndRadius(scaled, const Radius.circular(6)),
        Paint()
          ..color = color
          ..style = PaintingStyle.stroke
          ..strokeWidth = 2.0,
      );

      // Corner accents
      _drawCorners(canvas, scaled, color);

      // Label pill
      _drawPill(canvas, scaled, det.label, det.confidence, color);
    }

    // Hand skeleton — landmarks are 0–1, scale to canvas size
    final lm = handLandmarks;
    if (lm != null) {
      final bonePaint = Paint()
        ..color = const Color(0xFFE0E0FF).withOpacity(0.85)
        ..strokeWidth = 1.8
        ..strokeCap = StrokeCap.round;

      for (final conn in _handConnections) {
        final a = lm.points[conn[0]];
        final b = lm.points[conn[1]];
        canvas.drawLine(
          Offset(a.dx * size.width, a.dy * size.height),
          Offset(b.dx * size.width, b.dy * size.height),
          bonePaint,
        );
      }

      const tipIndices = {4, 8, 12, 16, 20};
      for (var i = 0; i < 21; i++) {
        final pt = lm.points[i];
        final px = Offset(pt.dx * size.width, pt.dy * size.height);
        final isTip = tipIndices.contains(i);
        canvas.drawCircle(
          px,
          isTip ? 5.0 : 3.5,
          Paint()..color = Colors.black.withOpacity(0.4),
        );
        canvas.drawCircle(
          px,
          isTip ? 4.0 : 2.5,
          Paint()..color = isTip ? const Color(0xFF56E39F) : Colors.white,
        );
      }
    }
  }

  void _drawCorners(Canvas canvas, Rect r, Color color) {
    final p = Paint()
      ..color = color
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3.5
      ..strokeCap = StrokeCap.round;
    const c = 14.0;
    // Top-left
    canvas.drawLine(r.topLeft, r.topLeft + const Offset(c, 0), p);
    canvas.drawLine(r.topLeft, r.topLeft + const Offset(0, c), p);
    // Top-right
    canvas.drawLine(r.topRight, r.topRight + const Offset(-c, 0), p);
    canvas.drawLine(r.topRight, r.topRight + const Offset(0, c), p);
    // Bottom-left
    canvas.drawLine(r.bottomLeft, r.bottomLeft + const Offset(c, 0), p);
    canvas.drawLine(r.bottomLeft, r.bottomLeft + const Offset(0, -c), p);
    // Bottom-right
    canvas.drawLine(r.bottomRight, r.bottomRight + const Offset(-c, 0), p);
    canvas.drawLine(r.bottomRight, r.bottomRight + const Offset(0, -c), p);
  }

  void _drawPill(
    Canvas canvas,
    Rect box,
    String label,
    double conf,
    Color color,
  ) {
    final text = '$label  ${(conf * 100).toStringAsFixed(0)}%';
    final tp = TextPainter(
      text: TextSpan(
        text: text,
        style: const TextStyle(
          color: Colors.black,
          fontSize: 11,
          fontWeight: FontWeight.w700,
          letterSpacing: 0.3,
        ),
      ),
      textDirection: TextDirection.ltr,
    )..layout();

    final pillW = tp.width + 12;
    const pillH = 20.0;
    final pillX = box.left;
    final pillY = box.top - pillH - 2;

    canvas.drawRRect(
      RRect.fromRectAndRadius(
        Rect.fromLTWH(pillX, pillY, pillW, pillH),
        const Radius.circular(4),
      ),
      Paint()..color = color,
    );
    tp.paint(canvas, Offset(pillX + 6, pillY + 4));
  }

  @override
  bool shouldRepaint(DetectionOverlayPainter old) =>
      old.detections != detections ||
      old.handLandmarks != handLandmarks ||
      old.dealingLabel != dealingLabel;
}

// ─────────────────────────────────────────────────────────────
// App
// ─────────────────────────────────────────────────────────────

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

// ─────────────────────────────────────────────────────────────
// Landing screen
// ─────────────────────────────────────────────────────────────

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
                      style: Theme.of(context).textTheme.headlineMedium
                          ?.copyWith(
                            fontWeight: FontWeight.w800,
                            letterSpacing: 0.4,
                          ),
                    ),
                    const SizedBox(height: 12),
                    Text(
                      'Live camera monitoring with bounding boxes, hand skeleton overlay, and anomaly notifications.',
                      textAlign: TextAlign.center,
                      style: Theme.of(context).textTheme.bodyLarge?.copyWith(
                        color: Colors.white70,
                        height: 1.4,
                      ),
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
                      'Bounding boxes and hand skeleton render live. A push notification fires on every anomaly detected.',
                      textAlign: TextAlign.center,
                      style: Theme.of(
                        context,
                      ).textTheme.bodySmall?.copyWith(color: Colors.white54),
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

// Holds a safe copy of a CameraImage frame so the platform buffer can be
// recycled without losing the pixel data.
class _CameraImageCopy {
  _CameraImageCopy({
    required this.width,
    required this.height,
    required this.format,
    required this.planes,
  });
  final int width;
  final int height;
  final ImageFormatGroup format;
  final List<Uint8List> planes;
}

// ─────────────────────────────────────────────────────────────
// Stream screen
// ─────────────────────────────────────────────────────────────

class StreamScreen extends StatefulWidget {
  const StreamScreen({super.key});

  @override
  State<StreamScreen> createState() => _StreamScreenState();
}

class _StreamScreenState extends State<StreamScreen> {
  final VisionPipeline _pipeline = VisionPipeline();
  CameraController? _controller;
  Future<void>? _initFuture;

  // Live camera stream — raw frame bytes copied in the callback so the
  // platform buffer can be recycled immediately.
  // Analysis processes the latest frame when idle
  Timer? _analysisTimer;
  bool _analysisBusy = false;
  _CameraImageCopy? _pendingImage;

  bool _streaming = false;

  List<VisionDetection> _detections = [];
  HandLandmarks? _handLandmarks;
  DealingLabel _currentLabel = DealingLabel.unknown;
  double _currentConfidence = 0.0;
  double _modelW = 640;
  double _modelH = 640;

  // Debug notes from latest frame — shown in a collapsible panel
  List<String> _debugNotes = [];
  bool _showDebug = false;

  final List<DetectionCapture> _captures = [];
  DealingLabel? _lastNotifiedLabel;
  DateTime _lastNotificationTime = DateTime(2000);
  // Recent label history for simple temporal smoothing
  final List<DealingLabel> _labelHistory = [];
  final int _labelHistoryMax = 5;
  // Recent numeric positions (combinedPos) for smoothing
  final List<double> _posHistory = [];
  final int _posHistoryMax = 6;

  @override
  void initState() {
    super.initState();
    _boot();
  }

  @override
  void dispose() {
    _controller?.stopImageStream();
    _analysisTimer?.cancel();
    _pendingImage = null;
    _controller?.dispose();
    super.dispose();
  }

  Future<void> _boot() async {
    await _pipeline.initialize();
    await _prepareCamera();
    if (mounted) setState(() {});
  }

  Future<void> _prepareCamera() async {
    if (kIsWeb || !(Platform.isAndroid || Platform.isIOS || Platform.isMacOS)) {
      return;
    }
    try {
      final cameras = await availableCameras();
      if (cameras.isEmpty) return;
      // Use a lower resolution to reduce per-frame processing cost.
      final ctrl = CameraController(
        cameras.first,
        ResolutionPreset.medium,
        enableAudio: false,
        imageFormatGroup: ImageFormatGroup.yuv420,
      );
      _controller = ctrl;
      _initFuture = ctrl.initialize();
      await _initFuture;
      if (mounted) setState(() {});
    } catch (e) {
      debugPrint('Camera error: $e');
    }
  }

  Future<void> _toggleStreaming() async {
    if (_streaming) {
      _stopStreaming();
      return;
    }
    if (_controller == null) await _prepareCamera();
    if (_initFuture != null) await _initFuture;
    if (!mounted) return;
    setState(() => _streaming = true);

    // Listen to the live camera preview stream. The callback copies raw
    // bytes so the platform buffer can be recycled immediately.
    await _controller!.startImageStream(_onCameraImage);

    // Poll for new frames every 50ms after analysis finishes.
    _analysisTimer?.cancel();
    _analysisTimer = Timer.periodic(
      const Duration(milliseconds: 50),
      (_) => _processFrame(),
    );
  }

  void _stopStreaming() {
    _controller?.stopImageStream();
    _analysisTimer?.cancel();
    _analysisTimer = null;
    _pendingImage = null;
    _labelHistory.clear();
    _posHistory.clear();
    if (!mounted) return;
    setState(() {
      _streaming = false;
      _detections = [];
      _handLandmarks = null;
      _currentLabel = DealingLabel.unknown;
    });
    Navigator.of(context).push(
      MaterialPageRoute<void>(
        builder: (_) =>
            ResultScreen(captures: List<DetectionCapture>.from(_captures)),
      ),
    );
  }

  /// Called by the camera plugin for every preview frame.  We copy the raw
  /// byte planes so the native buffer can be recycled straight away, then the
  /// analysis loop picks it up when it is ready.
  void _onCameraImage(CameraImage image) {
    if (_analysisBusy || !_streaming) return;
    _pendingImage = _CameraImageCopy(
      width: image.width,
      height: image.height,
      format: image.format.group,
      planes: image.planes.map((p) => Uint8List.fromList(p.bytes)).toList(),
    );
  }

  Future<void> _processFrame() async {
    if (!_streaming || _analysisBusy || _pendingImage == null) return;
    final imgCopy = _pendingImage!;
    _pendingImage = null;
    _analysisBusy = true;

    try {
      final image = VisionPipeline._yuvCopyToImage(imgCopy);

      final result = await _pipeline.analyzeImage(image);

      if (!mounted) return;

      // Parse numeric positions from notes (prefer combined pos when present)
      final parsedPos = _parsePosFromNotes(result.notes);
      final smoothedPos = parsedPos == null
          ? null
          : _applyPosSmoothing(parsedPos);
      final smoothedLabel = smoothedPos == null
          ? _applyLabelSmoothing(result.dealingLabel)
          : _labelFromPos(smoothedPos);

      setState(() {
        _detections = result.detections;
        _handLandmarks = result.handLandmarks;
        _currentLabel = smoothedLabel;
        _currentConfidence = result.confidence;
        _modelW = result.modelInputW;
        _modelH = result.modelInputH;
        _debugNotes = result.notes;
      });

      // Anomaly handling
      if (smoothedLabel.isAnomaly) {
        final storage = await getApplicationDocumentsDirectory();
        final dir = Directory(
          '${storage.path}${Platform.pathSeparator}anomalies',
        );
        if (!await dir.exists()) await dir.create(recursive: true);

        final ts = DateTime.now().millisecondsSinceEpoch.toString();
        final rawPath = '${dir.path}${Platform.pathSeparator}anomaly_$ts.jpg';
        await File(rawPath).writeAsBytes(img.encodeJpg(image), flush: true);

        String? procPath;
        try {
          final maskPath = await _pipeline._buildOpenCvMask(
            rawPath, result.notes,
          );
          if (maskPath != null) {
            procPath =
                '${dir.path}${Platform.pathSeparator}anomaly_${ts}_cv.png';
            await File(maskPath).copy(procPath);
          }
        } catch (_) {}

        final capture = DetectionCapture(
          label: smoothedLabel,
          timestamp: DateTime.now(),
          rawImagePath: rawPath,
          processedImagePath: procPath,
          confidence: result.confidence,
          notes: result.notes,
        );

        if (mounted) {
          setState(() {
            _captures.add(capture);
          });
        }
        await _HiveService.addCapture(capture);

        // Debounced notification
        final now = DateTime.now();
        if (_lastNotifiedLabel != smoothedLabel ||
            now.difference(_lastNotificationTime).inSeconds >= 3) {
          _lastNotifiedLabel = smoothedLabel;
          _lastNotificationTime = now;
          await sendAnomalyNotification(smoothedLabel);
        }

        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
              content: Text(
                '⚠️ ${smoothedLabel.title} — ${smoothedLabel.subtitle}',
              ),
              backgroundColor: smoothedLabel.color,
              behavior: SnackBarBehavior.floating,
              duration: const Duration(seconds: 2),
            ),
          );
        }
      }

      // Cleanup temp files
      if (result.processedImagePath != null && !result.dealingLabel.isAnomaly) {
        try {
          await File(result.processedImagePath!).delete();
        } catch (_) {}
      }
    } catch (e) {
      debugPrint('Process error: $e');
    } finally {
      _analysisBusy = false;
    }
  }

  // Simple majority-vote smoothing helper for DealingLabel
  DealingLabel _applyLabelSmoothing(DealingLabel incoming) {
    _labelHistory.add(incoming);
    if (_labelHistory.length > _labelHistoryMax) _labelHistory.removeAt(0);
    final counts = <DealingLabel, int>{};
    for (final l in _labelHistory) {
      counts[l] = (counts[l] ?? 0) + 1;
    }
    DealingLabel best = _labelHistory.last;
    var bestCount = 0;
    counts.forEach((k, v) {
      if (v > bestCount) {
        best = k;
        bestCount = v;
      }
    });
    return best;
  }

  // Parse a numeric combined position from VisionPipeline notes if present.
  double? _parsePosFromNotes(List<String> notes) {
    for (final n in notes) {
      if (n.startsWith('Combined pos:')) {
        final parts = n.split(':');
        if (parts.length > 1) {
          final val = double.tryParse(parts[1].trim());
          if (val != null) return val;
        }
      }
      // Fallback: card rel pos
      if (n.startsWith('Card rel pos in deck:')) {
        final parts = n.split(':');
        if (parts.length > 1) {
          final t = parts[1].trim();
          final tokens = t.split(RegExp(r'\s')); // maybe has fixed format
          final val = double.tryParse(tokens[0]);
          if (val != null) return val;
        }
      }
    }
    return null;
  }

  // Smooth combined position values with simple moving average
  double _applyPosSmoothing(double incoming) {
    _posHistory.add(incoming);
    if (_posHistory.length > _posHistoryMax) _posHistory.removeAt(0);
    // Use median to be robust to outliers
    final copy = List<double>.from(_posHistory)..sort();
    if (copy.isEmpty) return incoming;
    if (copy.length % 2 == 1) return copy[copy.length ~/ 2];
    final hi = copy[copy.length ~/ 2];
    final lo = copy[copy.length ~/ 2 - 1];
    return (hi + lo) / 2.0;
  }

  // Map numeric relative position to DealingLabel using same thresholds
  DealingLabel _labelFromPos(double pos) {
    if (pos < 0.30) return DealingLabel.normal;
    if (pos < 0.70) return DealingLabel.secondDealing;
    return DealingLabel.bottomDealing;
  }

  @override
  Widget build(BuildContext context) {
    final ready = _controller != null && _controller!.value.isInitialized;

    return Scaffold(
      backgroundColor: Colors.black,
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        title: const Text('Live monitor'),
        actions: [
          // Debug toggle
          IconButton(
            icon: Icon(
              Icons.bug_report_outlined,
              color: _showDebug ? const Color(0xFF56E39F) : Colors.white38,
            ),
            onPressed: () => setState(() => _showDebug = !_showDebug),
            tooltip: 'Debug notes',
          ),
          if (_captures.isNotEmpty)
            Padding(
              padding: const EdgeInsets.only(right: 12),
              child: Center(
                child: Container(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 10,
                    vertical: 4,
                  ),
                  decoration: BoxDecoration(
                    color: const Color(0xFFFF6B6B).withOpacity(0.2),
                    borderRadius: BorderRadius.circular(999),
                  ),
                  child: Text(
                    '${_captures.length} anomaly',
                    style: const TextStyle(
                      color: Color(0xFFFF6B6B),
                      fontSize: 12,
                    ),
                  ),
                ),
              ),
            ),
        ],
      ),
      body: SafeArea(
        child: Column(
          children: [
            // Camera + overlay
            Expanded(
              child: Padding(
                padding: const EdgeInsets.symmetric(horizontal: 12),
                child: ClipRRect(
                  borderRadius: BorderRadius.circular(20),
                  child: Stack(
                    fit: StackFit.expand,
                    children: [
                      Container(color: Colors.black),
                      if (ready) CameraPreview(_controller!),
                      if (!ready)
                        const Center(
                          child: Text(
                            'Camera preview appears here on a real device.',
                            textAlign: TextAlign.center,
                            style: TextStyle(color: Colors.white54),
                          ),
                        ),
                      // Detection overlay
                      if (_detections.isNotEmpty || _handLandmarks != null)
                        CustomPaint(
                          painter: DetectionOverlayPainter(
                            detections: _detections,
                            handLandmarks: _handLandmarks,
                            modelInputW: _modelW,
                            modelInputH: _modelH,
                            dealingLabel: _currentLabel,
                          ),
                        ),
                      // Status badges
                      Positioned(
                        top: 14,
                        left: 14,
                        right: 14,
                        child: Row(
                          children: [
                            _badge('OpenCV', Icons.filter_alt_outlined),
                            const SizedBox(width: 8),
                            _badge('YOLOv8', Icons.center_focus_strong),
                            const SizedBox(width: 8),
                            _badge('Skeleton', Icons.accessibility_new),
                          ],
                        ),
                      ),
                      // Detection banner
                      if (_streaming && _currentLabel != DealingLabel.unknown)
                        Positioned(
                          top: 60,
                          left: 14,
                          right: 14,
                          child: _DetectionBanner(
                            label: _currentLabel,
                            confidence: _currentConfidence,
                          ),
                        ),
                      // Debug notes panel
                      if (_showDebug && _debugNotes.isNotEmpty)
                        Positioned(
                          bottom: 120,
                          left: 14,
                          right: 14,
                          child: Container(
                            padding: const EdgeInsets.all(10),
                            decoration: BoxDecoration(
                              color: Colors.black.withOpacity(0.75),
                              borderRadius: BorderRadius.circular(10),
                              border: Border.all(color: Colors.white12),
                            ),
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: _debugNotes
                                  .map(
                                    (n) => Text(
                                      n,
                                      style: const TextStyle(
                                        color: Color(0xFF56E39F),
                                        fontSize: 10,
                                        fontFamily: 'monospace',
                                      ),
                                    ),
                                  )
                                  .toList(),
                            ),
                          ),
                        ),
                      // Start / stop button
                      Positioned(
                        left: 0,
                        right: 0,
                        bottom: 24,
                        child: Column(
                          children: [
                            GestureDetector(
                              onTap: _toggleStreaming,
                              child: Container(
                                width: 88,
                                height: 88,
                                decoration: BoxDecoration(
                                  color: _streaming
                                      ? const Color(0xFFFF6B6B)
                                      : const Color(0xFF56E39F),
                                  shape: BoxShape.circle,
                                  boxShadow: [
                                    BoxShadow(
                                      color:
                                          (_streaming
                                                  ? const Color(0xFFFF6B6B)
                                                  : const Color(0xFF56E39F))
                                              .withOpacity(0.4),
                                      blurRadius: 24,
                                      spreadRadius: 2,
                                    ),
                                  ],
                                ),
                                child: Icon(
                                  _streaming
                                      ? Icons.stop_rounded
                                      : Icons.play_arrow_rounded,
                                  size: 44,
                                  color: Colors.black,
                                ),
                              ),
                            ),
                            const SizedBox(height: 8),
                            Text(
                              _streaming ? 'Stop' : 'Start',
                              style: Theme.of(context).textTheme.labelLarge
                                  ?.copyWith(color: Colors.white70),
                            ),
                          ],
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ),
            // Status tiles
            Padding(
              padding: const EdgeInsets.fromLTRB(12, 10, 12, 14),
              child: Row(
                children: [
                  Expanded(
                    child: _InfoTile(
                      title: 'Status',
                      value: _streaming ? 'Monitoring' : 'Ready',
                    ),
                  ),
                  const SizedBox(width: 10),
                  Expanded(
                    child: _InfoTile(
                      title: 'Detections',
                      value: _streaming ? '${_detections.length} obj' : '—',
                    ),
                  ),
                  const SizedBox(width: 10),
                  Expanded(
                    child: _InfoTile(
                      title: 'Anomalies',
                      value: _captures.isEmpty
                          ? 'None yet'
                          : '${_captures.length} found',
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _badge(String label, IconData icon) {
    return Expanded(
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 6),
        decoration: BoxDecoration(
          color: Colors.black.withOpacity(0.45),
          borderRadius: BorderRadius.circular(999),
          border: Border.all(color: Colors.white12),
        ),
        child: Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(icon, size: 14, color: Colors.white60),
            const SizedBox(width: 5),
            Text(
              label,
              style: const TextStyle(fontSize: 11, color: Colors.white60),
            ),
          ],
        ),
      ),
    );
  }
}

// ─────────────────────────────────────────────────────────────
// Detection banner
// ─────────────────────────────────────────────────────────────

class _DetectionBanner extends StatelessWidget {
  const _DetectionBanner({required this.label, required this.confidence});
  final DealingLabel label;
  final double confidence;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
      decoration: BoxDecoration(
        color: label.color.withOpacity(0.18),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: label.color.withOpacity(0.5)),
      ),
      child: Row(
        children: [
          Icon(
            label.isAnomaly
                ? Icons.warning_amber_rounded
                : Icons.check_circle_outline,
            color: label.color,
            size: 18,
          ),
          const SizedBox(width: 8),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  label.title,
                  style: TextStyle(
                    color: label.color,
                    fontWeight: FontWeight.w700,
                    fontSize: 13,
                  ),
                ),
                Text(
                  label.subtitle,
                  style: const TextStyle(color: Colors.white60, fontSize: 11),
                ),
              ],
            ),
          ),
          Text(
            '${(confidence * 100).toStringAsFixed(0)}%',
            style: TextStyle(
              color: label.color,
              fontWeight: FontWeight.w700,
              fontSize: 13,
            ),
          ),
        ],
      ),
    );
  }
}

// ─────────────────────────────────────────────────────────────
// Info tile
// ─────────────────────────────────────────────────────────────

class _InfoTile extends StatelessWidget {
  const _InfoTile({required this.title, required this.value});
  final String title;
  final String value;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: const Color(0xFF10233A),
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: Colors.white10),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            title,
            style: Theme.of(
              context,
            ).textTheme.labelSmall?.copyWith(color: Colors.white54),
          ),
          const SizedBox(height: 4),
          Text(
            value,
            style: Theme.of(
              context,
            ).textTheme.bodySmall?.copyWith(fontWeight: FontWeight.w700),
          ),
        ],
      ),
    );
  }
}

// ─────────────────────────────────────────────────────────────
// Result screen
// ─────────────────────────────────────────────────────────────

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

// ─────────────────────────────────────────────────────────────
// History screen
// ─────────────────────────────────────────────────────────────

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
      _captures = _HiveService.getCaptures();
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
    await _HiveService.deleteCapture(c);
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
                              padding:
                                  const EdgeInsets.fromLTRB(12, 12, 12, 0),
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
                                        color:
                                            c.label.color.withOpacity(0.15),
                                        borderRadius:
                                            BorderRadius.circular(999),
                                        border: Border.all(
                                          color:
                                              c.label.color.withOpacity(0.4),
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
