import 'dart:async';
import 'dart:io';
import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:path_provider/path_provider.dart';

import '../../core/notifications.dart';
import '../../core/orientation_helper.dart';
import '../../data/camera_image_copy.dart';
import '../../data/hive_service.dart';
import '../../domain/enums/dealing_label.dart';
import '../../domain/models/analysis_result.dart';
import '../../domain/models/detection_capture.dart';
import '../../domain/models/hand_landmarks.dart';
import '../../domain/models/vision_detection.dart';
import '../../pipeline/vision_pipeline.dart';
import '../painters/detection_overlay_painter.dart';
import 'result_screen.dart';

class StreamScreen extends StatefulWidget {
  const StreamScreen({super.key});

  @override
  State<StreamScreen> createState() => _StreamScreenState();
}

class _StreamScreenState extends State<StreamScreen>
    with WidgetsBindingObserver {
  final VisionPipeline _pipeline = VisionPipeline();
  CameraController? _controller;
  Future<void>? _initFuture;

  bool _analysisBusy = false;
  CameraImageCopy? _pendingImage;

  bool _streaming = false;

  List<VisionDetection> _detections = [];
  HandLandmarks? _handLandmarks;
  DealingLabel _currentLabel = DealingLabel.unknown;
  double _currentConfidence = 0.0;
  double _modelW = 640;
  double _modelH = 640;

  List<String> _debugNotes = [];
  bool _showDebug = false;

  int _framesProcessed = 0;
  DateTime _fpsWindowStart = DateTime.now();
  double _fps = 0;

  DeviceOrientation _deviceOrientation = DeviceOrientation.portraitUp;

  final List<DetectionCapture> _captures = [];
  DealingLabel? _lastNotifiedLabel;
  DateTime _lastNotificationTime = DateTime(2000);
  final List<DealingLabel> _labelHistory = [];
  final int _labelHistoryMax = 5;
  final List<double> _posHistory = [];
  final int _posHistoryMax = 6;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _boot();
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _controller?.stopImageStream();
    _pendingImage = null;
    _controller?.dispose();
    super.dispose();
  }

  @override
  void didChangeMetrics() {
    super.didChangeMetrics();
    _syncDeviceOrientation();
  }

  void _syncDeviceOrientation() {
    final ctrl = _controller;
    if (ctrl == null || !ctrl.value.isInitialized) return;
    final newOrientation = ctrl.value.deviceOrientation;
    if (newOrientation != _deviceOrientation && mounted) {
      setState(() => _deviceOrientation = newOrientation);
    }
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
      final ctrl = CameraController(
        cameras.first,
        ResolutionPreset.low,
        enableAudio: false,
        imageFormatGroup: ImageFormatGroup.yuv420,
      );
      _controller = ctrl;
      _initFuture = ctrl.initialize();
      await _initFuture;
      _syncDeviceOrientation();
      if (mounted) setState(() {});
    } catch (e) {
      debugPrint('Camera error: $e');
    }
  }

  int get _quarterTurns {
    final ctrl = _controller;
    if (ctrl == null) return 0;
    return quarterTurnsForOrientation(
      sensorOrientation: ctrl.description.sensorOrientation,
      deviceOrientation: _deviceOrientation,
      isFrontFacing:
          ctrl.description.lensDirection == CameraLensDirection.front,
    );
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

    _framesProcessed = 0;
    _fpsWindowStart = DateTime.now();

    await _controller!.startImageStream(_onCameraImage);
  }

  void _stopStreaming() {
    _controller?.stopImageStream();
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

  void _onCameraImage(CameraImage image) {
    if (!_streaming) return;
    _pendingImage = CameraImageCopy(
      width: image.width,
      height: image.height,
      format: image.format.group,
      planes: image.planes.map((p) => Uint8List.fromList(p.bytes)).toList(),
    );
    if (!_analysisBusy) {
      unawaited(_processFrame());
    }
  }

  Future<void> _processFrame() async {
    if (!_streaming || _analysisBusy || _pendingImage == null) return;
    final imgCopy = _pendingImage!;
    _pendingImage = null;
    _analysisBusy = true;

    final turns = _quarterTurns;

    try {
      final result = await _pipeline.analyzeCameraCopy(
        imgCopy,
        rotationQuarterTurns: turns,
      );

      if (!mounted) return;

      final parsedPos = _parsePosFromNotes(result.notes);
      final smoothedPos = parsedPos == null
          ? null
          : _applyPosSmoothing(parsedPos);
      final smoothedLabel = smoothedPos == null
          ? _applyLabelSmoothing(result.dealingLabel)
          : _labelFromPos(smoothedPos);

      _framesProcessed++;
      final elapsed = DateTime.now().difference(_fpsWindowStart);
      if (elapsed.inMilliseconds >= 1000) {
        _fps = _framesProcessed * 1000 / elapsed.inMilliseconds;
        _framesProcessed = 0;
        _fpsWindowStart = DateTime.now();
      }

      setState(() {
        _detections = result.detections;
        _handLandmarks = result.handLandmarks;
        _currentLabel = smoothedLabel;
        _currentConfidence = result.confidence;
        _modelW = result.modelInputW;
        _modelH = result.modelInputH;
        _debugNotes = result.notes;
      });

      if (smoothedLabel.isAnomaly) {
        await _handleAnomaly(imgCopy, smoothedLabel, result, turns);
      }

      if (result.processedImagePath != null && !result.dealingLabel.isAnomaly) {
        try {
          await File(result.processedImagePath!).delete();
        } catch (_) {}
      }
    } catch (e) {
      debugPrint('Process error: $e');
    } finally {
      _analysisBusy = false;
      if (_streaming && _pendingImage != null) {
        unawaited(_processFrame());
      }
    }
  }

  Future<void> _handleAnomaly(
    CameraImageCopy imgCopy,
    DealingLabel smoothedLabel,
    AnalysisResult result,
    int rotationQuarterTurns,
  ) async {
    var image = VisionPipeline.yuvCopyToImage(imgCopy);
    if (rotationQuarterTurns != 0) {
      image = img.copyRotate(image, angle: rotationQuarterTurns * 90);
    }
    final storage = await getApplicationDocumentsDirectory();
    final dir = Directory('${storage.path}${Platform.pathSeparator}anomalies');
    if (!await dir.exists()) await dir.create(recursive: true);

    final ts = DateTime.now().millisecondsSinceEpoch.toString();
    final rawPath = '${dir.path}${Platform.pathSeparator}anomaly_$ts.jpg';
    await File(rawPath).writeAsBytes(img.encodeJpg(image), flush: true);

    String? procPath;
    try {
      final maskPath = await _pipeline.buildOpenCvMask(rawPath, result.notes);
      if (maskPath != null) {
        procPath = '${dir.path}${Platform.pathSeparator}anomaly_${ts}_cv.png';
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
    await HiveService.addCapture(capture);

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

  double? _parsePosFromNotes(List<String> notes) {
    for (final n in notes) {
      if (n.startsWith('Combined pos:')) {
        final parts = n.split(':');
        if (parts.length > 1) {
          final val = double.tryParse(parts[1].trim());
          if (val != null) return val;
        }
      }
      if (n.startsWith('Card rel pos in deck:')) {
        final parts = n.split(':');
        if (parts.length > 1) {
          final t = parts[1].trim();
          final tokens = t.split(RegExp(r'\s'));
          final val = double.tryParse(tokens[0]);
          if (val != null) return val;
        }
      }
    }
    return null;
  }

  double _applyPosSmoothing(double incoming) {
    _posHistory.add(incoming);
    if (_posHistory.length > _posHistoryMax) _posHistory.removeAt(0);
    final copy = List<double>.from(_posHistory)..sort();
    if (copy.isEmpty) return incoming;
    if (copy.length % 2 == 1) return copy[copy.length ~/ 2];
    final hi = copy[copy.length ~/ 2];
    final lo = copy[copy.length ~/ 2 - 1];
    return (hi + lo) / 2.0;
  }

  DealingLabel _labelFromPos(double pos) {
    if (pos < 0.30) return DealingLabel.normal;
    if (pos < 0.70) return DealingLabel.secondDealing;
    return DealingLabel.bottomDealing;
  }

  @override
  Widget build(BuildContext context) {
    final ready = _controller != null && _controller!.value.isInitialized;
    final isLandscape =
        MediaQuery.of(context).orientation == Orientation.landscape;

    return Scaffold(
      backgroundColor: Colors.black,
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        title: const Text('Live monitor'),
        actions: [
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
            Expanded(
              child: Padding(
                padding: const EdgeInsets.symmetric(horizontal: 12),
                child: ClipRRect(
                  borderRadius: BorderRadius.circular(20),
                  child: Stack(
                    fit: StackFit.expand,
                    children: [
                      Container(color: Colors.black),
                      if (ready)
                        Stack(
                          fit: StackFit.expand,
                          children: [
                            CameraPreview(_controller!),
                            if (_detections.isNotEmpty ||
                                _handLandmarks != null)
                              CustomPaint(
                                painter: DetectionOverlayPainter(
                                  detections: _detections,
                                  handLandmarks: _handLandmarks,
                                  modelInputW: _modelW,
                                  modelInputH: _modelH,
                                  dealingLabel: _currentLabel,
                                ),
                                size: Size.infinite,
                              ),
                          ],
                        ),
                      if (!ready)
                        const Center(
                          child: Text(
                            'Camera preview appears here on a real device.',
                            textAlign: TextAlign.center,
                            style: TextStyle(color: Colors.white54),
                          ),
                        ),
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
                            if (_streaming) ...[
                              const SizedBox(width: 8),
                              _badge(
                                '${_fps.toStringAsFixed(1)} fps',
                                Icons.speed,
                              ),
                            ],
                          ],
                        ),
                      ),
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
                      if (_showDebug && _debugNotes.isNotEmpty)
                        Positioned(
                          bottom: isLandscape ? 24 : 120,
                          left: 14,
                          right: isLandscape ? null : 14,
                          width: isLandscape ? 260 : null,
                          child: Container(
                            padding: const EdgeInsets.all(10),
                            decoration: BoxDecoration(
                              color: Colors.black.withOpacity(0.75),
                              borderRadius: BorderRadius.circular(10),
                              border: Border.all(color: Colors.white12),
                            ),
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Text(
                                  'Rotation: $_quarterTurns × 90° · ${_deviceOrientation.name}',
                                  style: const TextStyle(
                                    color: Colors.white38,
                                    fontSize: 9,
                                  ),
                                ),
                                const SizedBox(height: 4),
                                ..._debugNotes.map(
                                  (n) => Text(
                                    n,
                                    style: const TextStyle(
                                      color: Color(0xFF56E39F),
                                      fontSize: 10,
                                      fontFamily: 'monospace',
                                    ),
                                  ),
                                ),
                              ],
                            ),
                          ),
                        ),
                      Positioned(
                        left: 0,
                        right: 0,
                        bottom: isLandscape ? 12 : 24,
                        child: Column(
                          children: [
                            GestureDetector(
                              onTap: _toggleStreaming,
                              child: Container(
                                width: isLandscape ? 64 : 88,
                                height: isLandscape ? 64 : 88,
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
                                  size: isLandscape ? 32 : 44,
                                  color: Colors.black,
                                ),
                              ),
                            ),
                            if (!isLandscape) ...[
                              const SizedBox(height: 8),
                              Text(
                                _streaming ? 'Stop' : 'Start',
                                style: Theme.of(context).textTheme.labelLarge
                                    ?.copyWith(color: Colors.white70),
                              ),
                            ],
                          ],
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ),
            if (!isLandscape)
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
