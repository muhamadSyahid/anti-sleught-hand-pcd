import 'dart:async';
import 'dart:io';
import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:opencv_dart/opencv.dart' as cv;
import 'package:path_provider/path_provider.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

import '../core/constants.dart';
import '../data/camera_image_copy.dart';
import '../domain/enums/dealing_label.dart';
import '../domain/models/analysis_result.dart';
import '../domain/models/hand_landmarks.dart';
import '../domain/models/vision_detection.dart';
import 'plane_data.dart';

class VisionPipeline {
  Interpreter? _yoloInterpreter;
  Interpreter? _poseInterpreter;
  List<String> _labels = const ['card', 'deck', 'hand'];

  int _yoloW = 640;
  int _yoloH = 640;

  // Dimensions of the actual source frame (post-rotation, pre-resize) that
  // was last fed into YOLO. Detection boxes are rescaled into this space
  // before being returned, so the overlay painter can scale them onto the
  // camera preview using the *source frame's* aspect ratio rather than the
  // model's square 640x640 input — which is what was causing misaligned
  // boxes whenever the source frame wasn't already square.
  double _lastSourceW = 640;
  double _lastSourceH = 640;

  String modelStatus = 'Not initialised';
  String poseStatus = 'Not initialised';

  bool _yoloBusy = false;
  bool _poseBusy = false;

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
      modelInputW: _lastSourceW,
      modelInputH: _lastSourceH,
    );
  }

  Future<AnalysisResult> analyzeCameraCopy(
    CameraImageCopy copy, {
    int rotationQuarterTurns = 0,
  }) async {
    final notes = <String>[];

    // Use the simple, proven YUV->Image->resize path for both YOLO and hand
    // landmark instead of the separate fast float32 conversion path. This
    // trades a bit of speed for correctness while we confirm whether the
    // fast path's rotation/stride math was producing garbage input.
    var fullImage = yuvCopyToImage(copy);
    if (rotationQuarterTurns != 0) {
      fullImage = img.copyRotate(fullImage, angle: rotationQuarterTurns * 90);
    }

    final detections = await _runYoloOnImage(fullImage, notes);
    HandLandmarks? landmarks;
    if (detections.isNotEmpty) {
      landmarks = await _runHandLandmarkOnImage(fullImage, notes);
    } else {
      notes.add('No detections — skipping hand landmark');
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
      modelInputW: _lastSourceW,
      modelInputH: _lastSourceH,
    );
  }

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
        PlaneData(bytes: planes[0].bytes, stride: planes[0].bytesPerRow),
        PlaneData(bytes: planes[1].bytes, stride: planes[1].bytesPerRow),
        if (planes.length > 2)
          PlaneData(bytes: planes[2].bytes, stride: planes[2].bytesPerRow),
      ];
      _fillFromYuv(result, pd, w, h);
      return result;
    }

    return null;
  }

  static img.Image yuvCopyToImage(CameraImageCopy copy) {
    final w = copy.width;
    final h = copy.height;
    final result = img.Image(width: w, height: h);

    if (copy.format == ImageFormatGroup.bgra8888) {
      _fillFromBgra(result, copy.planes[0], w * 4);
      return result;
    }

    final planes = [
      PlaneData(bytes: copy.planes[0], stride: w),
      PlaneData(bytes: copy.planes[1], stride: w >> 1),
      if (copy.planes.length > 2)
        PlaneData(bytes: copy.planes[2], stride: w >> 1),
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
    List<PlaneData> planes,
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
          x,
          y,
          (yVal + 1.402 * vVal).clamp(0, 255).round(),
          (yVal - 0.344 * uVal - 0.714 * vVal).clamp(0, 255).round(),
          (yVal + 1.772 * uVal).clamp(0, 255).round(),
          255,
        );
      }
    }
  }

  static void _yuvCopyToFloat32(
    CameraImageCopy copy,
    Float32List out,
    int destW,
    int destH, {
    int rotationQuarterTurns = 0,
  }) {
    final srcW = copy.width;
    final srcH = copy.height;
    final turns = rotationQuarterTurns & 3;

    final rotatedW = (turns == 1 || turns == 3) ? srcH : srcW;
    final rotatedH = (turns == 1 || turns == 3) ? srcW : srcH;

    int srcXFor(int rx, int ry) {
      switch (turns) {
        case 1:
          return ry;
        case 2:
          return srcW - 1 - rx;
        case 3:
          return srcH - 1 - ry;
        default:
          return rx;
      }
    }

    int srcYFor(int rx, int ry) {
      switch (turns) {
        case 1:
          return srcW - 1 - rx;
        case 2:
          return srcH - 1 - ry;
        case 3:
          return rx;
        default:
          return ry;
      }
    }

    if (copy.format == ImageFormatGroup.bgra8888) {
      final bytes = copy.planes[0];
      final stride = srcW * 4;
      var o = 0;
      for (int dy = 0; dy < destH; dy++) {
        final ry = (dy * rotatedH / destH).floor().clamp(0, rotatedH - 1);
        for (int dx = 0; dx < destW; dx++) {
          final rx = (dx * rotatedW / destW).floor().clamp(0, rotatedW - 1);
          final sx = srcXFor(rx, ry).clamp(0, srcW - 1);
          final sy = srcYFor(rx, ry).clamp(0, srcH - 1);
          final i = sy * stride + sx * 4;
          out[o++] = bytes[i + 2] / 255.0;
          out[o++] = bytes[i + 1] / 255.0;
          out[o++] = bytes[i] / 255.0;
        }
      }
      return;
    }

    final yPlane = copy.planes[0];
    final uPlane = copy.planes[1];
    final vPlane = copy.planes.length > 2 ? copy.planes[2] : copy.planes[1];
    final yStride = srcW;
    final uStride = srcW >> 1;

    var o = 0;
    for (int dy = 0; dy < destH; dy++) {
      final ry = (dy * rotatedH / destH).floor().clamp(0, rotatedH - 1);
      for (int dx = 0; dx < destW; dx++) {
        final rx = (dx * rotatedW / destW).floor().clamp(0, rotatedW - 1);
        final sx = srcXFor(rx, ry).clamp(0, srcW - 1);
        final sy = srcYFor(rx, ry).clamp(0, srcH - 1);

        final yVal = yPlane[sy * yStride + sx];
        final uvx = sx >> 1;
        final uvy = sy >> 1;
        final uvi = uvy * uStride + uvx;
        final uVal = uPlane[uvi] - 128;
        final vVal = vPlane[uvi] - 128;

        final r = (yVal + 1.402 * vVal).clamp(0, 255);
        final g = (yVal - 0.344 * uVal - 0.714 * vVal).clamp(0, 255);
        final b = (yVal + 1.772 * uVal).clamp(0, 255);

        out[o++] = r / 255.0;
        out[o++] = g / 255.0;
        out[o++] = b / 255.0;
      }
    }
  }

  Future<void> initialize() async {
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

    _poseInterpreter = await _loadInterpreter(
      assetPath: 'assets/models/hand_landmark.tflite',
      onLoaded: (_) {
        poseStatus = 'Hand pose loaded';
      },
      onError: (e) => poseStatus = 'Hand pose missing: $e',
    );
  }

  Future<Interpreter?> _loadInterpreter({
    required String assetPath,
    required void Function(Interpreter) onLoaded,
    required void Function(Object) onError,
  }) async {
    try {
      final opts = InterpreterOptions()..threads = 4;
      try {
        opts.addDelegate(GpuDelegateV2());
      } catch (_) {}
      try {
        opts.addDelegate(XNNPackDelegate());
      } catch (_) {}
      final interp = await Interpreter.fromAsset(assetPath, options: opts);
      // Some delegates (GPU/XNNPack) require tensors to be explicitly
      // (re)allocated before the first invoke(), otherwise run() can throw
      // "Bad state: failed precondition" on the very first call.
      interp.allocateTensors();
      onLoaded(interp);
      return interp;
    } catch (_) {}

    try {
      final opts = InterpreterOptions()..threads = 4;
      try {
        opts.addDelegate(XNNPackDelegate());
      } catch (_) {}
      final interp = await Interpreter.fromAsset(assetPath, options: opts);
      interp.allocateTensors();
      onLoaded(interp);
      return interp;
    } catch (_) {}

    try {
      final opts = InterpreterOptions()..threads = 4;
      final interp = await Interpreter.fromAsset(assetPath, options: opts);
      interp.allocateTensors();
      onLoaded(interp);
      return interp;
    } catch (_) {}

    try {
      final interp = await Interpreter.fromAsset(assetPath);
      interp.allocateTensors();
      onLoaded(interp);
      return interp;
    } catch (e) {
      onError(e);
      return null;
    }
  }

  Future<AnalysisResult> analyzeImagePath(String imagePath) async {
    final notes = <String>[];

    final detections = await _runYolo(imagePath, notes);
    HandLandmarks? landmarks;
    if (detections.isNotEmpty) {
      landmarks = await _runHandLandmark(imagePath, notes);
    } else {
      notes.add('No detections — skipping hand landmark');
      landmarks = null;
    }

    final label = _classifyDeal(detections, landmarks, notes);

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
      modelInputW: _lastSourceW,
      modelInputH: _lastSourceH,
    );
  }

  Future<String?> buildOpenCvMask(String imagePath, List<String> notes) async {
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

  Future<String?> _buildOpenCvMask(String imagePath, List<String> notes) async {
    return buildOpenCvMask(imagePath, notes);
  }

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

    // If a previous frame's inference is still running, skip this one
    // instead of letting two calls race on the same interpreter/tensors.
    if (_yoloBusy) {
      notes.add('YOLO skipped — previous inference still running');
      return const [];
    }
    _yoloBusy = true;

    try {
      // Remember the *source* frame's dimensions (before the square
      // resize). The model always sees a stretched 640x640 image, but the
      // camera preview shows the original aspect ratio — boxes need to be
      // un-stretched back into this space so they line up on screen.
      _lastSourceW = image.width.toDouble();
      _lastSourceH = image.height.toDouble();

      final resized = img.copyResize(
        image,
        width: _yoloW,
        height: _yoloH,
        interpolation: img.Interpolation.linear,
      );

      // Build a genuine nested [1, H, W, 3] list (not a flat typed buffer
      // reshaped after the fact). tflite_flutter's native marshalling can
      // silently produce all-zero output for some plugin/device
      // combinations when fed a reshaped Float32List instead of real
      // nested Dart lists, even though it doesn't throw.
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

      final detections = _runYoloOnNestedInput(input, interp, notes);
      return _rescaleDetectionsToSource(detections);
    } catch (e, st) {
      notes.add('YOLO error: $e');
      debugPrint('YOLO error: $e\n$st');
      return const [];
    } finally {
      _yoloBusy = false;
    }
  }

  /// Rescales detection boxes from the model's square 640x640 input space
  /// back into the original (non-square) source frame's pixel space. The
  /// resize into the model was a plain stretch (not a letterbox), so the
  /// inverse is just two independent linear scale factors — one for X, one
  /// for Y.
  List<VisionDetection> _rescaleDetectionsToSource(
    List<VisionDetection> detections,
  ) {
    if (detections.isEmpty) return detections;
    final scaleX = _lastSourceW / _yoloW;
    final scaleY = _lastSourceH / _yoloH;
    if (scaleX == 1.0 && scaleY == 1.0) return detections;

    return detections.map((d) {
      final box = d.box;
      final rescaledBox = Rect.fromLTWH(
        box.left * scaleX,
        box.top * scaleY,
        box.width * scaleX,
        box.height * scaleY,
      );
      return VisionDetection(
        label: d.label,
        confidence: d.confidence,
        box: rescaledBox,
      );
    }).toList();
  }

  Future<List<VisionDetection>> _runYoloOnCameraCopy(
    CameraImageCopy copy,
    List<String> notes, {
    int rotationQuarterTurns = 0,
  }) async {
    final interp = _yoloInterpreter;
    if (interp == null) {
      notes.add('YOLO skipped — no model');
      return const [];
    }

    if (_yoloBusy) {
      notes.add('YOLO skipped — previous inference still running');
      return const [];
    }
    _yoloBusy = true;

    try {
      // Convert the camera frame to an img.Image first (proven-correct
      // path, shared with the hand-landmark code) rather than writing
      // straight into a flat float buffer with custom rotation math.
      var fullImage = yuvCopyToImage(copy);
      if (rotationQuarterTurns != 0) {
        fullImage = img.copyRotate(fullImage, angle: rotationQuarterTurns * 90);
      }
      final resized = img.copyResize(
        fullImage,
        width: _yoloW,
        height: _yoloH,
        interpolation: img.Interpolation.linear,
      );

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

      return _runYoloOnNestedInput(input, interp, notes);
    } catch (e, st) {
      notes.add('YOLO error: $e');
      debugPrint('YOLO error: $e\n$st');
      return const [];
    } finally {
      _yoloBusy = false;
    }
  }

  /// Runs inference using genuine nested Dart lists for both input and
  /// output, matching the layout `interp.run()` expects natively. This
  /// mirrors the proven-working implementation rather than relying on
  /// `Float32List.reshape()`, which produced silent all-zero output on
  /// some devices despite running without throwing.
  List<VisionDetection> _runYoloOnNestedInput(
    List<dynamic> input,
    Interpreter interp,
    List<String> notes,
  ) {
    final outShape = interp.getOutputTensor(0).shape;
    final inShape = interp.getInputTensor(0).shape;
    notes.add('YOLO in shape: $inShape, out shape: $outShape');

    if (outShape.length != 3) {
      notes.add('Unsupported output shape');
      return const [];
    }

    final dim1 = outShape[1];
    final dim2 = outShape[2];
    final numClasses = _labels.length;
    final expectedFeats = 4 + numClasses;

    final bool boxesFirst;
    if (dim2 == expectedFeats) {
      boxesFirst = true;
    } else if (dim1 == expectedFeats) {
      boxesFirst = false;
    } else {
      boxesFirst = dim1 > dim2;
    }

    final boxCount = boxesFirst ? dim1 : dim2;
    final featCount = boxesFirst ? dim2 : dim1;

    final output = boxesFirst
        ? [List.generate(boxCount, (_) => List<double>.filled(featCount, 0.0))]
        : [List.generate(featCount, (_) => List<double>.filled(boxCount, 0.0))];

    interp.run(input, output);

    final detections = <VisionDetection>[];
    double globalMaxScore = 0;

    double featAt(int i, int f) {
      if (boxesFirst) {
        return (output[0][i][f] as num).toDouble();
      } else {
        return (output[0][f][i] as num).toDouble();
      }
    }

    for (var i = 0; i < boxCount; i++) {
      double bestScore = 0.0;
      int bestIdx = 0;
      for (var ci = 0; ci < numClasses; ci++) {
        final score = featAt(i, 4 + ci);
        if (score > bestScore) {
          bestScore = score;
          bestIdx = ci;
        }
      }

      if (bestScore > globalMaxScore) globalMaxScore = bestScore;
      if (bestScore < confThreshold) continue;

      final cx = featAt(i, 0);
      final cy = featAt(i, 1);
      final bw = featAt(i, 2);
      final bh = featAt(i, 3);

      final bool isNorm = cx <= 1.01 && cy <= 1.01 && bw <= 1.01 && bh <= 1.01;
      final double px = isNorm ? cx * _yoloW : cx;
      final double py = isNorm ? cy * _yoloH : cy;
      final double pw = isNorm ? bw * _yoloW : bw;
      final double ph = isNorm ? bh * _yoloH : bh;

      final box = Rect.fromLTWH(px - pw / 2, py - ph / 2, pw, ph);

      final label = bestIdx < _labels.length ? _labels[bestIdx] : 'unknown';
      detections.add(
        VisionDetection(label: label, confidence: bestScore, box: box),
      );
    }

    notes.add('YOLO max score: ${globalMaxScore.toStringAsFixed(4)}');
    final filtered = _nms(detections, iouThreshold: 0.45);
    notes.add(
      'YOLO detections: ${filtered.length} (raw: ${detections.length})',
    );
    return filtered;
  }

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

    if (_poseBusy) {
      notes.add('Hand landmark skipped — previous inference still running');
      return null;
    }
    _poseBusy = true;

    try {
      const sz = 224;
      final resized = img.copyResize(image, width: sz, height: sz);

      // Genuine nested [1, sz, sz, 3] list, same reasoning as the YOLO
      // input above — avoids relying on Float32List.reshape().
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

      final poseOutShape = interp.getOutputTensor(0).shape;
      final flatOutLen = poseOutShape.fold<int>(1, (a, b) => a * b);
      final output = [List<double>.filled(flatOutLen, 0.0)];

      interp.run(input, output);

      final flatOut = output[0];
      final points = List.generate(
        21,
        (i) => Offset(flatOut[i * 3], flatOut[i * 3 + 1]),
      );
      notes.add('Hand landmarks: 21 pts');
      return HandLandmarks(points);
    } catch (e) {
      notes.add('Hand landmark error: $e');
      return null;
    } finally {
      _poseBusy = false;
    }
  }

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

    if (cards.isEmpty || decks.isEmpty) {
      notes.add('Need card + deck — returning unknown');
      return DealingLabel.unknown;
    }

    final deck = decks.reduce((a, b) => a.confidence >= b.confidence ? a : b);
    final card = cards.reduce((a, b) => a.confidence >= b.confidence ? a : b);

    final deckTop = deck.box.top;
    final deckBottom = deck.box.bottom;
    final deckHeight = deckBottom - deckTop;

    if (deckHeight < 1) {
      notes.add('Deck box too small');
      return DealingLabel.unknown;
    }

    final cardCenterY = card.box.center.dy;
    final relPos = (cardCenterY - deckTop) / deckHeight;

    notes.add('Card rel pos in deck: ${relPos.toStringAsFixed(3)}');

    if (landmarks != null) {
      // Landmark points are normalised 0-1 against the source image that
      // was fed into the pose model (see _runHandLandmarkOnImage), which is
      // the same source-frame space the boxes above are now in — so scale
      // by _lastSourceH, not the model's square input height.
      final wristY = landmarks.points[0].dy * _lastSourceH;
      final palmBaseY = landmarks.points[9].dy * _lastSourceH;
      final thumbTipY = landmarks.points[4].dy * _lastSourceH;
      final indexTipY = landmarks.points[8].dy * _lastSourceH;
      final gripY = (thumbTipY + indexTipY) / 2;

      final gripRelPos = (gripY - deckTop) / deckHeight;
      notes.add('Grip rel pos: ${gripRelPos.toStringAsFixed(3)}');
      notes.add(
        'Wrist Y: ${wristY.toStringAsFixed(1)}, palmBase Y: ${palmBaseY.toStringAsFixed(1)}',
      );

      final combinedPos = (relPos + gripRelPos) / 2;
      notes.add('Combined pos: ${combinedPos.toStringAsFixed(3)}');

      return _labelFromRelPos(combinedPos, notes);
    }

    if (hands.isNotEmpty) {
      final hand = hands.reduce((a, b) => a.confidence >= b.confidence ? a : b);
      final handCenterY = hand.box.center.dy;
      final handRelPos = (handCenterY - deckTop) / deckHeight;
      notes.add('Hand rel pos: ${handRelPos.toStringAsFixed(3)}');

      final combinedPos = (relPos * 0.6 + handRelPos * 0.4);
      return _labelFromRelPos(combinedPos, notes);
    }

    return _labelFromRelPos(relPos, notes);
  }

  DealingLabel _labelFromRelPos(double pos, List<String> notes) {
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
