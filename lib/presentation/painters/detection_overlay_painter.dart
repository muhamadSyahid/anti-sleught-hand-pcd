import 'package:flutter/material.dart';
import '../../core/constants.dart';
import '../../domain/enums/dealing_label.dart';
import '../../domain/models/hand_landmarks.dart';
import '../../domain/models/vision_detection.dart';

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

  @override
  void paint(Canvas canvas, Size size) {
    if (modelInputW <= 0 || modelInputH <= 0) return;

    final scaleX = size.width / modelInputW;
    final scaleY = size.height / modelInputH;

    for (final det in detections) {
      final color = labelColors[det.label] ?? Colors.white;

      final scaled = Rect.fromLTWH(
        det.box.left * scaleX,
        det.box.top * scaleY,
        det.box.width * scaleX,
        det.box.height * scaleY,
      );

      canvas.drawRRect(
        RRect.fromRectAndRadius(scaled, const Radius.circular(6)),
        Paint()
          ..color = color
          ..style = PaintingStyle.stroke
          ..strokeWidth = 2.0,
      );

      _drawCorners(canvas, scaled, color);
      _drawPill(canvas, scaled, det.label, det.confidence, color);
    }

    final lm = handLandmarks;
    if (lm != null) {
      final bonePaint = Paint()
        ..color = const Color(0xFFE0E0FF).withOpacity(0.85)
        ..strokeWidth = 1.8
        ..strokeCap = StrokeCap.round;

      for (final conn in handConnections) {
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
    canvas.drawLine(r.topLeft, r.topLeft + const Offset(c, 0), p);
    canvas.drawLine(r.topLeft, r.topLeft + const Offset(0, c), p);
    canvas.drawLine(r.topRight, r.topRight + const Offset(-c, 0), p);
    canvas.drawLine(r.topRight, r.topRight + const Offset(0, c), p);
    canvas.drawLine(r.bottomLeft, r.bottomLeft + const Offset(c, 0), p);
    canvas.drawLine(r.bottomLeft, r.bottomLeft + const Offset(0, -c), p);
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
