import 'package:flutter/material.dart';

class VisionDetection {
  VisionDetection({
    required this.label,
    required this.confidence,
    required this.box,
  });
  final String label;
  final double confidence;
  final Rect box;
}
