import 'package:flutter/material.dart';
import '../enums/dealing_label.dart';
import 'vision_detection.dart';
import 'hand_landmarks.dart';

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
  final double modelInputW;
  final double modelInputH;
}
