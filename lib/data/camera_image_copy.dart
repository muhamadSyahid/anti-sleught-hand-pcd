import 'dart:typed_data';
import 'package:camera/camera.dart';

class CameraImageCopy {
  CameraImageCopy({
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
