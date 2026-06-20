import 'dart:typed_data';

class PlaneData {
  const PlaneData({required this.bytes, required this.stride});
  final Uint8List bytes;
  final int stride;
}
