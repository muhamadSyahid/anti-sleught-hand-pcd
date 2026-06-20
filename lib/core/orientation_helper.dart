import 'package:flutter/material.dart';

int quarterTurnsForOrientation({
  required int sensorOrientation,
  required DeviceOrientation deviceOrientation,
  required bool isFrontFacing,
}) {
  const ccwDegrees = {
    DeviceOrientation.portraitUp: 0,
    DeviceOrientation.landscapeLeft: 90,
    DeviceOrientation.portraitDown: 180,
    DeviceOrientation.landscapeRight: 270,
  };
  final deviceDegrees = ccwDegrees[deviceOrientation] ?? 0;
  final sign = isFrontFacing ? 1 : -1;
  final rotation = (sensorOrientation - deviceDegrees * sign + 360) % 360;
  return (rotation ~/ 90) % 4;
}
