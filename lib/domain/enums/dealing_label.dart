import 'package:flutter/material.dart';

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
