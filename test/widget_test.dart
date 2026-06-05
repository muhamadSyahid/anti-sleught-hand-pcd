import 'package:flutter_test/flutter_test.dart';

import '../lib/main.dart';

void main() {
  testWidgets('Landing screen shows start button', (WidgetTester tester) async {
    await tester.pumpWidget(const AntiSleughtHandApp());

    expect(find.text('Start'), findsOneWidget);
    expect(find.text('Anti Sleught Hand TCG'), findsOneWidget);
  });
}
