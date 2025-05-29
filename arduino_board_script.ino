// Define the LED pin
const int ledPin = 13; // Use the built-in LED on pin 13 or connect an external LED

// Variables to control blinking
int blinkState = 0;         // 0: off, 1: rapid blink, 2: slow blink
unsigned long lastBlinkTime = 0;
const long rapidBlinkInterval = 100; // milliseconds (adjust for desired speed)
const long slowBlinkInterval = 1000; // milliseconds

void setup() {
  pinMode(ledPin, OUTPUT);
  Serial.begin(9600); // Start serial communication at 9600 bps
}

void loop() {
  // Check for incoming serial data
  if (Serial.available() > 0) {
    char command = Serial.read();
    if (command == 'a') {
      blinkState = 1; // Rapid blink for apple
    } else if (command == 'o') {
      blinkState = 2; // Slow blink for orange
    } else if (command == 'n') {
      blinkState = 0; // No blink (nothing detected or default)
      digitalWrite(ledPin, LOW);
    }
  }

  // Handle blinking based on the state
  unsigned long currentMillis = millis();
  if (blinkState == 1) { // Rapid blink
    if (currentMillis - lastBlinkTime >= rapidBlinkInterval) {
      lastBlinkTime = currentMillis;
      digitalWrite(ledPin, !digitalRead(ledPin)); // Toggle the LED
    }
  } else if (blinkState == 2) { // Slow blink
    if (currentMillis - lastBlinkTime >= slowBlinkInterval) {
      lastBlinkTime = currentMillis;
      digitalWrite(ledPin, !digitalRead(ledPin)); // Toggle the LED
    }
  }
}