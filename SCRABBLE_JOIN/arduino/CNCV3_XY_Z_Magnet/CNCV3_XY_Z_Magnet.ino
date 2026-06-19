#include <Servo.h>

// CNC Shield V3 default step/dir wiring on Arduino Uno.
const byte X_STEP_PIN = 2;
const byte Y_STEP_PIN = 3;
const byte X_DIR_PIN = 5;
const byte Y_DIR_PIN = 6;
const byte STEPPER_ENABLE_PIN = 8;  // Active LOW on CNC Shield V3.

// New direct Arduino pins. A1/A2 are used as digital control pins here.
const byte Z_SERVO_PIN = A1;  // Arduino A1 / D15, MG995 signal wire.
const byte RELAY_PIN = A2;    // Arduino A2 / D16, relay module input for electromagnet.

// Most relay modules are active LOW: LOW = ON, HIGH = OFF.
// Change this to true only if your relay module is high-level triggered.
const bool RELAY_ACTIVE_HIGH = false;

const bool X_DIR_INVERT = false;
const bool Y_DIR_INVERT = false;

float xStepsPerMm = 80.0;
float yStepsPerMm = 80.0;
float currentX = 0.0;
float currentY = 0.0;
float defaultFeedRate = 1500.0;

Servo zServo;
int zUpAngle = 20;
int zDownAngle = 80;
const byte Z_SERVO_REFRESH_WRITES = 6;
const unsigned long Z_SERVO_REFRESH_GAP_MS = 30;
const unsigned long Z_SERVO_SETTLE_MS = 650;

String inputLine;

void setup() {
  Serial.begin(115200);

  pinMode(X_STEP_PIN, OUTPUT);
  pinMode(Y_STEP_PIN, OUTPUT);
  pinMode(X_DIR_PIN, OUTPUT);
  pinMode(Y_DIR_PIN, OUTPUT);
  pinMode(STEPPER_ENABLE_PIN, OUTPUT);
  pinMode(RELAY_PIN, OUTPUT);

  digitalWrite(STEPPER_ENABLE_PIN, LOW);
  writeRelay(false);

  zServo.attach(Z_SERVO_PIN);
  zServo.write(zUpAngle);

  Serial.println("READY CNCV3 XY Z MAGNET");
}

void loop() {
  while (Serial.available() > 0) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      if (inputLine.length() > 0) {
        handleCommand(inputLine);
        inputLine = "";
      }
    } else {
      inputLine += c;
    }
  }
}

void handleCommand(String cmd) {
  cmd.trim();
  cmd.toUpperCase();
  if (cmd.length() == 0) {
    return;
  }

  if (cmd == "ZU") {
    zMoveUp();
    Serial.println("OK Z UP");
  } else if (cmd == "ZD") {
    zMoveDown();
    Serial.println("OK Z DOWN");
  } else if (cmd.startsWith("ZH")) {
    int angle = constrain(cmd.substring(2).toInt(), 0, 180);
    zDownAngle = angle;
    Serial.print("OK Z HEIGHT ");
    Serial.println(zDownAngle);
  } else if (cmd == "R1" || cmd == "M1") {
    magnetOn();
    Serial.println("OK MAGNET ON");
  } else if (cmd == "R0" || cmd == "M0") {
    magnetOff();
    Serial.println("OK MAGNET OFF");
  } else if (cmd.startsWith("G0") || cmd.startsWith("G1")) {
    handleMoveCommand(cmd);
  } else if (cmd.startsWith("G92")) {
    handleSetPositionCommand(cmd);
  } else if (cmd.startsWith("XS")) {
    xStepsPerMm = max(1.0, cmd.substring(2).toFloat());
    Serial.print("OK X STEPS/MM ");
    Serial.println(xStepsPerMm, 4);
  } else if (cmd.startsWith("YS")) {
    yStepsPerMm = max(1.0, cmd.substring(2).toFloat());
    Serial.print("OK Y STEPS/MM ");
    Serial.println(yStepsPerMm, 4);
  } else if (cmd == "RESET" || cmd == "$X") {
    currentX = 0.0;
    currentY = 0.0;
    digitalWrite(STEPPER_ENABLE_PIN, LOW);
    Serial.println("OK RESET");
  } else {
    Serial.print("ERR UNKNOWN COMMAND ");
    Serial.println(cmd);
  }
}

void handleMoveCommand(String cmd) {
  float targetX = currentX;
  float targetY = currentY;
  float feedRate = defaultFeedRate;

  readWord(cmd, 'X', targetX);
  readWord(cmd, 'Y', targetY);
  readWord(cmd, 'F', feedRate);

  moveTo(targetX, targetY, feedRate);
  Serial.println("OK");
}

void handleSetPositionCommand(String cmd) {
  readWord(cmd, 'X', currentX);
  readWord(cmd, 'Y', currentY);
  Serial.println("OK G92");
}

bool readWord(const String &cmd, char key, float &value) {
  int start = cmd.indexOf(key);
  if (start < 0) {
    return false;
  }

  int end = start + 1;
  while (end < cmd.length() && cmd.charAt(end) == ' ') {
    end++;
  }
  int numberStart = end;
  while (end < cmd.length()) {
    char c = cmd.charAt(end);
    if ((c >= '0' && c <= '9') || c == '-' || c == '+' || c == '.') {
      end++;
    } else {
      break;
    }
  }

  if (end == numberStart) {
    return false;
  }

  value = cmd.substring(numberStart, end).toFloat();
  return true;
}

void moveTo(float targetX, float targetY, float feedRate) {
  long xSteps = roundToLong((targetX - currentX) * xStepsPerMm);
  long ySteps = roundToLong((targetY - currentY) * yStepsPerMm);

  bool xForward = xSteps >= 0;
  bool yForward = ySteps >= 0;

  xSteps = labs(xSteps);
  ySteps = labs(ySteps);

  digitalWrite(X_DIR_PIN, (xForward ^ X_DIR_INVERT) ? HIGH : LOW);
  digitalWrite(Y_DIR_PIN, (yForward ^ Y_DIR_INVERT) ? HIGH : LOW);

  long maxSteps = max(xSteps, ySteps);
  if (maxSteps == 0) {
    currentX = targetX;
    currentY = targetY;
    return;
  }

  float dx = targetX - currentX;
  float dy = targetY - currentY;
  float distanceMm = sqrt(dx * dx + dy * dy);
  float stepsPerSecond = 1000.0;
  if (distanceMm > 0.0 && feedRate > 0.0) {
    stepsPerSecond = (feedRate / 60.0) * (maxSteps / distanceMm);
  }
  unsigned long halfPulseDelayUs = max(250UL, (unsigned long)(500000.0 / stepsPerSecond));

  long xError = 0;
  long yError = 0;
  for (long stepIndex = 0; stepIndex < maxSteps; stepIndex++) {
    xError += xSteps;
    yError += ySteps;

    bool stepX = false;
    bool stepY = false;

    if (xError >= maxSteps) {
      xError -= maxSteps;
      stepX = true;
    }
    if (yError >= maxSteps) {
      yError -= maxSteps;
      stepY = true;
    }

    if (stepX) {
      digitalWrite(X_STEP_PIN, HIGH);
    }
    if (stepY) {
      digitalWrite(Y_STEP_PIN, HIGH);
    }
    delayMicroseconds(halfPulseDelayUs);

    if (stepX) {
      digitalWrite(X_STEP_PIN, LOW);
    }
    if (stepY) {
      digitalWrite(Y_STEP_PIN, LOW);
    }
    delayMicroseconds(halfPulseDelayUs);
  }

  currentX = targetX;
  currentY = targetY;
}

void zMoveUp() {
  moveZServoTo(zUpAngle);
}

void zMoveDown() {
  moveZServoTo(zDownAngle);
}

void moveZServoTo(int angle) {
  for (byte index = 0; index < Z_SERVO_REFRESH_WRITES; index++) {
    zServo.write(angle);
    delay(Z_SERVO_REFRESH_GAP_MS);
  }
  delay(Z_SERVO_SETTLE_MS);
}

void magnetOn() {
  writeRelay(true);
}

void magnetOff() {
  writeRelay(false);
}

long roundToLong(float value) {
  if (value >= 0.0) {
    return (long)(value + 0.5);
  }
  return (long)(value - 0.5);
}

void writeRelay(bool enabled) {
  if (RELAY_ACTIVE_HIGH) {
    digitalWrite(RELAY_PIN, enabled ? HIGH : LOW);
  } else {
    digitalWrite(RELAY_PIN, enabled ? LOW : HIGH);
  }
}
