#include <AccelStepper.h>
#include <Servo.h>
#include <math.h>

// ================= CNC Shield V3 Pinout =================
const int X_STEP_PIN = 2;
const int Y_STEP_PIN = 3;
const int Z_STEP_PIN = 4;

const int X_DIR_PIN = 5;
const int Y_DIR_PIN = 6;
const int Z_DIR_PIN = 7;

const int EN_PIN = 8;   // CNC Shield enable pin, active LOW

// ================= Servo + Relay Pins =================
const byte Z_SERVO_PIN = A1;   // MG995 servo signal
const byte RELAY_PIN   = A2;   // Relay module for electromagnet

// Most relay modules are active LOW: LOW = ON, HIGH = OFF
const bool RELAY_ACTIVE_HIGH = false;

// ================= Stepper Setup =================
AccelStepper stepperX(AccelStepper::DRIVER, X_STEP_PIN, X_DIR_PIN);
AccelStepper stepperY(AccelStepper::DRIVER, Y_STEP_PIN, Y_DIR_PIN);
AccelStepper stepperZ(AccelStepper::DRIVER, Z_STEP_PIN, Z_DIR_PIN);

// App coordinates:
// App X = letters axis = physical Z stepper
// App Y = numbers axis = physical X and Y steppers together
float currentLettersMm = 0.0;
float currentNumbersMm = 0.0;

// Steps/mm settings
float stepsPerMmLetters = 10.0;  // App X -> physical Z
float stepsPerMmNumbers = 10.0;  // App Y -> physical X/Y pair

// Offsets
float offsetLettersMm = 0.0;
float offsetNumbersMm = 0.0;

// Direction inversion
const bool INVERT_X = false;
const bool INVERT_Y = true;
const bool INVERT_Z = false;

// Speed settings
float maxSpeed = 3000.0;
float acceleration =950.0;

// ================= Servo Settings =================
Servo zServo;

int zUpAngle = 20;
int zDownAngle = 80;
const byte Z_SERVO_REFRESH_WRITES = 6;
const unsigned long Z_SERVO_REFRESH_GAP_MS = 30;
const unsigned long Z_SERVO_SETTLE_MS = 650;

// ================= Serial =================
String inputLine = "";

void setup() {
  Serial.begin(115200);

  pinMode(EN_PIN, OUTPUT);
  digitalWrite(EN_PIN, LOW);  // Enable stepper drivers

  pinMode(RELAY_PIN, OUTPUT);
  writeRelay(false);

  zServo.attach(Z_SERVO_PIN);
  zServo.write(zUpAngle);

  stepperX.setMaxSpeed(maxSpeed);
  stepperY.setMaxSpeed(maxSpeed);
  stepperZ.setMaxSpeed(maxSpeed);

  stepperX.setAcceleration(acceleration);
  stepperY.setAcceleration(acceleration);
  stepperZ.setAcceleration(acceleration);

  Serial.println("READY CNCV3 XYZ SERVO MAGNET");
  Serial.println("ok");
}

void loop() {
  readSerialCommand();

  stepperX.run();
  stepperY.run();
  stepperZ.run();
}

// ================= Serial Reading =================

void readSerialCommand() {
  while (Serial.available() > 0) {
    char c = Serial.read();

    if (c == '\n' || c == '\r') {
      inputLine.trim();

      if (inputLine.length() > 0) {
        handleCommand(inputLine);
      }

      inputLine = "";
    } else {
      inputLine += c;
    }
  }
}

// ================= Command Handler =================

void handleCommand(String cmd) {
  cmd.trim();
  cmd.toUpperCase();

  if (cmd.length() == 0) {
    return;
  }

  Serial.print("RX: ");
  Serial.println(cmd);

  // Basic commands
  if (cmd == "PING") {
    Serial.println("ok");
    return;
  }

  if (cmd == "G90") {
    Serial.println("ok");
    return;
  }

  if (cmd == "RESET" || cmd == "$X") {
    digitalWrite(EN_PIN, LOW);
    magnetOff();
    zMoveUp();
    Serial.println("ok reset");
    return;
  }

  // Servo commands
  if (cmd == "ZU") {
    zMoveUp();
    Serial.println("ok z up");
    return;
  }

  if (cmd == "ZD") {
    zMoveDown();
    Serial.println("ok z down");
    return;
  }

  if (cmd.startsWith("ZH")) {
    int angle = constrain(cmd.substring(2).toInt(), 0, 180);
    zDownAngle = angle;

    Serial.print("ok z height ");
    Serial.println(zDownAngle);
    return;
  }

  // Magnet commands
  if (cmd == "R1" || cmd == "M1") {
    magnetOn();
    Serial.println("ok magnet on");
    return;
  }

  if (cmd == "R0" || cmd == "M0") {
    magnetOff();
    Serial.println("ok magnet off");
    return;
  }

  // Home to software zero
  if (cmd == "HOMEZERO") {
    moveToBoardPosition(0.0, 0.0);

    currentLettersMm = 0.0;
    currentNumbersMm = 0.0;

    Serial.println("ok");
    return;
  }

  // Test physical Z stepper axis
  if (cmd == "ZTEST") {
    moveToBoardPosition(currentLettersMm + 10.0, currentNumbersMm);
    currentLettersMm += 10.0;

    Serial.println("ok");
    return;
  }

  // Test X/Y paired movement
  if (cmd == "XYTEST") {
    moveToBoardPosition(currentLettersMm, currentNumbersMm + 10.0);
    currentNumbersMm += 10.0;

    Serial.println("ok");
    return;
  }

  // Set current position without moving
  if (cmd.startsWith("G92")) {
    handleSetPositionCommand(cmd);
    return;
  }

  // Steps/mm command from second code
  // Example: STEPS X10 Y10
  if (cmd.startsWith("STEPS")) {
    handleStepsCommand(cmd);
    return;
  }

  // Offset command
  // Example: OFFSET X5 Y10
  if (cmd.startsWith("OFFSET")) {
    handleOffsetCommand(cmd);
    return;
  }

  // Compatibility with first code
  // XS80 changes numbers X/Y pair steps/mm
  if (cmd.startsWith("XS")) {
    stepsPerMmNumbers = max(1.0, cmd.substring(2).toFloat());

    Serial.print("ok numbers steps/mm ");
    Serial.println(stepsPerMmNumbers, 4);
    return;
  }

  // YS80 changes letters/Z steps/mm
  if (cmd.startsWith("YS")) {
    stepsPerMmLetters = max(1.0, cmd.substring(2).toFloat());

    Serial.print("ok letters steps/mm ");
    Serial.println(stepsPerMmLetters, 4);
    return;
  }

  // G0 / G1 movement
  // App X = letters axis = physical Z stepper
  // App Y = numbers axis = physical X/Y pair
  if (cmd.startsWith("G0") || cmd.startsWith("G1")) {
    float targetLetters = currentLettersMm;
    float targetNumbers = currentNumbersMm;

    readWord(cmd, 'X', targetLetters);
    readWord(cmd, 'Y', targetNumbers);

    moveToBoardPosition(targetLetters, targetNumbers);

    currentLettersMm = targetLetters;
    currentNumbersMm = targetNumbers;

    Serial.println("ok");
    return;
  }

  Serial.println("error unknown command");
}

// ================= Movement =================

void moveToBoardPosition(float lettersMm, float numbersMm) {
  float lettersWithOffset = lettersMm + offsetLettersMm;
  float numbersWithOffset = numbersMm + offsetNumbersMm;

  long zSteps = mmToSteps(lettersWithOffset, stepsPerMmLetters, INVERT_Z);
  long xSteps = mmToSteps(numbersWithOffset, stepsPerMmNumbers, INVERT_X);
  long ySteps = mmToSteps(numbersWithOffset, stepsPerMmNumbers, INVERT_Y);

  Serial.print("Move Z steps: ");
  Serial.println(zSteps);

  Serial.print("Move X steps: ");
  Serial.println(xSteps);

  Serial.print("Move Y steps: ");
  Serial.println(ySteps);

  stepperZ.moveTo(zSteps);
  stepperX.moveTo(xSteps);
  stepperY.moveTo(ySteps);

  while (
    stepperX.distanceToGo() != 0 ||
    stepperY.distanceToGo() != 0 ||
    stepperZ.distanceToGo() != 0
  ) {
    stepperX.run();
    stepperY.run();
    stepperZ.run();
  }
}

long mmToSteps(float mm, float stepsPerMm, bool invertAxis) {
  long steps = lround(mm * stepsPerMm);
  return invertAxis ? -steps : steps;
}

// ================= G92 Position Set =================

void handleSetPositionCommand(String cmd) {
  readWord(cmd, 'X', currentLettersMm);
  readWord(cmd, 'Y', currentNumbersMm);

  float lettersWithOffset = currentLettersMm + offsetLettersMm;
  float numbersWithOffset = currentNumbersMm + offsetNumbersMm;

  long zSteps = mmToSteps(lettersWithOffset, stepsPerMmLetters, INVERT_Z);
  long xSteps = mmToSteps(numbersWithOffset, stepsPerMmNumbers, INVERT_X);
  long ySteps = mmToSteps(numbersWithOffset, stepsPerMmNumbers, INVERT_Y);

  stepperZ.setCurrentPosition(zSteps);
  stepperX.setCurrentPosition(xSteps);
  stepperY.setCurrentPosition(ySteps);

  Serial.println("ok g92");
}

// ================= Steps + Offset Commands =================

void handleStepsCommand(String cmd) {
  float newLettersSteps = stepsPerMmLetters;
  float newNumbersSteps = stepsPerMmNumbers;

  readWord(cmd, 'X', newLettersSteps);
  readWord(cmd, 'Y', newNumbersSteps);

  if (newLettersSteps <= 0.0 || newNumbersSteps <= 0.0) {
    Serial.println("error invalid steps");
    return;
  }

  stepsPerMmLetters = newLettersSteps;
  stepsPerMmNumbers = newNumbersSteps;

  Serial.print("steps X");
  Serial.print(stepsPerMmLetters, 3);
  Serial.print(" Y");
  Serial.println(stepsPerMmNumbers, 3);

  Serial.println("ok");
}

void handleOffsetCommand(String cmd) {
  readWord(cmd, 'X', offsetLettersMm);
  readWord(cmd, 'Y', offsetNumbersMm);

  Serial.print("offset X");
  Serial.print(offsetLettersMm, 3);
  Serial.print(" Y");
  Serial.println(offsetNumbersMm, 3);

  Serial.println("ok");
}

// ================= Parser =================

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

    if (
      (c >= '0' && c <= '9') ||
      c == '-' ||
      c == '+' ||
      c == '.'
    ) {
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

// ================= Servo =================

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

// ================= Magnet / Relay =================

void magnetOn() {
  writeRelay(true);
}

void magnetOff() {
  writeRelay(false);
}

void writeRelay(bool enabled) {
  if (RELAY_ACTIVE_HIGH) {
    digitalWrite(RELAY_PIN, enabled ? HIGH : LOW);
  } else {
    digitalWrite(RELAY_PIN, enabled ? LOW : HIGH);
  }
}
