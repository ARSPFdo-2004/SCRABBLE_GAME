#include <AccelStepper.h>
#include <math.h>

// CNC Shield V3 pins on Arduino Uno
const int X_STEP_PIN = 2;
const int Y_STEP_PIN = 3;
const int Z_STEP_PIN = 4;
const int X_DIR_PIN = 5;
const int Y_DIR_PIN = 6;
const int Z_DIR_PIN = 7;
const int EN_PIN = 8;

// X and Y move together for board numbers
AccelStepper stepperX(AccelStepper::DRIVER, X_STEP_PIN, X_DIR_PIN);
AccelStepper stepperY(AccelStepper::DRIVER, Y_STEP_PIN, Y_DIR_PIN);

// Z moves for board letters
AccelStepper stepperZ(AccelStepper::DRIVER, Z_STEP_PIN, Z_DIR_PIN);

// App motion coordinates
// X = letters axis -> physical Z
// Y = numbers axis -> physical X and Y together
float currentLettersMm = 0.0;
float currentNumbersMm = 0.0;

// Runtime-configurable settings
float stepsPerMmLetters = 10.0; // Z axis
float stepsPerMmNumbers = 10.0; // X/Y pair

float offsetLettersMm = 0.0;    // added to app X before motion
float offsetNumbersMm = 0.0;    // added to app Y before motion

// If one paired motor turns opposite, invert one of them
const bool INVERT_X = false;
const bool INVERT_Y = true;
const bool INVERT_Z = false;

String inputLine = "";

void setup() {
  Serial.begin(115200);

  pinMode(EN_PIN, OUTPUT);
  digitalWrite(EN_PIN, LOW); // enable A4988 drivers

  stepperX.setMaxSpeed(3000);
  stepperY.setMaxSpeed(3000);
  stepperZ.setMaxSpeed(3000);

  stepperX.setAcceleration(1200);
  stepperY.setAcceleration(1200);
  stepperZ.setAcceleration(1200);

  Serial.println("ready");
  Serial.println("ok");
}

void loop() {
  readSerialCommand();
  stepperX.run();
  stepperY.run();
  stepperZ.run();
}

void readSerialCommand() {
  while (Serial.available() > 0) {
    char c = Serial.read();

    if (c == '\n') {
      inputLine.trim();
      if (inputLine.length() > 0) {
        handleCommand(inputLine);
      }
      inputLine = "";
    } else if (c != '\r') {
      inputLine += c;
    }
  }
}

void handleCommand(String cmd) {
  cmd.trim();
  cmd.toUpperCase();

  Serial.print("RX: ");
  Serial.println(cmd);

  if (cmd == "G90") {
    Serial.println("ok");
    return;
  }

  if (cmd == "PING") {
    Serial.println("ok");
    return;
  }

  if (cmd == "HOMEZERO") {
    moveToBoardPosition(0.0, 0.0);
    currentLettersMm = 0.0;
    currentNumbersMm = 0.0;
    Serial.println("ok");
    return;
  }

  if (cmd == "ZTEST") {
    moveToBoardPosition(currentLettersMm + 10.0, currentNumbersMm);
    currentLettersMm += 10.0;
    Serial.println("ok");
    return;
  }

  if (cmd == "XYTEST") {
    moveToBoardPosition(currentLettersMm, currentNumbersMm + 10.0);
    currentNumbersMm += 10.0;
    Serial.println("ok");
    return;
  }

  if (cmd.startsWith("STEPS")) {
    handleStepsCommand(cmd);
    return;
  }

  if (cmd.startsWith("OFFSET")) {
    handleOffsetCommand(cmd);
    return;
  }

  if (cmd.startsWith("G0") || cmd.startsWith("G1")) {
    float targetLetters = currentLettersMm;
    float targetNumbers = currentNumbersMm;

    int xIndex = cmd.indexOf('X');
    int yIndex = cmd.indexOf('Y');

    if (xIndex >= 0) {
      targetLetters = parseNumber(cmd, xIndex + 1);
    }

    if (yIndex >= 0) {
      targetNumbers = parseNumber(cmd, yIndex + 1);
    }

    Serial.print("Letters target (Z axis): ");
    Serial.println(targetLetters);
    Serial.print("Numbers target (X/Y axes): ");
    Serial.println(targetNumbers);

    moveToBoardPosition(targetLetters, targetNumbers);

    currentLettersMm = targetLetters;
    currentNumbersMm = targetNumbers;

    Serial.println("ok");
    return;
  }

  Serial.println("error");
}

void handleStepsCommand(String cmd) {
  float newLettersSteps = stepsPerMmLetters;
  float newNumbersSteps = stepsPerMmNumbers;

  int xIndex = cmd.indexOf('X');
  int yIndex = cmd.indexOf('Y');

  if (xIndex >= 0) {
    newLettersSteps = parseNumber(cmd, xIndex + 1);
  }

  if (yIndex >= 0) {
    newNumbersSteps = parseNumber(cmd, yIndex + 1);
  }

  if (newLettersSteps <= 0.0 || newNumbersSteps <= 0.0) {
    Serial.println("error: invalid steps");
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
  float newLettersOffset = offsetLettersMm;
  float newNumbersOffset = offsetNumbersMm;

  int xIndex = cmd.indexOf('X');
  int yIndex = cmd.indexOf('Y');

  if (xIndex >= 0) {
    newLettersOffset = parseNumber(cmd, xIndex + 1);
  }

  if (yIndex >= 0) {
    newNumbersOffset = parseNumber(cmd, yIndex + 1);
  }

  offsetLettersMm = newLettersOffset;
  offsetNumbersMm = newNumbersOffset;

  Serial.print("offset X");
  Serial.print(offsetLettersMm, 3);
  Serial.print(" Y");
  Serial.println(offsetNumbersMm, 3);
  Serial.println("ok");
}

float parseNumber(String text, int startIndex) {
  String number = "";

  while (startIndex < text.length()) {
    char c = text[startIndex];
    if ((c >= '0' && c <= '9') || c == '.' || c == '-' || c == '+') {
      number += c;
      startIndex++;
    } else {
      break;
    }
  }

  return number.toFloat();
}

long mmToSteps(float mm, float stepsPerMm, bool invertAxis) {
  long steps = lround(mm * stepsPerMm);
  return invertAxis ? -steps : steps;
}

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