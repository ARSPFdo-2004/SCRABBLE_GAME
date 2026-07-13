#include <Servo.h>
#include <Adafruit_NeoPixel.h>
#include <U8g2lib.h>
#include <string.h>

#define LED_PIN 12
#define LED_COUNT 144
#define GRID_SIZE 12

const byte ON_BUTTON_PIN = 2;
const byte TIMER_BUTTON_PIN = 3;
const byte CHALLENGE_BUTTON_PIN = 4;
const byte PREVIOUS_BUTTON_PIN = 5;
const byte NEXT_BUTTON_PIN = 6;
const byte ACTUATOR_SERVO_PIN = 9;
const int ACTUATOR_DOWN_ANGLE = 20;
const int ACTUATOR_UP_ANGLE = 160;

const unsigned long BUTTON_DEBOUNCE_MS = 35;
const unsigned long SERVO_STEP_INTERVAL_MS = 20;
const unsigned long ANIMATION_ROW_INTERVAL_MS = 120;
const unsigned long ANIMATION_FINAL_HOLD_MS = 300;
const unsigned long ENDING_ANIMATION_STEP_MS = 85;
const unsigned long ENDING_ANIMATION_HOLD_MS = 300;

Adafruit_NeoPixel strip(LED_COUNT, LED_PIN, NEO_GRB + NEO_KHZ800);
Servo actuatorServo;
U8G2_ST7920_128X64_1_SW_SPI screen(U8G2_R0, 13, 11, 10, U8X8_PIN_NONE);

bool systemOn = false;
int player1Score = 0;
int player2Score = 0;
int countdownSeconds = 120;
bool displayDirty = true;

char serialLine[64];
byte serialLength = 0;

// The latest three words from the previous player are available for challenge.
const byte MAX_CHALLENGE_WORDS = 3;
const byte MAX_WORD_LENGTH = 16;
const byte CELL_BYTES = (LED_COUNT + 7) / 8;
char challengeWords[MAX_CHALLENGE_WORDS][MAX_WORD_LENGTH + 1];
byte challengeWordCells[MAX_CHALLENGE_WORDS][CELL_BYTES];
byte challengeWordCount = 0;
byte selectedWordIndex = 0;
bool challengeMode = false;
bool challengeChoicePending = false;
bool challengeLightsActive = false;
bool timerButtonPressed = false;

bool buttonState = HIGH;
bool lastButtonReading = HIGH;
unsigned long lastButtonChangeTime = 0;

struct DebouncedButton {
  byte pin;
  bool stableState;
  bool lastReading;
  bool pressedEvent;
  unsigned long lastChangeTime;

  DebouncedButton(byte buttonPin)
    : pin(buttonPin), stableState(HIGH), lastReading(HIGH),
      pressedEvent(false), lastChangeTime(0) {}

  void begin() {
    pinMode(pin, INPUT_PULLUP);
    stableState = digitalRead(pin);
    lastReading = stableState;
  }

  void update() {
    pressedEvent = false;
    bool reading = digitalRead(pin);
    if (reading != lastReading) {
      lastChangeTime = millis();
      lastReading = reading;
    }
    if (millis() - lastChangeTime >= BUTTON_DEBOUNCE_MS && reading != stableState) {
      stableState = reading;
      if (stableState == LOW) {
        pressedEvent = true;
      }
    }
  }

  bool pressed() {
    return pressedEvent;
  }
};

DebouncedButton challengeButton(CHALLENGE_BUTTON_PIN);
DebouncedButton previousButton(PREVIOUS_BUTTON_PIN);
DebouncedButton nextButton(NEXT_BUTTON_PIN);
DebouncedButton timerButton(TIMER_BUTTON_PIN);

int currentServoAngle = ACTUATOR_DOWN_ANGLE;
int targetServoAngle = ACTUATOR_DOWN_ANGLE;
unsigned long lastServoStepTime = 0;

int glowingRow = 0;
int clearingRow = 0;
bool clearingAnimation = false;
bool animationActive = false;
bool boardScanning = false;
bool endingAnimationActive = false;
byte endingAnimationStep = 0;
bool endingAnimationContracting = false;
unsigned long endingAnimationTime = 0;
unsigned long lastAnimationTime = 0;

int pixelIndex(int row, int column) {
  // The physical first LED is at the bottom, so reverse logical screen rows.
  int physicalRow = GRID_SIZE - 1 - row;

  // Serpentine columns: first column bottom-to-top, next top-to-bottom.
  if (column % 2 == 0) {
    return column * GRID_SIZE + physicalRow;
  }
  return column * GRID_SIZE + (GRID_SIZE - 1 - physicalRow);
}

void clearLights() {
  strip.clear();
  strip.show();
}

bool cellMatches(int row, int column, const byte cells[][2], byte count) {
  for (byte index = 0; index < count; index++) {
    if (cells[index][0] == row && cells[index][1] == column) {
      return true;
    }
  }
  return false;
}

void showPremiumLights() {
  if (!systemOn || animationActive || endingAnimationActive || boardScanning || challengeLightsActive) {
    return;
  }

  // Coordinates match the 12x12 premium layout used by the desktop app.
  static const byte tripleWord[][2] = {
    {0,0},{0,11},{11,0},{11,11},{0,5},{0,6},{5,0},{6,0},{11,5},{11,6},{5,11},{6,11}
  };
  static const byte doubleWord[][2] = {
    {1,1},{2,2},{4,4},{1,10},{2,9},{4,7},{10,1},{9,2},{7,4},{10,10},{9,9},{7,7}
  };
  static const byte tripleLetter[][2] = {
    {1,5},{1,6},{5,1},{6,1},{10,5},{10,6},{5,10},{6,10},{5,5},{6,6},{5,6},{6,5}
  };
  static const byte doubleLetter[][2] = {
    {0,3},{0,8},{3,0},{8,0},{11,3},{11,8},{3,11},{8,11},
    {2,6},{2,5},{3,3},{3,8},{8,3},{8,8},{6,2},{5,2},{6,9},{5,9}
  };

  strip.clear();
  for (int row = 0; row < GRID_SIZE; row++) {
    for (int column = 0; column < GRID_SIZE; column++) {
      uint32_t color = 0;
      if (cellMatches(row, column, doubleLetter, sizeof(doubleLetter) / sizeof(doubleLetter[0]))) {
        color = strip.Color(0, 20, 255);       // DL: deep blue
      } else if (cellMatches(row, column, tripleLetter, sizeof(tripleLetter) / sizeof(tripleLetter[0]))) {
        color = strip.Color(255, 0, 0);        // TL: red
      } else if (cellMatches(row, column, doubleWord, sizeof(doubleWord) / sizeof(doubleWord[0]))) {
        color = strip.Color(150, 0, 255);      // DW: purple
      } else if (cellMatches(row, column, tripleWord, sizeof(tripleWord) / sizeof(tripleWord[0]))) {
        color = strip.Color(255, 150, 0);      // TW: gold
      }
      if (color != 0) {
        // Board row labels run opposite to the animation's logical rows.
        strip.setPixelColor(pixelIndex(GRID_SIZE - 1 - row, column), color);
      }
    }
  }
  strip.show();
}

void drawEndingDiamond(byte radiusStep) {
  strip.clear();
  for (int row = 0; row < GRID_SIZE; row++) {
    for (int column = 0; column < GRID_SIZE; column++) {
      // Doubled coordinates keep the diamond centered between the four middle cells.
      int distanceFromCenter = abs(row * 2 - (GRID_SIZE - 1))
        + abs(column * 2 - (GRID_SIZE - 1));
      int edgeDistance = radiusStep * 2 + 2 - distanceFromCenter;
      if (edgeDistance < 0) {
        continue;
      }
      int brightness = max(70, 255 - edgeDistance * 10);
      strip.setPixelColor(
        pixelIndex(row, column),
        // Exact same blue palette as the board-up glow.
        strip.Color(38 * brightness / 255, 171 * brightness / 255, brightness)
      );
    }
  }
  strip.show();
}

void startEndingAnimation() {
  animationActive = false;
  clearingAnimation = false;
  boardScanning = false;
  challengeLightsActive = false;
  endingAnimationActive = true;
  endingAnimationStep = 0;
  endingAnimationContracting = false;
  endingAnimationTime = millis();
  drawEndingDiamond(endingAnimationStep);
}

void updateEndingAnimation() {
  if (!endingAnimationActive) {
    return;
  }

  unsigned long now = millis();
  unsigned long interval = endingAnimationContracting && endingAnimationStep == GRID_SIZE - 1
    ? ENDING_ANIMATION_HOLD_MS
    : ENDING_ANIMATION_STEP_MS;
  if (now - endingAnimationTime < interval) {
    return;
  }
  endingAnimationTime = now;

  if (!endingAnimationContracting) {
    if (endingAnimationStep < GRID_SIZE - 1) {
      endingAnimationStep++;
    } else {
      endingAnimationContracting = true;
    }
    drawEndingDiamond(endingAnimationStep);
    return;
  }

  if (endingAnimationStep > 0) {
    endingAnimationStep--;
    drawEndingDiamond(endingAnimationStep);
  } else {
    endingAnimationActive = false;
    systemOn = false;
    clearLights();
    if (!actuatorServo.attached()) {
      actuatorServo.attach(ACTUATOR_SERVO_PIN);
    }
    targetServoAngle = ACTUATOR_DOWN_ANGLE;
    lastServoStepTime = 0;
  }
}

void drawTopToBottomGlow(int activeRow) {
  strip.clear();

  for (int row = 0; row <= activeRow; row++) {
    int distanceBehindGlow = activeRow - row;
    int brightness = max(255 - distanceBehindGlow * 18, 70);
    int red = 38 * brightness / 255;
    int green = 171 * brightness / 255;
    int blue = brightness;

    for (int column = 0; column < GRID_SIZE; column++) {
      strip.setPixelColor(
        pixelIndex(row, column),
        strip.Color(red, green, blue)
      );
    }
  }

  strip.show();
}

void startBlueAnimation() {
  glowingRow = 0;
  clearingRow = 0;
  clearingAnimation = false;
  animationActive = true;
  lastAnimationTime = millis();
  drawTopToBottomGlow(glowingRow);
}

void updateBlueAnimation() {
  if (!systemOn || !animationActive) {
    return;
  }

  unsigned long now = millis();
  if (!clearingAnimation &&
    glowingRow < GRID_SIZE - 1
    && now - lastAnimationTime >= ANIMATION_ROW_INTERVAL_MS
  ) {
    lastAnimationTime = now;
    glowingRow++;
    drawTopToBottomGlow(glowingRow);
  } else if (!clearingAnimation &&
    glowingRow == GRID_SIZE - 1
    && now - lastAnimationTime >= ANIMATION_FINAL_HOLD_MS
  ) {
    clearingAnimation = true;
    clearingRow = 0;
    lastAnimationTime = now;

    for (int column = 0; column < GRID_SIZE; column++) {
      strip.setPixelColor(pixelIndex(clearingRow, column), 0);
    }
    strip.show();
    clearingRow++;
  } else if (
    clearingAnimation
    && now - lastAnimationTime >= ANIMATION_ROW_INTERVAL_MS
  ) {
    lastAnimationTime = now;

    if (clearingRow < GRID_SIZE) {
      for (int column = 0; column < GRID_SIZE; column++) {
        strip.setPixelColor(pixelIndex(clearingRow, column), 0);
      }
      strip.show();
      clearingRow++;
    }

    if (clearingRow >= GRID_SIZE) {
      animationActive = false;
      clearingAnimation = false;
      showPremiumLights();
    }
  }
}

void updateActuator() {
  unsigned long now = millis();
  if (currentServoAngle == targetServoAngle) {
    if (actuatorServo.attached()) {
      actuatorServo.detach();
    }
    return;
  }
  if (now - lastServoStepTime < SERVO_STEP_INTERVAL_MS) {
    return;
  }

  lastServoStepTime = now;
  currentServoAngle += currentServoAngle < targetServoAngle ? 1 : -1;
  actuatorServo.write(currentServoAngle);

  // Stop sending correction pulses at the endpoint to prevent vibration.
  if (currentServoAngle == targetServoAngle) {
    actuatorServo.detach();
  }
}

void turnSystemOn() {
  systemOn = true;
  endingAnimationActive = false;
  boardScanning = false;
  if (!actuatorServo.attached()) {
    actuatorServo.attach(ACTUATOR_SERVO_PIN);
  }
  targetServoAngle = ACTUATOR_UP_ANGLE;
  lastServoStepTime = 0;
  startBlueAnimation();
}

void turnSystemOff() {
  if (!systemOn || endingAnimationActive) {
    return;
  }
  startEndingAnimation();
}

void updateOnButton() {
  bool reading = digitalRead(ON_BUTTON_PIN);

  if (reading != lastButtonReading) {
    lastButtonChangeTime = millis();
    lastButtonReading = reading;
  }

  if (millis() - lastButtonChangeTime >= BUTTON_DEBOUNCE_MS && reading != buttonState) {
    buttonState = reading;

    if (buttonState == LOW) {
      if (systemOn) {
        turnSystemOff();
      } else {
        turnSystemOn();
      }
    }
  }
}

const char* selectedChallengeWord() {
  if (challengeWordCount == 0) {
    return "";
  }
  if (selectedWordIndex >= challengeWordCount) {
    selectedWordIndex = 0;
  }
  return challengeWords[selectedWordIndex];
}

void clearChallengeWords() {
  if (challengeLightsActive) {
    clearLights();
  }
  challengeWordCount = 0;
  selectedWordIndex = 0;
  challengeMode = false;
  challengeChoicePending = false;
  challengeLightsActive = false;
  memset(challengeWordCells, 0, sizeof(challengeWordCells));
  displayDirty = true;
}

bool addChallengeWord(const char* word) {
  if (challengeWordCount >= MAX_CHALLENGE_WORDS || word[0] == '\0') {
    return false;
  }
  strncpy(challengeWords[challengeWordCount], word, MAX_WORD_LENGTH);
  challengeWords[challengeWordCount][MAX_WORD_LENGTH] = '\0';
  challengeWordCount++;
  displayDirty = true;
  return true;
}

void selectPreviousWord() {
  if (challengeWordCount == 0) {
    return;
  }
  selectedWordIndex = (selectedWordIndex + challengeWordCount - 1) % challengeWordCount;
  displayDirty = true;
}

void selectNextWord() {
  if (challengeWordCount == 0) {
    return;
  }
  selectedWordIndex = (selectedWordIndex + 1) % challengeWordCount;
  displayDirty = true;
}

void setChallengeCell(byte wordIndex, int pixel) {
  if (wordIndex >= MAX_CHALLENGE_WORDS || pixel < 0 || pixel >= LED_COUNT) {
    return;
  }
  challengeWordCells[wordIndex][pixel / 8] |= (1 << (pixel % 8));
}

bool challengeCellIsSet(byte wordIndex, int pixel) {
  if (wordIndex >= challengeWordCount || pixel < 0 || pixel >= LED_COUNT) {
    return false;
  }
  return challengeWordCells[wordIndex][pixel / 8] & (1 << (pixel % 8));
}

void showSelectedWordInRed() {
  strip.clear();
  for (int pixel = 0; pixel < LED_COUNT; pixel++) {
    if (challengeCellIsSet(selectedWordIndex, pixel)) {
      strip.setPixelColor(pixel, strip.Color(255, 0, 0));
    }
  }
  strip.show();
  challengeLightsActive = true;
}

bool showCellsInRed(char* payload) {
  bool foundCell = false;
  strip.clear();
  char* square = strtok(payload, ",");
  while (square != NULL) {
    if (square[0] >= 'A' && square[0] < 'A' + GRID_SIZE) {
      int row = atoi(square + 1) - 1;
      int column = square[0] - 'A';
      if (row >= 0 && row < GRID_SIZE) {
        int boardRow = GRID_SIZE - 1 - row;
        strip.setPixelColor(pixelIndex(boardRow, column), strip.Color(255, 0, 0));
        foundCell = true;
      }
    }
    square = strtok(NULL, ",");
  }
  strip.show();
  challengeLightsActive = foundCell;
  return foundCell;
}

bool storeWordCells(char* payload) {
  char* cellsStart = NULL;
  long wordIndex = strtol(payload, &cellsStart, 10);
  if (cellsStart == payload || wordIndex < 0 || wordIndex >= MAX_CHALLENGE_WORDS) {
    return false;
  }

  while (*cellsStart == ' ') {
    cellsStart++;
  }
  memset(challengeWordCells[wordIndex], 0, CELL_BYTES);

  char* square = strtok(cellsStart, ",");
  while (square != NULL) {
    if (square[0] >= 'A' && square[0] < 'A' + GRID_SIZE) {
      int row = atoi(square + 1) - 1;
      int column = square[0] - 'A';
      if (row >= 0 && row < GRID_SIZE) {
        // Board labels count from the opposite edge to the animation rows.
        // Undo the animation flip so A2 lights physical A2, not A11.
        int boardRow = GRID_SIZE - 1 - row;
        setChallengeCell((byte)wordIndex, pixelIndex(boardRow, column));
      }
    }
    square = strtok(NULL, ",");
  }
  return true;
}

void updateChallengeButtons() {
  challengeButton.update();
  previousButton.update();
  nextButton.update();

  if (challengeMode && previousButton.pressed()) {
    selectPreviousWord();
    showSelectedWordInRed();
  }
  if (challengeMode && nextButton.pressed()) {
    selectNextWord();
    showSelectedWordInRed();
  }
  if (challengeButton.pressed()) {
    if (!challengeMode) {
      if (challengeWordCount > 0) {
        selectedWordIndex = 0;
        challengeMode = true;
        challengeChoicePending = false;
        showSelectedWordInRed();
        displayDirty = true;
      }
    } else {
      challengeMode = false;
      challengeChoicePending = true;
      showSelectedWordInRed();
      displayDirty = true;
    }
  }
}

void updateTimerButton() {
  timerButton.update();
  if (timerButton.pressed()) {
    timerButtonPressed = true;
  }
}

void drawScreen() {
  char timerText[8];
  int minutes = countdownSeconds / 60;
  int seconds = countdownSeconds % 60;
  snprintf(timerText, sizeof(timerText), "%d:%02d", minutes, seconds);

  screen.firstPage();
  do {
    if (challengeMode || challengeChoicePending) {
      screen.setFont(u8g2_font_ncenB08_tr);
      screen.setCursor(0, 11);
      if (challengeChoicePending) {
        screen.print(F("CHALLENGED"));
      } else {
        screen.print(F("CHALLENGE WORD "));
        screen.print(selectedWordIndex + 1);
        screen.print(F("/"));
        screen.print(challengeWordCount);
      }
      screen.drawHLine(0, 16, 128);

      screen.setFont(u8g2_font_ncenB14_tr);
      const char* word = selectedChallengeWord();
      int wordWidth = screen.getStrWidth(word);
      screen.setCursor(max(0, (128 - wordWidth) / 2), 42);
      screen.print(word);

      screen.setFont(u8g2_font_5x7_tr);
      screen.setCursor(4, 62);
      screen.print(challengeMode ? F("PREV   NEXT   CHALLENGE") : F("SENDING TO GUI..."));
      continue;
    }

    screen.setFont(u8g2_font_ncenB08_tr);
    screen.setCursor(0, 12);
    screen.print(F("P1: "));
    screen.print(player1Score);
    screen.setCursor(70, 12);
    screen.print(F("P2: "));
    screen.print(player2Score);
    screen.drawHLine(0, 18, 128);

    screen.setFont(u8g2_font_logisoso24_tn);
    int timerWidth = screen.getStrWidth(timerText);
    screen.setCursor((128 - timerWidth) / 2, 54);
    screen.print(timerText);
  } while (screen.nextPage());
}

void updateScreen() {
  if (!displayDirty) {
    return;
  }
  displayDirty = false;
  drawScreen();
}

void handleSerialCommand(char* command) {
  int newPlayer1Score = 0;
  int newPlayer2Score = 0;

  if (sscanf(command, "SCORE P1 %d P2 %d", &newPlayer1Score, &newPlayer2Score) == 2) {
    player1Score = newPlayer1Score;
    player2Score = newPlayer2Score;
    if (challengeLightsActive) {
      clearLights();
      challengeLightsActive = false;
      showPremiumLights();
    }
    displayDirty = true;
    Serial.println(F("ok score"));
  } else if (strncmp(command, "TIMER ", 6) == 0) {
    countdownSeconds = max(0, atoi(command + 6));
    displayDirty = true;
    Serial.println(F("ok timer"));
  } else if (strcmp(command, "TIMER_TAKE") == 0) {
    if (timerButtonPressed) {
      timerButtonPressed = false;
      Serial.println(F("ok timer pressed"));
    } else {
      Serial.println(F("ok timer none"));
    }
  } else if (strcmp(command, "WORD_CLEAR") == 0) {
    clearChallengeWords();
    Serial.println(F("ok word clear"));
  } else if (strncmp(command, "WORD_ADD ", 9) == 0) {
    if (addChallengeWord(command + 9)) {
      Serial.println(F("ok word add"));
    } else {
      Serial.println(F("err word list full"));
    }
  } else if (strncmp(command, "WORD_LIST ", 10) == 0) {
    clearChallengeWords();
    char* word = strtok(command + 10, ",");
    while (word != NULL && challengeWordCount < MAX_CHALLENGE_WORDS) {
      addChallengeWord(word);
      word = strtok(NULL, ",");
    }
    Serial.println(F("ok word list"));
  } else if (strncmp(command, "WORD_CELLS ", 11) == 0) {
    if (storeWordCells(command + 11)) {
      Serial.println(F("ok word cells"));
    } else {
      Serial.println(F("err invalid word cells"));
    }
  } else if (strcmp(command, "WORD_PREV") == 0) {
    selectPreviousWord();
    Serial.println(F("ok word prev"));
  } else if (strcmp(command, "WORD_NEXT") == 0) {
    selectNextWord();
    Serial.println(F("ok word next"));
  } else if (strcmp(command, "CHALLENGE_START") == 0) {
    if (challengeWordCount > 0) {
      selectedWordIndex = 0;
      challengeMode = true;
      challengeChoicePending = false;
      showSelectedWordInRed();
      displayDirty = true;
      Serial.println(F("ok challenge start"));
    } else {
      Serial.println(F("err no challenge words"));
    }
  } else if (strcmp(command, "CHALLENGE_CANCEL") == 0) {
    challengeMode = false;
    challengeChoicePending = false;
    clearLights();
    challengeLightsActive = false;
    showPremiumLights();
    displayDirty = true;
    Serial.println(F("ok challenge cancel"));
  } else if (strcmp(command, "CHALLENGE_TAKE") == 0) {
    if (challengeChoicePending) {
      Serial.print(F("ok challenge chosen "));
      Serial.print(selectedWordIndex);
      Serial.print(F(" "));
      Serial.println(selectedChallengeWord());
      challengeChoicePending = false;
      displayDirty = true;
    } else {
      Serial.println(F("ok challenge none"));
    }
  } else if (strcmp(command, "BOARD_UP") == 0) {
    turnSystemOn();
    Serial.println(F("ok board up"));
  } else if (strcmp(command, "BOARD_DOWN") == 0) {
    turnSystemOff();
    Serial.println(F("ok board down"));
  } else if (strcmp(command, "SCAN_START") == 0) {
    boardScanning = true;
    clearLights();
    Serial.println(F("ok scan start"));
  } else if (strcmp(command, "SCAN_END") == 0) {
    boardScanning = false;
    showPremiumLights();
    Serial.println(F("ok scan end"));
  } else if (strncmp(command, "LED_CELLS ", 10) == 0) {
    if (showCellsInRed(command + 10)) {
      Serial.println(F("ok led cells"));
    } else {
      Serial.println(F("err invalid led cells"));
    }
  } else if (strcmp(command, "LED_TEST") == 0) {
    turnSystemOn();
    Serial.println(F("ok led test"));
  } else if (strcmp(command, "LED_CLEAR") == 0) {
    turnSystemOff();
    Serial.println(F("ok led clear"));
  } else if (strcmp(command, "PING") == 0) {
    Serial.println(F("ok board actuator"));
  } else if (command[0] != '\0') {
    Serial.print(F("err unknown command "));
    Serial.println(command);
  }
}

void readSerialCommands() {
  while (Serial.available() > 0) {
    char incoming = Serial.read();

    if (incoming == '\n' || incoming == '\r') {
      if (serialLength > 0) {
        serialLine[serialLength] = '\0';
        handleSerialCommand(serialLine);
        serialLength = 0;
      }
    } else if (serialLength < sizeof(serialLine) - 1) {
      serialLine[serialLength++] = incoming;
    }
  }
}

void setup() {
  Serial.begin(115200);
  pinMode(ON_BUTTON_PIN, INPUT_PULLUP);
  timerButton.begin();
  challengeButton.begin();
  previousButton.begin();
  nextButton.begin();

  strip.begin();
  strip.setBrightness(80);
  clearLights();

  actuatorServo.attach(ACTUATOR_SERVO_PIN);
  actuatorServo.write(ACTUATOR_DOWN_ANGLE);
  delay(400);
  actuatorServo.detach();

  screen.begin();
  drawScreen();
  displayDirty = false;

  Serial.println(F("SCRABBLE_BOARD_ACTUATOR_READY"));
  Serial.println(F("ok ready"));
}

void loop() {
  readSerialCommands();
  updateOnButton();
  updateTimerButton();
  updateChallengeButtons();
  updateActuator();
  updateBlueAnimation();
  updateEndingAnimation();
  updateScreen();
}
