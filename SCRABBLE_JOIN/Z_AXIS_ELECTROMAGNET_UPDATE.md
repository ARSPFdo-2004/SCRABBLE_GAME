# Z Axis Actuator + Electromagnet Update

Implemented files in this workspace:

- `scrabble_plotter/gui.py` now adds the Z and relay controls inside the move plotter section.
- `arduino/CNCV3_XY_Z_Magnet/CNCV3_XY_Z_Magnet.ino` is the updated uploadable Arduino sketch.
- `Z_AXIS_IMPLEMENTED.md` has the final wiring and command summary.

## New Arduino pins

Use these Arduino pins directly, not through the CNC V3 shield motor outputs:

- MG995 servo signal/data: `A1` (`D15`)
- Relay module input for the electromagnet: `A2` (`D16`)

Keep the existing CNC shield pins for X/Y movement unchanged.

Important wiring notes:

- Do not power the MG995 from the Arduino 5V pin. Use an external 5V supply rated at least 2A, preferably 3A or more.
- Connect servo red to external 5V, servo brown/black to external GND, and servo orange/yellow signal to Arduino `A1`.
- Connect the external supply GND to Arduino GND.
- Do not connect the electromagnet directly to the Arduino pin. Arduino `A2` should only drive the relay module input.

## Arduino code to merge into the existing sketch

Add this near the top of the Arduino sketch:

```cpp
#include <Servo.h>

const byte Z_SERVO_PIN = A1;      // Arduino D15
const byte RELAY_PIN = A2;        // Arduino D16
const bool RELAY_ACTIVE_HIGH = false;

Servo zServo;

int zUpAngle = 20;                // original/up position
int zDownAngle = 80;              // adjustable down position
int zMinAngle = 0;
int zMaxAngle = 180;
```

Add this inside `setup()`:

```cpp
pinMode(RELAY_PIN, OUTPUT);
digitalWrite(RELAY_PIN, RELAY_ACTIVE_HIGH ? LOW : HIGH);

zServo.attach(Z_SERVO_PIN);
zServo.write(zUpAngle);
```

Add these helper functions:

```cpp
void zMoveUp() {
  zServo.write(zUpAngle);
}

void zMoveDown() {
  zServo.write(zDownAngle);
}

void zSetHeight(int angle) {
  zDownAngle = constrain(angle, zMinAngle, zMaxAngle);
}

void magnetOn() {
  digitalWrite(RELAY_PIN, RELAY_ACTIVE_HIGH ? HIGH : LOW);
}

void magnetOff() {
  digitalWrite(RELAY_PIN, RELAY_ACTIVE_HIGH ? LOW : HIGH);
}
```

Add these commands to the existing serial command parser:

```cpp
// ZU        -> move actuator back to original/up position
// ZD        -> move actuator down to the saved adjustable height
// ZH90      -> set Z down height/angle to 90 degrees
// M1        -> electromagnet ON
// M0        -> electromagnet OFF

if (cmd == "ZU") {
  zMoveUp();
  Serial.println("OK Z UP");
}
else if (cmd == "ZD") {
  zMoveDown();
  Serial.println("OK Z DOWN");
}
else if (cmd.startsWith("ZH")) {
  int angle = cmd.substring(2).toInt();
  zSetHeight(angle);
  Serial.print("OK Z HEIGHT ");
  Serial.println(zDownAngle);
}
else if (cmd == "M1") {
  magnetOn();
  Serial.println("OK MAGNET ON");
}
else if (cmd == "M0") {
  magnetOff();
  Serial.println("OK MAGNET OFF");
}
```

If the existing parser reads single characters instead of full command strings, add equivalent cases:

```cpp
case 'U':
  zMoveUp();
  break;

case 'D':
  zMoveDown();
  break;

case 'N':
  magnetOn();
  break;

case 'F':
  magnetOff();
  break;
```

For adjustable height with single-character parsing, send a full line such as `ZH80` from the app and handle that before the single-character switch.

## UI/app buttons to add

Add four buttons plus one numeric control:

- `Z Down`: sends `ZD`
- `Z Up`: sends `ZU`
- `Relay ON`: sends `R1`
- `Relay OFF`: sends `R0`
- `Z Height`: numeric value/slider from `0` to `180`; when changed, sends `ZH<value>`, for example `ZH80`

Start with:

- `zUpAngle = 20`
- `zDownAngle = 80`

Then adjust `zDownAngle` slowly until the actuator reaches the needed Scrabble tile pickup/drop height.
