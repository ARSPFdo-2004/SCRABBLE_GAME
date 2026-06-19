# Z Axis + Electromagnet Implementation

## New direct Arduino pins

- MG995 servo signal/data: `A1` / `D15`
- Relay module input for the electromagnet: `A2` / `D16`

Keep the CNC Shield V3 X/Y pins unchanged. The new pins are direct Arduino signal pins and are not connected to the X/Y stepper driver outputs.

## Wiring

- MG995 orange/yellow signal wire -> Arduino `A1`
- MG995 red wire -> external regulated `5V`
- MG995 brown/black wire -> external supply `GND`
- External supply `GND` -> Arduino `GND`
- Relay module input/IN -> Arduino `A2`
- Relay module VCC/GND -> relay control supply, with GND shared with Arduino if the module requires it
- Electromagnet power line -> relay `COM` and `NO` contacts, using a separate magnet power supply

Do not power the MG995 or electromagnet directly from the Arduino `5V` pin. `A2` only switches the relay input; the relay contacts switch the electromagnet supply.

## GUI commands

The controls are inside the normal move plotter section:

- `Z Down` -> sends `ZD`
- `Z Up` -> sends `ZU`
- `Set Height` -> sends `ZH<angle>`, for example `ZH80`
- `Relay ON` -> sends `R1`
- `Relay OFF` -> sends `R0`

The default down height is `80` degrees. Adjust it slowly from the GUI until the actuator reaches the required pickup/drop height.

## If the COM port says "Cannot configure port"

That error is from Windows before the app sends any movement command. Try this order:

1. Close Arduino IDE Serial Monitor, Arduino IDE Serial Plotter, PuTTY, and any other program connected to the Arduino.
2. Unplug the Arduino USB cable, wait a few seconds, and plug it back in directly to the computer.
3. Open Windows Device Manager -> `Ports (COM & LPT)` and find the Arduino/CH340/USB-Serial COM number.
4. In the Scrabble GUI, click refresh ports and select that COM port. Do not select `COM1` or a Bluetooth/unused serial port.
5. Use baud `115200` with `arduino/CNCV3_XY_Z_Magnet/CNCV3_XY_Z_Magnet.ino`.
6. If Device Manager shows a warning icon or the port keeps disappearing, reinstall the Arduino/CH340 USB driver or try another USB cable.

## Arduino sketch

Upload this sketch if you want the Arduino to handle both the existing CNC Shield V3 X/Y movement and the new Z/magnet commands:

`arduino/CNCV3_XY_Z_Magnet/CNCV3_XY_Z_Magnet.ino`

The sketch uses:

- X step: `D2`
- Y step: `D3`
- X direction: `D5`
- Y direction: `D6`
- CNC shield enable: `D8`
- MG995 signal: `A1` / `D15`
- Relay input for electromagnet: `A2` / `D16`

If either X or Y moves in the wrong physical direction after upload, change `X_DIR_INVERT` or `Y_DIR_INVERT` at the top of the sketch.

Most relay modules are active-low, so the sketch defaults to `RELAY_ACTIVE_HIGH = false`. If the relay works in reverse, change `RELAY_ACTIVE_HIGH` at the top of the sketch.

If the electromagnet is still always on even when the relay is off, move the magnet wire from relay `NC` to relay `NO`.
