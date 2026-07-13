# Scrabble Join

This folder contains the joined desktop app for the Scrabble project.

It combines:
- camera calibration, letter capture, and word detection from `SCRABBLE_GAME`
- plotter movement, COM-port control, step settings, cart movement, and reset from `Scrabble-2 new`

## Run

Install the Python dependencies once:

```powershell
cd "D:\Hardware project\SCRABBLE_GAME2\SCRABBLE_JOIN"
python -m venv .venv
.\.venv\Scripts\activate
pip install -r .\scrabble_plotter\requirements.txt
```

Open the GUI:

```powershell
python .\main.py
```

The GUI also starts a read-only website feed on port `8765`. It provides the scores, countdowns, current player, and latest 12x12 board grid. Its hotspot address is shown in the **Game State** panel. Paste that address into the Scrablify website on the other computer. If Windows asks, allow Python through the firewall for private networks.

You can also run the package command directly:

```powershell
python -m scrabble_plotter gui
```

## What The GUI Does

- Start and stop the camera preview.
- Calibrate the board corners from the camera.
- Take a still picture, capture letters, and detect horizontal or vertical words.
- Fill the 12x12 board matrix and calculate the board score.
- Start each turn and its timer immediately while the tile cart moves to the active player in the background.
- Preview and send plotter moves to board squares from A1 to L12.
- Send stepper scale settings, move to the cart position, or reset the controller.

## Challenge Display

The GUI keeps the actuator display synchronized with Player 1's word history.
Player 2 presses the physical Challenge button to open that list, uses Previous
and Next to select a word, then presses Challenge again to submit it. The GUI
validates the selected word, applies the challenge score, and sends the updated
scores back to the display. Outside challenge selection, the screen shows the
live scores and turn countdown. Re-upload
`arduino/scrabble_board_actuator/scrabble_board_actuator.ino` after updating
the GUI so the physical buttons use the current protocol.

The default calibration file is stored in this folder as `scrabble_plotter_calibration.json`.
