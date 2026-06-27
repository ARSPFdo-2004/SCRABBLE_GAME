w# Scrabble Join

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

You can also run the package command directly:

```powershell
python -m scrabble_plotter gui
```

## What The GUI Does

- Start and stop the camera preview.
- Calibrate the board corners from the camera.
- Take a still picture, capture letters, and detect horizontal or vertical words.
- Fill the 12x12 board matrix and calculate the board score.
- Preview and send plotter moves to board squares from A1 to L12.
- Send stepper scale settings, move to the cart position, or reset the controller.

The default calibration file is stored in this folder as `scrabble_plotter_calibration.json`.
