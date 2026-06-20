# Frequency Painter

A small program that lets you paint/draw the same image in both the spatial and frequency domains at the same time. Both views represent the same image and they sync automatically.

## Install
The only requirements are Python 3, numpy, matplotlib, and Pillow.

## Run

```bash
# Blank 256x256 canvas
python freq_painter.py

# Load an image file (any format Pillow can open). Resized to --size, square.
python freq_painter.py path/to/image.png

# Custom canvas size
python freq_painter.py --size 384
python freq_painter.py path/to/image.png --size 512
```

## Controls

- Left-click + drag on either pane: paint (additive).
- Right-click + drag on either pane: erase (subtractive).
- `[` / `]`: shrink / grow brush.
- `-` / `=`: decrease / increase brush strength.
- `c`: clear the canvas to black.
- `s`: save the current spatial image to `out.png`.

## Details
Phase is initialized as zero for a blank image and is preserved throughout the painting process (since frequency painting only affects amplitude). Due to the repeated conversion between the frequency and spatial domains, the program may have trouble maintining asymmetric images. Additionally, all images are in greyscale for simplicity.

*This program was created with heavy assistance from AI coding tools.*
