# freq-painter

A small toy that lets you paint/draw the same image in both the spatial and
frequency domains at the same time. Two panes appear side by side: the left
shows the image in the spatial domain, the right shows its (log-magnitude,
fft-shifted) Fourier transform. Painting on either pane updates the single
underlying image and both views stay in sync.

Single-channel (grayscale) only.

## Install

```bash
pip install -r requirements.txt
```

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

A stroke stays on the pane where it started until you release the button, so
you can drag off and back on without the brush jumping to the other pane.

## How freq-domain painting works

The canonical state is a single 2D real-valued grayscale array. The right
pane displays `log(1 + |fftshift(fft2(image))|)`.

When you paint on the right pane, the brush adjusts the magnitude of the FFT
in log space (so the brush is perceptually uniform with the displayed view)
while preserving the existing phase. The result is then converted back to
the spatial domain via `real(ifft2(ifftshift(F)))`, which both keeps the
spatial image real-valued and ensures the next FFT we display is
Hermitian-symmetric automatically.

### Limitations

- Because the freq-domain edit round-trips through `real(ifft2(...))`, the
  imaginary component caused by an asymmetric edit is silently discarded.
  In practice this means roughly half the "energy" of a single-pixel
  freq-domain stroke is lost. This is the intended trade-off for not having
  to manage Hermitian conjugate-symmetric mirroring by hand.
- Phase is preserved internally but not visualized.
- One color channel only.
