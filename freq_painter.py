"""freq-painter: paint a grayscale image simultaneously in the spatial and frequency domains.

Left pane shows the image in the spatial domain. Right pane shows
``log(1 + |fftshift(fft2(image))|)``. Painting on either pane updates a single
canonical 2D float array; both views are re-derived from it after every edit.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

SAVE_PATH = "out.png"
DEFAULT_SIZE = 256
DEFAULT_BRUSH_RADIUS = 5
DEFAULT_BRUSH_STRENGTH = 0.2


class FreqPainter:
    """Owns the canonical image, both matplotlib views, and all input handling."""

    def __init__(self, image: np.ndarray) -> None:
        if image.ndim != 2:
            raise ValueError(f"image must be 2D, got shape {image.shape}")

        self.image = np.clip(image.astype(np.float64, copy=True), 0.0, 1.0)

        self.brush_radius: int = DEFAULT_BRUSH_RADIUS
        self.brush_strength: float = DEFAULT_BRUSH_STRENGTH

        self._active_button: int | None = None
        self._active_ax: plt.Axes | None = None

        self._disable_conflicting_keymaps()

        self.fig, (self.ax_spatial, self.ax_freq) = plt.subplots(
            1, 2, figsize=(11.0, 5.5)
        )
        try:
            self.fig.canvas.manager.set_window_title("freq-painter")
        except Exception:
            pass

        self.ax_spatial.set_title("spatial")
        self.ax_freq.set_title("frequency (log magnitude)")
        for ax in (self.ax_spatial, self.ax_freq):
            ax.set_xticks([])
            ax.set_yticks([])

        log_mag = self._log_mag(self.image)
        self.im_spatial = self.ax_spatial.imshow(
            self.image, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest"
        )
        self.im_freq = self.ax_freq.imshow(
            log_mag,
            cmap="gray",
            vmin=0.0,
            vmax=max(1e-6, float(log_mag.max())),
            interpolation="nearest",
        )

        self.fig.canvas.mpl_connect("button_press_event", self._on_press)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        self.fig.tight_layout()

    @staticmethod
    def _disable_conflicting_keymaps() -> None:
        # matplotlib's default keymap binds 's' to save-figure and 'c' to navigation-back.
        # We use those keys ourselves, so strip them from the rc keymaps.
        for rc_key, conflicts in (
            ("keymap.save", {"s", "ctrl+s"}),
            ("keymap.back", {"c"}),
        ):
            current = plt.rcParams.get(rc_key, [])
            plt.rcParams[rc_key] = [k for k in current if k not in conflicts]

    @staticmethod
    def _log_mag(image: np.ndarray) -> np.ndarray:
        F_shift = np.fft.fftshift(np.fft.fft2(image))
        return np.log1p(np.abs(F_shift))

    def show(self) -> None:
        self._print_help()
        plt.show()

    def _print_help(self) -> None:
        print(
            "freq-painter controls:\n"
            "  Left-click + drag : paint (additive)\n"
            "  Right-click + drag: erase (subtractive)\n"
            "  [ / ]             : brush smaller / bigger\n"
            "  - / =             : brush weaker / stronger\n"
            "  c                 : clear to black\n"
            f"  s                 : save spatial image to {SAVE_PATH}\n"
            f"  starting brush: radius={self.brush_radius}, "
            f"strength={self.brush_strength:.2f}"
        )

    # ------------------------------------------------------------------ view

    def _refresh_views(self) -> None:
        log_mag = self._log_mag(self.image)
        self.im_spatial.set_data(self.image)
        self.im_freq.set_data(log_mag)
        self.im_freq.set_clim(vmin=0.0, vmax=max(1e-6, float(log_mag.max())))
        self.fig.canvas.draw_idle()

    # ----------------------------------------------------------------- brush

    def _disk_mask(
        self, cy: int, cx: int, shape: tuple[int, int]
    ) -> np.ndarray:
        """Cosine-falloff disk centered at (cy, cx), zero outside ``brush_radius``."""
        h, w = shape
        r = max(1, self.brush_radius)
        y0 = max(0, cy - r)
        y1 = min(h, cy + r + 1)
        x0 = max(0, cx - r)
        x1 = min(w, cx + r + 1)
        mask = np.zeros(shape, dtype=np.float64)
        if y1 <= y0 or x1 <= x0:
            return mask
        ys = np.arange(y0, y1) - cy
        xs = np.arange(x0, x1) - cx
        yy, xx = np.meshgrid(ys, xs, indexing="ij")
        d = np.sqrt(yy * yy + xx * xx) / r
        local = np.where(d < 1.0, 0.5 * (1.0 + np.cos(np.pi * d)), 0.0)
        mask[y0:y1, x0:x1] = local
        return mask

    # ----------------------------------------------------------------- paint

    def _paint_spatial(self, cy: int, cx: int, sign: int) -> None:
        mask = self._disk_mask(cy, cx, self.image.shape)
        self.image = np.clip(
            self.image + sign * self.brush_strength * mask, 0.0, 1.0
        )

    def _paint_freq(self, cy: int, cx: int, sign: int) -> None:
        F = np.fft.fftshift(np.fft.fft2(self.image))
        mag = np.abs(F)
        phase = np.angle(F)

        mask = self._disk_mask(cy, cx, F.shape)

        # Scale the log-magnitude brush by log(N*M) so a stroke produces
        # a visible spatial response across typical canvas sizes despite
        # numpy's unnormalized FFT (forward fft scales by N*M, inverse by 1).
        # The brush is still applied in log-magnitude space, matching what
        # the user sees on the right pane.
        freq_gain = float(np.log(max(self.image.size, 2)))
        log_mag = np.log1p(mag) + sign * self.brush_strength * freq_gain * mask
        new_mag = np.expm1(np.maximum(log_mag, 0.0))

        F_new = new_mag * np.exp(1j * phase)
        # Round-tripping through .real enforces a real-valued spatial image
        # without manual Hermitian-conjugate mirroring; the next _refresh_views
        # rederives a self-consistent symmetric F for display.
        new_image = np.fft.ifft2(np.fft.ifftshift(F_new)).real
        self.image = np.clip(new_image, 0.0, 1.0)

    # ---------------------------------------------------------------- events

    def _coords(self, event) -> tuple[int, int] | None:
        if event.xdata is None or event.ydata is None:
            return None
        h, w = self.image.shape
        cx = int(round(event.xdata))
        cy = int(round(event.ydata))
        if 0 <= cy < h and 0 <= cx < w:
            return cy, cx
        return None

    def _paint_at(self, ax: plt.Axes, cy: int, cx: int, sign: int) -> None:
        if ax is self.ax_spatial:
            self._paint_spatial(cy, cx, sign)
        else:
            self._paint_freq(cy, cx, sign)
        self._refresh_views()

    def _on_press(self, event) -> None:
        if event.inaxes not in (self.ax_spatial, self.ax_freq):
            return
        if event.button not in (1, 3):
            return
        coords = self._coords(event)
        if coords is None:
            return
        self._active_button = int(event.button)
        self._active_ax = event.inaxes
        sign = 1 if self._active_button == 1 else -1
        cy, cx = coords
        self._paint_at(self._active_ax, cy, cx, sign)

    def _on_motion(self, event) -> None:
        if self._active_button is None or self._active_ax is None:
            return
        if event.inaxes is not self._active_ax:
            return
        coords = self._coords(event)
        if coords is None:
            return
        sign = 1 if self._active_button == 1 else -1
        cy, cx = coords
        self._paint_at(self._active_ax, cy, cx, sign)

    def _on_release(self, event) -> None:
        if self._active_button is not None and int(event.button) == self._active_button:
            self._active_button = None
            self._active_ax = None

    def _on_key(self, event) -> None:
        key = event.key
        if key == "[":
            self.brush_radius = max(1, self.brush_radius - 1)
            print(f"brush radius = {self.brush_radius}")
        elif key == "]":
            max_r = max(1, min(self.image.shape) // 2)
            self.brush_radius = min(max_r, self.brush_radius + 1)
            print(f"brush radius = {self.brush_radius}")
        elif key == "-":
            self.brush_strength = max(0.01, self.brush_strength * 0.8)
            print(f"brush strength = {self.brush_strength:.3f}")
        elif key in ("=", "+"):
            self.brush_strength = min(4.0, self.brush_strength * 1.25)
            print(f"brush strength = {self.brush_strength:.3f}")
        elif key == "c":
            self.image[...] = 0.0
            self._refresh_views()
            print("cleared")
        elif key == "s":
            arr = (np.clip(self.image, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
            Image.fromarray(arr, mode="L").save(SAVE_PATH)
            print(f"saved {SAVE_PATH}")


def load_image(path: Path | None, size: int) -> np.ndarray:
    if path is None:
        return np.zeros((size, size), dtype=np.float64)
    with Image.open(path) as im:
        gray = im.convert("L").resize((size, size), Image.LANCZOS)
        arr = np.asarray(gray, dtype=np.float64) / 255.0
    return arr


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Paint a grayscale image simultaneously in the spatial and "
            "frequency domains."
        )
    )
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=None,
        help="Optional image file to load (anything Pillow can read). Defaults to a blank canvas.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=DEFAULT_SIZE,
        help=(
            "Canvas size NxN. If a path is given, the loaded image is resized "
            f"to this. Default {DEFAULT_SIZE}."
        ),
    )
    args = parser.parse_args()

    if args.size < 16:
        parser.error("--size must be at least 16")

    image = load_image(args.path, args.size)
    painter = FreqPainter(image)
    painter.show()


if __name__ == "__main__":
    main()
