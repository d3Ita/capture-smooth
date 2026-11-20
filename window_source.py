"""Helpers for the window capture source (PS Remote Play, or any window); the source itself is in smooth.py.

Works while the window is behind Capture Smooth (but not minimized). Only the video is kept: the title bar
and borders are cut off using the window's client area, then the picture is fitted to 16:9 (Remote Play
letterboxes the stream when the window is not 16:9).

Remote Play adds its own delay before Capture Smooth sees a frame (video encoding on the PS5, network,
decoding on the PC): typically 50 to 100 ms on a good home network, on top of Capture Smooth's own.
"""
import ctypes
import ctypes.wintypes as wt

user32 = ctypes.windll.user32
dwmapi = ctypes.windll.dwmapi
DWMWA_EXTENDED_FRAME_BOUNDS = 9


def find_windows(fragment, exclude=("Capture Smooth",)):
    """Visible top-level windows whose title contains fragment (case-insensitive): [(hwnd, title)]."""
    found = []
    frag = fragment.lower()

    @ctypes.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)
    def cb(hwnd, _):
        if user32.IsWindowVisible(hwnd):
            n = user32.GetWindowTextLengthW(hwnd)
            if n:
                buf = ctypes.create_unicode_buffer(n + 1)
                user32.GetWindowTextW(hwnd, buf, n + 1)
                title = buf.value
                if frag in title.lower() and not any(x.lower() in title.lower() for x in exclude):
                    found.append((hwnd, title))
        return True

    user32.EnumWindows(cb, 0)
    return found


def video_rect(hwnd, frame_w, frame_h):
    """Where the video is inside a captured frame of this window: (x, y, w, h) in frame pixels."""
    ext = wt.RECT()
    dwmapi.DwmGetWindowAttribute(hwnd, DWMWA_EXTENDED_FRAME_BOUNDS, ctypes.byref(ext), ctypes.sizeof(ext))
    client = wt.RECT()
    user32.GetClientRect(hwnd, ctypes.byref(client))
    origin = wt.POINT(0, 0)
    user32.ClientToScreen(hwnd, ctypes.byref(origin))
    ew, eh = max(ext.right - ext.left, 1), max(ext.bottom - ext.top, 1)
    sx, sy = frame_w / ew, frame_h / eh
    x, y = (origin.x - ext.left) * sx, (origin.y - ext.top) * sy
    w, h = client.right * sx, client.bottom * sy
    if w < 16 or h < 9:  # odd window: keep the whole frame
        x, y, w, h = 0, 0, frame_w, frame_h
    if w / h > 16 / 9:  # letterboxed video: keep the centred 16:9 part
        vw = h * 16 / 9
        x, w = x + (w - vw) / 2, vw
    else:
        vh = w * 9 / 16
        y, h = y + (h - vh) / 2, vh
    x, y = max(int(round(x)), 0), max(int(round(y)), 0)
    return x, y, min(int(round(w)), frame_w - x), min(int(round(h)), frame_h - y)
