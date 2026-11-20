"""Hides the Windows taskbar of the main screen while the app runs, and shows it again on exit.

Windows only hides the taskbar on its own for a window that covers the whole monitor, and such a window
makes the screen flash when focus moves to another monitor. So the app's window stays one pixel short of
fullscreen and the taskbar is hidden directly. The second screen's taskbar is left alone.

Run `python taskbar.py` to show the taskbar again if the app was killed before it could do it.
"""
import atexit
import ctypes

user32 = ctypes.windll.user32
SW_HIDE, SW_SHOW = 0, 5


def _main_taskbar():
    return user32.FindWindowW("Shell_TrayWnd", None)


def show_taskbar():
    hwnd = _main_taskbar()
    if hwnd:
        user32.ShowWindow(hwnd, SW_SHOW)


def hide_taskbar():
    hwnd = _main_taskbar()
    if not hwnd:
        return False
    atexit.register(show_taskbar)
    # Closing the console window skips atexit: show the taskbar from the console close handler too.
    handler = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_uint)(lambda event: (show_taskbar(), False)[1])
    hide_taskbar._handler = handler  # keep a reference so it is not garbage collected
    ctypes.windll.kernel32.SetConsoleCtrlHandler(handler, True)
    user32.ShowWindow(hwnd, SW_HIDE)
    return True


if __name__ == "__main__":
    show_taskbar()
    print("Barre des tâches réaffichée.")
