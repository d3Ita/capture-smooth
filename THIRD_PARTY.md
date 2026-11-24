# Third-party components

| Component | Use | Licence |
|---|---|---|
| [RIFE / Practical-RIFE](https://github.com/hzwer/Practical-RIFE) (hzwer et al.) | network architecture (`rife.py`) | MIT |
| RIFE 4.6 weights, via [vs-rife](https://github.com/HolyWu/vs-rife) | `models/flownet_v4.6.pkl` and the ONNX files exported from it | MIT |
| [PyTorch](https://pytorch.org) | CUDA engine | BSD-3-Clause |
| NVIDIA CUDA runtime, cuBLAS, cuDNN… (shipped inside PyTorch) | CUDA engine | NVIDIA redistributable licences |
| [ONNX Runtime](https://github.com/microsoft/onnxruntime) | DirectML engine | MIT |
| [DirectML](https://github.com/microsoft/DirectML) | DirectML engine | Microsoft DirectML licence (redistributable) |
| [OpenCV](https://opencv.org) | capture | Apache-2.0 |
| [PyAV](https://github.com/PyAV-Org/PyAV) | raw capture (NVIDIA build) | BSD-3-Clause |
| [FFmpeg](https://ffmpeg.org) libraries (shipped inside PyAV, unmodified shared libraries) | raw capture (NVIDIA build) | LGPL-3.0-or-later |
| [Qt for Python (PySide6)](https://www.qt.io/qt-for-python) and the Qt libraries it ships (unmodified shared libraries) | settings window | LGPL-3.0 |
| [QtPy](https://github.com/spyder-ide/qtpy) | settings window (Qt binding layer) | MIT |
| [windows-capture](https://github.com/NiiightmareXD/windows-capture) | PS Remote Play window capture | MIT |
| [NumPy](https://numpy.org) | arrays | BSD-3-Clause |
| [GLFW](https://www.glfw.org) and [pyGLFW](https://github.com/FlorianRhiem/pyGLFW) | window | zlib / MIT |
| [PyOpenGL](https://pyopengl.sourceforge.net) | display | BSD-3-Clause |
| [python-sounddevice](https://github.com/spatialaudio/python-sounddevice) and PortAudio | sound | MIT |
| [comtypes](https://github.com/enthought/comtypes) | capture device names | MIT |
| [PyInstaller](https://pyinstaller.org) | .exe packaging | GPL-2.0 with bootloader exception |

**Not shipped:** NVIDIA Video Effects SDK Python package [`nvidia-vfx`](https://pypi.org/project/nvidia-vfx/) (Video Frame Generation), NVIDIA proprietary licence ([NVIDIA Software License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-software-license-agreement/), [Product-Specific Terms for NVIDIA AI Products](https://www.nvidia.com/en-us/agreements/enterprise-software/product-specific-terms-for-ai-products/)). The user may download it from pypi.nvidia.com after accepting that licence in the settings window (`vfg_download.py`); its licence files are kept next to it. Capture Smooth's MIT licence does not cover it.

**Logo and icon** (`assets/logo_source.png`, made into `logo.png` and `icon.ico` by `tools/make_logo.py`): generated locally with the Qwen-Image model (through Prisme), for this project.

The full licence texts ship with each package (see the `_internal` folder of a release).
