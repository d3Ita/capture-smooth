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
| [NumPy](https://numpy.org) | arrays | BSD-3-Clause |
| [GLFW](https://www.glfw.org) and [pyGLFW](https://github.com/FlorianRhiem/pyGLFW) | window | zlib / MIT |
| [PyOpenGL](https://pyopengl.sourceforge.net) | display | BSD-3-Clause |
| [python-sounddevice](https://github.com/spatialaudio/python-sounddevice) and PortAudio | sound | MIT |
| [comtypes](https://github.com/enthought/comtypes) | capture device names | MIT |
| [PyInstaller](https://pyinstaller.org) | .exe packaging | GPL-2.0 with bootloader exception |

The full licence texts ship with each package (see the `_internal` folder of a release).
