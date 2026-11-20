"""CUDA -> OpenGL interop: frames computed on the GPU go straight into the display texture.

Without it, every displayed frame makes a round trip: GPU -> PC memory (download) -> GPU (glTexSubImage2D),
about 2 to 3 ms per frame and a lot of memory traffic at 90 fps. Uses the CUDA runtime that ships with
PyTorch, so it shares PyTorch's CUDA context. Only possible when the OpenGL context runs on the NVIDIA card.
"""
import ctypes
import glob
import os

GL_TEXTURE_2D = 0x0DE1
WRITE_DISCARD = 2            # cudaGraphicsRegisterFlagsWriteDiscard
DEVICE_TO_DEVICE = 3         # cudaMemcpyDeviceToDevice


class CudaGLTexture:
    def __init__(self, texture_id):
        import torch
        dll = sorted(glob.glob(os.path.join(os.path.dirname(torch.__file__), "lib", "cudart64_*.dll")))
        if not dll:
            raise RuntimeError("cudart introuvable")
        rt = ctypes.CDLL(dll[-1])
        vp, sz = ctypes.c_void_p, ctypes.c_size_t
        rt.cudaSetDevice.argtypes = [ctypes.c_int]
        rt.cudaGraphicsGLRegisterImage.argtypes = [ctypes.POINTER(vp), ctypes.c_uint, ctypes.c_uint, ctypes.c_uint]
        rt.cudaGraphicsMapResources.argtypes = [ctypes.c_int, ctypes.POINTER(vp), vp]
        rt.cudaGraphicsUnmapResources.argtypes = [ctypes.c_int, ctypes.POINTER(vp), vp]
        rt.cudaGraphicsSubResourceGetMappedArray.argtypes = [ctypes.POINTER(vp), vp, ctypes.c_uint, ctypes.c_uint]
        rt.cudaMemcpy2DToArray.argtypes = [vp, sz, sz, vp, sz, sz, sz, ctypes.c_int]
        rt.cudaGraphicsUnregisterResource.argtypes = [vp]
        self.rt = rt
        self.check(rt.cudaSetDevice(0), "cudaSetDevice")
        self.res = vp()
        self.check(rt.cudaGraphicsGLRegisterImage(ctypes.byref(self.res), texture_id, GL_TEXTURE_2D, WRITE_DISCARD),
                   "cudaGraphicsGLRegisterImage")

    @staticmethod
    def check(err, what):
        if err:
            raise RuntimeError(f"{what}: erreur CUDA {err}")

    def upload(self, device_ptr, width, height):
        """Copies an RGBA uint8 image (width x height, rows packed) from GPU memory into the texture."""
        rt = self.rt
        self.check(rt.cudaGraphicsMapResources(1, ctypes.byref(self.res), None), "map")
        try:
            arr = ctypes.c_void_p()
            self.check(rt.cudaGraphicsSubResourceGetMappedArray(ctypes.byref(arr), self.res, 0, 0), "array")
            self.check(rt.cudaMemcpy2DToArray(arr, 0, 0, device_ptr, width * 4, width * 4, height, DEVICE_TO_DEVICE),
                       "copy")
        finally:
            rt.cudaGraphicsUnmapResources(1, ctypes.byref(self.res), None)

    def close(self):
        try:
            self.rt.cudaGraphicsUnregisterResource(self.res)
        except Exception:
            pass
