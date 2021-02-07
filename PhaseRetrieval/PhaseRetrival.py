import ctypes
import numpy as np


class Offset(object):
    spx = 0
    spy = 0
    bgx = 0
    bgy = 0


class PhaseRetrieval(object):
    def __init__(self, width, height, offset):
        self.width = width
        self.height = height
        self.offset = Offset()
        self.offset.spx = offset.spx
        self.offset.spy = offset.spy
        self.offset.bgx = offset.bgx
        self.offset.bgy = offset.bgy

    def phase_retrieval_gpu(self, sp, bg):
        output_width = self.width // 4
        c_float_p = ctypes.POINTER(ctypes.c_float)
        c_uchar_p = ctypes.POINTER(ctypes.c_ubyte)
        dst = np.zeros((output_width, output_width)).astype(np.float32)

        my_dll = ctypes.CDLL("PhaseRetrieval.dll")
        function_func = my_dll.PhaseRetriever
        function_func.argtypes = [c_uchar_p, c_uchar_p, c_float_p, ctypes.c_int, ctypes.c_int]

        p1 = sp.ctypes.data_as(c_uchar_p)
        p2 = bg.ctypes.data_as(c_uchar_p)
        p3 = dst.ctypes.data_as(c_float_p)
        p4 = ctypes.c_int(self.width)
        p5 = ctypes.c_int(self.height)
        p6 = ctypes.c_int(self.offset.spx)
        p7 = ctypes.c_int(self.offset.spy)
        p8 = ctypes.c_int(self.offset.bgx)
        p9 = ctypes.c_int(self.offset.bgy)
        function_func(p1, p2, p3, p4, p5, p6, p7, p8, p9)
        return dst
