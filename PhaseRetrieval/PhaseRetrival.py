import ctypes
import numpy as np
from pathlib import Path as path_func


class Offset(object):
    spx = 0
    spy = 0
    bgx = 0
    bgy = 0


class PhaseRetrieval(object):
    def __init__(self, width, height, offset, dll_path):
        self.width = width
        self.height = height
        self.offset = Offset()
        self.offset.spx = offset.spx
        self.offset.spy = offset.spy
        self.offset.bgx = offset.bgx
        self.offset.bgy = offset.bgy
        check_file_exist(dll_path, "PhaseRetrieval.dll")
        self.dll_path = dll_path

    def phase_retrieval_gpu(self, sp, bg):
        output_width = self.width // 4
        c_float_p = ctypes.POINTER(ctypes.c_float)
        c_uchar_p = ctypes.POINTER(ctypes.c_ubyte)
        dst = np.zeros((output_width, output_width)).astype(np.float32)

        my_dll = ctypes.CDLL(self.dll_path)
        phase_retriever = my_dll.PhaseRetriever
        phase_retriever.argtypes = [c_uchar_p, c_uchar_p, c_float_p, ctypes.c_int, ctypes.c_int]

        p1 = sp.ctypes.data_as(c_uchar_p)
        p2 = bg.ctypes.data_as(c_uchar_p)
        p3 = dst.ctypes.data_as(c_float_p)
        p4 = ctypes.c_int(self.width)
        p5 = ctypes.c_int(self.height)
        p6 = ctypes.c_int(self.offset.spx)
        p7 = ctypes.c_int(self.offset.spy)
        p8 = ctypes.c_int(self.offset.bgx)
        p9 = ctypes.c_int(self.offset.bgy)
        phase_retriever(p1, p2, p3, p4, p5, p6, p7, p8, p9)
        return dst


def check_file_exist(this_path, text):
    my_file = path_func(this_path)
    if not my_file.exists():
        raise OSError("Cannot find " + str(text) + "!")