## Introduction

This repo is the implementation of phase retrieval for diffraction phase microscopy. 
In the case of test image, the total elapsed time, including retriving and unwrapping in both sample and background image(3072 * 3072 pixel), is 59ms.

## Dependencies

* OpenCV 4.2.0
* CUDA 10.2 (cufft, thrust)

## References

```
[1] yohschang repo: https://github.com/yohschang/phase_retrival
[2] Backoach, Ohad, et al. 
    "Fast phase processing in off-axis holography by CUDA including parallel phase unwrapping." 
    Optics express 24.4 (2016): 3177-3188.
```
