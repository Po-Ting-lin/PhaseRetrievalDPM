from PhaseRetrival import PhaseRetrieval, Offset
import cv2
from matplotlib import pyplot as plt
import time

def main():
    sp = cv2.imread(r"sp1.bmp", cv2.IMREAD_GRAYSCALE)
    bg = cv2.imread(r"bg1.bmp", cv2.IMREAD_GRAYSCALE)
    offset = Offset()
    offset.spx = 0
    offset.spy = 0
    offset.bgx = 0
    offset.bgy = 0
    image_width, image_height = sp.shape

    phase_retrieval = PhaseRetrieval(image_width, image_height, offset)
    time1 = time.time()
    result = phase_retrieval.phase_retrieval_gpu(sp, bg)
    time2 = time.time()
    print("time consume: ", time2 - time1)

    plt.figure()
    plt.imshow(result, cmap='jet', vmin=-0.2, vmax=3.5)
    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    main()