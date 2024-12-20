import numpy as np
import cv2


def gbtf(mosaic):
    mask = np.ones_like(mosaic)

    # list[start:end:step]
    # red
    mask[0, ::2, 0::2] = 0
    mask[0, 1::2, :] = 0

    # green
    mask[1, ::2, 1::2] = 0
    mask[1, 1::2, ::2] = 0

    # blue
    mask[2, 0::2, :] = 0
    mask[2, 1::2, 1::2] = 0
    green, dif = green_interpolation(mosaic, mask)
    red = red_interpolation(green, mosaic, mask, dif)
    blue = blue_interpolation(green, mosaic, mask, dif)

    # result image
    rgb_size = mosaic.shape
    rgb_dem = np.zeros((3, rgb_size[1], rgb_size[2]))
    rgb_dem[0, :, :] = red
    rgb_dem[1, :, :] = green
    rgb_dem[2, :, :] = blue

    return rgb_dem


def green_interpolation(mosaic, mask):
    # imask
    imask = (mask == 0)

    # raw CFA data
    rawq = np.sum(mosaic, axis=0)

    ### Calculate Horizontal and Vertical Color Differences ###
    # mask
    size_rawq = rawq.shape
    maskGr = np.zeros((size_rawq[0], size_rawq[1]))
    maskGb = np.zeros((size_rawq[0], size_rawq[1]))

    maskGr[0::2, 0::2] = 1
    maskGb[1::2, 1::2] = 1

    # guide image
    Kh = np.array([[1 / 2, 0, 1 / 2]])

    difh, difv, difv2, difh2 = haresidual(rawq, mask, maskGr, maskGb, mosaic)

    # directional weight
    Ke = np.array([[0, 0, 0, 0, 26, 24, 21, 17, 12]]) / 100
    Kw = np.array([[12, 17, 21, 24, 26, 0, 0, 0, 0]]) / 100
    Ks = Ke.T
    Kn = Kw.T
    Ws, Wn, Ww, We = Means4Weights(difh2, difv2)

    # combine directional color differences
    difn = cv2.filter2D(difv.astype('float32'), -1, kernel=Kn, borderType=cv2.BORDER_REPLICATE)
    difs = cv2.filter2D(difv.astype('float32'), -1, kernel=Ks, borderType=cv2.BORDER_REPLICATE)
    difw = cv2.filter2D(difh.astype('float32'), -1, kernel=Kw, borderType=cv2.BORDER_REPLICATE)
    dife = cv2.filter2D(difh.astype('float32'), -1, kernel=Ke, borderType=cv2.BORDER_REPLICATE)

    Wt = Ww + We + Wn + Ws
    dif = (Wn * difn + Ws * difs + Ww * difw + We * dife) / Wt

    # Calculate Green by adding bayer raw data
    green = dif + rawq
    green = green * imask[1, :, :] + rawq * mask[1, :, :]

    # clip to 0-255
    green = np.clip(green, 0, 1)

    return green, dif


def red_interpolation(green, mosaic, mask, dif):
    Prb = np.array([[0, 0, -1, 0, -1, 0, 0], [0, 0, 0, 0, 0, 0, 0], [-1, 0, 10, 0, 10, 0, -1],
                    [0, 0, 0, 0, 0, 0, 0], [-1, 0, 10, 0, 10, 0, -1], [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, -1, 0, -1, 0, 0]]) / 32
    Aknl = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]) / 4
    red = mosaic[0, :, :] + mask[2, :, :] * (
                green - cv2.filter2D(dif.astype('float32'), -1, kernel=Prb, borderType=cv2.BORDER_REPLICATE))
    tempimg = mosaic[1, :, :] - mask[1, :, :] * cv2.filter2D(green.astype('float32'), -1, kernel=Aknl,
                                                                borderType=cv2.BORDER_REPLICATE) \
                + mask[1, :, :] * cv2.filter2D(red.astype('float32'), -1, kernel=Aknl,
                                                borderType=cv2.BORDER_REPLICATE)
    red = red + tempimg

    # R interpolation
    red = np.clip(red, 0, 255)

    return red


def blue_interpolation(green, mosaic, mask, dif):
    Prb = np.array([[0, 0, -1, 0, -1, 0, 0], [0, 0, 0, 0, 0, 0, 0], [-1, 0, 10, 0, 10, 0, -1],
                    [0, 0, 0, 0, 0, 0, 0], [-1, 0, 10, 0, 10, 0, -1], [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, -1, 0, -1, 0, 0]]) / 32
    Aknl = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]) / 4
    blue = mosaic[2, :, :] + mask[0, :, :] * (
                green - cv2.filter2D(dif.astype('float32'), -1, kernel=Prb, borderType=cv2.BORDER_REPLICATE))
    tempimg = mosaic[1, :, :] - mask[1, :, :] * cv2.filter2D(green.astype('float32'), -1, kernel=Aknl,
                                                                borderType=cv2.BORDER_REPLICATE) \
                + mask[1, :, :] * cv2.filter2D(blue.astype('float32'), -1, kernel=Aknl,
                                                borderType=cv2.BORDER_REPLICATE)
    blue = blue + tempimg

    # blue interpolation
    blue = np.clip(blue, 0, 255)

    return blue


def haresidual(rawq, mask, maskGr, maskGb, mosaic):
    f = np.array([[1 / 4, 1 / 2, -1 / 2, 1 / 2, 1 / 4]])
    rawh = cv2.filter2D(rawq.astype('float32'), -1, kernel=f, borderType=cv2.BORDER_REPLICATE)
    rawv = cv2.filter2D(rawq.astype('float32'), -1, kernel=f.T, borderType=cv2.BORDER_REPLICATE)

    # tentative image
    Grh = rawh * mask[0, :, :]
    Gbh = rawh * mask[2, :, :]
    Rh = rawh * maskGr
    Bh = rawh * maskGb
    Grv = rawv * mask[0, :, :]
    Gbv = rawv * mask[2, :, :]
    Rv = rawv * maskGb
    Bv = rawv * maskGr

    # vertical and horizontal color difference
    difh = mosaic[1, :, :] + Grh + Gbh - mosaic[0, :, :] - mosaic[2, :, :] - Rh - Bh
    difv = mosaic[1, :, :] + Grv + Gbv - mosaic[0, :, :] - mosaic[2, :, :] - Rv - Bv

    ### Combine Vertical and Horizontal Color Differences ###
    # color difference gradient
    Kh = np.array([[1, 0, -1]])
    Kv = Kh.T
    AvK = np.array([[1, 1, 1]])
    difh2 = abs(cv2.filter2D(difh.astype('float32'), -1, kernel=Kh, borderType=cv2.BORDER_REPLICATE))
    difh2 = cv2.filter2D(difh2.astype('float32'), -1, kernel=AvK.T, borderType=cv2.BORDER_REPLICATE)
    difv2 = abs(cv2.filter2D(difv.astype('float32'), -1, kernel=Kv, borderType=cv2.BORDER_REPLICATE))
    difv2 = cv2.filter2D(difv2.astype('float32'), -1, kernel=AvK, borderType=cv2.BORDER_REPLICATE)

    return difh, difv, difv2, difh2


def Means4Weights(difh2, difv2):
    K = np.multiply(cv2.getGaussianKernel(5, 2), (cv2.getGaussianKernel(5, 2)).T)
    Kw = np.array([[1, 0, 0]])
    Ke = np.array([[0, 0, 1]])

    wh = cv2.filter2D(difh2.astype('float32'), -1, kernel=K, borderType=cv2.BORDER_REPLICATE)
    wv = cv2.filter2D(difv2.astype('float32'), -1, kernel=K, borderType=cv2.BORDER_REPLICATE)

    Ks = Ke.T
    Kn = Kw.T

    Ww = cv2.filter2D(wh.astype('float32'), -1, kernel=Kw, borderType=cv2.BORDER_REPLICATE)
    We = cv2.filter2D(wh.astype('float32'), -1, kernel=Ke, borderType=cv2.BORDER_REPLICATE)
    Wn = cv2.filter2D(wv.astype('float32'), -1, kernel=Kn, borderType=cv2.BORDER_REPLICATE)
    Ws = cv2.filter2D(wv.astype('float32'), -1, kernel=Ks, borderType=cv2.BORDER_REPLICATE)

    Ww = 1 / (Ww * Ww + 1e-32)
    We = 1 / (We * We + 1e-32)
    Ws = 1 / (Ws * Ws + 1e-32)
    Wn = 1 / (Wn * Wn + 1e-32)

    return Ws, Wn, Ww, We

