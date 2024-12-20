import numpy as np
import copy


def bayer(im, ismask=False):
    """Bayer mosaic.

  The patterned assumed is::

    G r
    b G

  Args:
    im (np.array): image to mosaic. Dimensions are [c, h, w]

  Returns:
    np.array: mosaicked image (if return_mask==False), or binary mask if (return_mask==True)
  """

    mask = np.ones_like(im)

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

    mosaic = im * mask
    imask = (mask == 0)

    if ismask:
        return mosaic, imask, mask
    else:
        return mosaic, imask


def bayer_to_mosaic(cfa):
    mask_one = np.zeros_like(cfa)
    # R
    mask_r = copy.deepcopy(mask_one)
    mask_r[0::2, 1::2] = 1
    mosaic_r = cfa * mask_r

    # G
    mask_g = copy.deepcopy(mask_one)
    mask_g[0::2, 0::2] = 1
    mask_g[1::2, 1::2] = 1
    mosaic_g = cfa * mask_g

    # B
    mask_b = copy.deepcopy(mask_one)
    mask_b[1::2, 0::2] = 1
    mosaic_b = cfa * mask_b

    mosaic = np.concatenate((mosaic_r[np.newaxis, ...], mosaic_g[np.newaxis, ...], mosaic_b[np.newaxis, ...]), 0)
    mask = np.concatenate((mask_r[np.newaxis, ...], mask_g[np.newaxis, ...], mask_b[np.newaxis, ...]), 0)
    imask = (mask == 0)

    return mosaic, imask

