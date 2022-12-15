from fastai.vision import *
from fastai.vision.data import ImageDataLoaders
from pathlib import Path

# Data Augmentations

def xtra_tfms(base_size = 75*5):
    box_dim = int(base_size/4)
    cutout_ = partial(cutout, p = .8, n_holes = (1,1), length = (box_dim, box_dim))
    jitter_ = partial(jitter, p = 0.5,  magnitude = (0.005, 0.01))
    skew_   = partial(skew,   p = 0.5, direction = (0, 7), magnitude = 0.2)
    squish_ = partial(squish, p = 0.5, row_pct = 0.25, col_pct = 0.25)
    tilt_   = partial(tilt,   p = 0.5, direction = (0, 3))
    perp_warp_ = partial(perspective_warp, p = 0.5, magnitude = (-0.2, 0.2))
    crop_pad_  = partial(crop_pad, p = 0.5, padding_mode = 'border', row_pct = 0.1, col_pct = 0.1)
    rgb_randomize_ = partial(rgb_randomize, thresh=0.05)

    xtra_tfms = [jitter_(), skew_(), squish_(), perp_warp_(),
                 tilt_(), cutout_(), crop_pad_()]

    return xtra_tfms

def get_tfms(): return get_transforms(do_flip = True,
                                      flip_vert = False,
                                      max_zoom = 1.,
                                      max_lighting = 0.4,
                                      max_warp = 0.3,
                                      p_affine = 0.85,
                                      p_lighting = 0.85,
                                      xtra_tfms = xtra_tfms())

def get_model_data(path):
    path = Path(path)
    data = ImageDataLoaders.from_folder(path, 'train', 'valid', size = (375, 666), ds_tfms = get_tfms(), bs=1,
                                      resize_method = ResizeMethod.SQUISH,
                                      num_workers = 0
                                     ).normalize(imagenet_stats)

    learn = cnn_learner(data, models.resnet50, metrics = [accuracy], pretrained=True)
    learn = learn.to_fp16()

    learn.load(path/'models'/'shot-type-classifier');

    return learn, data
