from .model import (
    get_cam_model, 
    get_vgg16_model
)

MODEL_FACTORY = {
    'vgg_16': get_vgg16_model,
    'vgg_16_cam': get_cam_model
}