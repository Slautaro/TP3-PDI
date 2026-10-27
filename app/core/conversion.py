import numpy as np
from PIL import Image

def cv2_a_pil(imagen: np.ndarray) -> Image.Image:
    if imagen is None:
        return None
    if imagen.ndim == 3:
        imagen = imagen[:, :, ::-1]
    return Image.fromarray(imagen)
