import cv2
import numpy as np

def _validar_coordenadas(imagen: np.ndarray, x: int, y: int) -> None:
    alto, ancho = imagen.shape[:2]
    if not (0 <= x < ancho and 0 <= y < alto):
        raise IndexError(f"Coordenada ({x}, {y}) fuera de rango ({ancho}x{alto}).")

def obtener_pixel(imagen: np.ndarray, x: int, y: int):
    _validar_coordenadas(imagen, x, y)
    if imagen.ndim == 3:
        return tuple(int(canal) for canal in imagen[y, x])
    return int(imagen[y, x])

def establecer_pixel(imagen: np.ndarray, x: int, y: int, valor) -> np.ndarray:
    _validar_coordenadas(imagen, x, y)
    resultado = imagen.copy()
    resultado[y, x] = valor
    return resultado

def copiar_region(imagen: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    x_min, x_max = sorted((x1, x2))
    y_min, y_max = sorted((y1, y2))
    alto, ancho = imagen.shape[:2]
    x_min, x_max = max(0, x_min), min(ancho, x_max + 1)
    y_min, y_max = max(0, y_min), min(alto, y_max + 1)
    if x_max <= x_min or y_max <= y_min:
        raise ValueError("La región seleccionada está vacía o fuera de la imagen.")
    return imagen[y_min:y_max, x_min:x_max].copy()

def redimensionar_imagen(imagen: np.ndarray, ancho: int, alto: int) -> np.ndarray:
    interpolacion = cv2.INTER_AREA if (ancho, alto) < imagen.shape[1::-1] else cv2.INTER_LINEAR
    return cv2.resize(imagen, (ancho, alto), interpolation=interpolacion)

def restar_imagenes(imagen_a: np.ndarray, imagen_b: np.ndarray) -> np.ndarray:
    if imagen_a.shape[:2] != imagen_b.shape[:2]:
        raise ValueError(
            "Las imágenes deben tener el mismo ancho y alto para restarlas "
            f"({imagen_a.shape[:2]} vs {imagen_b.shape[:2]})."
        )

    if imagen_a.ndim != imagen_b.ndim:
        if imagen_a.ndim == 2:
            imagen_a = cv2.cvtColor(imagen_a, cv2.COLOR_GRAY2BGR)
        else:
            imagen_b = cv2.cvtColor(imagen_b, cv2.COLOR_GRAY2BGR)

    return cv2.absdiff(imagen_a, imagen_b)

def estadisticas_region(imagen: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> dict:
    region = copiar_region(imagen, x1, y1, x2, y2)
    alto, ancho = region.shape[:2]
    cantidad_pixeles = alto * ancho

    estadisticas = {
        "ancho": ancho,
        "alto": alto,
        "cantidad_pixeles": cantidad_pixeles,
    }
    if region.ndim == 3:
        promedio_b, promedio_g, promedio_r = region.reshape(-1, 3).mean(axis=0)
        estadisticas["promedio_color"] = (float(promedio_r), float(promedio_g), float(promedio_b))
    else:
        estadisticas["promedio_gris"] = float(region.mean())
    return estadisticas
