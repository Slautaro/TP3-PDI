import os

import cv2
import numpy as np

try:
    import rawpy
except ImportError:
    rawpy = None

EXTENSIONES_RAW = {
    ".raw", ".cr2", ".cr3", ".nef", ".arw", ".dng", ".orf",
    ".rw2", ".raf", ".pef", ".srw", ".3fr", ".erf", ".kdc",
    ".mrw", ".x3f",
}

EXTENSIONES_RAW_SIN_ENCABEZADO = {".raw"}

class DimensionesRawRequeridas(IOError):
    def __init__(self, ruta: str):
        self.ruta = ruta
        super().__init__(
            "El archivo .raw no tiene encabezado con las dimensiones (no es "
            f"un RAW de cámara). Indicá ancho y alto para poder cargarlo: {ruta}"
        )

def _leer_bytes_archivo(ruta: str) -> np.ndarray | None:
    try:
        datos = np.fromfile(ruta, dtype=np.uint8)
    except OSError:
        return None
    return datos if datos.size else None

def _cargar_raw(ruta: str) -> np.ndarray:
    extension = os.path.splitext(ruta)[1].lower()
    puede_ser_sin_encabezado = extension in EXTENSIONES_RAW_SIN_ENCABEZADO

    if rawpy is None:
        if puede_ser_sin_encabezado:
            raise DimensionesRawRequeridas(ruta)
        raise IOError(
            "No se pudo cargar el archivo RAW porque falta la librería "
            "'rawpy'. Instalala con: pip install rawpy"
        )

    try:
        with rawpy.imread(ruta) as raw:
            rgb = raw.postprocess()
    except Exception as error:
        if puede_ser_sin_encabezado:
            raise DimensionesRawRequeridas(ruta) from error
        raise IOError(
            f"El archivo RAW es inválido o está corrupto: {ruta} ({error})"
        ) from error
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

def cargar_raw_sin_encabezado(
    ruta: str, ancho: int, alto: int, canales: int = 1
) -> np.ndarray:
    datos = _leer_bytes_archivo(ruta)
    if datos is None:
        raise IOError(f"No se pudo leer el archivo: {ruta}")

    bytes_esperados = ancho * alto * canales
    if datos.size != bytes_esperados:
        raise IOError(
            f"El tamaño del archivo no coincide con {ancho}x{alto} "
            f"({canales} canal/es): se esperaban {bytes_esperados} bytes y "
            f"el archivo tiene {datos.size}. Revisá ancho/alto: {ruta}"
        )

    forma = (alto, ancho) if canales == 1 else (alto, ancho, canales)
    return datos.reshape(forma).copy()

def _cargar_estandar(ruta: str) -> np.ndarray:
    datos = _leer_bytes_archivo(ruta)
    if datos is None:
        raise IOError(f"No se pudo leer el archivo: {ruta}")
    imagen = cv2.imdecode(datos, cv2.IMREAD_UNCHANGED)
    if imagen is None:
        raise IOError(
            f"El formato de imagen es inválido o el archivo está corrupto: {ruta}"
        )
    return imagen

def cargar_imagen(ruta: str) -> np.ndarray:
    if not os.path.isfile(ruta):
        raise IOError(f"El archivo no existe: {ruta}")

    extension = os.path.splitext(ruta)[1].lower()
    try:
        if extension in EXTENSIONES_RAW:
            imagen = _cargar_raw(ruta)
        else:
            imagen = _cargar_estandar(ruta)
    except IOError:
        raise
    except Exception as error:
        raise IOError(
            f"No se pudo cargar la imagen (formato inválido o archivo corrupto): "
            f"{ruta} ({error})"
        ) from error

    if imagen is None or imagen.size == 0:
        raise IOError(f"La imagen está vacía o corrupta: {ruta}")
    if imagen.ndim == 3 and imagen.shape[2] == 4:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGRA2BGR)
    elif imagen.ndim == 3 and imagen.shape[2] != 3:
        raise IOError(f"No se pudo interpretar la imagen (canales no soportados): {ruta}")
    return imagen

def guardar_imagen(ruta: str, imagen: np.ndarray) -> None:
    if imagen is None:
        raise ValueError("No hay imagen para guardar.")
    ok = cv2.imwrite(ruta, imagen)
    if not ok:
        raise IOError(f"No se pudo guardar la imagen en: {ruta}")
