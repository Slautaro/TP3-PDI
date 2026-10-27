import math

import cv2
import numpy as np
import customtkinter as ctk

from app.tps.tp1_operadores import ControladorTP1

SECCION_BORDES = "Bordes"
SECCION_FILTROS = "Filtros"
SECCION_UMBRAL = "Umbralización"

SIGMA_S = 2.0
SIGMA_R = 1000.0
LAMBDA_DIFUSION = 0.25

MASCARA_PREWITT = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1],
], dtype=float)

MASCARA_SOBEL = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1],
], dtype=float)

MASCARA_LAPLACE = np.array([
    [0, 1, 0],
    [1, -4, 1],
    [0, 1, 0],
], dtype=float)

class ControladorTP2(ControladorTP1):
    def __init__(self, app):
        super().__init__(app)

    def registrar(self, barra_lateral) -> None:
        self._construir_seccion_bordes(barra_lateral)
        self._construir_seccion_filtros(barra_lateral)
        self._construir_seccion_umbral(barra_lateral)


    def _construir_seccion_bordes(self, barra_lateral) -> None:
        barra_lateral.agregar_boton(SECCION_BORDES, "Prewitt", self.aplicar_prewitt)
        barra_lateral.agregar_boton(SECCION_BORDES, "Sobel", self.aplicar_sobel)
        barra_lateral.agregar_boton(SECCION_BORDES, "Laplaciano", self.metodo_laplaciano)
        barra_lateral.agregar_boton(SECCION_BORDES, "Laplaciano + pendiente", self.metodo_laplaciano_pendiente)
        barra_lateral.agregar_boton(SECCION_BORDES, "Marr-Hildreth (LoG)", self.metodo_marr_hildreth)

    def gradiente(self, imagen, mascara_x, mascara_y):
        gx = self.convolucionar(imagen, mascara_x)
        gy = self.convolucionar(imagen, mascara_y)
        magnitud = np.sqrt(gx ** 2 + gy ** 2)
        return gx, gy, magnitud

    def _aplicar_detector_gradiente(self, mascara_x, mascara_y):
        imagen = self.app.imagen_original
        if imagen is None:
            self.app.mostrar_error("Primero cargá una imagen.")
            return

        canales = [imagen] if imagen.ndim == 2 else cv2.split(imagen)
        bordes = []
        for canal in canales:
            _, _, magnitud = self.gradiente(canal, mascara_x, mascara_y)
            bordes.append(self.escalar_a_255(magnitud))

        self.app.establecer_resultado(bordes[0] if len(bordes) == 1 else cv2.merge(bordes))

    def aplicar_sobel(self):
        self._aplicar_detector_gradiente(MASCARA_SOBEL, MASCARA_SOBEL.T)

    def aplicar_prewitt(self):
        self._aplicar_detector_gradiente(MASCARA_PREWITT, MASCARA_PREWITT.T)

    def detectar_cruces_por_cero(self, laplaciano, umbral_pendiente=0.0):
        bordes = np.zeros(laplaciano.shape, dtype=bool)

        for eje in (0, 1):
            valores = np.moveaxis(laplaciano, eje, 0)
            marcas = np.zeros(valores.shape, dtype=bool)

            actual, siguiente = valores[:-1], valores[1:]
            cambio_directo = (actual * siguiente < 0) & (np.abs(actual - siguiente) > umbral_pendiente)
            marcas[:-1] |= cambio_directo

            anterior, centro, posterior = valores[:-2], valores[1:-1], valores[2:]
            cambio_con_cero = (
                (centro == 0) & (anterior * posterior < 0)
                & (np.abs(anterior - posterior) > umbral_pendiente)
            )
            marcas[1:-1] |= cambio_con_cero

            bordes |= np.moveaxis(marcas, 0, eje)

        return bordes.astype(np.uint8) * 255

    def mascara_log(self, tamano, sigma):
        centro = tamano // 2
        mascara = np.zeros((tamano, tamano), dtype=float)

        for i in range(tamano):
            for j in range(tamano):
                x, y = i - centro, j - centro
                r2 = x ** 2 + y ** 2
                mascara[i, j] = (
                    1 / (math.sqrt(2 * math.pi) * sigma ** 3)
                    * np.exp(-r2 / (2 * sigma ** 2))
                    * (r2 / sigma ** 2 - 2)
                )

        return mascara

    def metodo_laplaciano(self):
        imagen = self._obtener_original_gris()
        if imagen is None:
            return

        laplaciano = self.convolucionar(imagen, MASCARA_LAPLACE)
        self.app.establecer_resultado(self.detectar_cruces_por_cero(laplaciano, umbral_pendiente=0))

    def metodo_laplaciano_pendiente(self):
        imagen = self._obtener_original_gris()
        if imagen is None:
            return

        umbral = self._pedir_numero(
            "Laplaciano + pendiente", "Umbral mínimo de pendiente para aceptar un borde (> 0):"
        )
        if umbral is None:
            return
        if umbral <= 0:
            self.app.mostrar_error("El umbral debe ser mayor a 0.")
            return

        laplaciano = self.convolucionar(imagen, MASCARA_LAPLACE)
        self.app.establecer_resultado(self.detectar_cruces_por_cero(laplaciano, umbral_pendiente=umbral))

    def metodo_marr_hildreth(self):
        imagen = self._obtener_original_gris()
        if imagen is None:
            return

        sigma = self._pedir_numero(
            "Marr-Hildreth", "Ingresá el sigma del suavizado gaussiano (> 0):"
        )
        if sigma is None:
            return
        if sigma <= 0:
            self.app.mostrar_error("Sigma debe ser mayor a 0.")
            return

        tamano = math.ceil(4 * sigma + 1)
        if tamano % 2 == 0:
            tamano += 1

        log = self.convolucionar(imagen, self.mascara_log(tamano, sigma))
        self.app.establecer_resultado(self.detectar_cruces_por_cero(log, umbral_pendiente=0))


    def _construir_seccion_filtros(self, barra_lateral) -> None:
        barra_lateral.agregar_boton(SECCION_FILTROS, "Difusión Isotrópica", self.aplicar_difusion_isotropica)
        barra_lateral.agregar_boton(SECCION_FILTROS, "Difusión Anisotrópica", self.aplicar_difusion_anisotropica)
        barra_lateral.agregar_boton(SECCION_FILTROS, "Filtro bilateral", self.aplicar_filtro_bilateral)

    def detector_leclerc(self, valor, sigma):
        return math.exp(-(valor ** 2) / sigma ** 2)

    def detector_lorentz(self, valor, sigma):
        return 1 / ((valor ** 2) / sigma ** 2 + 1)

    def difusion_isotropica(self, imagen, iteraciones, lambda_):
        resultado = imagen.astype(float)
        alto, ancho = resultado.shape

        for _ in range(iteraciones):
            con_borde = np.pad(resultado, 1, mode="edge")
            nueva = np.zeros((alto, ancho), dtype=float)

            for i in range(alto):
                for j in range(ancho):
                    centro = con_borde[i + 1, j + 1]
                    norte = con_borde[i, j + 1] - centro
                    sur = con_borde[i + 2, j + 1] - centro
                    este = con_borde[i + 1, j + 2] - centro
                    oeste = con_borde[i + 1, j] - centro
                    nueva[i, j] = centro + lambda_ * (norte + sur + este + oeste)

            resultado = nueva

        return np.clip(resultado, 0, 255).astype(np.uint8)

    def difusion_anisotropica(self, imagen, iteraciones, lambda_, sigma, detector="leclerc"):
        g = self.detector_lorentz if detector == "lorentz" else self.detector_leclerc
        resultado = imagen.astype(float)
        alto, ancho = resultado.shape

        for _ in range(iteraciones):
            con_borde = np.pad(resultado, 1, mode="edge")
            nueva = np.zeros((alto, ancho), dtype=float)

            for i in range(alto):
                for j in range(ancho):
                    centro = con_borde[i + 1, j + 1]
                    norte = con_borde[i, j + 1] - centro
                    sur = con_borde[i + 2, j + 1] - centro
                    este = con_borde[i + 1, j + 2] - centro
                    oeste = con_borde[i + 1, j] - centro

                    c_norte = g(abs(norte), sigma)
                    c_sur = g(abs(sur), sigma)
                    c_este = g(abs(este), sigma)
                    c_oeste = g(abs(oeste), sigma)

                    nueva[i, j] = centro + lambda_ * (
                        c_norte * norte + c_sur * sur + c_este * este + c_oeste * oeste
                    )

            resultado = nueva

        return np.clip(resultado, 0, 255).astype(np.uint8)

    def _validar_parametros_difusion(self, datos):
        if datos["iteraciones"] <= 0 or datos["iteraciones"] != int(datos["iteraciones"]):
            self.app.mostrar_error("Las iteraciones deben ser un entero mayor a 0.")
            return False
        if not 0 <= datos["lambda"] <= 1:
            self.app.mostrar_error("Lambda debe cumplir 0 ≤ λ ≤ 1.")
            return False
        return True

    def aplicar_difusion_isotropica(self):
        imagen = self._obtener_original_gris()
        if imagen is None:
            return

        datos = self._pedir_formulario(
            "Difusión Isotrópica",
            [
                ("iteraciones", "Cantidad de iteraciones (t)", None),
                ("lambda", "λ (0 ≤ λ ≤ 1)", LAMBDA_DIFUSION),
            ],
        )
        if datos is None or not self._validar_parametros_difusion(datos):
            return

        resultado = self.difusion_isotropica(imagen, int(datos["iteraciones"]), datos["lambda"])
        self.app.establecer_resultado(resultado)

    def aplicar_difusion_anisotropica(self):
        imagen = self._obtener_original_gris()
        if imagen is None:
            return

        datos = self._pedir_formulario(
            "Difusión Anisotrópica",
            [
                ("iteraciones", "Cantidad de iteraciones (t)", None),
                ("lambda", "λ (0 ≤ λ ≤ 1)", LAMBDA_DIFUSION),
                ("detector", "Detector de borde g", ("leclerc", "lorentz")),
                ("sigma", "σ del detector (> 0)", None),
            ],
        )
        if datos is None or not self._validar_parametros_difusion(datos):
            return
        if datos["sigma"] <= 0:
            self.app.mostrar_error("σ debe ser mayor a 0.")
            return

        resultado = self.difusion_anisotropica(
            imagen, int(datos["iteraciones"]), datos["lambda"], datos["sigma"], datos["detector"]
        )
        self.app.establecer_resultado(resultado)

    def filtro_bilateral(self, imagen, sigma_s, sigma_r):
        mitad = max(1, int(round(sigma_s)))
        base = imagen.astype(float)
        alto, ancho = base.shape
        con_borde = np.pad(base, mitad, mode="edge")

        numerador = np.zeros((alto, ancho), dtype=float)
        normalizacion = np.zeros((alto, ancho), dtype=float)

        for dy in range(-mitad, mitad + 1):
            for dx in range(-mitad, mitad + 1):
                vecino = con_borde[mitad + dy:mitad + dy + alto, mitad + dx:mitad + dx + ancho]
                g_s = math.exp(-(dx ** 2 + dy ** 2) / (2 * sigma_s ** 2))
                g_r = np.exp(-((vecino - base) ** 2) / (2 * sigma_r ** 2))
                peso = g_s * g_r
                numerador += peso * vecino
                normalizacion += peso

        return np.clip(numerador / normalizacion, 0, 255).astype(np.uint8)

    def aplicar_filtro_bilateral(self):
        imagen = self._obtener_original_gris()
        if imagen is None:
            return

        datos = self._pedir_formulario(
            "Filtro bilateral",
            [("sigma_s", "sigma_s (espacial)", SIGMA_S), ("sigma_r", "sigma_r (intensidad)", SIGMA_R)],
        )
        if datos is None:
            return
        if datos["sigma_s"] <= 0 or datos["sigma_r"] <= 0:
            self.app.mostrar_error("sigma_s y sigma_r deben ser mayores a 0.")
            return

        self.app.establecer_resultado(self.filtro_bilateral(imagen, datos["sigma_s"], datos["sigma_r"]))


    def _construir_seccion_umbral(self, barra_lateral) -> None:
        barra_lateral.agregar_boton(SECCION_UMBRAL, "Umbral óptimo iterativo", self.aplicar_umbral_iterativo)
        barra_lateral.agregar_boton(SECCION_UMBRAL, "Umbral de Otsu", self.aplicar_umbral_otsu)
        barra_lateral.agregar_boton(SECCION_UMBRAL, "Segmentar color por bandas", self.segmentar_color_por_bandas)

    def umbral_iterativo(self, imagen, t_inicial, delta_t):
        umbral = float(t_inicial)
        iteraciones = 0

        while True:
            iteraciones += 1
            grupo_alto = imagen[imagen > umbral]
            grupo_bajo = imagen[imagen <= umbral]
            if grupo_alto.size == 0 or grupo_bajo.size == 0:
                umbral = float(imagen.mean())
                if iteraciones > 1000:
                    return umbral, iteraciones
                continue

            nuevo = 0.5 * (grupo_alto.mean() + grupo_bajo.mean())
            if abs(nuevo - umbral) < delta_t:
                return float(nuevo), iteraciones
            umbral = nuevo

    def umbral_otsu(self, imagen):
        frecuencias = np.bincount(imagen.ravel(), minlength=256).astype(float)
        p = frecuencias / frecuencias.sum()

        p1 = np.cumsum(p)
        m = np.cumsum(np.arange(256) * p)
        m_global = m[-1]

        denominador = p1 * (1 - p1)
        varianza_entre_clases = np.zeros(256, dtype=float)
        valido = denominador > 0
        varianza_entre_clases[valido] = (m_global * p1[valido] - m[valido]) ** 2 / denominador[valido]

        maximos = np.flatnonzero(varianza_entre_clases == varianza_entre_clases.max())
        return int(round(maximos.mean()))

    def binarizar(self, imagen, umbral):
        return np.where(imagen > umbral, 255, 0).astype(np.uint8)

    def _umbral_segun_metodo(self, imagen, metodo, t_inicial, delta_t):
        if metodo == "iterativo":
            umbral, iteraciones = self.umbral_iterativo(imagen, t_inicial, delta_t)
            return umbral, f"{umbral:.1f} ({iteraciones} it.)"
        umbral = self.umbral_otsu(imagen)
        return umbral, f"{umbral}"

    def _validar_umbral_iterativo(self, datos):
        if not 0 < datos["t_inicial"] < 255 or datos["delta_t"] <= 0:
            self.app.mostrar_error("T inicial debe cumplir 0 < T < 255 (T ≠ 0, 255) y ΔT debe ser mayor a 0.")
            return False
        return True

    def aplicar_umbral_iterativo(self):
        imagen = self._obtener_original_gris()
        if imagen is None:
            return

        datos = self._pedir_formulario(
            "Umbral óptimo iterativo",
            [
                ("t_inicial", "Umbral inicial T (T ≠ 0, 255)", None),
                ("delta_t", "ΔT: diferencia mínima entre T sucesivos", None),
            ],
        )
        if datos is None or not self._validar_umbral_iterativo(datos):
            return

        umbral, iteraciones = self.umbral_iterativo(imagen, datos["t_inicial"], datos["delta_t"])
        self.app.establecer_resultado(self.binarizar(imagen, umbral))
        self.app.mostrar_info(f"Umbral óptimo iterativo: T = {umbral:.2f} ({iteraciones} iteraciones).")

    def aplicar_umbral_otsu(self):
        imagen = self._obtener_original_gris()
        if imagen is None:
            return

        umbral = self.umbral_otsu(imagen)
        self.app.establecer_resultado(self.binarizar(imagen, umbral))
        self.app.mostrar_info(f"Umbral de Otsu: T = {umbral}.")

    def segmentar_color(self, imagen_color, metodo, t_inicial, delta_t):
        bandas = []
        umbrales = []
        for banda in cv2.split(imagen_color):
            umbral, _ = self._umbral_segun_metodo(banda, metodo, t_inicial, delta_t)
            umbrales.append(umbral)
            bandas.append(self.binarizar(banda, umbral))
        return cv2.merge(bandas), umbrales

    def segmentar_color_por_bandas(self):
        imagen = self.app.imagen_original
        if imagen is None:
            self.app.mostrar_error("Primero cargá una imagen.")
            return
        if imagen.ndim != 3:
            self.app.mostrar_error("Esta función necesita una imagen en color. Cargá una imagen en color.")
            return

        datos = self._pedir_formulario(
            "Segmentación en color por bandas",
            [
                ("metodo", "Método de umbral", ("otsu", "iterativo")),
                ("t_inicial", "Iterativo: umbral inicial T", None),
                ("delta_t", "Iterativo: ΔT", None),
            ],
            opcionales={"t_inicial", "delta_t"},
        )
        if datos is None:
            return
        if datos["metodo"] == "iterativo" and not self._validar_umbral_iterativo(datos):
            return

        segmentada, (u_b, u_g, u_r) = self.segmentar_color(
            imagen, datos["metodo"], datos["t_inicial"], datos["delta_t"]
        )
        self.app.establecer_resultado(segmentada)
        self.app.mostrar_info(f"Umbrales ({datos['metodo']}): R = {u_r:g}, G = {u_g:g}, B = {u_b:g}.")


    def _obtener_original_gris(self):
        if self.app.imagen_original is None:
            self.app.mostrar_error("Primero cargá una imagen.")
            return None
        if self.app.imagen_original.ndim != 2:
            self.app.mostrar_error(
                "Esta función trabaja sobre imágenes en escala de grises. "
                "Cargá una imagen en gris."
            )
            return None
        return self.app.imagen_original

    def _pedir_formulario(self, titulo, campos, opcionales=()):
        ventana = ctk.CTkToplevel(self.app)
        ventana.title(titulo)
        ventana.resizable(False, False)
        ventana.transient(self.app)
        ventana.after(150, ventana.grab_set)

        controles = {}
        etiquetas = {}
        for fila, (clave, etiqueta, defecto) in enumerate(campos):
            ctk.CTkLabel(ventana, text=etiqueta, anchor="w").grid(
                row=fila, column=0, padx=(16, 8), pady=5, sticky="w"
            )
            if isinstance(defecto, (list, tuple)):
                control = ctk.StringVar(value=defecto[0])
                ctk.CTkOptionMenu(ventana, values=list(defecto), variable=control, width=150).grid(
                    row=fila, column=1, padx=(0, 16), pady=5
                )
            else:
                control = ctk.CTkEntry(ventana, width=150)
                if defecto is not None:
                    control.insert(0, str(defecto))
                control.grid(row=fila, column=1, padx=(0, 16), pady=5)
            controles[clave] = control
            etiquetas[clave] = etiqueta

        resultado = {}

        def aceptar():
            valores = {}
            for clave, control in controles.items():
                if isinstance(control, ctk.StringVar):
                    valores[clave] = control.get()
                    continue
                if clave in opcionales and not control.get().strip():
                    valores[clave] = None
                    continue
                try:
                    valores[clave] = float(control.get().replace(",", "."))
                except ValueError:
                    self.app.mostrar_error(f"'{etiquetas[clave]}' no es un número válido.")
                    return
            resultado.update(valores)
            ventana.destroy()

        botones = ctk.CTkFrame(ventana, fg_color="transparent")
        botones.grid(row=len(campos), column=0, columnspan=2, pady=(8, 14))
        ctk.CTkButton(botones, text="Aceptar", width=110, command=aceptar).pack(side="left", padx=6)
        ctk.CTkButton(botones, text="Cancelar", width=110, command=ventana.destroy).pack(side="left", padx=6)

        ventana.wait_window()
        return resultado or None

    def convolucionar(self, imagen, mascara):
        alto_m, ancho_m = mascara.shape
        borde_y, borde_x = alto_m // 2, ancho_m // 2
        con_borde = np.pad(
            imagen.astype(float), ((borde_y, borde_y), (borde_x, borde_x)), mode="edge"
        )
        alto, ancho = imagen.shape
        resultado = np.zeros((alto, ancho), dtype=float)

        for i in range(alto):
            for j in range(ancho):
                ventana = con_borde[i:i + alto_m, j:j + ancho_m]
                resultado[i, j] = np.sum(ventana * mascara)

        return resultado
