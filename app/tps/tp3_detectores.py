import math
import os
import re
from collections import deque
from tkinter import filedialog

import cv2
import numpy as np
import customtkinter as ctk

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from app.core import image_io
from app.tps.tp2_filtros import ControladorTP2, MASCARA_SOBEL

SECCION_CANNY_SUSAN = "Canny y SUSAN"
SECCION_HOUGH = "Hough"
SECCION_CONTORNOS = "Contornos activos"
SECCION_VIDEO = "Seguimiento en video"

UMBRAL_SUSAN = 15.0

MASCARA_CIRCULAR_SUSAN = np.array([
    [0, 0, 1, 1, 1, 0, 0],
    [0, 1, 1, 1, 1, 1, 0],
    [1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1],
    [0, 1, 1, 1, 1, 1, 0],
    [0, 0, 1, 1, 1, 0, 0],
], dtype=bool)

PHI_FONDO = 3
PHI_LOUT = 1
PHI_LIN = -1
PHI_OBJETO = -3

COLOR_LIN = (255, 0, 0)
COLOR_LOUT = (0, 255, 255)
COLOR_RECTA = (0, 0, 255)
COLOR_ESQUINA = (0, 0, 255)

VECINOS_4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
VECINOS_8 = VECINOS_4 + ((-1, -1), (-1, 1), (1, -1), (1, 1))

DEMORA_ENTRE_CUADROS_MS = 80


class ControladorTP3(ControladorTP2):
    def __init__(self, app):
        super().__init__(app)
        self.rectangulo_objeto = None
        self.rectangulo_fondo = None

        self.cuadros = []
        self.cuadros_resultado = []
        self.indice_cuadro_inicial = 0
        self._indice_animacion = 0

    def registrar(self, barra_lateral) -> None:
        self._construir_seccion_canny_susan(barra_lateral)
        self._construir_seccion_hough(barra_lateral)
        self._construir_seccion_contornos(barra_lateral)
        self._construir_seccion_video(barra_lateral)


    def _construir_seccion_canny_susan(self, barra_lateral) -> None:
        barra_lateral.agregar_boton(SECCION_CANNY_SUSAN, "Canny", self.aplicar_canny)
        barra_lateral.agregar_boton(SECCION_CANNY_SUSAN, "SUSAN bordes", lambda: self.aplicar_susan("bordes"))
        barra_lateral.agregar_boton(SECCION_CANNY_SUSAN, "SUSAN esquinas", lambda: self.aplicar_susan("esquinas"))
        barra_lateral.agregar_boton(SECCION_CANNY_SUSAN, "Usar resultado como entrada", self.usar_resultado_como_entrada)

    def angulo_discretizado(self, gx, gy):
        angulo = np.full(gx.shape, 90.0)
        distinto_cero = gx != 0
        angulo[distinto_cero] = np.degrees(np.arctan(gy[distinto_cero] / gx[distinto_cero]))
        angulo[angulo < 0] += 180

        direccion = np.zeros(gx.shape, dtype=int)
        direccion[(angulo >= 22.5) & (angulo < 67.5)] = 45
        direccion[(angulo >= 67.5) & (angulo < 112.5)] = 90
        direccion[(angulo >= 112.5) & (angulo < 157.5)] = 135
        return direccion

    def supresion_no_maximos(self, magnitud, direccion):
        desplazamientos = {0: (0, 1), 45: (1, 1), 90: (1, 0), 135: (-1, 1)}
        alto, ancho = magnitud.shape
        con_borde = np.pad(magnitud, 1, mode="constant")
        resultado = magnitud.copy()

        for i in range(alto):
            for j in range(ancho):
                if magnitud[i, j] == 0:
                    continue
                di, dj = desplazamientos[direccion[i, j]]
                vecino_1 = con_borde[i + 1 + di, j + 1 + dj]
                vecino_2 = con_borde[i + 1 - di, j + 1 - dj]
                if vecino_1 > magnitud[i, j] or vecino_2 > magnitud[i, j]:
                    resultado[i, j] = 0

        return resultado

    def umbralizacion_histeresis(self, magnitud, t1, t2, conectividad):
        vecinos = VECINOS_4 if conectividad == 4 else VECINOS_8
        alto, ancho = magnitud.shape

        borde = magnitud > t2
        candidato = (magnitud >= t1) & ~borde

        pendientes = deque(zip(*np.nonzero(borde)))
        while pendientes:
            i, j = pendientes.popleft()
            for di, dj in vecinos:
                ni, nj = i + di, j + dj
                if 0 <= ni < alto and 0 <= nj < ancho and candidato[ni, nj]:
                    candidato[ni, nj] = False
                    borde[ni, nj] = True
                    pendientes.append((ni, nj))

        return borde.astype(np.uint8) * 255

    def canny(self, imagen, sigma, t1, t2, conectividad):
        tamano = round(2 * sigma + 1)
        suavizada = self.convolucionar(imagen, self.mascara_gauss(tamano, sigma))

        gx = self.convolucionar(suavizada, MASCARA_SOBEL)
        gy = self.convolucionar(suavizada, MASCARA_SOBEL.T)
        magnitud = self.escalar_a_255(np.sqrt(gx ** 2 + gy ** 2)).astype(float)

        direccion = self.angulo_discretizado(gx, gy)
        sin_no_maximos = self.supresion_no_maximos(magnitud, direccion)
        return self.umbralizacion_histeresis(sin_no_maximos, t1, t2, conectividad)

    def aplicar_canny(self):
        imagen = self._obtener_original_gris()
        if imagen is None:
            return

        datos = self._pedir_formulario(
            "Detector de Canny",
            [
                ("sigma", "σ del filtro de Gauss", None),
                ("t1", "Umbral bajo t1 (0 a 255)", None),
                ("t2", "Umbral alto t2 (0 a 255)", None),
                ("conectividad", "Conectividad", ("8", "4")),
            ],
        )
        if datos is None:
            return
        if datos["sigma"] <= 0:
            self.app.mostrar_error("σ debe ser mayor a 0.")
            return
        if round(2 * datos["sigma"] + 1) % 2 == 0:
            self.app.mostrar_error(
                f"Con σ = {datos['sigma']:g} el tamaño de la máscara 2σ + 1 da par y no tiene "
                "píxel central. Ingresá un σ que dé un tamaño impar (por ejemplo, entero)."
            )
            return
        if not 0 <= datos["t1"] < datos["t2"] <= 255:
            self.app.mostrar_error("Los umbrales deben cumplir 0 ≤ t1 < t2 ≤ 255.")
            return

        resultado = self.canny(
            imagen, datos["sigma"], datos["t1"], datos["t2"], int(datos["conectividad"])
        )
        self.app.establecer_resultado(resultado)


    def valores_s_susan(self, imagen, t):
        base = imagen.astype(float)
        alto, ancho = base.shape[:2]
        radio = MASCARA_CIRCULAR_SUSAN.shape[0] // 2
        relleno = ((radio, radio), (radio, radio)) + ((0, 0),) * (base.ndim - 2)
        con_borde = np.pad(base, relleno, mode="edge")

        n = np.zeros((alto, ancho), dtype=float)
        for di in range(-radio, radio + 1):
            for dj in range(-radio, radio + 1):
                if not MASCARA_CIRCULAR_SUSAN[di + radio, dj + radio] or (di == 0 and dj == 0):
                    continue
                vecino = con_borde[radio + di:radio + di + alto, radio + dj:radio + dj + ancho]
                diferencia = vecino - base
                if base.ndim == 3:
                    diferencia = np.sqrt((diferencia ** 2).sum(axis=2))
                n += np.abs(diferencia) < t

        return 1 - n / MASCARA_CIRCULAR_SUSAN.sum()

    def aplicar_susan(self, modo):
        imagen = self.app.imagen_original
        if imagen is None:
            self.app.mostrar_error("Primero cargá una imagen.")
            return

        datos = self._pedir_formulario(
            f"SUSAN ({modo})",
            [
                ("t", "Umbral t de nivel de gris", UMBRAL_SUSAN),
                ("tolerancia", "Tolerancia para s ≈ " + ("0,5" if modo == "bordes" else "0,75"), None),
            ],
        )
        if datos is None:
            return
        if datos["t"] <= 0 or datos["tolerancia"] <= 0:
            self.app.mostrar_error("t y la tolerancia deben ser mayores a 0.")
            return

        s = self.valores_s_susan(imagen, datos["t"])
        objetivo = 0.5 if modo == "bordes" else 0.75
        marcados = np.abs(s - objetivo) < datos["tolerancia"]

        if modo == "bordes":
            self.app.establecer_resultado(marcados.astype(np.uint8) * 255)
            return

        resultado = self._a_color(imagen)
        for i, j in zip(*np.nonzero(marcados)):
            cv2.circle(resultado, (int(j), int(i)), 2, COLOR_ESQUINA, -1)
        self.app.establecer_resultado(resultado)
        self.app.mostrar_info(f"SUSAN: se marcaron {int(marcados.sum())} pixels como esquina.")


    def _construir_seccion_hough(self, barra_lateral) -> None:
        barra_lateral.agregar_boton(SECCION_HOUGH, "Hough rectas", self.aplicar_hough)

    def acumulador_hough(self, binaria, cantidad_theta, cantidad_r, epsilon):
        alto, ancho = binaria.shape
        d = max(ancho, alto)

        thetas = np.linspace(-np.pi / 2, np.pi / 2, cantidad_theta)
        rs = np.linspace(-math.sqrt(2) * d, math.sqrt(2) * d, cantidad_r)

        filas, columnas = np.nonzero(binaria)
        x, y = columnas.astype(float), filas.astype(float)

        acumulador = np.zeros((cantidad_r, cantidad_theta), dtype=int)
        for j, theta in enumerate(thetas):
            rho = np.sort(x * math.cos(theta) + y * math.sin(theta))
            desde = np.searchsorted(rho, rs - epsilon, side="right")
            hasta = np.searchsorted(rho, rs + epsilon, side="left")
            acumulador[:, j] = hasta - desde

        return acumulador, rs, thetas

    def dibujar_recta(self, imagen, r, theta, color):
        alto, ancho = imagen.shape[:2]
        coseno, seno = math.cos(theta), math.sin(theta)

        if abs(seno) >= abs(coseno):
            for x in range(ancho):
                y = round((r - x * coseno) / seno)
                if 0 <= y < alto:
                    imagen[y, x] = color
        else:
            for y in range(alto):
                x = round((r - y * seno) / coseno)
                if 0 <= x < ancho:
                    imagen[y, x] = color

    def aplicar_hough(self):
        imagen = self._obtener_original_gris()
        if imagen is None:
            return
        if not np.isin(imagen, (0, 255)).all():
            self.app.mostrar_error(
                "La imagen de entrada debe ser binaria (0 y 255). Aplicá primero un detector "
                "de bordes y umbralizá, y usá el resultado como entrada."
            )
            return

        datos = self._pedir_formulario(
            "Transformada de Hough (rectas)",
            [
                ("epsilon", "ε", None),
                ("cantidad_theta", "Discretización de θ (cantidad)", None),
                ("cantidad_r", "Discretización de r (cantidad)", None),
                ("umbral", "Umbral del acumulador (votos)", None),
            ],
        )
        if datos is None:
            return
        if datos["epsilon"] <= 0:
            self.app.mostrar_error("ε debe ser mayor a 0.")
            return
        cantidades = (datos["cantidad_theta"], datos["cantidad_r"])
        if any(c < 2 or c != int(c) for c in cantidades):
            self.app.mostrar_error("Las discretizaciones deben ser enteros mayores o iguales a 2.")
            return
        if datos["umbral"] <= 0:
            self.app.mostrar_error("El umbral del acumulador debe ser mayor a 0.")
            return

        acumulador, rs, thetas = self.acumulador_hough(
            imagen, int(datos["cantidad_theta"]), int(datos["cantidad_r"]), datos["epsilon"]
        )

        resultado = self._a_color(imagen)
        celdas = np.argwhere(acumulador >= datos["umbral"])
        for i, j in celdas:
            self.dibujar_recta(resultado, rs[i], thetas[j], COLOR_RECTA)

        self.app.establecer_resultado(resultado)
        self.mostrar_acumulador(acumulador, rs, thetas)
        self.app.mostrar_info(
            f"Hough: {len(celdas)} rectas con al menos {datos['umbral']:g} votos "
            f"(máximo del acumulador: {acumulador.max()})."
        )

    def mostrar_acumulador(self, acumulador, rs, thetas):
        ventana = ctk.CTkToplevel(self.app)
        ventana.title("Acumulador de Hough")
        ventana.geometry("700x500")

        figura = Figure()
        eje = figura.add_subplot()
        grafico = eje.imshow(
            acumulador, aspect="auto", cmap="gray", origin="lower",
            extent=(np.degrees(thetas[0]), np.degrees(thetas[-1]), rs[0], rs[-1]),
        )
        figura.colorbar(grafico, ax=eje, label="Votos")
        eje.set_title("Acumulador A(r, θ)")
        eje.set_xlabel("θ (grados)")
        eje.set_ylabel("r")

        canvas = FigureCanvasTkAgg(figura, master=ventana)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)


    def _construir_seccion_contornos(self, barra_lateral) -> None:
        barra_lateral.agregar_boton(SECCION_CONTORNOS, "Marcar objeto (curva inicial)", self.marcar_objeto)
        barra_lateral.agregar_boton(SECCION_CONTORNOS, "Marcar fondo", self.marcar_fondo)
        barra_lateral.agregar_boton(
            SECCION_CONTORNOS, "Segmentar (objeto y fondo)", lambda: self.aplicar_contornos_activos(con_fondo=True)
        )
        barra_lateral.agregar_boton(
            SECCION_CONTORNOS, "Segmentar (solo objeto)", lambda: self.aplicar_contornos_activos(con_fondo=False)
        )

    def _rectangulo_seleccionado(self):
        seleccion = self.app.ultima_seleccion
        if seleccion is None or seleccion["origen"] != "original":
            self.app.mostrar_error("Primero seleccioná un rectángulo arrastrando el mouse sobre la imagen original.")
            return None
        x_min, x_max = sorted((seleccion["x1"], seleccion["x2"]))
        y_min, y_max = sorted((seleccion["y1"], seleccion["y2"]))
        if x_max - x_min < 2 or y_max - y_min < 2:
            self.app.mostrar_error("El rectángulo es demasiado chico.")
            return None
        return x_min, y_min, x_max, y_max

    def marcar_objeto(self):
        rectangulo = self._rectangulo_seleccionado()
        if rectangulo is not None:
            self.rectangulo_objeto = rectangulo

    def marcar_fondo(self):
        rectangulo = self._rectangulo_seleccionado()
        if rectangulo is not None:
            self.rectangulo_fondo = rectangulo

    def color_promedio(self, imagen, rectangulo):
        x_min, y_min, x_max, y_max = rectangulo
        region = imagen[y_min:y_max + 1, x_min:x_max + 1].astype(float)
        return region.reshape(-1, region.shape[2]).mean(axis=0) if region.ndim == 3 else region.mean()

    def color_promedio_fondo(self, imagen, phi):
        region = imagen[phi == PHI_FONDO].astype(float)
        return region.mean(axis=0)

    def funcion_fd(self, imagen, theta_0, theta_1):
        colores = imagen.astype(float)
        if colores.ndim == 3:
            distancia_fondo = np.sqrt(((colores - theta_0) ** 2).sum(axis=2))
            distancia_objeto = np.sqrt(((colores - theta_1) ** 2).sum(axis=2))
        else:
            distancia_fondo = np.abs(colores - theta_0)
            distancia_objeto = np.abs(colores - theta_1)

        with np.errstate(divide="ignore", invalid="ignore"):
            fd = np.log(distancia_fondo / distancia_objeto)
        return np.nan_to_num(fd, nan=0.0)

    def inicializar_curva(self, forma, rectangulo):
        x_min, y_min, x_max, y_max = rectangulo
        alto, ancho = forma

        interior = np.zeros(forma, dtype=bool)
        interior[y_min:y_max + 1, x_min:x_max + 1] = True

        phi = np.where(interior, PHI_OBJETO, PHI_FONDO).astype(np.int8)
        l_in, l_out = set(), set()
        for i in range(alto):
            for j in range(ancho):
                for vi, vj in self._vecinos_4(i, j, alto, ancho):
                    if interior[i, j] and not interior[vi, vj]:
                        l_in.add((i, j))
                    elif not interior[i, j] and interior[vi, vj]:
                        l_out.add((i, j))

        for punto in l_in:
            phi[punto] = PHI_LIN
        for punto in l_out:
            phi[punto] = PHI_LOUT
        return phi, l_in, l_out

    def _vecinos_4(self, i, j, alto, ancho):
        for di, dj in VECINOS_4:
            ni, nj = i + di, j + dj
            if 0 <= ni < alto and 0 <= nj < ancho:
                yield ni, nj

    def iteracion_intercambio(self, phi, l_in, l_out, fd):
        alto, ancho = phi.shape

        for x in [p for p in l_out if fd[p] > 0]:
            l_out.discard(x)
            l_in.add(x)
            phi[x] = PHI_LIN
            for y in self._vecinos_4(*x, alto, ancho):
                if phi[y] == PHI_FONDO:
                    l_out.add(y)
                    phi[y] = PHI_LOUT

        for x in [p for p in l_in if all(phi[y] < 0 for y in self._vecinos_4(*p, alto, ancho))]:
            l_in.discard(x)
            phi[x] = PHI_OBJETO

        for x in [p for p in l_in if fd[p] < 0]:
            l_in.discard(x)
            l_out.add(x)
            phi[x] = PHI_LOUT
            for y in self._vecinos_4(*x, alto, ancho):
                if phi[y] == PHI_OBJETO:
                    l_in.add(y)
                    phi[y] = PHI_LIN

        for x in [p for p in l_out if all(phi[y] > 0 for y in self._vecinos_4(*p, alto, ancho))]:
            l_out.discard(x)
            phi[x] = PHI_FONDO

    def evolucionar_curva(self, imagen, phi, l_in, l_out, theta_0, theta_1, max_iteraciones):
        fd = self.funcion_fd(imagen, theta_0, theta_1)

        for iteracion in range(1, max_iteraciones + 1):
            self.iteracion_intercambio(phi, l_in, l_out, fd)

            if all(fd[p] < 0 for p in l_out) and all(fd[p] > 0 for p in l_in):
                return iteracion
        return max_iteraciones

    def dibujar_curva(self, imagen, l_in, l_out):
        resultado = self._a_color(imagen)
        for punto in l_out:
            resultado[punto] = COLOR_LOUT
        for punto in l_in:
            resultado[punto] = COLOR_LIN
        return resultado

    def _pedir_parametros_contornos(self, titulo, con_fondo):
        if self.rectangulo_objeto is None:
            self.app.mostrar_error("Primero marcá el rectángulo del objeto.")
            return None
        if con_fondo and self.rectangulo_fondo is None:
            self.app.mostrar_error("Primero marcá el rectángulo del objeto y el del fondo.")
            return None

        datos = self._pedir_formulario(titulo, [("na", "Máximo de iteraciones (Na)", None)])
        if datos is None:
            return None
        if datos["na"] <= 0 or datos["na"] != int(datos["na"]):
            self.app.mostrar_error("Na debe ser un entero mayor a 0.")
            return None
        return int(datos["na"])

    def _rectangulos_dentro_de(self, imagen, con_fondo):
        alto, ancho = imagen.shape[:2]
        rectangulos = (self.rectangulo_objeto, self.rectangulo_fondo) if con_fondo else (self.rectangulo_objeto,)
        for x_min, y_min, x_max, y_max in rectangulos:
            if x_max >= ancho or y_max >= alto:
                self.app.mostrar_error("Los rectángulos marcados no entran en la imagen. Volvé a marcarlos.")
                return False
        return True

    def _calcular_thetas(self, imagen, phi, con_fondo):
        theta_1 = self.color_promedio(imagen, self.rectangulo_objeto)
        if con_fondo:
            return self.color_promedio(imagen, self.rectangulo_fondo), theta_1
        if not (phi == PHI_FONDO).any():
            self.app.mostrar_error("El rectángulo cubre toda la imagen: no queda fondo para calcular θ0.")
            return None
        return self.color_promedio_fondo(imagen, phi), theta_1

    def aplicar_contornos_activos(self, con_fondo):
        imagen = self.app.imagen_original
        if imagen is None:
            self.app.mostrar_error("Primero cargá una imagen.")
            return

        na = self._pedir_parametros_contornos("Contornos activos", con_fondo)
        if na is None or not self._rectangulos_dentro_de(imagen, con_fondo):
            return

        phi, l_in, l_out = self.inicializar_curva(imagen.shape[:2], self.rectangulo_objeto)
        thetas = self._calcular_thetas(imagen, phi, con_fondo)
        if thetas is None:
            return
        theta_0, theta_1 = thetas

        iteraciones = self.evolucionar_curva(imagen, phi, l_in, l_out, theta_0, theta_1, na)

        self.app.establecer_resultado(self.dibujar_curva(imagen, l_in, l_out))
        self.app.mostrar_info(
            f"Contornos activos: terminó en {iteraciones} iteraciones.\n"
            f"Lin (azul): {len(l_in)} pixels. Lout (amarillo): {len(l_out)} pixels."
        )


    def _construir_seccion_video(self, barra_lateral) -> None:
        barra_lateral.agregar_boton(SECCION_VIDEO, "Cargar secuencia (carpeta)", self.cargar_secuencia)
        barra_lateral.agregar_boton(SECCION_VIDEO, "Elegir cuadro inicial", self.elegir_cuadro_inicial)
        barra_lateral.agregar_boton(SECCION_VIDEO, "Marcar objeto (curva inicial)", self.marcar_objeto)
        barra_lateral.agregar_boton(SECCION_VIDEO, "Seguir objeto", self.seguir_objeto)
        barra_lateral.agregar_boton(SECCION_VIDEO, "Repetir animación", self.repetir_animacion)
        barra_lateral.agregar_boton(SECCION_VIDEO, "Guardar video", self.guardar_video)

    def _clave_orden_natural(self, nombre):
        return [int(parte) if parte.isdigit() else parte.lower() for parte in re.split(r"(\d+)", nombre)]

    def cargar_secuencia(self):
        carpeta = filedialog.askdirectory(title="Seleccionar carpeta con los cuadros de la secuencia")
        if not carpeta:
            return

        extensiones = (".jpg", ".jpeg", ".png", ".bmp", ".pgm", ".ppm", ".tif", ".tiff")
        nombres = sorted(
            (n for n in os.listdir(carpeta) if n.lower().endswith(extensiones)),
            key=self._clave_orden_natural,
        )
        if not nombres:
            self.app.mostrar_error("La carpeta no tiene imágenes.")
            return

        try:
            cuadros = [image_io.cargar_imagen(os.path.join(carpeta, n)) for n in nombres]
        except IOError as error:
            self.app.mostrar_error(str(error))
            return
        if any(c.shape != cuadros[0].shape for c in cuadros):
            self.app.mostrar_error("Todos los cuadros de la secuencia deben tener el mismo tamaño.")
            return

        self.cuadros = cuadros
        self.cuadros_resultado = []
        self.nombre_secuencia = os.path.basename(carpeta)
        self._mostrar_cuadro_inicial(0)

    def _mostrar_cuadro_inicial(self, indice):
        self.indice_cuadro_inicial = indice
        self.app.establecer_original(self.cuadros[indice])
        self.app.establecer_resultado(self.cuadros[indice].copy())

    def elegir_cuadro_inicial(self):
        if not self.cuadros:
            self.app.mostrar_error("Primero cargá una secuencia de imágenes.")
            return

        numero = self._pedir_numero(
            "Cuadro inicial", f"Número de cuadro donde se elige el objeto (1 a {len(self.cuadros)}):", int
        )
        if numero is None:
            return
        if not 1 <= numero <= len(self.cuadros):
            self.app.mostrar_error(f"El cuadro debe estar entre 1 y {len(self.cuadros)}.")
            return
        self._mostrar_cuadro_inicial(numero - 1)

    def seguir_objeto(self):
        if not self.cuadros:
            self.app.mostrar_error("Primero cargá una secuencia de imágenes.")
            return

        inicial = self.indice_cuadro_inicial
        na = self._pedir_parametros_contornos("Seguimiento en video", con_fondo=False)
        if na is None or not self._rectangulos_dentro_de(self.cuadros[inicial], con_fondo=False):
            return

        phi, l_in, l_out = self.inicializar_curva(self.cuadros[inicial].shape[:2], self.rectangulo_objeto)
        thetas = self._calcular_thetas(self.cuadros[inicial], phi, con_fondo=False)
        if thetas is None:
            return
        theta_0, theta_1 = thetas

        self.cuadros_resultado = []
        self._procesar_cuadro(inicial, phi, l_in, l_out, theta_0, theta_1, na)

    def _procesar_cuadro(self, indice, phi, l_in, l_out, theta_0, theta_1, na):
        cuadro = self.cuadros[indice]
        self.evolucionar_curva(cuadro, phi, l_in, l_out, theta_0, theta_1, na)

        resultado = self.dibujar_curva(cuadro, l_in, l_out)
        self.cuadros_resultado.append(resultado)
        self.app.establecer_original(cuadro)
        self.app.establecer_resultado(resultado)

        if indice + 1 < len(self.cuadros):
            self.app.after(1, self._procesar_cuadro, indice + 1, phi, l_in, l_out, theta_0, theta_1, na)

    def repetir_animacion(self):
        if not self.cuadros_resultado:
            self.app.mostrar_error("Primero ejecutá el seguimiento.")
            return
        self._indice_animacion = 0
        self._mostrar_siguiente_cuadro()

    def _mostrar_siguiente_cuadro(self):
        if self._indice_animacion >= len(self.cuadros_resultado):
            return
        self.app.establecer_original(self.cuadros[self.indice_cuadro_inicial + self._indice_animacion])
        self.app.establecer_resultado(self.cuadros_resultado[self._indice_animacion])
        self._indice_animacion += 1
        self.app.after(DEMORA_ENTRE_CUADROS_MS, self._mostrar_siguiente_cuadro)

    def guardar_video(self):
        if not self.cuadros_resultado:
            self.app.mostrar_error("Primero ejecutá el seguimiento.")
            return

        ruta = filedialog.asksaveasfilename(
            title="Guardar video del seguimiento",
            defaultextension=".avi",
            initialfile=f"seguimiento_{self.nombre_secuencia}.avi",
            filetypes=[("Video AVI", "*.avi")],
        )
        if not ruta:
            return

        cuadros_por_segundo = 1000 / DEMORA_ENTRE_CUADROS_MS
        alto, ancho = self.cuadros_resultado[0].shape[:2]
        escritor = cv2.VideoWriter(ruta, cv2.VideoWriter_fourcc(*"MJPG"), cuadros_por_segundo, (ancho, alto))
        if not escritor.isOpened():
            self.app.mostrar_error("No se pudo crear el archivo de video.")
            return
        for cuadro in self.cuadros_resultado:
            escritor.write(self._a_color(cuadro))
        escritor.release()

        self.app.mostrar_info(f"Video guardado con {len(self.cuadros_resultado)} cuadros:\n{ruta}")


    def _a_color(self, imagen):
        return cv2.cvtColor(imagen, cv2.COLOR_GRAY2BGR) if imagen.ndim == 2 else imagen.copy()
