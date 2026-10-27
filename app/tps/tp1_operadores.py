import numpy as np
import customtkinter as ctk

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

SECCION_FUNCIONES = "Funciones"
SECCION_RUIDO = "Ruido"
SECCION_FILTROS = "Filtros"

MASCARA_MEDIANA_PONDERADA = np.array([
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1],
])

MASCARA_REALCE_BORDES = np.array([
    [-1, -1, -1],
    [-1, 8, -1],
    [-1, -1, -1],
], dtype=float)

class ControladorTP1:
    def __init__(self, app):
        self.app = app

    def registrar(self, barra_lateral) -> None:
        self._construir_seccion_funciones(barra_lateral)
        self._construir_seccion_filtros(barra_lateral)

    def _construir_seccion_funciones(self, barra_lateral) -> None:
        barra_lateral.agregar_boton(SECCION_FUNCIONES, "Transformación Gamma", self.aplicar_gamma)
        barra_lateral.agregar_boton(SECCION_FUNCIONES, "Aplicar Negativo", self.aplicar_negativo)
        barra_lateral.agregar_boton(SECCION_FUNCIONES, "Histograma", lambda: self.mostrar_histograma(self.app.imagen_original))
        barra_lateral.agregar_boton(SECCION_FUNCIONES, "Umbral", self.umbral_interactivo)
        barra_lateral.agregar_boton(SECCION_FUNCIONES, "Ecualizar Histograma", self.aplicar_contraste)
        barra_lateral.agregar_boton(SECCION_FUNCIONES, "Ecualizar Dos Veces", self.ecualizar_dos_veces)
        barra_lateral.agregar_boton(SECCION_FUNCIONES, "Generar Números Gauss", self.generar_numeros_gauss)
        barra_lateral.agregar_boton(SECCION_FUNCIONES, "Generar Números Exponencial", self.generar_numeros_exponencial)

        barra_lateral.agregar_boton(SECCION_RUIDO, "Ruido Gaussiano Aditivo", self.ruido_gaussiano_aditivo)
        barra_lateral.agregar_boton(SECCION_RUIDO, "Ruido Exponencial Multiplicativo", self.ruido_exponencial_multiplicativo)
        barra_lateral.agregar_boton(SECCION_RUIDO, "Ruido Sal y Pimienta", self.aplicar_sal_pimienta)

    def _construir_seccion_filtros(self, barra_lateral) -> None:
        barra_lateral.agregar_boton(SECCION_FILTROS, "Filtro de la Media", self.filtro_media)
        barra_lateral.agregar_boton(SECCION_FILTROS, "Filtro de la Mediana", self.filtro_mediana)
        barra_lateral.agregar_boton(SECCION_FILTROS, "Filtro Mediana Ponderada", self.filtro_mediana_ponderada)
        barra_lateral.agregar_boton(SECCION_FILTROS, "Filtro de Gauss", self.filtro_gauss)
        barra_lateral.agregar_boton(SECCION_FILTROS, "Realce de Bordes", self.filtro_realce_bordes)
        barra_lateral.agregar_boton(SECCION_FILTROS, "Usar resultado como entrada", self.usar_resultado_como_entrada)


    def _pedir_numero(self, titulo, texto, convertir=float):
        entrada = ctk.CTkInputDialog(title=titulo, text=texto).get_input()
        if not entrada:
            return None
        try:
            return convertir(entrada)
        except ValueError:
            self.app.mostrar_error("El valor ingresado no es un número válido.")
            return None

    def escalar_a_255(self, matriz):
        minimo = matriz.min()
        maximo = matriz.max()
        if maximo == minimo:
            return np.zeros_like(matriz, dtype=np.uint8)
        escalada = 255 * (matriz - minimo) / (maximo - minimo)
        return escalada.astype(np.uint8)

    def usar_resultado_como_entrada(self):

        if self.app.imagen_resultado is None:
            self.app.mostrar_error("Todavía no hay una imagen resultado.")
            return
        self.app.establecer_original(self.app.imagen_resultado.copy())


    def aplicar_gamma(self):
        if self.app.imagen_original is None:
            self.app.mostrar_error("Primero cargá una imagen.")
            return

        gamma = self._pedir_numero("Transformación Gamma", "Ingresá el valor de gamma:")
        if gamma is None:
            return

        if gamma <= 0 or gamma >= 2 or gamma == 1:
            self.app.mostrar_error(
                "Gamma debe cumplir 0 < gamma < 2 y gamma debe ser distinto de 1."
            )
            return

        imagen_transformada = self.transformacion_gamma(
            self.app.imagen_original,
            gamma
        )

        self.app.establecer_resultado(imagen_transformada)
    def transformacion_gamma(self, imagen, gamma):
        c = 255 ** (1 - gamma)
        return (c * (imagen.astype(float) ** gamma)).astype(np.uint8)

    def aplicar_negativo(self):
        if self.app.imagen_original is None:
            self.app.mostrar_error("Primero cargá una imagen.")
            return

        imagen_negativa = self.negativo(
            self.app.imagen_original
        )

        self.app.establecer_resultado(imagen_negativa)

    def negativo(self, imagen):
        return 255 - imagen


    def mostrar_histograma(self, imagen=None, titulo="Histograma de niveles de gris"):
        if imagen is None:
            imagen = self.app.imagen_original

        if imagen is None:
            self.app.mostrar_error("Primero cargá una imagen.")
            return

        es_color = imagen.ndim == 3

        ventana = ctk.CTkToplevel(self.app)
        ventana.title(titulo)
        ventana.geometry("700x500")

        figura = Figure()
        eje = figura.add_subplot()

        if es_color:
            for canal, (nombre, color) in enumerate((("Azul", "tab:blue"), ("Verde", "tab:green"), ("Rojo", "tab:red"))):
                eje.plot(range(256), self.histograma_grises(imagen[:, :, canal]), color=color, label=nombre)
            eje.legend()
        else:
            eje.bar(range(256), self.histograma_grises(imagen), width=1.0)

        eje.set_title(titulo)
        eje.set_xlabel("Intensidad" if es_color else "Nivel de gris")
        eje.set_ylabel("Frecuencia relativa")
        eje.set_xlim(0, 255)
        eje.set_xticks(range(0, 256, 50))

        canvas = FigureCanvasTkAgg(
            figura,
            master=ventana
        )

        canvas.draw()
        canvas.get_tk_widget().pack(
            fill="both",
            expand=True
        )

    def histograma_grises(self, imagen):
        histograma = np.bincount(imagen.flatten(), minlength=256)
        histograma = histograma / imagen.size

        return histograma

    def mostrar_histograma_numeros(self, numeros, titulo):
        ventana = ctk.CTkToplevel(self.app)
        ventana.title(titulo)
        ventana.geometry("700x500")

        figura = Figure()
        eje = figura.add_subplot()

        eje.hist(numeros, bins=50)

        eje.set_title(titulo)
        eje.set_xlabel("Valor")
        eje.set_ylabel("Frecuencia")

        canvas = FigureCanvasTkAgg(
            figura,
            master=ventana
        )

        canvas.draw()
        canvas.get_tk_widget().pack(
            fill="both",
            expand=True
        )


    def umbral_interactivo(self):
        if self.app.imagen_original is None:
            self.app.mostrar_error("Primero cargá una imagen.")
            return

        umbral = self._pedir_numero("Umbral", "Ingresá el umbral (0 - 255):", int)
        if umbral is None:
            return

        if umbral < 0 or umbral > 255:
            self.app.mostrar_error("El umbral debe estar entre 0 y 255.")
            return

        resultado = self.aplicar_umbral(self.app.imagen_original, umbral)

        self.app.establecer_resultado(resultado)

    def aplicar_umbral(self, imagen, umbral):
        imagen_binaria = np.zeros_like(imagen)
        imagen_binaria[imagen >= umbral] = 255

        return imagen_binaria


    def ecualizar_histograma(self, imagen):
        if imagen.ndim == 3:
            return np.stack([self.ecualizar_histograma(imagen[:, :, c]) for c in range(imagen.shape[2])], axis=2)

        histograma = self.histograma_grises(imagen)

        s = np.cumsum(histograma)

        s_min = s[s > 0].min()

        if s_min >= 1:
            return imagen.copy()

        transformacion = 255 * ((s - s_min) / (1 - s_min))
        transformacion = np.clip(transformacion, 0, 255)

        return transformacion[imagen].astype(np.uint8)

    def aplicar_contraste(self):
        if self.app.imagen_original is None:
            self.app.mostrar_error("Primero cargá una imagen.")
            return

        resultado = self.ecualizar_histograma(self.app.imagen_original)

        self.app.establecer_resultado(resultado)
        self.mostrar_histograma(resultado, "Histograma ecualizado (1 vez)")

    def ecualizar_dos_veces(self):
        if self.app.imagen_original is None:
            self.app.mostrar_error("Primero cargá una imagen.")
            return

        una_vez = self.ecualizar_histograma(self.app.imagen_original)
        dos_veces = self.ecualizar_histograma(una_vez)

        self.app.establecer_resultado(dos_veces)

        self.mostrar_histograma(una_vez, "Histograma ecualizado (1 vez)")
        self.mostrar_histograma(dos_veces, "Histograma ecualizado (2 veces)")

        gris = self.app.imagen_original
        cambio_primera = np.abs(gris.astype(int) - una_vez.astype(int))
        cambio_segunda = np.abs(una_vez.astype(int) - dos_veces.astype(int))

        self.app.mostrar_info(
            "Ecualización aplicada dos veces.\n\n"
            f"Original -> 1a ecualización: cambio promedio de "
            f"{cambio_primera.mean():.1f} niveles de gris (máximo {cambio_primera.max()}).\n"
            f"1a -> 2a ecualización: cambio promedio de "
            f"{cambio_segunda.mean():.1f} niveles de gris (máximo {cambio_segunda.max()}).\n\n"
            "Explicación: la primera ecualización reparte los niveles de gris y deja el "
            "histograma acumulado aproximadamente como una recta. Al ecualizar de nuevo, "
            "la transformación que se calcula sobre ese histograma ya es casi la "
            "identidad, así que la segunda pasada casi no modifica la imagen.\n\n"
            "No queda exactamente igual porque la imagen es discreta: al ecualizar la "
            "primera vez varios niveles de gris distintos se mezclan en un mismo valor "
            "y ya no se pueden volver a separar. Por eso el histograma nunca llega a ser "
            "perfectamente uniforme y lo que queda es un redondeo de uno o dos niveles, "
            "imperceptible a simple vista."
        )


    def generar_numeros_gauss(self):
        mu = self._pedir_numero("Valor Medio", "Ingresá el valor medio mu:")
        if mu is None:
            return

        desviacion = self._pedir_numero("Desviacion Estándar", "Ingresá la desviación estándar > 0:")
        if desviacion is None:
            return
        if desviacion <= 0:
            self.app.mostrar_error("La desviación estándar debe ser mayor a 0.")
            return

        cantidad = self._pedir_numero("Cantidad", "¿Cuántos números querés generar?", int)
        if cantidad is None:
            return
        if cantidad <= 0:
            self.app.mostrar_error("La cantidad debe ser mayor a 0.")
            return

        numeros = np.random.normal(loc=mu, scale=desviacion, size=cantidad)
        self.mostrar_histograma_numeros(numeros, "Distribución Gaussiana")

    def generar_numeros_exponencial(self):
        lambdaa = self._pedir_numero("Lambda", "Ingresá el parámetro lambda (> 0):")
        if lambdaa is None:
            return
        if lambdaa <= 0:
            self.app.mostrar_error("Lambda debe ser mayor a 0.")
            return

        cantidad = self._pedir_numero("Cantidad", "¿Cuántos números querés generar?", int)
        if cantidad is None:
            return
        if cantidad <= 0:
            self.app.mostrar_error("La cantidad debe ser mayor a 0.")
            return

        numeros = np.random.exponential(scale=1 / lambdaa, size=cantidad)
        self.mostrar_histograma_numeros(numeros, "Distribución Exponencial")


    def _pedir_porcentaje(self):
        porcentaje = self._pedir_numero(
            "Porcentaje Ruido", "Ingresá el porcentaje de la imagen a contaminar (0 - 100):"
        )
        if porcentaje is None:
            return None
        if porcentaje <= 0 or porcentaje > 100:
            self.app.mostrar_error("El porcentaje debe estar entre 0 y 100.")
            return None
        return porcentaje

    def _elegir_pixeles(self, imagen, porcentaje):
        alto, ancho = imagen.shape[:2]
        cant_pix = alto * ancho
        cant_ruido = int(cant_pix * (porcentaje / 100))

        indices = np.random.choice(cant_pix, size=cant_ruido, replace=False)
        return [(indice // ancho, indice % ancho) for indice in indices]

    def ruido_gaussiano_aditivo(self):
        if self.app.imagen_original is None:
            self.app.mostrar_error("Primero cargá una imagen.")
            return

        desviacion = self._pedir_numero("Desviacion Estándar", "Ingresá la desviación estándar > 0:")
        if desviacion is None:
            return
        if desviacion <= 0:
            self.app.mostrar_error("La desviación estándar debe ser mayor a 0.")
            return

        porcentaje = self._pedir_porcentaje()
        if porcentaje is None:
            return

        imagen = self.app.imagen_original
        pixeles = self._elegir_pixeles(imagen, porcentaje)
        numeros = np.random.normal(loc=0, scale=desviacion, size=len(pixeles))

        resultado = imagen.copy()
        for k, (i, j) in enumerate(pixeles):
            nuevo_valor = resultado[i, j].astype(float) + numeros[k]
            resultado[i, j] = np.clip(nuevo_valor, 0, 255)

        self.mostrar_histograma_numeros(numeros, "Ruido Gaussiano aditivo (mu = 0)")
        self.app.establecer_resultado(resultado.astype(np.uint8))

    def ruido_exponencial_multiplicativo(self):
        if self.app.imagen_original is None:
            self.app.mostrar_error("Primero cargá una imagen.")
            return

        lambdaa = self._pedir_numero("Lambda", "Ingresá el parámetro lambda (> 0):")
        if lambdaa is None:
            return
        if lambdaa <= 0:
            self.app.mostrar_error("Lambda debe ser mayor a 0.")
            return

        porcentaje = self._pedir_porcentaje()
        if porcentaje is None:
            return

        imagen = self.app.imagen_original
        pixeles = self._elegir_pixeles(imagen, porcentaje)
        numeros = np.random.exponential(scale=1 / lambdaa, size=len(pixeles))

        resultado = imagen.copy()
        for k, (i, j) in enumerate(pixeles):
            nuevo_valor = resultado[i, j].astype(float) * numeros[k]
            resultado[i, j] = np.clip(nuevo_valor, 0, 255)

        self.mostrar_histograma_numeros(numeros, "Ruido Exponencial multiplicativo")
        self.app.establecer_resultado(resultado.astype(np.uint8))


    def aplicar_sal_pimienta(self):
        if self.app.imagen_original is None:
            self.app.mostrar_error("Primero cargá una imagen.")
            return

        p = self._pedir_numero("Ruido Sal y Pimienta", "Ingresá p entre 0 y 0.5:")
        if p is None:
            return
        if p < 0 or p > 0.5:
            self.app.mostrar_error("p debe estar entre 0 y 0.5.")
            return

        resultado = self.app.imagen_original.copy()

        alto = resultado.shape[0]
        ancho = resultado.shape[1]

        for i in range(alto):
            for j in range(ancho):
                x = np.random.uniform(0, 1)

                if x <= p:
                    resultado[i, j] = 0
                elif x > 1 - p:
                    resultado[i, j] = 255

        self.app.establecer_resultado(resultado)


    def ventana_deslizante(self, imagen, tamano, funcion):
        if imagen.ndim == 3:
            return np.stack(
                [self.ventana_deslizante(imagen[:, :, c], tamano, funcion) for c in range(imagen.shape[2])],
                axis=2,
            )

        borde = tamano // 2
        con_borde = np.pad(imagen, borde, mode="edge")

        alto, ancho = imagen.shape
        resultado = np.zeros((alto, ancho), dtype=float)

        for i in range(alto):
            for j in range(ancho):
                ventana = con_borde[i:i + tamano, j:j + tamano]
                resultado[i, j] = funcion(ventana)

        return resultado

    def _pedir_tamano_mascara(self):
        tamano = self._pedir_numero(
            "Tamaño de la máscara",
            "Ingresá el tamaño de la ventana (impar, ej: 3, 5, 7):",
            int,
        )
        if tamano is None:
            return None
        if tamano < 3 or tamano % 2 == 0:
            self.app.mostrar_error("El tamaño debe ser un número impar mayor o igual a 3.")
            return None
        return tamano

    def filtro_media(self):
        if self.app.imagen_original is None:
            self.app.mostrar_error("Primero cargá una imagen.")
            return

        tamano = self._pedir_tamano_mascara()
        if tamano is None:
            return

        resultado = self.ventana_deslizante(self.app.imagen_original, tamano, np.mean)
        self.app.establecer_resultado(np.clip(resultado, 0, 255).astype(np.uint8))

    def filtro_mediana(self):
        if self.app.imagen_original is None:
            self.app.mostrar_error("Primero cargá una imagen.")
            return

        tamano = self._pedir_tamano_mascara()
        if tamano is None:
            return

        resultado = self.ventana_deslizante(self.app.imagen_original, tamano, np.median)
        self.app.establecer_resultado(np.clip(resultado, 0, 255).astype(np.uint8))

    def filtro_mediana_ponderada(self):
        if self.app.imagen_original is None:
            self.app.mostrar_error("Primero cargá una imagen.")
            return

        pesos = MASCARA_MEDIANA_PONDERADA.flatten()

        def mediana_ponderada(ventana):
            valores = np.repeat(ventana.flatten(), pesos)
            return np.median(valores)

        resultado = self.ventana_deslizante(self.app.imagen_original, 3, mediana_ponderada)
        self.app.establecer_resultado(np.clip(resultado, 0, 255).astype(np.uint8))

    def mascara_gauss(self, tamano, sigma):
        centro = tamano // 2
        mascara = np.zeros((tamano, tamano), dtype=float)

        for i in range(tamano):
            for j in range(tamano):
                x = i - centro
                y = j - centro
                mascara[i, j] = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))

        return mascara / mascara.sum()

    def filtro_gauss(self):
        if self.app.imagen_original is None:
            self.app.mostrar_error("Primero cargá una imagen.")
            return

        sigma = self._pedir_numero("Filtro de Gauss", "Ingresá el valor de sigma (> 0):")
        if sigma is None:
            return
        if sigma <= 0:
            self.app.mostrar_error("Sigma debe ser mayor a 0.")
            return

        tamano = round(2 * sigma + 1)

        if tamano % 2 == 0:
            self.app.mostrar_error(
                f"Con sigma = {sigma} el tamaño de la máscara da {tamano} (par) y no tiene "
                "píxel central. Ingresá un sigma que dé un tamaño impar (por ejemplo, entero)."
            )
            return

        mascara = self.mascara_gauss(tamano, sigma)

        def promedio_pesado(ventana):
            return np.sum(ventana * mascara)

        resultado = self.ventana_deslizante(self.app.imagen_original, tamano, promedio_pesado)
        self.app.establecer_resultado(np.clip(resultado, 0, 255).astype(np.uint8))

    def filtro_realce_bordes(self):
        if self.app.imagen_original is None:
            self.app.mostrar_error("Primero cargá una imagen.")
            return

        def realce(ventana):
            return np.sum(ventana * MASCARA_REALCE_BORDES)

        resultado = self.ventana_deslizante(self.app.imagen_original, 3, realce)

        self.app.establecer_resultado(self.escalar_a_255(resultado))
