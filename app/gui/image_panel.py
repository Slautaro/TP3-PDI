import tkinter as tk

import customtkinter as ctk
from PIL import Image, ImageTk

from app.core.conversion import cv2_a_pil

TAMANO_CANVAS = (480, 480)
MAX_AMPLIACION = 10.0
TOLERANCIA_CLICK_PX = 3

FUENTE_TITULO = ("Roboto", 14, "bold")

class PanelImagen(ctk.CTkFrame):
    def __init__(self, master, titulo, al_hacer_click_pixel=None, al_seleccionar_region=None, **kwargs):
        super().__init__(master, **kwargs)
        self.al_hacer_click_pixel = al_hacer_click_pixel
        self.al_seleccionar_region = al_seleccionar_region

        self._imagen_cv = None
        self._imagen_tk = None
        self._escala = 1.0
        self._desplazamiento = (0, 0)
        self._tamano_mostrado = (0, 0)
        self._punto_click_canvas = None
        self._id_rectangulo_seleccion = None

        self.etiqueta_titulo = ctk.CTkLabel(self, text=titulo, font=FUENTE_TITULO)
        self.etiqueta_titulo.pack(pady=(6, 2))

        self.canvas = tk.Canvas(
            self, width=TAMANO_CANVAS[0], height=TAMANO_CANVAS[1],
            bg="#2b2b2b", highlightthickness=1, highlightbackground="#555555",
        )
        self.canvas.pack(padx=8, pady=(0, 8))

        self.canvas.bind("<ButtonPress-1>", self._al_presionar)
        self.canvas.bind("<B1-Motion>", self._al_arrastrar)
        self.canvas.bind("<ButtonRelease-1>", self._al_soltar)

    def establecer_imagen(self, imagen_cv):
        self._imagen_cv = imagen_cv
        self._limpiar_seleccion()
        self._dibujar()

    def obtener_imagen(self):
        return self._imagen_cv

    def _dibujar(self):
        self.canvas.delete("imagen")
        if self._imagen_cv is None:
            return

        imagen_pil = cv2_a_pil(self._imagen_cv)
        ancho_img, alto_img = imagen_pil.size
        ancho_canvas, alto_canvas = TAMANO_CANVAS

        escala = min(ancho_canvas / ancho_img, alto_canvas / alto_img, MAX_AMPLIACION)
        ancho_disp = max(1, round(ancho_img * escala))
        alto_disp = max(1, round(alto_img * escala))
        redimensionada = imagen_pil.resize((ancho_disp, alto_disp), Image.NEAREST)
        self._imagen_tk = ImageTk.PhotoImage(redimensionada)

        despl_x, despl_y = (ancho_canvas - ancho_disp) // 2, (alto_canvas - alto_disp) // 2
        self._escala = escala
        self._desplazamiento = (despl_x, despl_y)
        self._tamano_mostrado = (ancho_disp, alto_disp)

        self.canvas.create_image(despl_x, despl_y, anchor="nw", image=self._imagen_tk, tags="imagen")

    def _canvas_a_coordenadas_imagen(self, canvas_x, canvas_y, ajustar=False):
        if self._imagen_cv is None:
            return None

        despl_x, despl_y = self._desplazamiento
        ancho_disp, alto_disp = self._tamano_mostrado

        if ajustar:
            canvas_x = min(max(canvas_x, despl_x), despl_x + ancho_disp - 1)
            canvas_y = min(max(canvas_y, despl_y), despl_y + alto_disp - 1)
        elif not (despl_x <= canvas_x < despl_x + ancho_disp and despl_y <= canvas_y < despl_y + alto_disp):
            return None

        imagen_x = int((canvas_x - despl_x) / self._escala)
        imagen_y = int((canvas_y - despl_y) / self._escala)
        alto, ancho = self._imagen_cv.shape[:2]
        return min(max(imagen_x, 0), ancho - 1), min(max(imagen_y, 0), alto - 1)

    def _al_presionar(self, evento):
        self._punto_click_canvas = (evento.x, evento.y)
        self._limpiar_seleccion()

    def _al_arrastrar(self, evento):
        if self._punto_click_canvas is None:
            return
        x0, y0 = self._punto_click_canvas
        self._dibujar_rectangulo_seleccion(x0, y0, evento.x, evento.y)

    def _al_soltar(self, evento):
        if self._punto_click_canvas is None:
            return
        x0, y0 = self._punto_click_canvas
        x1, y1 = evento.x, evento.y
        self._punto_click_canvas = None

        se_movio = abs(x1 - x0) > TOLERANCIA_CLICK_PX or abs(y1 - y0) > TOLERANCIA_CLICK_PX
        if not se_movio:
            self._limpiar_seleccion()
            punto = self._canvas_a_coordenadas_imagen(x1, y1, ajustar=False)
            if punto and self.al_hacer_click_pixel:
                self.al_hacer_click_pixel(*punto)
            return

        inicio = self._canvas_a_coordenadas_imagen(x0, y0, ajustar=True)
        fin = self._canvas_a_coordenadas_imagen(x1, y1, ajustar=True)
        if inicio and fin and self.al_seleccionar_region:
            self.al_seleccionar_region(inicio[0], inicio[1], fin[0], fin[1])

    def _dibujar_rectangulo_seleccion(self, x0, y0, x1, y1):
        if self._id_rectangulo_seleccion is not None:
            self.canvas.delete(self._id_rectangulo_seleccion)
        self._id_rectangulo_seleccion = self.canvas.create_rectangle(
            x0, y0, x1, y1, outline="#ff3b30", width=2, tags="seleccion"
        )

    def _limpiar_seleccion(self):
        if self._id_rectangulo_seleccion is not None:
            self.canvas.delete(self._id_rectangulo_seleccion)
            self._id_rectangulo_seleccion = None
