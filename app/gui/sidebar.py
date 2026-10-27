import customtkinter as ctk

FUENTE_SECCION = ("Roboto", 13, "bold")
FUENTE_BOTON = ("Roboto", 12)

ALTO_BOTON = 32
RADIO_BOTON = 10

COLOR_BORDE_SECUNDARIO = ("gray70", "gray40")
COLOR_HOVER_SECUNDARIO = ("gray85", "gray25")

class BarraLateral(ctk.CTkScrollableFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self._secciones = {}
        self._entradas = []
        self._grupos = []
        self._selector = None
        self._grupo_visible = None

        self.grupo_actual = None

    def agregar_seccion(self, titulo: str) -> ctk.CTkFrame:
        clave = (self.grupo_actual, titulo)
        if clave in self._secciones:
            return self._secciones[clave]

        etiqueta = ctk.CTkLabel(self, text=titulo, font=FUENTE_SECCION)
        marco = ctk.CTkFrame(self, fg_color="transparent")

        self._secciones[clave] = marco
        self._entradas.append((self.grupo_actual, etiqueta, marco))
        if self.grupo_actual is not None and self.grupo_actual not in self._grupos:
            self._grupos.append(self.grupo_actual)
        self._refrescar()
        return marco

    def fijar_siempre_visible(self, titulo: str) -> None:
        self._entradas = [
            (None if self._secciones[(self.grupo_actual, titulo)] is marco else grupo, etiqueta, marco)
            for grupo, etiqueta, marco in self._entradas
        ]
        self._refrescar()

    def mostrar_grupo(self, grupo: str) -> None:
        self._grupo_visible = grupo
        if self._selector is not None:
            self._selector.set(grupo)
        self._refrescar()
        try:
            self._parent_canvas.yview_moveto(0)
        except Exception:
            pass

    def construir_selector(self, grupo_inicial: str) -> None:
        self._selector = ctk.CTkSegmentedButton(
            self, values=self._grupos, command=self.mostrar_grupo, font=FUENTE_BOTON,
        )
        self.mostrar_grupo(grupo_inicial)

    def _refrescar(self) -> None:
        for _, etiqueta, marco in self._entradas:
            etiqueta.pack_forget()
            marco.pack_forget()
        if self._selector is not None:
            self._selector.pack_forget()

        for grupo, etiqueta, marco in self._entradas:
            if grupo is None:
                self._empaquetar_seccion(etiqueta, marco)
        if self._selector is not None:
            self._selector.pack(fill="x", padx=15, pady=(15, 0))
        for grupo, etiqueta, marco in self._entradas:
            if grupo is not None and (self._selector is None or grupo == self._grupo_visible):
                self._empaquetar_seccion(etiqueta, marco)

    def _empaquetar_seccion(self, etiqueta, marco) -> None:
        etiqueta.pack(fill="x", padx=15, pady=(15, 4))
        marco.pack(fill="x", padx=15, pady=(0, 10))

    def agregar_boton(self, titulo_seccion: str, texto: str, accion, primario: bool = False) -> ctk.CTkButton:
        marco = self.agregar_seccion(titulo_seccion)
        estilo = {} if primario else {
            "fg_color": "transparent",
            "border_width": 1,
            "border_color": COLOR_BORDE_SECUNDARIO,
            "hover_color": COLOR_HOVER_SECUNDARIO,
        }
        boton = ctk.CTkButton(
            marco, text=texto, command=accion,
            font=FUENTE_BOTON, corner_radius=RADIO_BOTON, height=ALTO_BOTON,
            **estilo,
        )
        boton.pack(fill="x", pady=6)
        return boton
