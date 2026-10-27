import customtkinter as ctk

FUENTE_INFO = ("Roboto", 11)

class PanelInformacion(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)

        self.etiqueta_pixel = ctk.CTkLabel(self, text="Píxel: -", anchor="w", font=FUENTE_INFO)
        self.etiqueta_pixel.pack(fill="x", padx=15, pady=(10, 4))

        self.etiqueta_region = ctk.CTkLabel(self, text="Región: -", anchor="w", font=FUENTE_INFO)
        self.etiqueta_region.pack(fill="x", padx=15, pady=4)

        self.etiqueta_estadisticas = ctk.CTkLabel(self, text="Estadísticas: -", anchor="w", font=FUENTE_INFO)
        self.etiqueta_estadisticas.pack(fill="x", padx=15, pady=(4, 10))

    def mostrar_pixel(self, origen: str, x: int, y: int, valor) -> None:
        self.etiqueta_pixel.configure(text=f"Píxel [{origen}] ({x}, {y}) = {valor}")

    def mostrar_region(self, origen: str, estadisticas: dict, coordenadas) -> None:
        x1, y1, x2, y2 = coordenadas
        self.etiqueta_region.configure(
            text=f"Región [{origen}]: ({x1}, {y1}) -> ({x2}, {y2})  |  "
                 f"{estadisticas['ancho']}x{estadisticas['alto']} px"
        )
        if "promedio_color" in estadisticas:
            r, g, b = estadisticas["promedio_color"]
            texto_promedio = f"Promedio color: R={r:.1f}, G={g:.1f}, B={b:.1f}"
        else:
            texto_promedio = f"Promedio gris: {estadisticas['promedio_gris']:.1f}"
        texto_estadisticas = f"Píxeles: {estadisticas['cantidad_pixeles']}  |  {texto_promedio}"
        self.etiqueta_estadisticas.configure(text=texto_estadisticas)

    def limpiar(self) -> None:
        self.etiqueta_pixel.configure(text="Píxel: -")
        self.etiqueta_region.configure(text="Región: -")
        self.etiqueta_estadisticas.configure(text="Estadísticas: -")
