import threading
import time

import cv2
import numpy as np
from libreyolo import LibreYOLO


class DepthEstimator:
    """
    Estima o mapa de profundidade dos frames da câmera usando o modelo
    de depth do LibreYOLO (MiDaS Small, o mais leve/rápido para CPU).

    A inferência roda em uma thread separada para não travar o vídeo
    nem o controle por teclado.
    """

    def __init__(self, model_name: str = "LibreMiDaSs-depth.pt") -> None:
        """
        Args:
            model_name (str): Nome do checkpoint de depth do LibreYOLO.
        """
        print("Carregando modelo de depth...")
        self.model = LibreYOLO(model_name)
        print("Modelo de depth carregado.")

        self._lock = threading.Lock()
        self._latest_frame = None
        self._depth_color = None
        self._depth_raw = None
        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def update(self, frame) -> None:
        """
        Disponibiliza o frame mais recente para a thread de inferência.
        Args:
            frame: Frame BGR (numpy array) vindo da câmera do drone.
        """
        with self._lock:
            self._latest_frame = frame.copy()

    def get_depth(self):
        """
        Retorna o último mapa de profundidade colorido (BGR) pronto para exibição,
        ou None se ainda não houve inferência.
        """
        with self._lock:
            return self._depth_color

    def get_center_depth(self, square: int = 50):
        """
        Retorna a média de profundidade relativa em um quadrado no centro do mapa.
        Como o modelo gera inverse-depth relativo: quanto MAIOR o valor, mais PERTO.
        Args:
            square (int): Tamanho do lado do quadrado central, em pixels.
        Returns:
            float: Média de depth na região, ou None se ainda não houve inferência.
        """
        with self._lock:
            depth = self._depth_raw

        if depth is None:
            return None

        h, w = depth.shape[:2]
        cx, cy = w // 2, h // 2
        r = square // 2
        region = depth[cy - r:cy + r, cx - r:cx + r]
        if region.size == 0:
            return None
        return float(region.mean())

    def stop(self) -> None:
        """Para a thread de inferência."""
        self._running = False
        self._thread.join(timeout=2)

    def _worker(self) -> None:
        """Loop da thread: processa sempre o frame mais recente disponível."""
        while self._running:
            with self._lock:
                frame = self._latest_frame
                self._latest_frame = None

            if frame is None:
                time.sleep(0.01)
                continue

            depth_color, depth_raw = self._infer(frame)
            with self._lock:
                self._depth_color = depth_color
                self._depth_raw = depth_raw

    def _infer(self, frame):
        """
        Roda o modelo no frame e converte o depth_map em imagem BGR colorida.
        Args:
            frame: Frame BGR (numpy array).
        Returns:
            tuple: (imagem BGR colorida do depth, mapa de depth bruto (H, W) float).
                   No mapa bruto, maior = mais perto.
        """
        result = self.model(frame)
        depth = result.depth_map.data.cpu().numpy().squeeze()  # (H, W) float, maior = mais perto

        # Normaliza para 0-255 e aplica colormap para visualização
        depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO), depth
