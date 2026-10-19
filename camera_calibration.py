import json
import os

import cv2
import numpy as np


class CameraCalibration:
    """
    Carrega os parâmetros intrínsecos da câmera de um arquivo de calibração
    no formato OpenCV/YAML (ex.: config/tello_02.yaml) e os adapta para a
    resolução de trabalho.
    """

    def __init__(self, path: str, width: int, height: int) -> None:
        """
        Args:
            path (str): Caminho do arquivo de calibração (.yaml).
            width (int): Largura da imagem de trabalho (px).
            height (int): Altura da imagem de trabalho (px).
        """
        fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
        if not fs.isOpened():
            raise FileNotFoundError(f"Arquivo de calibração não encontrado: {path}")

        fx = fs.getNode("Camera1.fx").real()
        fy = fs.getNode("Camera1.fy").real()
        cx = fs.getNode("Camera1.cx").real()
        cy = fs.getNode("Camera1.cy").real()
        cal_w = fs.getNode("Camera.width").real()
        cal_h = fs.getNode("Camera.height").real()
        dist = np.array([
            fs.getNode("Camera1.k1").real(),
            fs.getNode("Camera1.k2").real(),
            fs.getNode("Camera1.p1").real(),
            fs.getNode("Camera1.p2").real(),
            fs.getNode("Camera1.k3").real(),
        ], dtype=np.float64)
        fs.release()

        # Escala os intrínsecos da resolução da calibração para a de trabalho
        sx = width / cal_w
        sy = height / cal_h
        self.K = np.array([
            [fx * sx,      0, cx * sx],
            [0,      fy * sy, cy * sy],
            [0,            0,       1],
        ], dtype=np.float64)
        self.dist = dist
        self.width = width
        self.height = height

        # Mapas de undistort pré-computados (o remap por frame é barato)
        self._map1, self._map2 = cv2.initUndistortRectifyMap(
            self.K, self.dist, None, self.K, (width, height), cv2.CV_32FC1
        )

    def undistort(self, frame):
        """Corrige a distorção da lente usando os parâmetros de calibração."""
        return cv2.remap(frame, self._map1, self._map2, cv2.INTER_LINEAR)


class DepthScaleCalibrator:
    """
    Converte o depth relativo do modelo (inverse-depth: maior = mais perto)
    em distância métrica, a partir de uma calibração de 1 ponto: aponta-se o
    quadrado central para um objeto a uma distância conhecida.

    Relação: distancia (m) = scale / valor_relativo
    """

    def __init__(self, path: str = "config/depth_scale.json") -> None:
        """
        Args:
            path (str): Arquivo onde a escala calibrada é salva/carregada.
        """
        self.path = path
        self.scale = None
        if os.path.exists(path):
            with open(path) as f:
                self.scale = json.load(f).get("scale")

    @property
    def is_calibrated(self) -> bool:
        return self.scale is not None

    def calibrate(self, relative_value: float, known_distance_m: float) -> None:
        """
        Define a escala a partir de uma leitura a distância conhecida e salva em disco.
        Args:
            relative_value (float): Valor relativo lido no quadrado central.
            known_distance_m (float): Distância real do objeto, em metros.
        """
        if relative_value <= 0:
            raise ValueError("Valor relativo inválido para calibração.")
        self.set_scale(known_distance_m * relative_value, method="manual",
                       known_distance_m=known_distance_m)

    def set_scale(self, scale: float, method: str = "manual", **extra) -> None:
        """
        Define a escala diretamente e salva em disco.
        Args:
            scale (float): Fator de escala (distancia_m = scale / valor_relativo).
            method (str): Como a escala foi obtida ("manual", "motion", ...).
        """
        if scale <= 0:
            raise ValueError("Escala inválida.")
        self.scale = scale
        with open(self.path, "w") as f:
            json.dump({"scale": self.scale, "method": method, **extra}, f)

    def to_meters(self, relative_value: float):
        """
        Converte um valor relativo em metros.
        Returns:
            float: Distância em metros, ou None se ainda não calibrado.
        """
        if self.scale is None or relative_value <= 0:
            return None
        return self.scale / relative_value
