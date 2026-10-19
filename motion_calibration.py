import threading
import time


class MotionCalibrator:
    """
    Calibração automática de escala métrica usando os sensores do próprio drone:
      - TOF (distância ao chão): verifica se o drone está voando a uma altura
        segura antes de iniciar a manobra;
      - IMU/odometria (velocidades vgx/vgy do estado do Tello): mede o
        deslocamento real (cm) durante um voo curto para frente;
      - Modelo de depth: mede o depth relativo do centro antes e depois.

    Como o modelo gera inverse-depth (v = escala / Z), o deslocamento ΔZ
    medido pela IMU fornece a escala métrica:
        escala = ΔZ / (1/v1 - 1/v2)
    """

    FORWARD_CM = 50      # Distância comandada para frente durante a calibração
    MIN_HEIGHT_CM = 80   # Altura mínima (TOF) para permitir a manobra
    SETTLE_TIME_S = 1.5  # Espera para estabilizar após o movimento
    SAMPLES = 10         # Amostras de depth por medição (média)

    def __init__(self, tello, depth, scale_calibrator, suspend_rc: threading.Event) -> None:
        """
        Args:
            tello (TelloZune): Instância do drone.
            depth (DepthEstimator): Estimador de depth em execução.
            scale_calibrator (DepthScaleCalibrator): Onde a escala é salva/lida.
            suspend_rc (threading.Event): Suspende o envio de RC do teclado
                durante a manobra (para não conflitar com o comando de voo).
        """
        self.tello = tello
        self.depth = depth
        self.scale_calibrator = scale_calibrator
        self.suspend_rc = suspend_rc
        self.running = False

    def start(self) -> bool:
        """Dispara a calibração em thread separada. False se já estiver rodando."""
        if self.running:
            print("Calibração já em andamento.")
            return False
        threading.Thread(target=self._run, daemon=True).start()
        return True

    def _center_depth_avg(self):
        """Média de várias leituras do depth relativo do centro (mais estável)."""
        vals = []
        for _ in range(self.SAMPLES):
            v = self.depth.get_center_depth()
            if v is not None:
                vals.append(v)
            time.sleep(0.1)
        return sum(vals) / len(vals) if vals else None

    def _measure_displacement_cm(self, stop_ev: threading.Event) -> float:
        """Integra a velocidade horizontal (IMU/odometria) até stop_ev ser setado."""
        dist = 0.0
        last = time.time()
        while not stop_ev.is_set():
            vx, vy, _ = self.tello.get_speed()
            now = time.time()
            dist += (vx * vx + vy * vy) ** 0.5 * (now - last)  # cm/s * s = cm
            last = now
            time.sleep(0.02)
        return dist

    def _run(self) -> None:
        self.running = True
        self.suspend_rc.set()  # Pausa o RC do teclado durante a manobra
        try:
            # 1) TOF: confirma que o drone está no ar, alto o suficiente
            try:
                height = float(self.tello.get_state_field('tof') or 0)
            except ValueError:
                height = 0
            if height < self.MIN_HEIGHT_CM:
                print(f"Calibração cancelada: altura {height:.0f}cm < {self.MIN_HEIGHT_CM}cm. "
                      "Decole (T) e aponte o centro para uma parede/objeto à frente.")
                return

            # 2) Depth relativo inicial
            print("Calibrando: medindo depth inicial do centro...")
            v1 = self._center_depth_avg()
            if v1 is None or v1 <= 0:
                print("Calibração cancelada: sem leitura de depth.")
                return

            # 3) Move para frente, medindo o deslocamento real via IMU/odometria
            print(f"Movendo {self.FORWARD_CM}cm para frente...")
            stop_ev = threading.Event()
            result = {}

            def integrate():
                result['dist'] = self._measure_displacement_cm(stop_ev)

            t = threading.Thread(target=integrate, daemon=True)
            t.start()
            self.tello.send_cmd_return(f"forward {self.FORWARD_CM}", timeout=8.0)
            stop_ev.set()
            t.join()
            dz_m = result.get('dist', 0.0) / 100.0

            # 4) Estabiliza e mede o depth relativo novamente
            time.sleep(self.SETTLE_TIME_S)
            v2 = self._center_depth_avg()
            if v2 is None or v2 <= 0:
                print("Calibração falhou: sem leitura de depth após o movimento.")
                return

            # 5) escala = ΔZ / (1/v1 - 1/v2)  (v = escala/Z  =>  1/v = Z/escala)
            denom = (1.0 / v1) - (1.0 / v2)
            if dz_m < 0.2 or denom <= 0:
                print(f"Calibração falhou (ΔZ={dz_m:.2f}m). "
                      "Aponte o centro para uma parede/objeto à frente, com espaço livre, e tente de novo.")
                return

            scale = dz_m / denom
            self.scale_calibrator.set_scale(scale, method="motion")
            print(f"Calibrado via movimento: ΔZ={dz_m:.2f}m, escala={scale:.2f}. "
                  "Distâncias em metros na tela.")
        finally:
            self.suspend_rc.clear()
            self.running = False
