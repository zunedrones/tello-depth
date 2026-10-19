import cv2
import numpy as np

from tello_zune.tello_zune import TelloZune
from keyboard_controller import KeyboardController
from depth_estimator import DepthEstimator
from camera_calibration import CameraCalibration, DepthScaleCalibrator
from motion_calibration import MotionCalibrator

WIDTH, HEIGHT = 960, 720 # Mesma resolução do arquivo de calibração (config/tello_02.yaml)
CENTER_SQUARE = 50 # Tamanho do quadrado central (px) usado para medir o depth
KNOWN_DISTANCE_M = 1.0 # Distância do objeto usado na calibração instantânea (tecla C)

tello = TelloZune() # Cria objeto da classe TelloZune
tello.set_image_size((WIDTH, HEIGHT)) # Resolução da imagem do drone
tello.start_tello() # Inicia a comunicação com o drone

calibration = CameraCalibration("config/tello_02.yaml", WIDTH, HEIGHT) # Intrínsecos da câmera
calibrator = DepthScaleCalibrator("config/depth_scale.json") # Escala p/ converter depth relativo em metros
depth = DepthEstimator() # Estimador de profundidade (LibreYOLO), roda em thread separada

latest_center = {} # Guarda o último depth relativo do centro para a calibração instantânea

def calibrate() -> None:
    """Calibração instantânea (sem manobra): aponte o quadrado central para um
    objeto a KNOWN_DISTANCE_M metros e pressione C. Funciona até com o drone no chão."""
    value = latest_center.get("value")
    if value is None:
        print("Calibração ignorada: ainda não há leitura de depth.")
        return
    calibrator.calibrate(value, KNOWN_DISTANCE_M)
    print(f"Calibrado: {value:.2f} (relativo) = {KNOWN_DISTANCE_M} m")

controller = KeyboardController(tello) # Controlador de teclado para pilotar o drone
motion_cal = MotionCalibrator(tello, depth, calibrator, controller.suspend_rc) # Calibração automática via TOF+IMU
controller.actions = {'c': calibrate, 'v': motion_cal.start} # C: instantânea | V: manobra automática

try:
    while True:
        # Captura
        frame = tello.get_frame()
        depth.update(calibration.undistort(frame)) # Envia frame corrigido (calibração) para a thread de depth

        # Tratamento
        bat, height, temph, pres, time_elapsed = tello.get_info()
        fps = tello.calc_fps()
        cv2.putText(frame, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (10, 255, 0), 2)
        cv2.putText(frame, f"Bat: {bat}%", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (10, 255, 0), 2)
        cv2.putText(frame, f"Height: {height}cm", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (10, 255, 0), 2)
        cv2.putText(frame, f"Max. Temp.: {temph}C", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (10, 255, 0), 2)
        cv2.putText(frame, f"Press.: {pres}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (10, 255, 0), 2)
        cv2.putText(frame, f"TOF: {time_elapsed}s", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (10, 255, 0), 2)
        controller.draw_help(frame)

        # Quadrado central para medição de depth (média da área)
        center_depth = depth.get_center_depth(CENTER_SQUARE)
        latest_center["value"] = center_depth
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        r = CENTER_SQUARE // 2
        cv2.rectangle(frame, (cx - r, cy - r), (cx + r, cy + r), (0, 255, 255), 2)
        if center_depth is not None:
            dist_m = calibrator.to_meters(center_depth)
            if dist_m is not None:
                text = f"Dist: {dist_m:.2f} m"
            else:
                text = f"Depth: {center_depth:.1f} (C: calibrar @{KNOWN_DISTANCE_M}m)"
            cv2.putText(frame, text, (cx - r, cy - r - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Exibição: câmera à esquerda, depth à direita
        depth_frame = depth.get_depth()
        if depth_frame is not None:
            depth_frame = cv2.resize(depth_frame, (w, h))
            cv2.rectangle(depth_frame, (cx - r, cy - r), (cx + r, cy + r), (0, 255, 255), 2)
            display = np.hstack((frame, depth_frame))
        else:
            display = frame
        cv2.imshow('Tello', display)

        # Controle pelo teclado (ESC para sair)
        key = cv2.waitKey(1)
        if not controller.handle_key(key):
            break
        controller.update()
finally:
    # Finalização
    depth.stop()
    tello.end_tello()
    cv2.destroyAllWindows()
