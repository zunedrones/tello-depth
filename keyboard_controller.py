import threading
import time

import cv2


class KeyboardController:
    """
    Controla o drone Tello via teclado, usando a janela do OpenCV.

    Teclas:
        T      - Takeoff (decolar)
        L      - Land (pousar)
        W/S    - Frente / Trás
        A/D    - Esquerda / Direita
        R/F    - Subir / Descer
        Q/E    - Girar anti-horário / horário (yaw)
        Espaço - Parada de emergência (motores desligam imediatamente!)
        ESC    - Sair do programa
    """

    SPEED = 50         # Velocidade RC (0-100)
    KEY_TIMEOUT = 0.3  # Segundos que uma tecla permanece ativa após o último pressionamento

    def __init__(self, tello, actions: dict | None = None) -> None:
        """
        Args:
            tello (TelloZune): Instância do drone a ser controlado.
            actions (dict, opcional): Ações extras no formato {tecla: callable},
                executadas quando a tecla é pressionada (ex.: {'c': calibrar}).
        """
        self.tello = tello
        self.actions = actions or {}
        self.active_keys: dict[str, float] = {}  # tecla -> timestamp do último pressionamento
        self.suspend_rc = threading.Event()  # Quando setado, pausa o envio de RC (ex.: calibração em voo)

    def handle_key(self, key: int) -> bool:
        """
        Processa uma tecla vinda do cv2.waitKey().
        Args:
            key (int): Código da tecla retornado por cv2.waitKey().
        Returns:
            bool: True para continuar o loop, False para encerrar o programa.
        """
        if key == -1:
            return True

        key = key & 0xFF

        if key == 27:  # ESC
            return False

        if key == ord('t'):
            self.tello.takeoff()
        elif key == ord('l'):
            self.tello.land()
        elif key == ord(' '):
            self.active_keys.clear()
            self.tello.emergency_stop()
        elif chr(key) in self.actions:
            self.actions[chr(key)]()
        elif chr(key) in 'wsadrfqe':
            # Registra o pressionamento; a tecla permanece ativa por KEY_TIMEOUT segundos
            self.active_keys[chr(key)] = time.time()

        return True

    def update(self) -> None:
        """
        Recalcula as velocidades RC a partir das teclas ativas e envia ao drone.
        Deve ser chamada a cada iteração do loop principal. Envia rc 0 0 0 0
        quando nenhuma tecla está ativa, garantindo que o drone pare.
        Ignorado enquanto suspend_rc estiver setado (ex.: calibração em voo).
        """
        if self.suspend_rc.is_set():
            self.active_keys.clear()
            return

        now = time.time()
        self.active_keys = {
            k: t for k, t in self.active_keys.items()
            if now - t < self.KEY_TIMEOUT
        }

        s = self.SPEED
        left_right = s * ('d' in self.active_keys) - s * ('a' in self.active_keys)
        forward_back = s * ('w' in self.active_keys) - s * ('s' in self.active_keys)
        up_down = s * ('r' in self.active_keys) - s * ('f' in self.active_keys)
        yaw = s * ('e' in self.active_keys) - s * ('q' in self.active_keys)

        self.tello.send_rc_control(left_right, forward_back, up_down, yaw)

    def draw_help(self, frame) -> None:
        """
        Desenha as instruções de controle sobre o frame exibido.
        Args:
            frame: Frame do vídeo (numpy array) onde o texto será desenhado.
        """
        lines = [
            "T: takeoff | L: land | C: calibrar | V: calibrar auto | ESC: sair",
            "W/S/A/D: move | R/F: sobe/desce | Q/E: gira | ESPACO: emerg.",
        ]
        for i, line in enumerate(lines):
            y = frame.shape[0] - 40 + (i * 25)
            cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 255), 1)
