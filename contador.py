import cv2
import mediapipe as mp

# Inicializa a captura de vídeo da câmera
video = cv2.VideoCapture(0)  # Índice 0 representa a câmera padrão do sistema

# Inicializa o módulo Hands do Mediapipe
hands = mp.solutions.hands
Hands = hands.Hands(max_num_hands=1)
mpDwaw = mp.solutions.drawing_utils

while True:
    # Lê um quadro do vídeo
    success, img = video.read()

    # Converte o espaço de cores do quadro de BGR para RGB
    frameRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Processa o quadro usando o módulo Hands do Mediapipe
    results = Hands.process(frameRGB)
    handPoints = results.multi_hand_landmarks
    h, w, _ = img.shape
    pontos = []

    if handPoints:
        for points in handPoints:
            # Desenha as landmarks das mãos no quadro
            mpDwaw.draw_landmarks(img, points, hands.HAND_CONNECTIONS)

            # Extrai as coordenadas das landmarks e as adiciona à lista de pontos
            for id, cord in enumerate(points.landmark):
                cx, cy = int(cord.x * w), int(cord.y * h)
                pontos.append((cx, cy))

            dedos = [8, 12, 16, 20]
            contador = 0

            if pontos:
                # Verifica se o polegar está dobrado em relação ao indicador
                if pontos[4][0] < pontos[3][0]:
                    contador += 1

                # Verifica se os outros dedos estão dobrados em relação aos dedos anteriores
                for x in dedos:
                    if pontos[x][1] < pontos[x - 2][1]:
                        contador += 1

            # Desenha um retângulo e exibe o contador de dedos na tela
            cv2.rectangle(img, (80, 10), (200, 110), (255, 0, 0), -1)
            cv2.putText(img, str(contador), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 5)

    # Exibe o quadro na janela
    cv2.imshow('Imagem', img)

    # Aguarda o pressionamento de uma tecla (1 ms)
    cv2.waitKey(1)
