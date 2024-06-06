import cv2
import mediapipe as mp
import numpy as np

correct = cv2.imread('right.png')
correct = cv2.cvtColor(correct, cv2.COLOR_BGR2RGB)
incorrect = cv2.imread('wrong.png')
incorrect = cv2.cvtColor(incorrect, cv2.COLOR_BGR2RGB)
counter = cv2.imread('counter.png')
counter = cv2.cvtColor(counter, cv2.COLOR_BGR2RGB)


def draw_rounded_rect(img, rect_start, rect_end, corner_width, box_color):
    """
    Dibuja un rectángulo con esquinas redondeadas en una imagen.

    Args:
        img: Imagen donde se dibujará el rectángulo.
        rect_start: Coordenada superior izquierda del rectángulo (x1, y1).
        rect_end: Coordenada inferior derecha del rectángulo (x2, y2).
        corner_width: Ancho de las esquinas redondeadas.
        box_color: Color del rectángulo.

    Returns:
        Imagen con el rectángulo dibujado.
    """

    x1, y1 = rect_start
    x2, y2 = rect_end
    w = corner_width

    # Dibujar rectángulos rellenos
    cv2.rectangle(img, (x1 + w, y1), (x2 - w, y1 + w), box_color, -1)
    cv2.rectangle(img, (x1 + w, y2 - w), (x2 - w, y2), box_color, -1)
    cv2.rectangle(img, (x1, y1 + w), (x1 + w, y2 - w), box_color, -1)
    cv2.rectangle(img, (x2 - w, y1 + w), (x2, y2 - w), box_color, -1)
    cv2.rectangle(img, (x1 + w, y1 + w), (x2 - w, y2 - w), box_color, -1)

    # Dibujar elipses rellenas
    cv2.ellipse(img, (x1 + w, y1 + w), (w, w),
                angle=0,
                startAngle=-90,
                endAngle=-180,
                color=box_color,
                thickness=-1)

    cv2.ellipse(img, (x2 - w, y1 + w), (w, w),
                angle=0,
                startAngle=0,
                endAngle=-90,
                color=box_color,
                thickness=-1)

    cv2.ellipse(img, (x1 + w, y2 - w), (w, w),
                angle=0,
                startAngle=90,
                endAngle=180,
                color=box_color,
                thickness=-1)

    cv2.ellipse(img, (x2 - w, y2 - w), (w, w),
                angle=0,
                startAngle=0,
                endAngle=90,
                color=box_color,
                thickness=-1)

    return img


def draw_dotted_line(frame, lm_coord, start, end, line_color):
    """
   Dibuja una línea de puntos en una imagen.

   Args:
       frame: Imagen donde se dibujará la línea.
       lm_coord: Coordenadas de inicio de la línea.
       start: Posición de inicio en el eje Y.
       end: Posición de fin en el eje Y.
       line_color: Color de la línea.

   Returns:
       Imagen con la línea dibujada.
   """
    pix_step = 0

    for i in range(start, end + 1, 8):
        cv2.circle(frame, (lm_coord[0], i + pix_step),
                   2,
                   line_color,
                   -1,
                   lineType=cv2.LINE_AA)

    return frame


def draw_text(img,
              msg,
              width=7,
              font=cv2.FONT_HERSHEY_SIMPLEX,
              pos=(0, 0),
              font_scale=1,
              font_thickness=2,
              text_color=(0, 255, 0),
              text_color_bg=(0, 0, 0),
              box_offset=(20, 10),
              overlay_image=False,
              overlay_type=None):
    """
    Dibuja un texto en una imagen, con la opción de superponer una imagen.

    Args:
        img: Imagen donde se dibujará el texto.
        msg: Mensaje de texto.
        width: Ancho del borde redondeado.
        font: Tipo de fuente del texto.
        pos: Posición del texto en la imagen (x, y).
        font_scale: Escala de la fuente.
        font_thickness: Grosor de la fuente.
        text_color: Color del texto.
        text_color_bg: Color de fondo del texto.
        box_offset: Desplazamiento del cuadro de texto.
        overlay_image: Si se debe superponer una imagen.
        overlay_type: Tipo de imagen a superponer ("correct", "incorrect", "counter").

    Returns:
        Tamaño del texto.
    """

    offset = box_offset
    x, y = pos
    text_size, _ = cv2.getTextSize(msg, font, font_scale, font_thickness)
    text_w, text_h = text_size

    rec_start = tuple(p - o for p, o in zip(pos, offset))
    rec_end = tuple(
        m + n - o
        for m, n, o in zip((x + text_w, y + text_h), offset, (25, 0)))

    resize_height = 0

    if overlay_image:
        resize_height = rec_end[1] - rec_start[1]

        img = draw_rounded_rect(img, rec_start,
                                (rec_end[0] + resize_height, rec_end[1]),
                                width, text_color_bg)
        if overlay_type == "correct":
            overlay_res = cv2.resize(correct, (resize_height, resize_height),
                                     interpolation=cv2.INTER_AREA)
        elif overlay_type == "incorrect":
            overlay_res = cv2.resize(incorrect, (resize_height, resize_height),
                                     interpolation=cv2.INTER_AREA)
        elif overlay_type == "counter":
            overlay_res = cv2.resize(counter, (resize_height, resize_height),
                                     interpolation=cv2.INTER_AREA)

        img[rec_start[1]:rec_start[1] + resize_height, rec_start[0] +
            width:rec_start[0] + width + resize_height] = overlay_res

    else:
        img = draw_rounded_rect(img, rec_start, rec_end, width, text_color_bg)

    cv2.putText(
        img,
        msg,
        (int(rec_start[0] + resize_height + 8),
         int(y + text_h + font_scale - 1)),
        font,
        font_scale,
        text_color,
        font_thickness,
        cv2.LINE_AA,
    )

    return text_size


def find_angle(p1, p2, ref_pt=np.array([0, 0])):
    """
   Calcula el ángulo entre dos puntos con respecto a un punto de referencia.

   Args:
       p1: Primer punto (x, y).
       p2: Segundo punto (x, y).
       ref_pt: Punto de referencia (x, y). Por defecto es el origen (0, 0).

   Returns:
       Ángulo en grados entre los puntos con respecto al punto de referencia.
   """
    p1_ref = p1 - ref_pt
    p2_ref = p2 - ref_pt

    cos_theta = (np.dot(p1_ref, p2_ref)) / (1.0 * np.linalg.norm(p1_ref) *
                                            np.linalg.norm(p2_ref))
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))

    degree = int(180 / np.pi) * theta

    return int(degree)


def get_landmark_array(pose_landmark, key, frame_width, frame_height):
    """
    Obtiene las coordenadas de un punto de referencia de una pose normalizadas al tamaño del marco.

    Args:
        pose_landmark: Resultado de los puntos de referencia de la pose.
        key: Clave del punto de referencia a obtener.
        frame_width: Ancho del marco.
        frame_height: Altura del marco.

    Returns:
        Coordenadas del punto de referencia (x, y) en el marco.
    """

    denorm_x = int(pose_landmark[key].x * frame_width)
    denorm_y = int(pose_landmark[key].y * frame_height)

    return np.array([denorm_x, denorm_y])


def get_landmark_features(kp_results, dict_features, feature, frame_width,
                          frame_height):
    """
   Obtiene las coordenadas de los puntos de referencia para una característica específica de la pose.

   Args:
       kp_results: Resultados de los puntos clave de la pose.
       dict_features: Diccionario de características con las claves de los puntos de referencia.
       feature: Característica específica a obtener ('nose', 'left', 'right').
       frame_width: Ancho del marco.
       frame_height: Altura del marco.

   Returns:
       Coordenadas de los puntos de referencia para la característica especificada.

   Raises:
       ValueError: Si la característica no es 'nose', 'left' o 'right'.
   """

    if feature == 'nose':
        return get_landmark_array(kp_results, dict_features[feature],
                                  frame_width, frame_height)

    elif feature == 'left' or 'right':
        shldr_coord = get_landmark_array(kp_results,
                                         dict_features[feature]['shoulder'],
                                         frame_width, frame_height)
        elbow_coord = get_landmark_array(kp_results,
                                         dict_features[feature]['elbow'],
                                         frame_width, frame_height)
        wrist_coord = get_landmark_array(kp_results,
                                         dict_features[feature]['wrist'],
                                         frame_width, frame_height)
        hip_coord = get_landmark_array(kp_results,
                                       dict_features[feature]['hip'],
                                       frame_width, frame_height)
        knee_coord = get_landmark_array(kp_results,
                                        dict_features[feature]['knee'],
                                        frame_width, frame_height)
        ankle_coord = get_landmark_array(kp_results,
                                         dict_features[feature]['ankle'],
                                         frame_width, frame_height)
        foot_coord = get_landmark_array(kp_results,
                                        dict_features[feature]['foot'],
                                        frame_width, frame_height)

        return shldr_coord, elbow_coord, wrist_coord, hip_coord, knee_coord, ankle_coord, foot_coord

    else:
        raise ValueError("feature needs to be either 'nose', 'left' or 'right")


def get_mediapipe_pose(static_image_mode=False,
                       model_complexity=1,
                       smooth_landmarks=True,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5):
    """
    Configura y devuelve un objeto de pose de MediaPipe.

    Args:
        static_image_mode: Si se debe usar el modo de imagen estática.
        model_complexity: Complejidad del modelo (0, 1, 2).
        smooth_landmarks: Si se deben suavizar los puntos de referencia.
        min_detection_confidence: Confianza mínima para la detección.
        min_tracking_confidence: Confianza mínima para el seguimiento.

    Returns:
        Objeto de pose de MediaPipe configurado.
    """

    pose = mp.solutions.pose.Pose(
        static_image_mode=static_image_mode,
        model_complexity=model_complexity,
        smooth_landmarks=smooth_landmarks,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence)
    return pose
