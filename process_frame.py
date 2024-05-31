import time
import cv2
import numpy as np
from utils import find_angle, get_landmark_features, draw_text, draw_dotted_line


class ProcessFrame:
    def __init__(self, thresholds, flip_frame=False):

        # Configuramos si la orientación de la cama se invierte
        self.flip_frame = flip_frame

        # Umbrales
        self.thresholds = thresholds

        # TIPO DE FUENTE
        self.font = cv2.FONT_HERSHEY_SIMPLEX

        # Tipo de linea
        self.linetype = cv2.LINE_AA

        # Establecer radio para dibujar el arco
        self.radius = 20

        # Los colores se encuentran en un formato RGB
        self.COLORS = {
            'blue': (0, 127, 255),
            'red': (255, 50, 50),
            'green': (0, 255, 127),
            'light_green': (100, 233, 127),
            'yellow': (255, 255, 0),
            'magenta': (255, 0, 255),
            'white': (255, 255, 255),
            'cyan': (0, 255, 255),
            'light_blue': (102, 204, 255)
        }

        # Diccionario para mantener las diversas características de los puntos de referencia.
        self.dict_features = {}
        self.left_features = {
            'shoulder': 11,
            'elbow': 13,
            'wrist': 15,
            'hip': 23,
            'knee': 25,
            'ankle': 27,
            'foot': 31
        }

        self.right_features = {
            'shoulder': 12,
            'elbow': 14,
            'wrist': 16,
            'hip': 24,
            'knee': 26,
            'ankle': 28,
            'foot': 32
        }

        self.dict_features['left'] = self.left_features
        self.dict_features['right'] = self.right_features
        self.dict_features['nose'] = 0

        # Para rastrear contadores y compartir estados dentro y fuera de las funciones de callback.
        self.state_tracker = {
            'state_seq': [],

            'start_inactive_time': time.perf_counter(),
            'start_inactive_time_front': time.perf_counter(),
            'INACTIVE_TIME': 0.0,
            'INACTIVE_TIME_FRONT': 0.0,

            # 0 --> INCLINATE HACIA ATRÁS, 1 --> INCLINATE HACIA ADELANT, 2 --> RODILLA SOBREPASANDO EL DEDO DEL PIE, 3 --> SENTADILLA DEMASIADO PROFUNDA
            'DISPLAY_TEXT': np.full((4,), False),
            'COUNT_FRAMES': np.zeros((4,), dtype=np.int64),

            'LOWER_HIPS': False,

            'INCORRECT_POSTURE': False,

            'prev_state': None,
            'curr_state': None,

            'SQUAT_COUNT': 0,
            'IMPROPER_SQUAT': 0,
            'TOTAL_SQUAT_COUNT': 0

        }

        self.FEEDBACK_ID_MAP = {
            0: ('INCLINATE HACIA ATRÁS', 215, (0, 153, 255)),
            1: ('INCLINATE HACIA ADELANTE', 215, (0, 153, 255)),
            2: ('RODILLA SOBREPASANDO EL DEDO DEL PIE', 170, (255, 80, 80)),
            3: ('SENTADILLA DEMASIADO PROFUNDA', 125, (255, 80, 80))
        }

    def _get_state(self, knee_angle):
        # Determina el estado actual basado en el ángulo de la rodilla.
        knee = None

        if self.thresholds['HIP_KNEE_VERT']['NORMAL'][0] <= knee_angle <= self.thresholds['HIP_KNEE_VERT']['NORMAL'][1]:
            knee = 1
        elif self.thresholds['HIP_KNEE_VERT']['TRANS'][0] <= knee_angle <= self.thresholds['HIP_KNEE_VERT']['TRANS'][1]:
            knee = 2
        elif self.thresholds['HIP_KNEE_VERT']['PASS'][0] <= knee_angle <= self.thresholds['HIP_KNEE_VERT']['PASS'][1]:
            knee = 3

        return f's{knee}' if knee else None

    def _update_state_sequence(self, state):
        # Actualiza la secuencia de estados para rastrear la progresión de los movimientos de sentadilla.
        if state == 's2':
            if (('s3' not in self.state_tracker['state_seq']) and (self.state_tracker['state_seq'].count('s2')) == 0) or \
                    (('s3' in self.state_tracker['state_seq']) and (self.state_tracker['state_seq'].count('s2') == 1)):
                self.state_tracker['state_seq'].append(state)


        elif state == 's3':
            if (state not in self.state_tracker['state_seq']) and 's2' in self.state_tracker['state_seq']:
                self.state_tracker['state_seq'].append(state)

    def _show_feedback(self, frame, c_frame, dict_maps, lower_hips_disp):
        # Muestra comentarios en el marco basado en la postura detectada.
        if lower_hips_disp:
            draw_text(
                frame,
                'BAJA LAS CADERAS',
                pos=(30, 80),
                text_color=(0, 0, 0),
                font_scale=0.6,
                font_thickness=2,
                text_color_bg=(255, 255, 0)
            )

        for idx in np.where(c_frame)[0]:
            draw_text(
                frame,
                dict_maps[idx][0],
                pos=(30, dict_maps[idx][1]),
                text_color=(255, 255, 230),
                font_scale=0.6,
                font_thickness=2,
                text_color_bg=dict_maps[idx][2]
            )

        return frame

    def process(self, frame: np.array, pose):
        """
        Procesa un fotograma de video para detectar y analizar los puntos de referencia de la postura,
        proporciona retroalimentación visual y lleva un seguimiento del conteo de sentadillas.
        :param frame: El frame de video a ser procesado
        :param pose: El modelo de ML - Mediapipe pose
        :return:

        Description: Esta función procesa un fotograma de video para detectar puntos de referencia de la postura
        utilizando el modelo de estimación de postura proporcionado. Calcula varios ángulos y coordenadas para los
        puntos de referencia del cuerpo, proporciona retroalimentación visual en el fotograma, lleva un seguimiento
        de las sentadillas correctas e incorrectas y actualiza el rastreador de estado en consecuencia. También
        verifica la alineación de la cámara y la inactividad para reiniciar los contadores de sentadillas si es
        necesario.
        """


        play_sound = None

        frame_height, frame_width, _ = frame.shape

        # Procesa la imagen.
        keypoints = pose.process(frame)

        if keypoints.pose_landmarks:
            ps_lm = keypoints.pose_landmarks
            # Obtiene las coordenadas de la nariz y varios puntos de las articulaciones.
            nose_coord = get_landmark_features(ps_lm.landmark, self.dict_features, 'nose', frame_width, frame_height)
            left_shldr_coord, left_elbow_coord, left_wrist_coord, left_hip_coord, left_knee_coord, left_ankle_coord, left_foot_coord = \
                get_landmark_features(ps_lm.landmark, self.dict_features, 'left', frame_width, frame_height)
            right_shldr_coord, right_elbow_coord, right_wrist_coord, right_hip_coord, right_knee_coord, right_ankle_coord, right_foot_coord = \
                get_landmark_features(ps_lm.landmark, self.dict_features, 'right', frame_width, frame_height)

            # Calcula el ángulo de desviación entre los hombros y la nariz.
            offset_angle = find_angle(left_shldr_coord, right_shldr_coord, nose_coord)

            # Verifica la alineación de la cámara.
            if offset_angle > self.thresholds['OFFSET_THRESH']:

                display_inactivity = False

                end_time = time.perf_counter()
                self.state_tracker['INACTIVE_TIME_FRONT'] += end_time - self.state_tracker['start_inactive_time_front']
                self.state_tracker['start_inactive_time_front'] = end_time

                # Reinicia el conteo de sentadillas si la cámara está desalineada por demasiado tiempo.
                if self.state_tracker['INACTIVE_TIME_FRONT'] >= self.thresholds['INACTIVE_THRESH']:
                    self.state_tracker['SQUAT_COUNT'] = 0
                    self.state_tracker['IMPROPER_SQUAT'] = 0
                    display_inactivity = True

                cv2.circle(frame, nose_coord, 7, self.COLORS['white'], -1)
                cv2.circle(frame, left_shldr_coord, 7, self.COLORS['yellow'], -1)
                cv2.circle(frame, right_shldr_coord, 7, self.COLORS['magenta'], -1)

                if self.flip_frame:
                    frame = cv2.flip(frame, 1)

                if display_inactivity:
                    play_sound = 'reset_counters'
                    self.state_tracker['INACTIVE_TIME_FRONT'] = 0.0
                    self.state_tracker['start_inactive_time_front'] = time.perf_counter()

                draw_text(
                    frame,
                    'REPS. CORRECTAS: ' + str(self.state_tracker['SQUAT_COUNT']),
                    pos=(int(frame_width * 0.70), 30),
                    text_color=(255, 255, 230),
                    font_scale=0.6,
                    text_color_bg=(18, 185, 0),
                    font_thickness=2,
                    overlay_image=True,
                    overlay_type='correct'
                )

                draw_text(
                    frame,
                    'REPS. INCORRECTAS: ' + str(self.state_tracker['IMPROPER_SQUAT']),
                    pos=(int(frame_width * 0.70), 100),
                    text_color=(255, 255, 230),
                    font_scale=0.6,
                    text_color_bg=(221, 0, 0),
                    font_thickness=2,
                    overlay_image=True,
                    overlay_type='incorrect'
                )

                draw_text(
                    frame,
                    'CAMARA NO ALINEADA',
                    pos=(30, frame_height - 60),
                    text_color=(255, 255, 230),
                    font_scale=0.6,
                    font_thickness=2,
                    text_color_bg=(255, 153, 0),
                )

                draw_text(
                    frame,
                    'ANGULO DESVIADO POR: ' + str(offset_angle),
                    pos=(30, frame_height - 30),
                    text_color=(255, 255, 230),
                    font_scale=0.6,
                    font_thickness=2,
                    text_color_bg=(255, 153, 0),
                )

                # Reinicia los tiempos de inactividad para la vista lateral.
                self.state_tracker['start_inactive_time'] = time.perf_counter()
                self.state_tracker['INACTIVE_TIME'] = 0.0
                self.state_tracker['prev_state'] = None
                self.state_tracker['curr_state'] = None

            # La cámara está alineada correctamente.
            else:

                self.state_tracker['INACTIVE_TIME_FRONT'] = 0.0
                self.state_tracker['start_inactive_time_front'] = time.perf_counter()

                dist_l_sh_hip = abs(left_foot_coord[1] - left_shldr_coord[1])
                dist_r_sh_hip = abs(right_foot_coord[1] - right_shldr_coord)[1]

                shldr_coord = None
                elbow_coord = None
                wrist_coord = None
                hip_coord = None
                knee_coord = None
                ankle_coord = None
                foot_coord = None

                if dist_l_sh_hip > dist_r_sh_hip:
                    shldr_coord = left_shldr_coord
                    elbow_coord = left_elbow_coord
                    wrist_coord = left_wrist_coord
                    hip_coord = left_hip_coord
                    knee_coord = left_knee_coord
                    ankle_coord = left_ankle_coord
                    foot_coord = left_foot_coord

                    multiplier = -1


                else:
                    shldr_coord = right_shldr_coord
                    elbow_coord = right_elbow_coord
                    wrist_coord = right_wrist_coord
                    hip_coord = right_hip_coord
                    knee_coord = right_knee_coord
                    ankle_coord = right_ankle_coord
                    foot_coord = right_foot_coord

                    multiplier = 1

                # ------------------- Calculo angulo vertical  --------------

                hip_vertical_angle = find_angle(shldr_coord, np.array([hip_coord[0], 0]), hip_coord)
                cv2.ellipse(frame, hip_coord, (30, 30),
                            angle=0, startAngle=-90, endAngle=-90 + multiplier * hip_vertical_angle,
                            color=self.COLORS['white'], thickness=3, lineType=self.linetype)

                draw_dotted_line(frame, hip_coord, start=hip_coord[1] - 80, end=hip_coord[1] + 20,
                                 line_color=self.COLORS['blue'])

                knee_vertical_angle = find_angle(hip_coord, np.array([knee_coord[0], 0]), knee_coord)
                cv2.ellipse(frame, knee_coord, (20, 20),
                            angle=0, startAngle=-90, endAngle=-90 - multiplier * knee_vertical_angle,
                            color=self.COLORS['white'], thickness=3, lineType=self.linetype)

                draw_dotted_line(frame, knee_coord, start=knee_coord[1] - 50, end=knee_coord[1] + 20,
                                 line_color=self.COLORS['blue'])

                ankle_vertical_angle = find_angle(knee_coord, np.array([ankle_coord[0], 0]), ankle_coord)
                cv2.ellipse(frame, ankle_coord, (30, 30),
                            angle=0, startAngle=-90, endAngle=-90 + multiplier * ankle_vertical_angle,
                            color=self.COLORS['white'], thickness=3, lineType=self.linetype)

                draw_dotted_line(frame, ankle_coord, start=ankle_coord[1] - 50, end=ankle_coord[1] + 20,
                                 line_color=self.COLORS['blue'])

                # ------------------------------------------------------------

                # Join landmarks.
                cv2.line(frame, shldr_coord, elbow_coord, self.COLORS['light_blue'], 5, lineType=self.linetype)
                cv2.line(frame, wrist_coord, elbow_coord, self.COLORS['light_blue'], 5, lineType=self.linetype)
                cv2.line(frame, shldr_coord, hip_coord, self.COLORS['light_blue'], 5, lineType=self.linetype)
                cv2.line(frame, knee_coord, hip_coord, self.COLORS['light_blue'], 5, lineType=self.linetype)
                cv2.line(frame, ankle_coord, knee_coord, self.COLORS['light_blue'], 5, lineType=self.linetype)
                cv2.line(frame, ankle_coord, foot_coord, self.COLORS['light_blue'], 5, lineType=self.linetype)

                # Plot landmark points
                cv2.circle(frame, shldr_coord, 7, self.COLORS['yellow'], -1, lineType=self.linetype)
                cv2.circle(frame, elbow_coord, 7, self.COLORS['yellow'], -1, lineType=self.linetype)
                cv2.circle(frame, wrist_coord, 7, self.COLORS['yellow'], -1, lineType=self.linetype)
                cv2.circle(frame, hip_coord, 7, self.COLORS['yellow'], -1, lineType=self.linetype)
                cv2.circle(frame, knee_coord, 7, self.COLORS['yellow'], -1, lineType=self.linetype)
                cv2.circle(frame, ankle_coord, 7, self.COLORS['yellow'], -1, lineType=self.linetype)
                cv2.circle(frame, foot_coord, 7, self.COLORS['yellow'], -1, lineType=self.linetype)

                current_state = self._get_state(int(knee_vertical_angle))
                self.state_tracker['curr_state'] = current_state
                self._update_state_sequence(current_state)

                # -------------------------------------- COMPUTE COUNTERS --------------------------------------

                if current_state == 's1':

                    if len(self.state_tracker['state_seq']) == 3 and not self.state_tracker['INCORRECT_POSTURE']:
                        self.state_tracker['SQUAT_COUNT'] += 1
                        play_sound = str(self.state_tracker['SQUAT_COUNT'])

                    elif 's2' in self.state_tracker['state_seq'] and len(self.state_tracker['state_seq']) == 1:
                        self.state_tracker['IMPROPER_SQUAT'] += 1
                        play_sound = 'incorrect'

                    elif self.state_tracker['INCORRECT_POSTURE']:
                        self.state_tracker['IMPROPER_SQUAT'] += 1
                        play_sound = 'incorrect'

                    self.state_tracker['state_seq'] = []
                    self.state_tracker['INCORRECT_POSTURE'] = False

                    self.state_tracker['TOTAL_SQUAT_COUNT'] = self.state_tracker['SQUAT_COUNT'] + self.state_tracker['IMPROPER_SQUAT']



                # ----------------------------------------------------------------------------------------------------

                # -------------------------------------- PERFORM FEEDBACK ACTIONS --------------------------------------

                else:
                    if hip_vertical_angle > self.thresholds['HIP_THRESH'][1]:
                        self.state_tracker['DISPLAY_TEXT'][0] = True


                    elif hip_vertical_angle < self.thresholds['HIP_THRESH'][0] and \
                            self.state_tracker['state_seq'].count('s2') == 1:
                        self.state_tracker['DISPLAY_TEXT'][1] = True

                    if self.thresholds['KNEE_THRESH'][0] < knee_vertical_angle < self.thresholds['KNEE_THRESH'][1] and \
                            self.state_tracker['state_seq'].count('s2') == 1:
                        self.state_tracker['LOWER_HIPS'] = True


                    elif knee_vertical_angle > self.thresholds['KNEE_THRESH'][2]:
                        self.state_tracker['DISPLAY_TEXT'][3] = True
                        self.state_tracker['INCORRECT_POSTURE'] = True

                    if (ankle_vertical_angle > self.thresholds['ANKLE_THRESH']):
                        self.state_tracker['DISPLAY_TEXT'][2] = True
                        self.state_tracker['INCORRECT_POSTURE'] = True

                # ----------------------------------------------------------------------------------------------------

                # ----------------------------------- COMPUTE INACTIVITY ---------------------------------------------

                display_inactivity = False

                if self.state_tracker['curr_state'] == self.state_tracker['prev_state']:

                    end_time = time.perf_counter()
                    self.state_tracker['INACTIVE_TIME'] += end_time - self.state_tracker['start_inactive_time']
                    self.state_tracker['start_inactive_time'] = end_time

                    if self.state_tracker['INACTIVE_TIME'] >= self.thresholds['INACTIVE_THRESH']:
                        self.state_tracker['SQUAT_COUNT'] = 0
                        self.state_tracker['IMPROPER_SQUAT'] = 0
                        display_inactivity = True


                else:

                    self.state_tracker['start_inactive_time'] = time.perf_counter()
                    self.state_tracker['INACTIVE_TIME'] = 0.0

                # -------------------------------------------------------------------------------------------------------

                hip_text_coord_x = hip_coord[0] + 10
                knee_text_coord_x = knee_coord[0] + 15
                ankle_text_coord_x = ankle_coord[0] + 10

                if self.flip_frame:
                    frame = cv2.flip(frame, 1)
                    hip_text_coord_x = frame_width - hip_coord[0] + 10
                    knee_text_coord_x = frame_width - knee_coord[0] + 15
                    ankle_text_coord_x = frame_width - ankle_coord[0] + 10

                if 's3' in self.state_tracker['state_seq'] or current_state == 's1':
                    self.state_tracker['LOWER_HIPS'] = False

                self.state_tracker['COUNT_FRAMES'][self.state_tracker['DISPLAY_TEXT']] += 1

                frame = self._show_feedback(frame, self.state_tracker['COUNT_FRAMES'], self.FEEDBACK_ID_MAP,
                                            self.state_tracker['LOWER_HIPS'])

                if display_inactivity:
                    play_sound = 'reset_counters'
                    self.state_tracker['start_inactive_time'] = time.perf_counter()
                    self.state_tracker['INACTIVE_TIME'] = 0.0

                cv2.putText(frame, str(int(hip_vertical_angle)), (hip_text_coord_x, hip_coord[1]), self.font, 0.8,
                            (187, 51, 255), 2, lineType=self.linetype)
                cv2.putText(frame, str(int(knee_vertical_angle)), (knee_text_coord_x, knee_coord[1] + 10), self.font,
                            0.8, (187, 51, 255), 2, lineType=self.linetype)
                cv2.putText(frame, str(int(ankle_vertical_angle)), (ankle_text_coord_x, ankle_coord[1]), self.font, 0.8,
                            (187, 51, 255), 2, lineType=self.linetype)

                draw_text(
                    frame,
                    'REP. CORRECTAS: ' + str(self.state_tracker['SQUAT_COUNT']),
                    pos=(int(frame_width * 0.70), 30),
                    text_color=(255, 255, 230),
                    font_scale=0.6,
                    text_color_bg=(18, 185, 0),
                    font_thickness=2,
                    overlay_image=True,
                    overlay_type='correct'
                )

                draw_text(
                    frame,
                    'REP. INCORRECTAS: ' + str(self.state_tracker['IMPROPER_SQUAT']),
                    pos=(int(frame_width * 0.70), 100),
                    text_color=(255, 255, 230),
                    font_scale=0.6,
                    text_color_bg=(221, 0, 0),
                    font_thickness=2,
                    overlay_image=True,
                    overlay_type='incorrect'
                )

                # CONTEO TOTAL DE REPS.
                draw_text(
                    frame,
                    'REPS. TOTALES: ' + str(self.state_tracker['TOTAL_SQUAT_COUNT']),
                    pos=(int(frame_width * 0.70), 170),
                    text_color=(255, 255, 230),
                    font_scale=0.6,
                    text_color_bg=(48, 107, 233),
                    font_thickness=2,
                    overlay_image=True,
                    overlay_type='counter'
                )



                self.state_tracker['DISPLAY_TEXT'][
                    self.state_tracker['COUNT_FRAMES'] > self.thresholds['CNT_FRAME_THRESH']] = False
                self.state_tracker['COUNT_FRAMES'][
                    self.state_tracker['COUNT_FRAMES'] > self.thresholds['CNT_FRAME_THRESH']] = 0
                self.state_tracker['prev_state'] = current_state




        else:

            if self.flip_frame:
                frame = cv2.flip(frame, 1)

            end_time = time.perf_counter()
            self.state_tracker['INACTIVE_TIME'] += end_time - self.state_tracker['start_inactive_time']

            display_inactivity = False

            if self.state_tracker['INACTIVE_TIME'] >= self.thresholds['INACTIVE_THRESH']:
                self.state_tracker['SQUAT_COUNT'] = 0
                self.state_tracker['IMPROPER_SQUAT'] = 0

                display_inactivity = True

            self.state_tracker['start_inactive_time'] = end_time

            draw_text(
                frame,
                'REPS. CORRECTAS: ' + str(self.state_tracker['SQUAT_COUNT']),
                pos=(int(frame_width * 0.70), 30),
                text_color=(255, 255, 230),
                font_scale=0.6,
                text_color_bg=(18, 185, 0),
                font_thickness=2,
                overlay_image=True,
                overlay_type='correct'
            )

            draw_text(
                frame,
                'REPS. INCORRECTAS: ' + str(self.state_tracker['IMPROPER_SQUAT']),
                pos=(int(frame_width * 0.70), 100),
                text_color=(255, 255, 230),
                font_scale=0.6,
                text_color_bg=(221, 0, 0),
                font_thickness=2,
                overlay_image=True,
                overlay_type='incorrect'
            )

            if display_inactivity:
                play_sound = 'reset_counters'
                self.state_tracker['start_inactive_time'] = time.perf_counter()
                self.state_tracker['INACTIVE_TIME'] = 0.0

            # Reinicia todas las demás variables de estado

            self.state_tracker['prev_state'] = None
            self.state_tracker['curr_state'] = None
            self.state_tracker['INACTIVE_TIME_FRONT'] = 0.0
            self.state_tracker['INCORRECT_POSTURE'] = False
            self.state_tracker['DISPLAY_TEXT'] = np.full((5,), False)
            self.state_tracker['COUNT_FRAMES'] = np.zeros((5,), dtype=np.int64)
            self.state_tracker['start_inactive_time_front'] = time.perf_counter()

        return frame, play_sound
