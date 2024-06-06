import os
import time

import gradio as gr
import cv2
import subprocess

from utils import get_mediapipe_pose
from process_frame import ProcessFrame
from thresholds import get_thresholds_beginner, get_thresholds_pro

sample_video = os.path.join(os.path.dirname(__file__),
                            "samples/sample-squats.mp4")
sample_video_woman = os.path.join(os.path.dirname(__file__),
                                  "samples/sample_squats_2.mp4")
banner = os.path.join(os.path.dirname(__file__), "iron-assist.png")

footer = """
    ### <p style="text-align: center;">UMG - Proyecto Final</p>
    """

# Initialize face mesh solution
POSE = get_mediapipe_pose()

title = "IronAssist - Entrenador Personal"


def process_video(video_path, mode="Principiante", progress=gr.Progress()):

    progress(0, desc="Inicializando modelo...")
    time.sleep(3)

    output_video_file = f"output_recorded.mp4"

    if mode == 'Principiante':
        thresholds = get_thresholds_beginner()

    elif mode == 'Pro':
        thresholds = get_thresholds_pro()

    progress(0.10, desc="Obteniendo parametros de dificultad....")
    time.sleep(2)

    upload_process_frame = ProcessFrame(thresholds=thresholds)

    vf = cv2.VideoCapture(video_path)

    fps = int(vf.get(cv2.CAP_PROP_FPS))
    width = int(vf.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vf.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (width, height)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_output = cv2.VideoWriter(output_video_file, fourcc, fps, frame_size)

    progress(0.90, desc="Comenzando procesamiento de datos...")
    time.sleep(2)

    progress(1, desc="Todo listo!...")
    time.sleep(1)
    count = 0
    while vf.isOpened():
        ret, frame = vf.read()
        if not ret:
            break

        # convert frame from BGR to RGB before processing it.
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out_frame, _ = upload_process_frame.process(frame, POSE)

        video_output.write(cv2.cvtColor(out_frame, cv2.COLOR_RGB2BGR))

        if not count % 12:
            yield out_frame, None

        count += 1

    vf.release()
    video_output.release()

    # convertedVideo = f"output_h264.mp4"
    # subprocess.call(args=f"ffmpeg -y -i {output_video_file} -c:v libx264 {convertedVideo}".split(" "))

    yield None, output_video_file


input_video = gr.Video(label="Selecciona un archivo, o grabate",
                       format="mp4",
                       show_download_button=True,
                       mirror_webcam=False)

output_frames_up = gr.Image(label="Frames Procesados")
output_video_file_up = gr.Video(label="Resultado: Análisis", )

demo = gr.Interface(fn=process_video,
                    inputs=[
                        input_video,
                        gr.Radio(choices=["Principiante", "Pro"],
                                 label="Dificultad",
                                 info="Selecciona un modo")
                    ],
                    outputs=[output_frames_up, output_video_file_up],
                    submit_btn="Comenzar Analisis",
                    stop_btn="Parar",
                    clear_btn="Limpiar",
                    description=f"""
                        <div style="text-align: center;">
                            <h3>Potenciado con Inteligencia Artificial</h3>
                            <p>Sube un video o grabate tu mismo para analizar tu entrenamiento y recibir retroalimentación en tiempo real.</p>
                            <div style="display: flex; justify-content: center; align-items: center;">
                                <img src="file/iron-assist.png" alt="IronAssist Interface" style="max-width: 40%; height: auto;">
                            </div>
                        </div>
                        """,
                    article=footer,
                    theme=gr.themes.Soft(),
                    allow_flagging="never",
                    examples=[[sample_video], [sample_video_woman]],
                    title="IronAssist - Entrenador Personal")

demo.queue().launch(allowed_paths=["."], auth=["jorge", "tester"])
