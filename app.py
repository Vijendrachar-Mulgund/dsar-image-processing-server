import socket
import time

import cv2
import numpy as np
import threading
import uuid
import os
import signal
import boto3

from flask import Flask, Response
from ultralytics import YOLO
from pathlib import Path
from dotenv import load_dotenv

from config import (SERVER_HOST, SERVER_SOCKET_PORT, SERVER_FLASK_PORT, SERVER_SOCKET_ADDRESS, SERVER_MAX_QUEUE_SIZE,
                    IMAGE_ENCODE_DECODE_FORMAT, VIDEO_IMAGE_ENCODE_DECODE_FORMAT,
                    VIDEO_RECORDING_FRAME_RATE, SOCKET_TRANSMISSION_SIZE, YOLOv8_MODEL, VIDEO_RECORDING_CODEC_FORMAT,
                    YOLOv8_MINIMUM_CONFIDENCE_SCORE, AWS_S3_STORE_OBJECT_PARAMETER_CONTENT_TYPE,
                    AWS_S3_STORE_OBJECT_PARAMETER_CONTENT_DISPOSITION)

app = Flask(__name__)

# Load the Environment variables
load_dotenv()

# Create an S3 client
s3 = boto3.client('s3')

# Define bucket and file names

bucket_name = os.getenv('AWS_S3_BUCKET_NAME')
remote_file = None

# Shared variable to store the most recent frame
current_frame = None
frame_lock = threading.Lock()

# Create a new Case ID
case_id = None

# Local file path
local_file = None

# Initialise the YOLO model
model = YOLO(YOLOv8_MODEL)

# Store the total number of people / Re-initialise at every connection
total_number_of_people_found = []


# Server Initialization
def init_socket_server():
    # Create a socket object
    socket_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Bind to the port
    socket_server.bind(SERVER_SOCKET_ADDRESS)
    # Queue up to 5 requests
    socket_server.listen(SERVER_MAX_QUEUE_SIZE)

    print("Server listening on {}:{}".format(SERVER_HOST, SERVER_SOCKET_PORT))

    # Establish a connection
    client_connection, client_address = socket_server.accept()

    print("Got a connection from {}".format(client_address))

    # Initiate the Video receiving process
    receive_video(client_connection, socket_server)

    return client_connection, socket_server


def receive_video(client_conn, server_conn):
    global current_frame
    global case_id
    global total_number_of_people_found
    global local_file
    global remote_file

    case_id = str(uuid.uuid4())
    total_number_of_people_found = []

    # Receive resolution from server
    resolution = client_conn.recv(SOCKET_TRANSMISSION_SIZE).decode()
    width, height = map(int, resolution.split(','))

    fourcc = cv2.VideoWriter_fourcc(*VIDEO_RECORDING_CODEC_FORMAT)
    Path("recordings").mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(f"recordings/{case_id}.mp4", fourcc, VIDEO_RECORDING_FRAME_RATE, (int(width), int(height)))

    while True:
        # Receive data from the client
        length = client_conn.recv(SOCKET_TRANSMISSION_SIZE)
        if not length:
            break
        length = int(length.decode(IMAGE_ENCODE_DECODE_FORMAT))

        data = b''
        while len(data) < length:
            packet = client_conn.recv(length - len(data))
            if not packet:
                break
            data += packet

        frame_data = np.frombuffer(data, dtype=np.uint8)
        frame = cv2.imdecode(frame_data, cv2.IMREAD_COLOR)

        if frame is not None:
            # Each frame processed here
            processed_frame = frame_track(frame)

            # Record the video / Write the frame
            writer.write(processed_frame)

            # Update current_frame with the new frame
            with frame_lock:
                current_frame = processed_frame

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    writer.release()
    print("Video recording is completed ✅")

    client_conn.close()
    server_conn.close()
    cv2.destroyAllWindows()

    time.sleep(5)

    print("Uploading file to S3 🚀 ...")

    # Create the file name to be stored
    local_file = f'recordings/{case_id}.mp4'
    remote_file = f'{case_id}.mp4'

    extra_args = {
        'ContentType': AWS_S3_STORE_OBJECT_PARAMETER_CONTENT_TYPE,
        'Metadata': {
            'Content-Disposition': AWS_S3_STORE_OBJECT_PARAMETER_CONTENT_DISPOSITION
        },
    }

    # Upload the file
    s3.upload_file(local_file, bucket_name, remote_file, ExtraArgs=extra_args)

    # Construct the URL
    print(f"Video File uploaded successfully to Amazon S3 bucket {bucket_name} ✅")

    # Exit the server to restart
    stop_server()


def generate_frames():
    global current_frame

    # Get each frame for the stream
    while True:
        with frame_lock:
            if current_frame is not None:
                frame = current_frame
            else:
                continue

        if frame is not None:
            # Encode for stream
            ret, buffer = cv2.imencode(VIDEO_IMAGE_ENCODE_DECODE_FORMAT, frame)
            frame = buffer.tobytes()

            # Return each frame for stream
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def frame_track(frame):
    global model
    # Apply the YOLOv8 model for Object detection
    result = model.track(frame)
    updated_frame = result[0].plot()

    # Access confidence scores
    for res in result:
        boxes = res.boxes  # Boxes object for bounding box outputs

        if len(boxes) > len(total_number_of_people_found):
            for box in boxes:
                confidence = box.conf.item()  # Confidence score
                obj_cls = box.cls.item()  # Class item

                if confidence > YOLOv8_MINIMUM_CONFIDENCE_SCORE:
                    total_number_of_people_found.append(confidence)

                print(f"Confidence: {confidence:.2f}")
                print(f"Class: {obj_cls:.2f}")

    print(f"Total number of people found: {total_number_of_people_found}")
    return updated_frame


@app.route('/')
def index():
    return """
        <html>
            <body>
                <h1>Live Video Stream</h1>
                <img width="1280" height="720" src="/video_feed">
            </body>
        </html>
        """


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


def stop_server():
    os.kill(os.getpid(), signal.SIGINT)


if __name__ == "__main__":
    try:
        while True:
            # Start Socket server
            receive_thread = threading.Thread(target=init_socket_server)
            receive_thread.daemon = True
            receive_thread.start()

            # Start Flask server
            app.run(host=SERVER_HOST, port=SERVER_FLASK_PORT)

    except KeyboardInterrupt:
        print("Server shut down 🛑")
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"An error occurred {e}")
