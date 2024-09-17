import socket
import time
import datetime

import cv2
import numpy as np
import threading
import os
import signal
import boto3
import requests
import pickle

from flask import Flask, Response
from ultralytics import YOLO
from pathlib import Path
from dotenv import load_dotenv

from config import (SERVER_HOST, SERVER_SOCKET_PORT, SERVER_FLASK_PORT, SERVER_SOCKET_ADDRESS, SERVER_MAX_QUEUE_SIZE,
                    IMAGE_ENCODE_DECODE_FORMAT, VIDEO_IMAGE_ENCODE_DECODE_FORMAT, AWS_S3_SERVICE_NAME,
                    VIDEO_RECORDING_FRAME_RATE, SOCKET_TRANSMISSION_SIZE, YOLOv8_MODEL, VIDEO_RECORDING_CODEC_FORMAT,
                    YOLOv8_MINIMUM_CONFIDENCE_SCORE, AWS_S3_STORE_OBJECT_PARAMETER_VIDEO_CONTENT_TYPE,
                    IMAGE_FILE_FORMAT, AWS_S3_STORE_OBJECT_PARAMETER_CONTENT_DISPOSITION, VIDEO_RECORDING_FILE_FORMAT,
                    AWS_S3_STORE_OBJECT_PARAMETER_IMAGE_CONTENT_TYPE)

app = Flask(__name__)

# Load the Environment variables
load_dotenv()

# Create an S3 client
s3 = boto3.client(
        service_name=AWS_S3_SERVICE_NAME,
        region_name=os.getenv('AWS_S3_BUCKET_REGION'),
        aws_access_key_id=os.getenv('AWS_S3_BUCKET_ACCESS_KEY'),
        aws_secret_access_key=os.getenv('AWS_S3_BUCKET_SECRET_KEY')
)

# Define bucket and file names
bucket_name = os.getenv('AWS_S3_BUCKET_NAME')

# Server running status
server_running = True

# Shared variable to store the most recent frame
current_frame = None
frame_lock = threading.Lock()

# Save the Image where the people are found
final_image = None

# Create a new Case ID
case_id = None

# Initialise the YOLO model
model = YOLO(YOLOv8_MODEL)

# Store the total number of people / Re-initialise at every connection
total_number_of_people_found = []

# Mock a registered drone
registered_drone_id = '66c933d3464fcfba82454de3'

# Location Coordinates # Longitude, Latitude
location_coordinates = None

# Server Initialization
def init_socket_server():
    global case_id

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

    try:
        # Get the Case ID
        create_case_payload = {
            'title': 'This is a New Missing Persons Case',
            'description': 'More details coming soon',
            'isLive': True,
            'liveVideoURL': os.getenv('WEB_LIVE_VIDEO_URL')
        }


        create_case_response = requests.post(f'{os.getenv("WEB_PORTAL_URL")}/case/create-new-case',
                                             json=create_case_payload)

        case_response = create_case_response.json()

        print("New Case created ✅ _id", case_response['case']['_id'])

        case_id = case_response['case']['_id']

    except Exception as error:
        print("Something went wrong : ", error)

    # Initiate the Video receiving process
    receive_video(client_connection, socket_server)

    return client_connection, socket_server


def receive_video(client_conn, server_conn):
    global current_frame
    global case_id
    global total_number_of_people_found
    global final_image
    global location_coordinates

    total_number_of_people_found = []

    width = 1280
    height = 720

    fourcc = cv2.VideoWriter_fourcc(*VIDEO_RECORDING_CODEC_FORMAT)

    # Create the folder if it doesn't exist
    Path("recordings").mkdir(parents=True, exist_ok=True)
    Path("images").mkdir(parents=True, exist_ok=True)

    # Begin the video writer
    writer = cv2.VideoWriter(f"recordings/{case_id}-video.{VIDEO_RECORDING_FILE_FORMAT}", fourcc,
                             VIDEO_RECORDING_FRAME_RATE, (int(width), int(height)))

    # Image file path
    image_file_path = f"images/{case_id}-image.{IMAGE_FILE_FORMAT}"

    while True:
        # Receive data from the client
        length = client_conn.recv(SOCKET_TRANSMISSION_SIZE)
        print("The location ", length)
        try :
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

        except UnicodeDecodeError:
            location_array = pickle.loads(length)
            print("Location Received ✅", location_array, type(location_array), type(location_array[0]))
            location_coordinates = location_array
            break

        if frame is not None:
            # Each frame processed here
            processed_frame = frame_track(frame)

            # Record the video / Write the frame
            writer.write(processed_frame)

            # Update current_frame with the new frame
            with frame_lock:
                current_frame = processed_frame

            if len(total_number_of_people_found) == 0:
                message = "NEXT_FRAME"
                client_conn.sendall(message.encode(IMAGE_ENCODE_DECODE_FORMAT))
            else:
                message = "LOCATION_COORDINATES"
                client_conn.sendall(message.encode(IMAGE_ENCODE_DECODE_FORMAT))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    writer.release()
    print("Video recording is completed ✅")

    if final_image is None:
        final_image = frame


    cv2.imwrite(image_file_path, final_image)
    print("The Image file is saved ✅")

    client_conn.close()
    server_conn.close()
    cv2.destroyAllWindows()

    # Pause for the video to finish encoding
    time.sleep(5)

    print("Uploading file to S3 🚀 ...")

    # Upload the video to AWS
    upload_video_to_cloud()

    try:
        if  location_coordinates and len(total_number_of_people_found) > 0:
            print("The necessary info is present")

            # Initiate the AI call
            initiate_ai_chat_payload = {
                'location': {
                    "type": "Point",
                    "coordinates": location_coordinates
                },
                "description": f"Success ✅ {len(total_number_of_people_found)} people located",
                "isLive": False,
                "imageURL": f'{os.getenv("AWS_CLOUDFRONT_URL")}/{case_id}-image.{IMAGE_FILE_FORMAT}',
                "numberOfPeopleFound": len(total_number_of_people_found),
                "droneId": registered_drone_id,
                "videoURL": f'{os.getenv("AWS_CLOUDFRONT_URL")}/{case_id}-video.{VIDEO_RECORDING_FILE_FORMAT}'
            }

            print(f"All the information is available ✅ Initiated the API Call to the AI 🤖 - Case ID - {case_id}")

            requests.put(f'{os.getenv("WEB_PORTAL_URL")}/case/initiate-ai-checklist/{case_id}',
                          json=initiate_ai_chat_payload)

        elif len(total_number_of_people_found) == 0:
            print("The necessary info is not present")

            # Initiate the AI call
            initiate_case_update_payload = {
                "isLive": False,
                "droneId": registered_drone_id,
                "numberOfPeopleFound": len(total_number_of_people_found),
                "description": f"Failure ❌ No people located",
                "videoURL": f'{os.getenv("AWS_CLOUDFRONT_URL")}/{case_id}-video.{VIDEO_RECORDING_FILE_FORMAT}',
                "status": "CLOSED"
            }

            requests.put(f'{os.getenv("WEB_PORTAL_URL")}/case/update-case/{case_id}',
                         json=initiate_case_update_payload)

    except Exception as aiError:
        print("The AI could not generate a response, Please try again", aiError)

    # Exit the server to restart
    stop_server()


def upload_video_to_cloud():
    global case_id

    # Create the file name to be stored
    local_video_file = f'recordings/{case_id}-video.{VIDEO_RECORDING_FILE_FORMAT}'
    remote_video_file = f'{case_id}-video.{VIDEO_RECORDING_FILE_FORMAT}'

    # Create the file name to be stored
    local_image_file = f'images/{case_id}-image.{IMAGE_FILE_FORMAT}'
    remote_image_file = f'{case_id}-image.{IMAGE_FILE_FORMAT}'

    extra_video_args = {
        'ContentType': AWS_S3_STORE_OBJECT_PARAMETER_VIDEO_CONTENT_TYPE,
        'Metadata': {
            'Content-Disposition': AWS_S3_STORE_OBJECT_PARAMETER_CONTENT_DISPOSITION
        },
    }

    # Upload the video file
    s3.upload_file(local_video_file, bucket_name, remote_video_file, ExtraArgs=extra_video_args)

    # Acknowledge
    print(f"Video File uploaded successfully to Amazon S3 bucket {bucket_name} ✅")

    extra_image_args = {
        'ContentType': AWS_S3_STORE_OBJECT_PARAMETER_IMAGE_CONTENT_TYPE,
        'Metadata': {
            'Content-Disposition': AWS_S3_STORE_OBJECT_PARAMETER_CONTENT_DISPOSITION
        }
    }

    # Upload the Image file
    s3.upload_file(local_image_file, bucket_name, remote_image_file, ExtraArgs=extra_image_args)

    # Acknowledge
    print(f"Image File uploaded successfully to Amazon S3 bucket {bucket_name} ✅")


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
    global final_image
    global total_number_of_people_found

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
                    final_image = frame

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


@app.route('/video_feed', methods=["GET"])
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/health-check', methods=['GET'])
def health_check():
    return {
        "statusCode": 200,
        "message": "The server is running ✅",
        "data": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

@app.route('/server-shutdown/<admin_key>', methods=['POST'])
def shutdown(admin_key):
    global server_running
    if admin_key:
        server_running = False
        print("Server shutting down...")
        os.kill(os.getpid(), signal.SIGTERM)

def stop_server():
    os.kill(os.getpid(), signal.SIGINT)


if __name__ == "__main__":
    try:
        while server_running:
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
