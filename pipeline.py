import kfp
import kfp.dsl as dsl

from kfp import compiler
from kfp.dsl import Artifact, Input, Output

from typing import Dict, List

@dsl.component(
    base_image='python:3.11',
    packages_to_install=['google-cloud-videointelligence']
)
def track_objects(video: str, objects: Output[Artifact]):
    """Detects objects in the video."""
    import json

    from google.cloud import videointelligence

    video_client = videointelligence.VideoIntelligenceServiceClient()
    features = [videointelligence.Feature.OBJECT_TRACKING]
    operation = video_client.annotate_video(
        request={
            "features": features,
            "input_uri": video
        }
    )
    print("\nProcessing video for object annotations.")

    result = operation.result(timeout=7200)
    print("\nFinished processing.\n")

    # The first result is retrieved because a single video was processed.
    object_annotations = result.annotation_results[0].object_annotations

    objects_data = []
    for object_annotation in object_annotations:
        objects_data.append({
            "entity": {
                "description": object_annotation.entity.description,
                "entity_id": object_annotation.entity.entity_id,
            },
            "confidence": object_annotation.confidence,
            "frames": [
                {
                    "normalized_bounding_box": {
                        "left": frame.normalized_bounding_box.left,
                        "top": frame.normalized_bounding_box.top,
                        "right": frame.normalized_bounding_box.right,
                        "bottom": frame.normalized_bounding_box.bottom,
                    },
                    "time_offset": frame.time_offset.seconds + frame.time_offset.microseconds / 1e6,
                }
                for frame in object_annotation.frames
            ],
            "segment":{
                "start_time_offset": object_annotation.segment.start_time_offset.seconds + object_annotation.segment.start_time_offset.microseconds / 1e6,
                "end_time_offset": object_annotation.segment.end_time_offset.seconds + object_annotation.segment.end_time_offset.microseconds / 1e6,
            }
        })

    with open(objects.path, 'w') as f:
        json.dump(objects_data, f)


@dsl.component(
    base_image='python:3.11',
    packages_to_install=['av']
)
def extract_metadata(video: str) -> Dict:
    import av

    container = av.open(video.replace("gs://", "/gcs/"))

    video_stream = next(s for s in container.streams if s.type == 'video')
    return {
        "width": video_stream.width,
        "height": video_stream.height,
        "duration": float(video_stream.duration * video_stream.time_base),
        "frame_rate": float(video_stream.average_rate)
    }

@dsl.component(
    base_image='python:3.11',
    packages_to_install=['opencv-python-headless']
)
def extract_images(video: str, metadata: Dict, objects: Input[Artifact], catalog: Output[Artifact]):
    import base64
    import cv2
    import json
    import os

    with open(objects.path, 'r') as f:
        object_annotations = json.loads(f.read())

    images = []
    for object_annotation in object_annotations:
        for frame in object_annotation["frames"]:
            # Time offset (in seconds)
            time_offset = frame["time_offset"]
            bounding_left = int(frame["normalized_bounding_box"]["left"]*metadata["width"])
            bounding_top = int(frame["normalized_bounding_box"]["top"]*metadata["height"])
            bounding_right = int(frame["normalized_bounding_box"]["right"]*metadata["width"])
            bounding_bottom = int(frame["normalized_bounding_box"]["bottom"]*metadata["height"])
            bounding_left = bounding_left if bounding_left >= 0 else 0
            bounding_top = bounding_top if bounding_top >= 0 else 0
            bounding_right = bounding_right if bounding_right <= metadata["width"] else metadata["width"]
            bounding_bottom = bounding_bottom if bounding_bottom <= metadata["height"] else metadata["height"]

            # Open video file
            cap = cv2.VideoCapture(video.replace("gs://", "/gcs/"))

            if not cap.isOpened():
                print("Could not open the video file")
            else:
                # Get frames per second (fps) from the video
                fps = cap.get(cv2.CAP_PROP_FPS)

                # Calculate frame number
                frame_number = int(time_offset * fps)

                # Set the frame position
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

                # Read frame
                ret, frame = cap.read()

                if ret:
                    # If frame was read successfully, crop and save as an image
                    cropped_frame = frame[bounding_top:bounding_bottom, bounding_left:bounding_right]
                    # Save the cropped frame as a base64 endoded string
                    frame = {
                        "time_offset": time_offset,
                        "top": bounding_top,
                        "left": bounding_left,
                        "bottom": bounding_bottom,
                        "right": bounding_right,
                    }
                    # base64 encode the time offset and bounding box
                    directory_string = video.split("/")[-1].replace(".mp4", "")
                    frame_string = base64.b64encode(json.dumps(frame).encode('utf-8')).decode('utf-8')
                    # create directory if it doesn't exist
                    os.makedirs(f"/gcs/YOUR BUCKET HERE/{directory_string}", exist_ok=True)
                    cv2.imwrite(f"/gcs/YOUR BUCKET HERE/{directory_string}/{frame_string}.png", cropped_frame)
                else:
                    print("Could not read frame at time offset", time_offset)

                # Release video
                cap.release()
                images.append(f"gs://YOUR BUCKET HERE/{directory_string}/{frame_string}.png")

    with open(catalog.path, 'w') as f:
        json.dump(images, f)

@dsl.component(
    base_image='python:3.11',
    packages_to_install=['google-auth', 'requests']
)
def generate_embeddings(video: str, metadata: Dict, objects: Input[Artifact], catalog: Input[Artifact], embeddings: Output[Artifact]):
    import base64
    import google.auth
    import google.auth.transport.requests
    import json
    import requests
    import subprocess
    import time

    PROJECT_ID = "YOUR PROJECT HERE"

    creds, project = google.auth.default()

    auth_req = google.auth.transport.requests.Request()
    creds.refresh(auth_req)

    with open(catalog.path, 'r') as f:
        images = json.loads(f.read())

    embeddings_data = []
    for image in images:
        with open(image.replace("gs://", "/gcs/"), 'rb') as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
            response = requests.post(
                f"https://us-central1-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/us-central1/publishers/google/models/multimodalembedding@001:predict",
                headers={
                    "Authorization": f"Bearer {creds.token}"
                },
                json={
                    "instances": [
                        {"image": {"bytesBase64Encoded": encoded_image}}
                    ]
                }
            )
            # sleep to avoid rate limiting
            time.sleep(0.5)
            if response.status_code != 200:
                print(response.status_code)
                print(response.json())
            else:
                embeddings_data.append({
                    "image": image,
                    "embedding": response.json()["predictions"][0]["imageEmbedding"]
                })

    with open(embeddings.path, 'w') as f:
        json.dump(embeddings_data, f)

@dsl.component(
    base_image='python:3.11',
    packages_to_install=['google-auth', 'requests']
)
def compile_results(
    video: str,
    metadata: Dict,
    objects: Input[Artifact],
    embeddings: Input[Artifact],
) -> str:
    import base64
    import json
    
    with open(objects.path, 'r') as f:
        object_annotations = json.loads(f.read())

    with open(embeddings.path, 'r') as f:
        embeddings = json.loads(f.read())

    embedding_results = []
    for embedding in embeddings:
        frame_details = {}
        for object_annotation in object_annotations:
            for frame in object_annotation["frames"]:
                bounding_left = int(frame["normalized_bounding_box"]["left"]*metadata["width"])
                bounding_top = int(frame["normalized_bounding_box"]["top"]*metadata["height"])
                bounding_right = int(frame["normalized_bounding_box"]["right"]*metadata["width"])
                bounding_bottom = int(frame["normalized_bounding_box"]["bottom"]*metadata["height"])
                bounding_left = bounding_left if bounding_left >= 0 else 0
                bounding_top = bounding_top if bounding_top >= 0 else 0
                bounding_right = bounding_right if bounding_right <= metadata["width"] else metadata["width"]
                bounding_bottom = bounding_bottom if bounding_bottom <= metadata["height"] else metadata["height"]
                frame_string = base64.b64encode(json.dumps({
                    "time_offset": frame["time_offset"],
                    "top": bounding_top,
                    "left": bounding_left,
                    "bottom": bounding_bottom,
                    "right": bounding_right,
                }).encode('utf-8')).decode('utf-8')
                if embedding["image"].split("/")[-1].replace(".png", "") == frame_string:
                    frame_details = {
                        "time_offset": frame["time_offset"],
                        "top": bounding_top,
                        "left": bounding_left,
                        "bottom": bounding_bottom,
                        "right": bounding_right,
                        "description": object_annotation["entity"]["description"],
                        "entity_id": object_annotation["entity"]["entity_id"],
                        "confidence": object_annotation["confidence"]
                    }
        assert frame_details != {}, "Could not find frame details for image"
        embedding_results.append({
            "embedding": embedding["embedding"],
            "image": embedding["image"],
            "frame_details": frame_details
        })

    with open(video.replace("gs://", "/gcs/").replace(".mp4", ".json"), 'w') as f:
        json.dump({
            "metadata": metadata,
            "embeddings": embedding_results
        }, f)
    
    return video.replace(".mp4", ".json")

@dsl.pipeline(
    name="video-entity-similarity"
)
def video_entity_similarity(videos: List[str]):
    with dsl.ParallelFor(
        name="videos",
        items=["gs://YOUR BUCKET HERE/JaneGoodall.mp4"]
    ) as video:
        track_objects_task = track_objects(video=video)
        extract_metadata_task = extract_metadata(video=video)
        extract_images_task = extract_images(
            video=video,
            metadata=extract_metadata_task.output,
            objects=track_objects_task.outputs["objects"]
        )
        generate_embeddings_task = generate_embeddings(
            video=video,
            metadata=extract_metadata_task.output,
            objects=track_objects_task.outputs["objects"],
            catalog=extract_images_task.outputs["catalog"]
        )
        compile_results_task = compile_results(
            video=video,
            metadata=extract_metadata_task.output,
            objects=track_objects_task.outputs["objects"],
            embeddings=generate_embeddings_task.outputs["embeddings"]
        )

compiler.Compiler().compile(video_entity_similarity, 'pipeline.json')