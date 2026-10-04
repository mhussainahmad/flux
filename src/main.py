"""Serve the FLUX.1-schnell pipeline over a local Unix socket.

The server loads the pipeline once, then answers requests from a single
client: each request is a JSON-encoded ``TextToImageRequest`` and each reply
is the generated image as JPEG bytes. The loop ends when the client
disconnects.
"""

import atexit
from io import BytesIO
from multiprocessing.connection import Listener
from os import chmod, remove
from os.path import abspath, exists
from pathlib import Path

import torch

from pipeline import infer, load_pipeline
from request import TextToImageRequest

SOCKET = abspath(Path(__file__).parent.parent / "inferences.sock")


def main() -> None:
    atexit.register(torch.cuda.empty_cache)

    print("Loading pipeline")
    pipeline = load_pipeline()

    if exists(SOCKET):
        remove(SOCKET)

    print(f"Pipeline ready, listening on {SOCKET}")
    with Listener(SOCKET) as listener:
        chmod(SOCKET, 0o777)
        with listener.accept() as connection:
            print("Client connected")
            while True:
                try:
                    payload = connection.recv_bytes()
                except EOFError:
                    print("Client disconnected, shutting down")
                    return

                request = TextToImageRequest.model_validate_json(payload.decode("utf-8"))
                image = infer(request, pipeline)

                buffer = BytesIO()
                image.save(buffer, format="JPEG")
                connection.send_bytes(buffer.getvalue())


if __name__ == "__main__":
    main()
