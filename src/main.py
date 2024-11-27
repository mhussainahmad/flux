import atexit
from io import BytesIO
from multiprocessing.connection import Listener
from os import chmod, remove
from os.path import abspath, exists
from pathlib import Path

import torch

from PIL.JpegImagePlugin import JpegImageFile
from pipelines.models import TextToImageRequest

from pipeline import load_pipeline, infer

SOCKET = abspath(Path(__file__).parent.parent / "inferences.sock")


def at_exit():
    torch.cuda.empty_cache()


def main():
    atexit.register(at_exit)

    print(f"Loading pipeline")
    pipeline = load_pipeline()

    print(f"Pipeline loaded, creating socket at '{SOCKET}'")

    if exists(SOCKET):
        remove(SOCKET)

    with Listener(SOCKET) as listener:
        chmod(SOCKET, 0o777)

        print(f"Awaiting connections")
        with listener.accept() as connection:
            print(f"Connected")

            while True:
                try:
                    request = TextToImageRequest.model_validate_json(connection.recv_bytes().decode("utf-8"))
                except EOFError:
                    print(f"Inference socket exiting")

                    return

                image = infer(request, pipeline)

                data = BytesIO()
                image.save(data, format=JpegImageFile.format)

                packet = data.getvalue()

                connection.send_bytes(packet)


if __name__ == '__main__':
    main()
