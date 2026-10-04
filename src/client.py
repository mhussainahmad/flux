"""Send one prompt to a running server and save the result.

    uv run python src/client.py "a red tractor in a wheat field" out.jpg
"""

import sys
from multiprocessing.connection import Client
from os.path import abspath
from pathlib import Path

from request import TextToImageRequest

SOCKET = abspath(Path(__file__).parent.parent / "inferences.sock")


def main() -> None:
    prompt = sys.argv[1] if len(sys.argv) > 1 else "A beautiful sunset over the mountains"
    out = sys.argv[2] if len(sys.argv) > 2 else "out.jpg"

    with Client(SOCKET) as connection:
        request = TextToImageRequest(prompt=prompt, seed=0)
        connection.send_bytes(request.model_dump_json().encode("utf-8"))
        with open(out, "wb") as f:
            f.write(connection.recv_bytes())
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
