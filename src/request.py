from pydantic import BaseModel


class TextToImageRequest(BaseModel):
    """One generation request, sent to the server as JSON."""

    prompt: str
    seed: int | None = None
    width: int | None = None
    height: int | None = None
