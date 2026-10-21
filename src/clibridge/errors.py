from __future__ import annotations


class BridgeError(Exception):
    """A request could not be served by the CLI backend.

    ``status`` is the HTTP status the server reports. CLI failures default to
    424 (Failed Dependency) so that SDKs don't retry them three times.
    """

    def __init__(self, message: str, status: int = 424):
        super().__init__(message)
        self.status = status
