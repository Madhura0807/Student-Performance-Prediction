class CustomException(Exception):
    """Raised when the application encounters a predictable runtime error."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message
