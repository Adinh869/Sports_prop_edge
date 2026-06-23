"""Production-safe validation exceptions."""


class ValidationError(ValueError):
    """Input failed semantic validation (range, required fields, format)."""


class SchemaError(TypeError):
    """Input could not be parsed into the expected data contract."""


class SafetyError(RuntimeError):
    """Safety gate blocked processing due to unacceptable input batch conditions."""
