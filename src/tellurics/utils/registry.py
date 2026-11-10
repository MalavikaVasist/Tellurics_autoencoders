"""Model registry for dynamic architecture selection."""

from typing import Any

from tellurics.utils.logging import get_logger

logger = get_logger(__name__)


class ModelRegistry:
    """Registry for model architectures.

    Allows registration and retrieval of model classes by name.
    """

    _registry: dict[str, type] = {}

    @classmethod
    def register(cls, name: str) -> Any:
        """Decorator to register a model class.

        Args:
            name: Unique identifier for the model.

        Returns:
            Decorator function.
        """

        def decorator(model_class: type) -> type:
            if name in cls._registry:
                logger.warning(f"Model '{name}' already registered. Overwriting.")
            cls._registry[name] = model_class
            return model_class

        return decorator

    @classmethod
    def get(cls, name: str) -> type:
        """Retrieve a registered model class.

        Args:
            name: Model identifier.

        Returns:
            The model class.

        Raises:
            KeyError: If model is not registered.
        """
        if name not in cls._registry:
            available = list(cls._registry.keys())
            msg = f"Model '{name}' not found. Available: {available}"
            raise KeyError(msg)
        return cls._registry[name]

    @classmethod
    def list_models(cls) -> list[str]:
        """List all registered model names."""
        return list(cls._registry.keys())
