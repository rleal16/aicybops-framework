import logging
from typing import Dict, List, Optional, Self, Type

from .base_model import BaseModel

logger = logging.getLogger(__name__)

class ModelRegistry:
    _instance = None
    _models: Dict[str, Type[BaseModel]] = {}

    def __new__(cls) -> Self:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._models = {}
        return cls._instance
    
    @classmethod
    def register(cls, name: str, model_class: Type[BaseModel]) -> None:
        if not issubclass(model_class, BaseModel):
            raise ValueError(f"Model class must be a subclass of BaseModel. Got {model_class}.")
        cls._models[name] = model_class
        logger.info("Registered model %s of type %s.", name, model_class.__name__)
    
    @classmethod
    def get(cls, name: str) -> Optional[Type[BaseModel]]:
        return cls._models.get(name)
    
    @classmethod
    def list(cls) -> List[str]:
        return list(cls._models.keys())
    
    @classmethod
    def clear(cls) -> None:
        cls._models.clear()
        logger.info("Cleared all registered models.")
    
    @classmethod
    def delete(cls, name: str) -> None:
        deleted = cls._models.pop(name, None)
        if deleted is not None:
            logger.info("Model %s deleted.", name)
        else:
            logger.info("Model %s not found.", name)