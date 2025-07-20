"""
Module: serializers.py
Description: Serializers industriels (JSON, dict, model, validation, custom encoder, FastAPI/Pydantic-ready).
"""
import json
from typing import Any
from pydantic import BaseModel

def to_json(obj: Any) -> str:
    if isinstance(obj, BaseModel):
        return obj.json()
    return json.dumps(obj, default=str)

def from_json(data: str, model: Any = None) -> Any:
    dct = json.loads(data)
    if model and issubclass(model, BaseModel):
        return model(**dct)
    return dct

# Exemples d'utilisation
# to_json({"a": 1})
# from_json('{"a": 1}')
