from ariadne import ScalarType
import datetime
import json
import logging

datetime_scalar = ScalarType("DateTime")
json_scalar = ScalarType("JSON")
logger = logging.getLogger("GraphQLScalars")

@datetime_scalar.serializer
def serialize_datetime(value):
    if isinstance(value, datetime.datetime):
        return value.isoformat()
    raise ValueError("Invalid DateTime")

@datetime_scalar.value_parser
def parse_datetime_value(value):
    try:
        return datetime.datetime.fromisoformat(value)
    except Exception as e:
        logger.error(f"Erreur parsing DateTime: {e}")
        raise

@json_scalar.serializer
def serialize_json(value):
    return value

@json_scalar.value_parser
def parse_json_value(value):
    try:
        return json.loads(value) if isinstance(value, str) else value
    except Exception as e:
        logger.error(f"Erreur parsing JSON: {e}")
        raise
