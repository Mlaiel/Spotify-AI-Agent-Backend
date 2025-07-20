#!/usr/bin/env python3
"""
Data Transformer for PagerDuty Integration.

Advanced data transformation utilities for converting between different
data formats, schemas, and representations used in PagerDuty integration.

Features:
- Multi-format data transformation (JSON, XML, YAML, CSV)
- Schema mapping and validation
- Data normalization and cleaning
- Custom transformation pipelines
- Streaming data processing
- Template-based transformations
- Data enrichment and augmentation
- Validation and error handling
"""

import asyncio
import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union, Callable, Type
from dataclasses import dataclass, field
from enum import Enum
import copy

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

try:
    import xml.etree.ElementTree as ET
    XML_AVAILABLE = True
except ImportError:
    XML_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)


class TransformationError(Exception):
    """Base exception for transformation errors."""
    pass


class SchemaValidationError(TransformationError):
    """Exception raised for schema validation errors."""
    pass


class DataFormatError(TransformationError):
    """Exception raised for data format errors."""
    pass


class TransformationFormat(Enum):
    """Supported transformation formats."""
    JSON = "json"
    XML = "xml"
    YAML = "yaml"
    CSV = "csv"
    PLAIN_TEXT = "text"
    PAGERDUTY_EVENT = "pagerduty_event"
    PAGERDUTY_INCIDENT = "pagerduty_incident"


@dataclass
class TransformationRule:
    """Data transformation rule definition."""
    source_field: str
    target_field: str
    transformer: Optional[Callable] = None
    default_value: Any = None
    required: bool = False
    validation_regex: Optional[str] = None
    description: str = ""


@dataclass
class TransformationSchema:
    """Schema definition for data transformation."""
    name: str
    source_format: TransformationFormat
    target_format: TransformationFormat
    rules: List[TransformationRule] = field(default_factory=list)
    validators: List[Callable] = field(default_factory=list)
    preprocessors: List[Callable] = field(default_factory=list)
    postprocessors: List[Callable] = field(default_factory=list)
    description: str = ""


class PagerDutyEventSchema(BaseModel):
    """Pydantic schema for PagerDuty events."""
    routing_key: str
    event_action: str  # trigger, acknowledge, resolve
    dedup_key: Optional[str] = None
    payload: Dict[str, Any]
    client: Optional[str] = None
    client_url: Optional[str] = None
    
    class Config:
        extra = "allow"


class PagerDutyIncidentSchema(BaseModel):
    """Pydantic schema for PagerDuty incidents."""
    title: str
    service: Dict[str, str]  # {id: str, type: "service_reference"}
    urgency: str = "high"
    incident_key: Optional[str] = None
    body: Optional[Dict[str, str]] = None
    
    class Config:
        extra = "allow"


class DataTransformer:
    """
    Advanced data transformer with support for multiple formats and schemas.
    
    Features:
    - Multi-format transformation
    - Schema-based mapping
    - Custom transformation pipelines
    - Data validation and cleaning
    - Template-based transformations
    - Streaming processing support
    """
    
    def __init__(self):
        """Initialize data transformer."""
        self.schemas: Dict[str, TransformationSchema] = {}
        self.custom_transformers: Dict[str, Callable] = {}
        self.templates: Dict[str, str] = {}
        
        # Register built-in schemas
        self._register_builtin_schemas()
        
        # Register built-in transformers
        self._register_builtin_transformers()
        
        logger.info("Data transformer initialized")
    
    def _register_builtin_schemas(self):
        """Register built-in transformation schemas."""
        # Alert to PagerDuty Event schema
        alert_to_event_schema = TransformationSchema(
            name="alert_to_pagerduty_event",
            source_format=TransformationFormat.JSON,
            target_format=TransformationFormat.PAGERDUTY_EVENT,
            rules=[
                TransformationRule(
                    source_field="routing_key",
                    target_field="routing_key",
                    required=True,
                    description="PagerDuty integration routing key"
                ),
                TransformationRule(
                    source_field="action",
                    target_field="event_action",
                    default_value="trigger",
                    description="Event action type"
                ),
                TransformationRule(
                    source_field="alert_id",
                    target_field="dedup_key",
                    description="Deduplication key"
                ),
                TransformationRule(
                    source_field="summary",
                    target_field="payload.summary",
                    required=True,
                    description="Alert summary"
                ),
                TransformationRule(
                    source_field="severity",
                    target_field="payload.severity",
                    default_value="error",
                    description="Alert severity"
                ),
                TransformationRule(
                    source_field="source",
                    target_field="payload.source",
                    default_value="monitoring",
                    description="Alert source"
                ),
                TransformationRule(
                    source_field="timestamp",
                    target_field="payload.timestamp",
                    transformer=self._timestamp_transformer,
                    description="Alert timestamp"
                )
            ]
        )
        
        self.register_schema(alert_to_event_schema)
        
        # JSON to XML schema
        json_to_xml_schema = TransformationSchema(
            name="json_to_xml",
            source_format=TransformationFormat.JSON,
            target_format=TransformationFormat.XML,
            preprocessors=[self._flatten_nested_objects],
            postprocessors=[self._format_xml_output]
        )
        
        self.register_schema(json_to_xml_schema)
    
    def _register_builtin_transformers(self):
        """Register built-in transformer functions."""
        self.custom_transformers.update({
            'timestamp': self._timestamp_transformer,
            'severity_mapping': self._severity_mapping_transformer,
            'url_validator': self._url_validator_transformer,
            'text_cleaner': self._text_cleaner_transformer,
            'json_parser': self._json_parser_transformer,
            'base64_encoder': self._base64_encoder_transformer,
            'base64_decoder': self._base64_decoder_transformer,
            'hash_generator': self._hash_generator_transformer
        })
    
    def register_schema(self, schema: TransformationSchema):
        """Register a transformation schema."""
        self.schemas[schema.name] = schema
        logger.debug(f"Registered transformation schema: {schema.name}")
    
    def register_transformer(self, name: str, transformer: Callable):
        """Register a custom transformer function."""
        self.custom_transformers[name] = transformer
        logger.debug(f"Registered custom transformer: {name}")
    
    def register_template(self, name: str, template: str):
        """Register a transformation template."""
        self.templates[name] = template
        logger.debug(f"Registered transformation template: {name}")
    
    def transform(self, 
                  data: Any, 
                  schema_name: Optional[str] = None,
                  source_format: Optional[TransformationFormat] = None,
                  target_format: Optional[TransformationFormat] = None,
                  custom_rules: Optional[List[TransformationRule]] = None) -> Any:
        """
        Transform data using specified schema or format conversion.
        
        Args:
            data: Input data to transform
            schema_name: Name of registered schema to use
            source_format: Source data format
            target_format: Target data format
            custom_rules: Custom transformation rules
            
        Returns:
            Transformed data
        """
        try:
            # Use schema if provided
            if schema_name:
                if schema_name not in self.schemas:
                    raise TransformationError(f"Schema not found: {schema_name}")
                
                schema = self.schemas[schema_name]
                return self._transform_with_schema(data, schema)
            
            # Use format conversion
            elif source_format and target_format:
                return self._transform_format(data, source_format, target_format)
            
            # Use custom rules
            elif custom_rules:
                return self._transform_with_rules(data, custom_rules)
            
            else:
                raise TransformationError("No transformation method specified")
                
        except Exception as e:
            logger.error(f"Transformation failed: {e}")
            raise TransformationError(f"Transformation failed: {e}")
    
    def _transform_with_schema(self, data: Any, schema: TransformationSchema) -> Any:
        """Transform data using a registered schema."""
        # Apply preprocessors
        for preprocessor in schema.preprocessors:
            data = preprocessor(data)
        
        # Apply transformation rules
        result = self._transform_with_rules(data, schema.rules)
        
        # Validate against target schema if available
        if schema.target_format == TransformationFormat.PAGERDUTY_EVENT:
            PagerDutyEventSchema(**result)
        elif schema.target_format == TransformationFormat.PAGERDUTY_INCIDENT:
            PagerDutyIncidentSchema(**result)
        
        # Apply postprocessors
        for postprocessor in schema.postprocessors:
            result = postprocessor(result)
        
        # Apply validators
        for validator in schema.validators:
            if not validator(result):
                raise SchemaValidationError("Schema validation failed")
        
        return result
    
    def _transform_with_rules(self, data: Any, rules: List[TransformationRule]) -> Dict[str, Any]:
        """Transform data using transformation rules."""
        result = {}
        
        for rule in rules:
            try:
                # Extract source value
                source_value = self._extract_field_value(data, rule.source_field)
                
                if source_value is None and rule.required:
                    raise TransformationError(f"Required field missing: {rule.source_field}")
                
                # Use default value if source is None
                if source_value is None:
                    source_value = rule.default_value
                
                # Skip if still None
                if source_value is None:
                    continue
                
                # Apply transformer if specified
                if rule.transformer:
                    source_value = rule.transformer(source_value)
                
                # Validate with regex if specified
                if rule.validation_regex and isinstance(source_value, str):
                    if not re.match(rule.validation_regex, source_value):
                        raise TransformationError(f"Validation failed for field {rule.source_field}")
                
                # Set target value
                self._set_field_value(result, rule.target_field, source_value)
                
            except Exception as e:
                logger.error(f"Rule transformation failed for {rule.source_field}: {e}")
                if rule.required:
                    raise
        
        return result
    
    def _transform_format(self, data: Any, source_format: TransformationFormat, 
                         target_format: TransformationFormat) -> Any:
        """Transform data between different formats."""
        # Parse source format
        if source_format == TransformationFormat.JSON:
            if isinstance(data, str):
                data = json.loads(data)
        elif source_format == TransformationFormat.YAML:
            if not YAML_AVAILABLE:
                raise DataFormatError("YAML library not available")
            if isinstance(data, str):
                data = yaml.safe_load(data)
        elif source_format == TransformationFormat.XML:
            if not XML_AVAILABLE:
                raise DataFormatError("XML library not available")
            if isinstance(data, str):
                data = self._xml_to_dict(data)
        
        # Convert to target format
        if target_format == TransformationFormat.JSON:
            return json.dumps(data, indent=2, default=str)
        elif target_format == TransformationFormat.YAML:
            if not YAML_AVAILABLE:
                raise DataFormatError("YAML library not available")
            return yaml.dump(data, default_flow_style=False)
        elif target_format == TransformationFormat.XML:
            if not XML_AVAILABLE:
                raise DataFormatError("XML library not available")
            return self._dict_to_xml(data)
        elif target_format == TransformationFormat.PLAIN_TEXT:
            return str(data)
        
        return data
    
    def _extract_field_value(self, data: Any, field_path: str) -> Any:
        """Extract value from nested data structure using dot notation."""
        if not field_path:
            return data
        
        current = data
        
        for key in field_path.split('.'):
            if isinstance(current, dict):
                current = current.get(key)
            elif isinstance(current, list) and key.isdigit():
                index = int(key)
                current = current[index] if 0 <= index < len(current) else None
            else:
                return None
            
            if current is None:
                break
        
        return current
    
    def _set_field_value(self, data: Dict[str, Any], field_path: str, value: Any):
        """Set value in nested data structure using dot notation."""
        keys = field_path.split('.')
        current = data
        
        # Navigate to parent of target key
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # Set the value
        current[keys[-1]] = value
    
    def _xml_to_dict(self, xml_string: str) -> Dict[str, Any]:
        """Convert XML string to dictionary."""
        def element_to_dict(element):
            result = {}
            
            # Add attributes
            if element.attrib:
                result['@attributes'] = element.attrib
            
            # Add text content
            if element.text and element.text.strip():
                if len(element) == 0:
                    return element.text.strip()
                result['text'] = element.text.strip()
            
            # Add children
            for child in element:
                child_data = element_to_dict(child)
                
                if child.tag in result:
                    # Multiple children with same tag - convert to list
                    if not isinstance(result[child.tag], list):
                        result[child.tag] = [result[child.tag]]
                    result[child.tag].append(child_data)
                else:
                    result[child.tag] = child_data
            
            return result
        
        root = ET.fromstring(xml_string)
        return {root.tag: element_to_dict(root)}
    
    def _dict_to_xml(self, data: Dict[str, Any], root_name: str = "root") -> str:
        """Convert dictionary to XML string."""
        def dict_to_element(name, value):
            element = ET.Element(name)
            
            if isinstance(value, dict):
                # Handle attributes
                if '@attributes' in value:
                    element.attrib.update(value['@attributes'])
                
                # Handle text content
                if 'text' in value:
                    element.text = str(value['text'])
                
                # Handle child elements
                for key, val in value.items():
                    if key not in ('@attributes', 'text'):
                        if isinstance(val, list):
                            for item in val:
                                element.append(dict_to_element(key, item))
                        else:
                            element.append(dict_to_element(key, val))
            
            elif isinstance(value, list):
                for item in value:
                    element.append(dict_to_element('item', item))
            
            else:
                element.text = str(value)
            
            return element
        
        if isinstance(data, dict) and len(data) == 1:
            root_name, root_data = next(iter(data.items()))
            root = dict_to_element(root_name, root_data)
        else:
            root = dict_to_element(root_name, data)
        
        return ET.tostring(root, encoding='unicode')
    
    # Built-in transformer functions
    
    def _timestamp_transformer(self, value: Any) -> str:
        """Transform various timestamp formats to ISO format."""
        if isinstance(value, datetime):
            return value.isoformat()
        elif isinstance(value, str):
            try:
                # Try parsing common formats
                dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
                return dt.isoformat()
            except ValueError:
                try:
                    # Try Unix timestamp
                    dt = datetime.fromtimestamp(float(value), tz=timezone.utc)
                    return dt.isoformat()
                except (ValueError, TypeError):
                    return value
        elif isinstance(value, (int, float)):
            dt = datetime.fromtimestamp(value, tz=timezone.utc)
            return dt.isoformat()
        else:
            return str(value)
    
    def _severity_mapping_transformer(self, value: Any) -> str:
        """Transform various severity formats to PagerDuty format."""
        severity_map = {
            'critical': 'critical',
            'high': 'error',
            'medium': 'warning',
            'low': 'info',
            'error': 'error',
            'warning': 'warning',
            'info': 'info',
            '1': 'critical',
            '2': 'error',
            '3': 'warning',
            '4': 'info'
        }
        
        return severity_map.get(str(value).lower(), 'error')
    
    def _url_validator_transformer(self, value: Any) -> str:
        """Validate and normalize URLs."""
        url = str(value)
        
        if not url.startswith(('http://', 'https://')):
            url = f'https://{url}'
        
        # Basic URL validation
        url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        
        if not url_pattern.match(url):
            raise TransformationError(f"Invalid URL: {url}")
        
        return url
    
    def _text_cleaner_transformer(self, value: Any) -> str:
        """Clean and normalize text content."""
        text = str(value)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove control characters
        text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
        
        return text
    
    def _json_parser_transformer(self, value: Any) -> Any:
        """Parse JSON string to object."""
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        return value
    
    def _base64_encoder_transformer(self, value: Any) -> str:
        """Encode value to base64."""
        import base64
        return base64.b64encode(str(value).encode()).decode()
    
    def _base64_decoder_transformer(self, value: Any) -> str:
        """Decode base64 value."""
        import base64
        try:
            return base64.b64decode(str(value)).decode()
        except Exception:
            return str(value)
    
    def _hash_generator_transformer(self, value: Any) -> str:
        """Generate hash for value."""
        import hashlib
        return hashlib.md5(str(value).encode()).hexdigest()
    
    # Built-in preprocessor and postprocessor functions
    
    def _flatten_nested_objects(self, data: Any) -> Any:
        """Flatten nested objects for simpler processing."""
        if not isinstance(data, dict):
            return data
        
        result = {}
        
        def flatten_recursive(obj, prefix=""):
            for key, value in obj.items():
                new_key = f"{prefix}.{key}" if prefix else key
                
                if isinstance(value, dict):
                    flatten_recursive(value, new_key)
                else:
                    result[new_key] = value
        
        flatten_recursive(data)
        return result
    
    def _format_xml_output(self, data: Any) -> str:
        """Format XML output with proper indentation."""
        if isinstance(data, str) and data.startswith('<'):
            try:
                import xml.dom.minidom
                dom = xml.dom.minidom.parseString(data)
                return dom.toprettyxml(indent="  ")
            except Exception:
                return data
        return data
    
    async def transform_async(self, *args, **kwargs) -> Any:
        """Async version of transform method."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.transform, *args, **kwargs)
    
    def transform_stream(self, data_stream: List[Any], **kwargs) -> List[Any]:
        """Transform a stream of data items."""
        results = []
        
        for item in data_stream:
            try:
                result = self.transform(item, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to transform stream item: {e}")
                # Continue with next item
                continue
        
        return results
    
    def validate_transformation(self, 
                              source_data: Any, 
                              target_data: Any, 
                              schema_name: str) -> bool:
        """Validate that transformation was successful."""
        try:
            # Get schema
            if schema_name not in self.schemas:
                return False
            
            schema = self.schemas[schema_name]
            
            # Apply all validators
            for validator in schema.validators:
                if not validator(target_data):
                    return False
            
            # Schema-specific validation
            if schema.target_format == TransformationFormat.PAGERDUTY_EVENT:
                PagerDutyEventSchema(**target_data)
            elif schema.target_format == TransformationFormat.PAGERDUTY_INCIDENT:
                PagerDutyIncidentSchema(**target_data)
            
            return True
            
        except Exception as e:
            logger.error(f"Transformation validation failed: {e}")
            return False
    
    def get_available_schemas(self) -> List[str]:
        """Get list of available transformation schemas."""
        return list(self.schemas.keys())
    
    def get_schema_info(self, schema_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific schema."""
        if schema_name not in self.schemas:
            return None
        
        schema = self.schemas[schema_name]
        
        return {
            'name': schema.name,
            'description': schema.description,
            'source_format': schema.source_format.value,
            'target_format': schema.target_format.value,
            'rules_count': len(schema.rules),
            'rules': [
                {
                    'source_field': rule.source_field,
                    'target_field': rule.target_field,
                    'required': rule.required,
                    'description': rule.description
                }
                for rule in schema.rules
            ]
        }


# Global transformer instance
_global_transformer = None

def get_data_transformer() -> DataTransformer:
    """Get global data transformer instance."""
    global _global_transformer
    if _global_transformer is None:
        _global_transformer = DataTransformer()
    return _global_transformer


# Convenience functions
def transform_to_pagerduty_event(alert_data: Dict[str, Any]) -> Dict[str, Any]:
    """Transform alert data to PagerDuty event format."""
    transformer = get_data_transformer()
    return transformer.transform(alert_data, schema_name="alert_to_pagerduty_event")


def transform_json_to_xml(json_data: Union[str, Dict[str, Any]]) -> str:
    """Transform JSON data to XML format."""
    transformer = get_data_transformer()
    return transformer.transform(
        json_data,
        source_format=TransformationFormat.JSON,
        target_format=TransformationFormat.XML
    )
