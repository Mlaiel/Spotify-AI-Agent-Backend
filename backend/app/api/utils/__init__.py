"""
🎵 Spotify AI Agent - Enterprise Utilities Package
==================================================

Suite complète d'utilitaires enterprise pour le backend Spotify AI Agent.
Développé avec l'excellence technique et l'architecture moderne.

Architecture modulaire:
- Data Transformation & Validation
- String Processing & Text Analytics  
- DateTime Management & Timezone Handling
- File Management & Upload Security
- Cryptographic Security & Hashing
- Performance Monitoring & Optimization
- Network Communication & Health Checks
- Validation Framework & Data Integrity
- Formatting & Export Utilities

🎖️ Développé par l'équipe d'experts enterprise
"""

# Version et métadonnées
__version__ = "1.0.0"
__author__ = "Spotify AI Agent Enterprise Team"
__license__ = "MIT"
__status__ = "Production"

# Imports des modules utilitaires
from .data_transform import *
from .string_utils import *
from .datetime_utils import *
from .crypto_utils import *
from .file_utils import *
from .performance_utils import *
from .network_utils import *
from .validators import *
from .formatters import *

# Exports principaux pour l'utilisation externe
__all__ = [
    # Data Transformation
    "transform_data",
    "validate_data_structure", 
    "normalize_data",
    "sanitize_input",
    "deep_merge",
    "flatten_dict",
    "unflatten_dict",
    "filter_dict",
    "safe_cast",
    "serialize_for_json",
    
    # String Utilities
    "slugify",
    "camel_to_snake",
    "snake_to_camel",
    "pascal_to_snake",
    "snake_to_pascal",
    "extract_emails",
    "extract_urls", 
    "extract_phone_numbers",
    "generate_hash",
    "generate_random_string",
    "mask_sensitive_data",
    "truncate_text",
    "clean_text",
    "validate_string_format",
    "format_template",
    "get_text_statistics",
    "is_palindrome",
    "reverse_words",
    "count_words",
    "remove_duplicates",
    
    # DateTime Utilities
    "format_datetime",
    "parse_datetime",
    "get_timezone_offset",
    "convert_timezone",
    "humanize_datetime",
    "calculate_duration",
    "is_business_day",
    "get_us_holidays",
    "is_valid_date_range",
    "is_future_date",
    "get_week_boundaries",
    "get_month_boundaries",
    "get_quarter_boundaries",
    "generate_date_range",
    "is_within_business_hours",
    "next_business_day",
    "add_business_days",
    
    # Cryptographic Utilities
    "SecureEncryption",
    "generate_encryption_key",
    "aes_encrypt",
    "aes_decrypt",
    "hash_password",
    "verify_password",
    "secure_hash",
    "hmac_sign",
    "verify_hmac",
    "generate_secure_token",
    "generate_api_key",
    "generate_session_id",
    "generate_csrf_token",
    "generate_rsa_keypair",
    "rsa_encrypt",
    "rsa_decrypt",
    "rsa_sign",
    "rsa_verify_signature",
    "constant_time_compare",
    "secure_random_choice",
    "generate_salt",
    "derive_key_from_password",
    
    # File Management
    "FileValidator",
    "sanitize_filename",
    "generate_safe_path",
    "FileUploadManager",
    "calculate_file_hash",
    "get_file_metadata",
    "compress_file",
    "decompress_file",
    "create_archive",
    "extract_archive",
    "read_file_chunks",
    "copy_file_chunked",
    "split_file",
    "join_file_parts",
    "get_directory_size",
    "clean_directory",
    "ensure_directory",
    
    # Performance Monitoring
    "PerformanceMetrics",
    "PerformanceMonitor",
    "performance_monitor",
    "monitor_performance",
    "benchmark",
    "rate_limit",
    "PerformanceCache",
    "memoize",
    "profile_code",
    "FunctionProfiler",
    "SystemMonitor",
    "BottleneckDetector",
    "force_garbage_collection",
    "memory_usage_mb",
    "memory_tracker",
    
    # Network Utilities
    "NetworkConfig",
    "RequestMetrics",
    "EnterpriseHttpClient",
    "is_valid_url",
    "is_valid_domain",
    "is_valid_ip",
    "is_private_ip",
    "extract_domain",
    "normalize_url",
    "resolve_dns",
    "get_mx_records",
    "check_domain_exists",
    "check_http_health",
    "check_port_open",
    "check_ssl_certificate",
    "ConnectivityMonitor",
    
    # Validators
    "validate_email",
    "validate_phone",
    "validate_url",
    "validate_ip_address",
    "validate_user_password",
    "validate_username",
    "validate_audio_metadata",
    "validate_playlist_data",
    "validate_audio_file",
    "validate_image_file",
    "DataValidator",
    
    # Formatters
    "EnterpriseJSONEncoder",
    "format_json",
    "format_json_response",
    "dict_to_xml",
    "format_xml_response",
    "format_csv",
    "format_csv_from_dataframe",
    "format_table",
    "format_list",
    "format_key_value",
    "TemplateFormatter",
    "EMAIL_TEMPLATE",
    "REPORT_TEMPLATE",
    "format_file_size",
    "format_duration",
    "format_percentage",
    "format_currency",
    "MultiFormatExporter",
]
