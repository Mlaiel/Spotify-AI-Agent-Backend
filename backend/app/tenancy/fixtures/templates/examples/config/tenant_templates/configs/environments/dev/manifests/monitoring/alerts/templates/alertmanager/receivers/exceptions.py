"""
Exceptions personnalisées pour le module receivers d'alertes.

Ce module définit toutes les exceptions spécifiques au système
de gestion des receivers d'alertes Alertmanager.
"""

class ReceiverBaseException(Exception):
    """Exception de base pour tous les receivers"""
    
    def __init__(self, message: str, error_code: str = None, details: dict = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
    
    def to_dict(self) -> dict:
        """Convertit l'exception en dictionnaire"""
        return {
            'error_type': self.__class__.__name__,
            'error_code': self.error_code,
            'message': self.message,
            'details': self.details
        }

class ReceiverConfigError(ReceiverBaseException):
    """Erreur de configuration des receivers"""
    
    def __init__(self, message: str, config_section: str = None, validation_errors: list = None):
        super().__init__(message, "CONFIG_ERROR")
        self.config_section = config_section
        self.validation_errors = validation_errors or []
        
        if config_section:
            self.details['config_section'] = config_section
        if validation_errors:
            self.details['validation_errors'] = validation_errors

class NotificationError(ReceiverBaseException):
    """Erreur lors de l'envoi de notifications"""
    
    def __init__(
        self, 
        message: str, 
        receiver_name: str = None, 
        channel_type: str = None,
        http_status: int = None,
        is_retryable: bool = None
    ):
        super().__init__(message, "NOTIFICATION_ERROR")
        self.receiver_name = receiver_name
        self.channel_type = channel_type
        self.http_status = http_status
        self.is_retryable = is_retryable
        
        if receiver_name:
            self.details['receiver_name'] = receiver_name
        if channel_type:
            self.details['channel_type'] = channel_type
        if http_status:
            self.details['http_status'] = http_status
        if is_retryable is not None:
            self.details['is_retryable'] = is_retryable

class TemplateRenderError(ReceiverBaseException):
    """Erreur lors du rendu des templates"""
    
    def __init__(self, message: str, template_name: str = None, channel_type: str = None):
        super().__init__(message, "TEMPLATE_ERROR")
        self.template_name = template_name
        self.channel_type = channel_type
        
        if template_name:
            self.details['template_name'] = template_name
        if channel_type:
            self.details['channel_type'] = channel_type

class EscalationError(ReceiverBaseException):
    """Erreur lors de l'escalade d'alertes"""
    
    def __init__(
        self, 
        message: str, 
        policy_name: str = None, 
        escalation_level: int = None,
        failed_receivers: list = None
    ):
        super().__init__(message, "ESCALATION_ERROR")
        self.policy_name = policy_name
        self.escalation_level = escalation_level
        self.failed_receivers = failed_receivers or []
        
        if policy_name:
            self.details['policy_name'] = policy_name
        if escalation_level is not None:
            self.details['escalation_level'] = escalation_level
        if failed_receivers:
            self.details['failed_receivers'] = failed_receivers

class ThrottleError(ReceiverBaseException):
    """Erreur liée au throttling des notifications"""
    
    def __init__(
        self, 
        message: str, 
        throttle_key: str = None, 
        current_count: int = None,
        limit: int = None,
        reset_time: str = None
    ):
        super().__init__(message, "THROTTLE_ERROR")
        self.throttle_key = throttle_key
        self.current_count = current_count
        self.limit = limit
        self.reset_time = reset_time
        
        if throttle_key:
            self.details['throttle_key'] = throttle_key
        if current_count is not None:
            self.details['current_count'] = current_count
        if limit is not None:
            self.details['limit'] = limit
        if reset_time:
            self.details['reset_time'] = reset_time

class SecurityError(ReceiverBaseException):
    """Erreur de sécurité (secrets, authentification, etc.)"""
    
    def __init__(self, message: str, security_context: str = None):
        super().__init__(message, "SECURITY_ERROR")
        self.security_context = security_context
        
        if security_context:
            self.details['security_context'] = security_context

class ConfigurationError(ReceiverBaseException):
    """Erreur de configuration générale"""
    
    def __init__(self, message: str, component: str = None):
        super().__init__(message, "CONFIGURATION_ERROR")
        self.component = component
        
        if component:
            self.details['component'] = component

class ValidationError(ReceiverBaseException):
    """Erreur de validation des données"""
    
    def __init__(self, message: str, field_name: str = None, field_value: str = None):
        super().__init__(message, "VALIDATION_ERROR")
        self.field_name = field_name
        self.field_value = field_value
        
        if field_name:
            self.details['field_name'] = field_name
        if field_value:
            self.details['field_value'] = field_value

class ConnectivityError(ReceiverBaseException):
    """Erreur de connectivité réseau"""
    
    def __init__(
        self, 
        message: str, 
        endpoint: str = None, 
        timeout: float = None,
        retry_attempt: int = None
    ):
        super().__init__(message, "CONNECTIVITY_ERROR")
        self.endpoint = endpoint
        self.timeout = timeout
        self.retry_attempt = retry_attempt
        
        if endpoint:
            self.details['endpoint'] = endpoint
        if timeout:
            self.details['timeout'] = timeout
        if retry_attempt is not None:
            self.details['retry_attempt'] = retry_attempt

class AuthenticationError(ReceiverBaseException):
    """Erreur d'authentification"""
    
    def __init__(self, message: str, auth_type: str = None, endpoint: str = None):
        super().__init__(message, "AUTHENTICATION_ERROR")
        self.auth_type = auth_type
        self.endpoint = endpoint
        
        if auth_type:
            self.details['auth_type'] = auth_type
        if endpoint:
            self.details['endpoint'] = endpoint

class RateLimitError(ReceiverBaseException):
    """Erreur de limite de taux"""
    
    def __init__(
        self, 
        message: str, 
        rate_limit: int = None, 
        reset_time: str = None,
        retry_after: int = None
    ):
        super().__init__(message, "RATE_LIMIT_ERROR")
        self.rate_limit = rate_limit
        self.reset_time = reset_time
        self.retry_after = retry_after
        
        if rate_limit:
            self.details['rate_limit'] = rate_limit
        if reset_time:
            self.details['reset_time'] = reset_time
        if retry_after:
            self.details['retry_after'] = retry_after

class CircuitBreakerError(ReceiverBaseException):
    """Erreur de circuit breaker"""
    
    def __init__(
        self, 
        message: str, 
        receiver_name: str = None, 
        failure_count: int = None,
        state: str = None
    ):
        super().__init__(message, "CIRCUIT_BREAKER_ERROR")
        self.receiver_name = receiver_name
        self.failure_count = failure_count
        self.state = state
        
        if receiver_name:
            self.details['receiver_name'] = receiver_name
        if failure_count is not None:
            self.details['failure_count'] = failure_count
        if state:
            self.details['state'] = state

class TenantError(ReceiverBaseException):
    """Erreur liée à la gestion multi-tenant"""
    
    def __init__(self, message: str, tenant_id: str = None, tenant_status: str = None):
        super().__init__(message, "TENANT_ERROR")
        self.tenant_id = tenant_id
        self.tenant_status = tenant_status
        
        if tenant_id:
            self.details['tenant_id'] = tenant_id
        if tenant_status:
            self.details['tenant_status'] = tenant_status

class ResourceError(ReceiverBaseException):
    """Erreur de ressource (mémoire, CPU, etc.)"""
    
    def __init__(
        self, 
        message: str, 
        resource_type: str = None, 
        current_usage: float = None,
        limit: float = None
    ):
        super().__init__(message, "RESOURCE_ERROR")
        self.resource_type = resource_type
        self.current_usage = current_usage
        self.limit = limit
        
        if resource_type:
            self.details['resource_type'] = resource_type
        if current_usage is not None:
            self.details['current_usage'] = current_usage
        if limit is not None:
            self.details['limit'] = limit

class TimeoutError(ReceiverBaseException):
    """Erreur de timeout"""
    
    def __init__(
        self, 
        message: str, 
        operation: str = None, 
        timeout_duration: float = None,
        elapsed_time: float = None
    ):
        super().__init__(message, "TIMEOUT_ERROR")
        self.operation = operation
        self.timeout_duration = timeout_duration
        self.elapsed_time = elapsed_time
        
        if operation:
            self.details['operation'] = operation
        if timeout_duration:
            self.details['timeout_duration'] = timeout_duration
        if elapsed_time:
            self.details['elapsed_time'] = elapsed_time

class DataIntegrityError(ReceiverBaseException):
    """Erreur d'intégrité des données"""
    
    def __init__(
        self, 
        message: str, 
        data_type: str = None, 
        checksum_expected: str = None,
        checksum_actual: str = None
    ):
        super().__init__(message, "DATA_INTEGRITY_ERROR")
        self.data_type = data_type
        self.checksum_expected = checksum_expected
        self.checksum_actual = checksum_actual
        
        if data_type:
            self.details['data_type'] = data_type
        if checksum_expected:
            self.details['checksum_expected'] = checksum_expected
        if checksum_actual:
            self.details['checksum_actual'] = checksum_actual

class SerializationError(ReceiverBaseException):
    """Erreur de sérialisation/désérialisation"""
    
    def __init__(
        self, 
        message: str, 
        serialization_format: str = None, 
        data_sample: str = None
    ):
        super().__init__(message, "SERIALIZATION_ERROR")
        self.serialization_format = serialization_format
        self.data_sample = data_sample
        
        if serialization_format:
            self.details['serialization_format'] = serialization_format
        if data_sample:
            self.details['data_sample'] = data_sample

class VersionCompatibilityError(ReceiverBaseException):
    """Erreur de compatibilité de version"""
    
    def __init__(
        self, 
        message: str, 
        component: str = None, 
        required_version: str = None,
        current_version: str = None
    ):
        super().__init__(message, "VERSION_COMPATIBILITY_ERROR")
        self.component = component
        self.required_version = required_version
        self.current_version = current_version
        
        if component:
            self.details['component'] = component
        if required_version:
            self.details['required_version'] = required_version
        if current_version:
            self.details['current_version'] = current_version

class HealthCheckError(ReceiverBaseException):
    """Erreur de contrôle de santé"""
    
    def __init__(
        self, 
        message: str, 
        receiver_name: str = None, 
        health_score: float = None,
        last_success: str = None
    ):
        super().__init__(message, "HEALTH_CHECK_ERROR")
        self.receiver_name = receiver_name
        self.health_score = health_score
        self.last_success = last_success
        
        if receiver_name:
            self.details['receiver_name'] = receiver_name
        if health_score is not None:
            self.details['health_score'] = health_score
        if last_success:
            self.details['last_success'] = last_success

# Exceptions spécifiques aux channels

class SlackError(NotificationError):
    """Erreur spécifique à Slack"""
    
    def __init__(self, message: str, slack_error_code: str = None, webhook_url: str = None):
        super().__init__(message, channel_type="slack")
        self.slack_error_code = slack_error_code
        self.webhook_url = webhook_url
        
        if slack_error_code:
            self.details['slack_error_code'] = slack_error_code
        if webhook_url:
            # Masquer l'URL pour la sécurité
            self.details['webhook_url'] = webhook_url[:30] + "..."

class EmailError(NotificationError):
    """Erreur spécifique à l'email"""
    
    def __init__(
        self, 
        message: str, 
        smtp_error_code: int = None, 
        smtp_server: str = None,
        recipients_count: int = None
    ):
        super().__init__(message, channel_type="email")
        self.smtp_error_code = smtp_error_code
        self.smtp_server = smtp_server
        self.recipients_count = recipients_count
        
        if smtp_error_code:
            self.details['smtp_error_code'] = smtp_error_code
        if smtp_server:
            self.details['smtp_server'] = smtp_server
        if recipients_count:
            self.details['recipients_count'] = recipients_count

class PagerDutyError(NotificationError):
    """Erreur spécifique à PagerDuty"""
    
    def __init__(
        self, 
        message: str, 
        pd_error_code: str = None, 
        incident_key: str = None,
        dedup_key: str = None
    ):
        super().__init__(message, channel_type="pagerduty")
        self.pd_error_code = pd_error_code
        self.incident_key = incident_key
        self.dedup_key = dedup_key
        
        if pd_error_code:
            self.details['pd_error_code'] = pd_error_code
        if incident_key:
            self.details['incident_key'] = incident_key
        if dedup_key:
            self.details['dedup_key'] = dedup_key

class WebhookError(NotificationError):
    """Erreur spécifique aux webhooks"""
    
    def __init__(
        self, 
        message: str, 
        webhook_url: str = None, 
        http_method: str = None,
        response_body: str = None
    ):
        super().__init__(message, channel_type="webhook")
        self.webhook_url = webhook_url
        self.http_method = http_method
        self.response_body = response_body
        
        if webhook_url:
            # Masquer l'URL pour la sécurité
            self.details['webhook_url'] = webhook_url[:50] + "..."
        if http_method:
            self.details['http_method'] = http_method
        if response_body:
            # Limiter la taille du body
            self.details['response_body'] = response_body[:200] + "..." if len(response_body) > 200 else response_body

class TeamsError(NotificationError):
    """Erreur spécifique à Microsoft Teams"""
    
    def __init__(self, message: str, teams_error_code: str = None, card_type: str = None):
        super().__init__(message, channel_type="teams")
        self.teams_error_code = teams_error_code
        self.card_type = card_type
        
        if teams_error_code:
            self.details['teams_error_code'] = teams_error_code
        if card_type:
            self.details['card_type'] = card_type

class DiscordError(NotificationError):
    """Erreur spécifique à Discord"""
    
    def __init__(self, message: str, discord_error_code: str = None, embed_count: int = None):
        super().__init__(message, channel_type="discord")
        self.discord_error_code = discord_error_code
        self.embed_count = embed_count
        
        if discord_error_code:
            self.details['discord_error_code'] = discord_error_code
        if embed_count:
            self.details['embed_count'] = embed_count

# Mapping des codes d'erreur HTTP vers les types d'exception
HTTP_ERROR_MAPPING = {
    400: ValidationError,
    401: AuthenticationError,
    403: AuthenticationError,
    404: ConfigurationError,
    408: TimeoutError,
    429: RateLimitError,
    500: NotificationError,
    502: ConnectivityError,
    503: ConnectivityError,
    504: TimeoutError
}

def create_http_error(status_code: int, message: str, **kwargs) -> ReceiverBaseException:
    """Crée une exception appropriée basée sur le code de statut HTTP"""
    error_class = HTTP_ERROR_MAPPING.get(status_code, NotificationError)
    return error_class(message, http_status=status_code, **kwargs)

def handle_exception(func):
    """Décorateur pour gérer les exceptions de manière standardisée"""
    import functools
    import logging
    
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except ReceiverBaseException:
            # Re-lever les exceptions connues
            raise
        except Exception as e:
            # Convertir les exceptions inconnues
            logger = logging.getLogger(func.__module__)
            logger.error(f"Unexpected error in {func.__name__}: {e}")
            raise ReceiverBaseException(
                f"Unexpected error in {func.__name__}: {str(e)}",
                error_code="UNEXPECTED_ERROR",
                details={'function': func.__name__, 'original_error': str(e)}
            )
    
    return wrapper
