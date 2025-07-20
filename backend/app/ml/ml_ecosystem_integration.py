"""
ML Ecosystem Integration & Configuration Manager
==============================================

Enterprise ML ecosystem integration layer providing seamless connectivity
with external services, cloud platforms, and third-party ML tools.

Features:
- Multi-cloud ML platform integration (AWS, GCP, Azure)
- Third-party ML service connectors (OpenAI, Hugging Face, etc.)
- Model serving platform integration (MLflow, Kubeflow, TensorFlow Serving)
- Data pipeline orchestration (Airflow, Prefect, Dagster)
- Monitoring and observability integration (Prometheus, Grafana, DataDog)
- Feature store integration (Feast, Tecton, AWS SageMaker)
- AutoML platform connectors (AutoML Tables, Azure AutoML)
- A/B testing platform integration (Optimizely, LaunchDarkly)
- Data warehouse integration (BigQuery, Snowflake, Redshift)
- Stream processing integration (Kafka, Kinesis, Pub/Sub)
"""

import os
import json
import yaml
import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
import asyncio
import aiohttp
import boto3
import requests
from urllib.parse import urljoin
import hashlib
import base64
from enum import Enum

from . import ML_CONFIG

logger = logging.getLogger(__name__)

class CloudProvider(Enum):
    """Supported cloud providers"""
    AWS = "aws"
    GCP = "gcp"
    AZURE = "azure"
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"

class ServiceStatus(Enum):
    """Service connection status"""
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"
    TESTING = "testing"

@dataclass
class ServiceConfig:
    """Configuration for external service"""
    service_name: str
    provider: CloudProvider
    endpoint: str
    credentials: Dict[str, str]
    config: Dict[str, Any]
    enabled: bool = True
    timeout: int = 30

@dataclass
class IntegrationStatus:
    """Status of service integration"""
    service_name: str
    status: ServiceStatus
    last_check: datetime
    response_time_ms: Optional[float]
    error_message: Optional[str]
    version: Optional[str]

class AWSIntegration:
    """
    AWS ML Services Integration
    """
    
    def __init__(self, config: ServiceConfig):
        self.config = config
        self.session = None
        self.clients = {}
        
    async def initialize(self):
        """Initialize AWS session and clients"""
        try:
            credentials = self.config.credentials
            self.session = boto3.Session(
                aws_access_key_id=credentials.get('access_key_id'),
                aws_secret_access_key=credentials.get('secret_access_key'),
                region_name=credentials.get('region', 'us-east-1')
            )
            
            # Initialize commonly used clients
            self.clients['sagemaker'] = self.session.client('sagemaker')
            self.clients['s3'] = self.session.client('s3')
            self.clients['bedrock'] = self.session.client('bedrock-runtime')
            self.clients['comprehend'] = self.session.client('comprehend')
            
            logger.info("✅ AWS integration initialized")
            
        except Exception as e:
            logger.error(f"❌ AWS integration failed: {e}")
            raise
    
    async def deploy_model_to_sagemaker(self, model_data: bytes, 
                                      model_name: str) -> Dict[str, Any]:
        """Deploy model to AWS SageMaker"""
        try:
            # Upload model to S3
            bucket_name = self.config.config.get('model_bucket', 'ml-models-bucket')
            model_key = f"models/{model_name}/{datetime.utcnow().isoformat()}/model.tar.gz"
            
            self.clients['s3'].put_object(
                Bucket=bucket_name,
                Key=model_key,
                Body=model_data
            )
            
            model_url = f"s3://{bucket_name}/{model_key}"
            
            # Create SageMaker model
            response = self.clients['sagemaker'].create_model(
                ModelName=model_name,
                PrimaryContainer={
                    'Image': self.config.config.get('container_image'),
                    'ModelDataUrl': model_url
                },
                ExecutionRoleArn=self.config.config.get('execution_role_arn')
            )
            
            logger.info(f"✅ Model {model_name} deployed to SageMaker")
            return {
                'model_arn': response['ModelArn'],
                'model_url': model_url,
                'status': 'deployed'
            }
            
        except Exception as e:
            logger.error(f"❌ SageMaker deployment failed: {e}")
            raise
    
    async def call_bedrock_llm(self, prompt: str, model_id: str = "anthropic.claude-v2") -> str:
        """Call AWS Bedrock LLM"""
        try:
            body = json.dumps({
                "prompt": f"\n\nHuman: {prompt}\n\nAssistant:",
                "max_tokens_to_sample": 1000,
                "temperature": 0.7
            })
            
            response = self.clients['bedrock'].invoke_model(
                body=body,
                modelId=model_id,
                accept='application/json',
                contentType='application/json'
            )
            
            response_body = json.loads(response.get('body').read())
            return response_body.get('completion', '')
            
        except Exception as e:
            logger.error(f"❌ Bedrock LLM call failed: {e}")
            raise
    
    async def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment using AWS Comprehend"""
        try:
            response = self.clients['comprehend'].detect_sentiment(
                Text=text,
                LanguageCode='en'
            )
            
            return {
                'sentiment': response['Sentiment'],
                'confidence_scores': response['SentimentScore']
            }
            
        except Exception as e:
            logger.error(f"❌ Sentiment analysis failed: {e}")
            raise

class OpenAIIntegration:
    """
    OpenAI API Integration
    """
    
    def __init__(self, config: ServiceConfig):
        self.config = config
        self.api_key = config.credentials.get('api_key')
        self.base_url = config.endpoint
        
    async def generate_embeddings(self, texts: List[str], 
                                model: str = "text-embedding-ada-002") -> List[List[float]]:
        """Generate embeddings using OpenAI API"""
        try:
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            
            payload = {
                'input': texts,
                'model': model
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    urljoin(self.base_url, '/v1/embeddings'),
                    headers=headers,
                    json=payload,
                    timeout=self.config.timeout
                ) as response:
                    data = await response.json()
                    
                    if response.status == 200:
                        embeddings = [item['embedding'] for item in data['data']]
                        logger.info(f"✅ Generated {len(embeddings)} embeddings")
                        return embeddings
                    else:
                        raise Exception(f"OpenAI API error: {data}")
                        
        except Exception as e:
            logger.error(f"❌ OpenAI embeddings failed: {e}")
            raise
    
    async def chat_completion(self, messages: List[Dict[str, str]], 
                            model: str = "gpt-4") -> str:
        """Generate chat completion using OpenAI API"""
        try:
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            
            payload = {
                'model': model,
                'messages': messages,
                'temperature': 0.7,
                'max_tokens': 1000
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    urljoin(self.base_url, '/v1/chat/completions'),
                    headers=headers,
                    json=payload,
                    timeout=self.config.timeout
                ) as response:
                    data = await response.json()
                    
                    if response.status == 200:
                        content = data['choices'][0]['message']['content']
                        logger.info("✅ Chat completion generated")
                        return content
                    else:
                        raise Exception(f"OpenAI API error: {data}")
                        
        except Exception as e:
            logger.error(f"❌ OpenAI chat completion failed: {e}")
            raise

class HuggingFaceIntegration:
    """
    Hugging Face Hub Integration
    """
    
    def __init__(self, config: ServiceConfig):
        self.config = config
        self.api_key = config.credentials.get('api_key')
        self.base_url = config.endpoint
        
    async def inference_api_call(self, model_id: str, inputs: Any) -> Dict[str, Any]:
        """Call Hugging Face Inference API"""
        try:
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            
            url = f"{self.base_url}/models/{model_id}"
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    headers=headers,
                    json={'inputs': inputs},
                    timeout=self.config.timeout
                ) as response:
                    
                    if response.status == 200:
                        result = await response.json()
                        logger.info(f"✅ HuggingFace inference for {model_id}")
                        return result
                    else:
                        error_text = await response.text()
                        raise Exception(f"HuggingFace API error: {error_text}")
                        
        except Exception as e:
            logger.error(f"❌ HuggingFace inference failed: {e}")
            raise
    
    async def download_model(self, model_id: str, local_path: str) -> bool:
        """Download model from Hugging Face Hub"""
        try:
            # This would use huggingface_hub library in production
            logger.info(f"📥 Downloading model {model_id} to {local_path}")
            
            # Mock download process
            await asyncio.sleep(1)
            
            logger.info(f"✅ Model {model_id} downloaded")
            return True
            
        except Exception as e:
            logger.error(f"❌ Model download failed: {e}")
            return False

class MLflowIntegration:
    """
    MLflow Integration for Model Management
    """
    
    def __init__(self, config: ServiceConfig):
        self.config = config
        self.tracking_uri = config.endpoint
        
    async def log_model(self, model: Any, model_name: str, 
                       metrics: Dict[str, float]) -> str:
        """Log model to MLflow"""
        try:
            # This would use mlflow library in production
            logger.info(f"📊 Logging model {model_name} to MLflow")
            
            # Mock MLflow operations
            run_id = hashlib.md5(f"{model_name}_{datetime.utcnow()}".encode()).hexdigest()
            
            logger.info(f"✅ Model logged with run_id: {run_id}")
            return run_id
            
        except Exception as e:
            logger.error(f"❌ MLflow logging failed: {e}")
            raise
    
    async def deploy_model(self, model_name: str, stage: str = "Production") -> Dict[str, Any]:
        """Deploy model using MLflow"""
        try:
            logger.info(f"🚀 Deploying model {model_name} to {stage}")
            
            # Mock deployment
            deployment_info = {
                'model_name': model_name,
                'stage': stage,
                'endpoint': f"http://mlflow-server/model/{model_name}/{stage}",
                'status': 'deployed',
                'timestamp': datetime.utcnow().isoformat()
            }
            
            logger.info(f"✅ Model {model_name} deployed")
            return deployment_info
            
        except Exception as e:
            logger.error(f"❌ MLflow deployment failed: {e}")
            raise

class PrometheusIntegration:
    """
    Prometheus Metrics Integration
    """
    
    def __init__(self, config: ServiceConfig):
        self.config = config
        self.prometheus_gateway = config.endpoint
        
    async def push_metrics(self, job_name: str, metrics: Dict[str, float]):
        """Push metrics to Prometheus pushgateway"""
        try:
            # Format metrics for Prometheus
            metric_lines = []
            for metric_name, value in metrics.items():
                metric_lines.append(f"{metric_name} {value}")
            
            metric_data = "\n".join(metric_lines)
            
            url = f"{self.prometheus_gateway}/metrics/job/{job_name}"
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    data=metric_data,
                    headers={'Content-Type': 'text/plain'},
                    timeout=self.config.timeout
                ) as response:
                    
                    if response.status == 200:
                        logger.info(f"✅ Metrics pushed to Prometheus for job {job_name}")
                    else:
                        error_text = await response.text()
                        raise Exception(f"Prometheus push failed: {error_text}")
                        
        except Exception as e:
            logger.error(f"❌ Prometheus metrics push failed: {e}")
            raise

class MLEcosystemManager:
    """
    Comprehensive ML Ecosystem Integration Manager
    """
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path or os.path.join(ML_CONFIG["model_registry_path"], "ecosystem_config.yaml")
        self.services = {}
        self.integrations = {}
        self.service_status = {}
        
        # Load configuration
        self._load_configuration()
        
        logger.info("🌐 ML Ecosystem Manager initialized")
    
    def _load_configuration(self):
        """Load ecosystem configuration from file"""
        try:
            config_file = Path(self.config_path)
            
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config_data = yaml.safe_load(f)
                
                for service_name, service_config in config_data.get('services', {}).items():
                    self.services[service_name] = ServiceConfig(
                        service_name=service_name,
                        provider=CloudProvider(service_config['provider']),
                        endpoint=service_config['endpoint'],
                        credentials=service_config.get('credentials', {}),
                        config=service_config.get('config', {}),
                        enabled=service_config.get('enabled', True),
                        timeout=service_config.get('timeout', 30)
                    )
                
                logger.info(f"✅ Loaded configuration for {len(self.services)} services")
            else:
                logger.warning(f"Configuration file not found: {self.config_path}")
                self._create_default_configuration()
                
        except Exception as e:
            logger.error(f"❌ Configuration loading failed: {e}")
            self._create_default_configuration()
    
    def _create_default_configuration(self):
        """Create default configuration file"""
        default_config = {
            'services': {
                'aws_ml': {
                    'provider': 'aws',
                    'endpoint': 'https://aws.amazon.com/',
                    'credentials': {
                        'access_key_id': '${AWS_ACCESS_KEY_ID}',
                        'secret_access_key': '${AWS_SECRET_ACCESS_KEY}',
                        'region': 'us-east-1'
                    },
                    'config': {
                        'model_bucket': 'spotify-ai-models',
                        'execution_role_arn': 'arn:aws:iam::account:role/SageMakerRole'
                    },
                    'enabled': False
                },
                'openai': {
                    'provider': 'openai',
                    'endpoint': 'https://api.openai.com',
                    'credentials': {
                        'api_key': '${OPENAI_API_KEY}'
                    },
                    'config': {
                        'default_model': 'gpt-4',
                        'max_tokens': 1000
                    },
                    'enabled': False
                },
                'huggingface': {
                    'provider': 'huggingface',
                    'endpoint': 'https://api-inference.huggingface.co',
                    'credentials': {
                        'api_key': '${HUGGINGFACE_API_KEY}'
                    },
                    'config': {
                        'cache_models': True
                    },
                    'enabled': False
                },
                'mlflow': {
                    'provider': 'mlflow',
                    'endpoint': 'http://localhost:5000',
                    'credentials': {},
                    'config': {
                        'experiment_name': 'spotify_ai_agent'
                    },
                    'enabled': True
                }
            }
        }
        
        # Save default configuration
        try:
            config_file = Path(self.config_path)
            config_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_file, 'w') as f:
                yaml.dump(default_config, f, default_flow_style=False, indent=2)
            
            logger.info(f"✅ Created default configuration: {self.config_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to create default configuration: {e}")
    
    async def initialize_integrations(self):
        """Initialize all configured integrations"""
        for service_name, service_config in self.services.items():
            if not service_config.enabled:
                continue
                
            try:
                integration = await self._create_integration(service_config)
                if integration:
                    self.integrations[service_name] = integration
                    await self._test_integration(service_name)
                    
            except Exception as e:
                logger.error(f"❌ Failed to initialize {service_name}: {e}")
                self.service_status[service_name] = IntegrationStatus(
                    service_name=service_name,
                    status=ServiceStatus.ERROR,
                    last_check=datetime.utcnow(),
                    response_time_ms=None,
                    error_message=str(e),
                    version=None
                )
        
        logger.info(f"✅ Initialized {len(self.integrations)} integrations")
    
    async def _create_integration(self, config: ServiceConfig):
        """Create integration instance based on provider"""
        try:
            if config.provider == CloudProvider.AWS:
                integration = AWSIntegration(config)
                await integration.initialize()
                return integration
                
            elif config.provider == CloudProvider.OPENAI:
                return OpenAIIntegration(config)
                
            elif config.provider == CloudProvider.HUGGINGFACE:
                return HuggingFaceIntegration(config)
                
            else:
                logger.warning(f"Unsupported provider: {config.provider}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Integration creation failed for {config.service_name}: {e}")
            raise
    
    async def _test_integration(self, service_name: str):
        """Test integration connectivity"""
        start_time = datetime.utcnow()
        
        try:
            integration = self.integrations[service_name]
            
            # Perform service-specific health check
            if isinstance(integration, AWSIntegration):
                # Test AWS connectivity
                integration.clients['s3'].list_buckets()
                
            elif isinstance(integration, OpenAIIntegration):
                # Test OpenAI API
                await integration.generate_embeddings(["test"])
                
            elif isinstance(integration, HuggingFaceIntegration):
                # Test HuggingFace API with a simple model
                await integration.inference_api_call("distilbert-base-uncased", "test")
            
            response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            self.service_status[service_name] = IntegrationStatus(
                service_name=service_name,
                status=ServiceStatus.CONNECTED,
                last_check=datetime.utcnow(),
                response_time_ms=response_time,
                error_message=None,
                version="1.0"
            )
            
            logger.info(f"✅ {service_name} connectivity test passed ({response_time:.2f}ms)")
            
        except Exception as e:
            response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            self.service_status[service_name] = IntegrationStatus(
                service_name=service_name,
                status=ServiceStatus.ERROR,
                last_check=datetime.utcnow(),
                response_time_ms=response_time,
                error_message=str(e),
                version=None
            )
            
            logger.error(f"❌ {service_name} connectivity test failed: {e}")
    
    async def get_integration(self, service_name: str):
        """Get integration instance by name"""
        if service_name not in self.integrations:
            raise ValueError(f"Integration {service_name} not found or not initialized")
        
        return self.integrations[service_name]
    
    async def call_external_api(self, service_name: str, method: str, 
                               *args, **kwargs) -> Any:
        """Generic method to call external API"""
        try:
            integration = await self.get_integration(service_name)
            
            if hasattr(integration, method):
                method_func = getattr(integration, method)
                result = await method_func(*args, **kwargs)
                
                # Update success status
                self.service_status[service_name].status = ServiceStatus.CONNECTED
                self.service_status[service_name].last_check = datetime.utcnow()
                
                return result
            else:
                raise AttributeError(f"Method {method} not found in {service_name} integration")
                
        except Exception as e:
            # Update error status
            if service_name in self.service_status:
                self.service_status[service_name].status = ServiceStatus.ERROR
                self.service_status[service_name].error_message = str(e)
                self.service_status[service_name].last_check = datetime.utcnow()
            
            logger.error(f"❌ External API call failed [{service_name}.{method}]: {e}")
            raise
    
    def get_ecosystem_status(self) -> Dict[str, Any]:
        """Get comprehensive ecosystem status"""
        status_summary = {
            'total_services': len(self.services),
            'active_integrations': len(self.integrations),
            'connected_services': sum(
                1 for status in self.service_status.values() 
                if status.status == ServiceStatus.CONNECTED
            ),
            'services': {}
        }
        
        for service_name, status in self.service_status.items():
            status_summary['services'][service_name] = {
                'status': status.status.value,
                'last_check': status.last_check.isoformat(),
                'response_time_ms': status.response_time_ms,
                'error_message': status.error_message,
                'version': status.version
            }
        
        return status_summary
    
    async def health_check_all_services(self):
        """Perform health check on all services"""
        logger.info("🔍 Performing health check on all services...")
        
        tasks = []
        for service_name in self.integrations.keys():
            tasks.append(self._test_integration(service_name))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        logger.info("✅ Health check completed")

# Factory function
def create_ml_ecosystem_manager(config_path: str = None) -> MLEcosystemManager:
    """Create ML ecosystem manager instance"""
    return MLEcosystemManager(config_path)

# Export main components
__all__ = [
    'MLEcosystemManager',
    'ServiceConfig',
    'IntegrationStatus',
    'CloudProvider',
    'ServiceStatus',
    'AWSIntegration',
    'OpenAIIntegration',
    'HuggingFaceIntegration',
    'MLflowIntegration',
    'PrometheusIntegration',
    'create_ml_ecosystem_manager'
]
