from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

tracer_provider = TracerProvider()
tracer = trace.get_tracer(__name__)

# Console exporter (dev)
tracer_provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter())

# OTLP exporter (prod, vers Jaeger, Tempo, etc.)
otlp_exporter = OTLPSpanExporter(endpoint="http://localhost:4318/v1/traces")
tracer_provider.add_span_processor(BatchSpanProcessor(otlp_exporter)

trace.set_tracer_provider(tracer_provider)

# Exemple d'utilisation :
# with tracer.start_as_current_span("websocket_message"):
#     ... traitement ...
