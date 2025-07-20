import logging
from concurrent import futures
import grpc
from backend.app.api.v2.grpc import services_pb2
from backend.app.api.v2.grpc import services_pb2_grpc

class AnalyticsServiceServicer(services_pb2_grpc.AnalyticsServiceServicer):
    """
    Service gRPC Analytics : stats, logs, monitoring.
    """
    def __init__(self):
        self.logger = logging.getLogger("AnalyticsService")

    def GetStats(self, request, context):
        artist_id = request.artist_id
        period = request.period or "monthly"
        # Logique stats réelle (ex: DB, cache)
        stats = services_pb2.StatsReply(
            listeners=42000,
            streams=120000,
            top_countries={"FR": 20000, "US": 15000, "DE": 7000}
        )
        self.logger.info(f"gRPC GetStats pour {artist_id}")
        return stats

    def LogEvent(self, request, context):
        self.logger.info(f"gRPC LogEvent: {request.event_type} by {request.user_id}")
        return services_pb2.StatusReply(status="ok", message="Event logged")

# Démarrage serveur (exemple)
def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10)
    services_pb2_grpc.add_AnalyticsServiceServicer_to_server(AnalyticsServiceServicer(), server)
    server.add_insecure_port('[:]:50052')
    server.start()
    server.wait_for_termination()

# if __name__ == "__main__":
#     serve()
