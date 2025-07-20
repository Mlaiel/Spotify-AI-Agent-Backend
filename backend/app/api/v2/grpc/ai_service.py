import logging
from concurrent import futures
import grpc
from backend.app.api.v2.grpc import services_pb2
from backend.app.api.v2.grpc import services_pb2_grpc

class AIServiceServicer(services_pb2_grpc.AIServiceServicer):
    """
    Service gRPC IA : génération de texte, recommandations personnalisées.
    """
    def __init__(self):
        self.logger = logging.getLogger("AIService")

    def Generate(self, request, context):
        prompt = request.prompt
        max_length = request.max_length or 128
        language = request.language or "fr"
        # Logique IA réelle (ex: HuggingFace, OpenAI)
        result = f"[IA-{language}] Généré pour: {prompt[:30]}... (max_length={max_length})"
        self.logger.info(f"gRPC Generate: {prompt}")
        return services_pb2.GenerateReply(result=result)

    def Recommend(self, request, context):
        user_id = request.user_id
        top_k = request.top_k or 5
        # Logique de reco réelle (ex: collaborative filtering)
        recos = [f"track_{i}" for i in range(top_k)]
        self.logger.info(f"gRPC Recommend pour user {user_id}")
        return services_pb2.RecommendReply(recommendations=recos)

# Démarrage serveur (exemple)
def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    services_pb2_grpc.add_AIServiceServicer_to_server(AIServiceServicer(), server)
    server.add_insecure_port('[:]:50051')
    server.start()
    server.wait_for_termination()

# if __name__ == "__main__":
#     serve()
