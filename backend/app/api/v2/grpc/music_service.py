import logging
from concurrent import futures
import grpc
from backend.app.api.v2.grpc import services_pb2
from backend.app.api.v2.grpc import services_pb2_grpc

class MusicServiceServicer(services_pb2_grpc.MusicServiceServicer):
    """
    Service gRPC Music : génération musicale, mastering, séparation de stems.
    """
    def __init__(self):
        self.logger = logging.getLogger("MusicService")

    def GenerateMusic(self, request, context):
        prompt = request.prompt
        duration = request.duration or 30
        style = request.style or "lofi"
        # Logique IA réelle (ex: génération audio)
        audio = b"AUDIO_BYTES_FAKE"  # À remplacer par vrai audio
        self.logger.info(f"gRPC GenerateMusic: {prompt} ({style}, {duration}s)")
        return services_pb2.MusicReply(audio=audio, format="wav")

    def MasterTrack(self, request, context):
        # Logique mastering réelle
        mastered = b"MASTERED_AUDIO_FAKE"
        self.logger.info(f"gRPC MasterTrack")
        return services_pb2.MusicReply(audio=mastered, format="wav")

    def SeparateStems(self, request, context):
        # Logique stems réelle
        stems = [b"STEM1_FAKE", b"STEM2_FAKE"]
        self.logger.info(f"gRPC SeparateStems")
        return services_pb2.StemsReply(stems=stems)

# Démarrage serveur (exemple)
def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    services_pb2_grpc.add_MusicServiceServicer_to_server(MusicServiceServicer(), server)
    server.add_insecure_port('[:]:50053')
    server.start()
    server.wait_for_termination()

# if __name__ == "__main__":
#     serve()
