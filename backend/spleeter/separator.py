"""Mock Spleeter Separator"""
import numpy as np

class Separator:
    def __init__(self, *args, **kwargs):
        pass
    
    def separate(self, audio_data):
        return {'vocals': np.zeros((1000,)), 'accompaniment': np.zeros((1000,))}
