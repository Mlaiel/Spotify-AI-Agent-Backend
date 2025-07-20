import requests
from typing import Optional

class SpleeterClient:
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key

    def separate(self, audio_path: str) -> Optional[bytes]:
        url = f"{self.base_url}/separate"
        with open(audio_path, "rb") as f:
            files = {"file": (audio_path, f, "audio/wav")}
            data = {"api_key": self.api_key}
            resp = requests.post(url, files=files, data=data, timeout=60)
            if resp.status_code == 200:
                return resp.content
            else:
                raise Exception(f"Spleeter error: {resp.status_code} {resp.text}")
