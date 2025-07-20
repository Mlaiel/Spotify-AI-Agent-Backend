"""
Sécurité avancée pour le microservice Spleeter (API key, audit, extension JWT ready).
"""


from fastapi import Request, HTTPException, Depends
from starlette.status import HTTP_401_UNAUTHORIZED
import os
import logging
# Pour extension future :
# from fastapi.security import OAuth2PasswordBearer
# from jose import jwt, JWTError

API_KEY = os.environ.get("SPLEETER_API_KEY", "changeme")

async def verify_api_key(request: Request):
    api_key = request.headers.get("X-API-KEY")
    if not api_key or api_key != API_KEY:
        logging.warning(f"Tentative d'accès refusée : {request.client.host}")
        raise HTTPException(
            status_code=HTTP_401_UNAUTHORIZED,
            detail="Clé API invalide ou manquante")
        )

# Extension JWT/OAuth2 ready (commenté)
# oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
# async def verify_jwt(token: str = Depends(oauth2_scheme):
#     try:
#         payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
#         return payload
#     except JWTError:
#         raise HTTPException(status_code=401, detail="Token invalide")
