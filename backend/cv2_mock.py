"""
OpenCV Mock for testing - Spotify AI Agent
==========================================
Mock minimal d'OpenCV pour éviter les problèmes de dépendances

Auteur: Équipe Lead Dev + Architecte IA
"""

import numpy as np
import sys
from typing import Any, Tuple, Optional

__version__ = "4.8.1.78"

# Ajout au module cv2 dans sys.modules pour être trouvé par les imports
sys.modules['cv2'] = sys.modules[__name__]

# Constants simulés
INTER_LINEAR = 1
INTER_CUBIC = 2
INTER_NEAREST = 0
IMREAD_COLOR = 1
IMREAD_GRAYSCALE = 0

def imread(filename: str, flags: int = IMREAD_COLOR) -> Optional[np.ndarray]:
    """Mock imread - retourne une image factice"""
    if flags == IMREAD_GRAYSCALE:
        return np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    else:
        return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

def imwrite(filename: str, img: np.ndarray) -> bool:
    """Mock imwrite"""
    return True

def resize(src: np.ndarray, dsize: Tuple[int, int], **kwargs) -> np.ndarray:
    """Mock resize"""
    return np.random.randint(0, 255, (*dsize, 3), dtype=np.uint8)

def cvtColor(src: np.ndarray, code: int) -> np.ndarray:
    """Mock cvtColor"""
    if len(src.shape) == 3:
        return np.random.randint(0, 255, src.shape[:2], dtype=np.uint8)
    return src

def GaussianBlur(src: np.ndarray, ksize: Tuple[int, int], sigmaX: float) -> np.ndarray:
    """Mock GaussianBlur"""
    return src

def threshold(src: np.ndarray, thresh: float, maxval: float, type: int) -> Tuple[float, np.ndarray]:
    """Mock threshold"""
    return thresh, (src > thresh).astype(np.uint8) * 255

def findContours(image: np.ndarray, mode: int, method: int) -> Tuple[list, np.ndarray]:
    """Mock findContours"""
    return [], np.array([])

def drawContours(image: np.ndarray, contours: list, contourIdx: int, color: Tuple[int, int, int], thickness: int) -> None:
    """Mock drawContours"""
    pass

def rectangle(img: np.ndarray, pt1: Tuple[int, int], pt2: Tuple[int, int], color: Tuple[int, int, int], thickness: int) -> None:
    """Mock rectangle"""
    pass

def putText(img: np.ndarray, text: str, org: Tuple[int, int], fontFace: int, fontScale: float, color: Tuple[int, int, int], thickness: int) -> None:
    """Mock putText"""
    pass

# Constants pour les modes
COLOR_BGR2GRAY = 6
COLOR_BGR2RGB = 4
COLOR_RGB2BGR = 4
THRESH_BINARY = 0
RETR_EXTERNAL = 0
CHAIN_APPROX_SIMPLE = 2
FONT_HERSHEY_SIMPLEX = 0
