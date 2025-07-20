# Module Services de Stockage (FR)

## Présentation
Ce module fournit une abstraction unifiée, sécurisée et extensible pour tous les besoins de stockage de fichiers et d’objets de la plateforme Spotify AI Agent. Il supporte le stockage local, S3-compatible et CDN, avec sécurité, audit, scalabilité industrielle.

### Fonctionnalités principales
- Interface unifiée pour le stockage (local, S3, CDN)
- Upload/download sécurisé, contrôle d’accès, chiffrement, antivirus
- Versioning, audit, politiques de rétention
- Hooks ML/IA pour analyse/modération de contenu
- Prêt pour la production, scalable, cloud-native
- Conformité totale (RGPD, audit, logs d’accès)

### Structure
- `file_service.py` : Abstraction, interface, logique métier
- `local_storage_service.py` : Stockage local sécurisé (dev/test)
- `s3_service.py` : Stockage objet S3 (AWS, MinIO, etc.)
- `cdn_service.py` : Intégration CDN (CloudFront, Cloudflare, etc.)

### Exemple d’utilisation
```python
from .s3_service import S3StorageService
s3 = S3StorageService(bucket="my-bucket", region="eu-west-1")
url = s3.upload_file("/tmp/song.mp3", key="audio/2025/song.mp3")
```

### Sécurité & conformité
- Toutes les opérations sont loguées et auditables
- Chiffrement au repos et en transit
- Contrôle d’accès, URLs signées, antivirus
- Suppression et rétention RGPD

### Extensibilité
- Ajoutez de nouveaux backends en étendant `FileStorageService`
- Hooks ML/IA pour analyse, modération, indexation

---
Pour la documentation détaillée, voir les docstrings dans chaque fichier service.

