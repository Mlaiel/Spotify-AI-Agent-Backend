"""
🎵 Spotify AI Agent - Tests File Utils Module
=============================================

Tests enterprise complets pour le module file_utils
avec validation de fichiers, sécurité et performance.

🎖️ Développé par l'équipe d'experts enterprise
"""

import pytest
import os
import tempfile
import shutil
import json
import csv
from pathlib import Path
from unittest.mock import patch, Mock, mock_open
import hashlib

# Import du module à tester
from backend.app.api.utils.file_utils import (
    read_file,
    write_file,
    append_file,
    file_exists,
    create_directory,
    delete_file,
    delete_directory,
    copy_file,
    move_file,
    get_file_size,
    get_file_info,
    list_directory,
    find_files,
    compress_file,
    decompress_file,
    calculate_file_hash,
    verify_file_integrity,
    safe_file_path,
    get_file_extension,
    change_file_extension,
    read_json_file,
    write_json_file,
    read_csv_file,
    write_csv_file,
    backup_file,
    rotate_log_files,
    clean_temp_files,
    monitor_file_changes
)

from . import TestUtils, security_test, performance_test, integration_test


class TestFileUtils:
    """Tests pour le module file_utils"""
    
    def setup_method(self):
        """Configuration avant chaque test"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.temp_dir, "test.txt")
        self.test_content = "Hello, World!\nThis is a test file."
    
    def teardown_method(self):
        """Nettoyage après chaque test"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_read_file_basic(self):
        """Test lecture fichier basique"""
        # Créer fichier test
        with open(self.test_file, 'w', encoding='utf-8') as f:
            f.write(self.test_content)
        
        content = read_file(self.test_file)
        
        assert content == self.test_content
    
    def test_read_file_binary(self):
        """Test lecture fichier binaire"""
        binary_content = b"Binary content\x00\x01\x02"
        binary_file = os.path.join(self.temp_dir, "test.bin")
        
        with open(binary_file, 'wb') as f:
            f.write(binary_content)
        
        content = read_file(binary_file, mode='rb')
        
        assert content == binary_content
    
    def test_read_file_encoding(self):
        """Test lecture avec encodage spécifique"""
        unicode_content = "Café à Paris 🎵"
        unicode_file = os.path.join(self.temp_dir, "unicode.txt")
        
        with open(unicode_file, 'w', encoding='utf-8') as f:
            f.write(unicode_content)
        
        content = read_file(unicode_file, encoding='utf-8')
        
        assert content == unicode_content
    
    def test_read_file_nonexistent(self):
        """Test lecture fichier inexistant"""
        nonexistent = os.path.join(self.temp_dir, "nonexistent.txt")
        
        content = read_file(nonexistent)
        
        assert content is None  # Ou exception selon implémentation
    
    def test_write_file_basic(self):
        """Test écriture fichier basique"""
        content = "Test content to write"
        
        success = write_file(self.test_file, content)
        
        assert success is True
        assert os.path.exists(self.test_file)
        
        # Vérifier contenu
        with open(self.test_file, 'r', encoding='utf-8') as f:
            written_content = f.read()
        
        assert written_content == content
    
    def test_write_file_binary(self):
        """Test écriture fichier binaire"""
        binary_content = b"Binary data\x00\x01\x02"
        binary_file = os.path.join(self.temp_dir, "binary.bin")
        
        success = write_file(binary_file, binary_content, mode='wb')
        
        assert success is True
        
        with open(binary_file, 'rb') as f:
            written_content = f.read()
        
        assert written_content == binary_content
    
    def test_write_file_create_directory(self):
        """Test écriture avec création répertoire"""
        nested_file = os.path.join(self.temp_dir, "nested", "dir", "file.txt")
        content = "Content in nested directory"
        
        success = write_file(nested_file, content, create_dirs=True)
        
        assert success is True
        assert os.path.exists(nested_file)
        assert os.path.isdir(os.path.dirname(nested_file))
    
    def test_write_file_overwrite_protection(self):
        """Test protection écrasement"""
        # Créer fichier existant
        with open(self.test_file, 'w') as f:
            f.write("Original content")
        
        # Tenter écrasement avec protection
        success = write_file(self.test_file, "New content", overwrite=False)
        
        assert success is False  # Ne doit pas écraser
        
        # Vérifier contenu original
        content = read_file(self.test_file)
        assert content == "Original content"
    
    def test_append_file_basic(self):
        """Test ajout contenu fichier"""
        initial_content = "Initial content\n"
        append_content = "Appended content\n"
        
        # Créer fichier initial
        write_file(self.test_file, initial_content)
        
        # Ajouter contenu
        success = append_file(self.test_file, append_content)
        
        assert success is True
        
        # Vérifier contenu final
        final_content = read_file(self.test_file)
        assert final_content == initial_content + append_content
    
    def test_append_file_nonexistent(self):
        """Test ajout fichier inexistant"""
        nonexistent = os.path.join(self.temp_dir, "new_file.txt")
        content = "New file content"
        
        success = append_file(nonexistent, content)
        
        assert success is True
        assert os.path.exists(nonexistent)
        assert read_file(nonexistent) == content
    
    def test_file_exists_basic(self):
        """Test vérification existence fichier"""
        # Fichier inexistant
        assert file_exists(self.test_file) is False
        
        # Créer fichier
        write_file(self.test_file, "content")
        assert file_exists(self.test_file) is True
        
        # Répertoire
        assert file_exists(self.temp_dir) is True
    
    def test_create_directory_basic(self):
        """Test création répertoire"""
        new_dir = os.path.join(self.temp_dir, "new_directory")
        
        success = create_directory(new_dir)
        
        assert success is True
        assert os.path.isdir(new_dir)
    
    def test_create_directory_nested(self):
        """Test création répertoire imbriqué"""
        nested_dir = os.path.join(self.temp_dir, "level1", "level2", "level3")
        
        success = create_directory(nested_dir, recursive=True)
        
        assert success is True
        assert os.path.isdir(nested_dir)
    
    def test_create_directory_existing(self):
        """Test création répertoire existant"""
        success = create_directory(self.temp_dir)
        
        # Ne doit pas échouer si existe déjà
        assert success is True
    
    def test_delete_file_basic(self):
        """Test suppression fichier"""
        # Créer fichier
        write_file(self.test_file, "content to delete")
        assert os.path.exists(self.test_file)
        
        # Supprimer
        success = delete_file(self.test_file)
        
        assert success is True
        assert not os.path.exists(self.test_file)
    
    def test_delete_file_nonexistent(self):
        """Test suppression fichier inexistant"""
        nonexistent = os.path.join(self.temp_dir, "nonexistent.txt")
        
        success = delete_file(nonexistent)
        
        # Ne doit pas échouer
        assert success is True  # Ou False selon implémentation
    
    @security_test
    def test_delete_file_safety(self):
        """Test sécurité suppression fichier"""
        # Tenter suppression fichier système critique
        system_file = "/etc/passwd"  # Linux
        
        # Ne doit pas permettre suppression
        success = delete_file(system_file)
        assert success is False  # Protection système
    
    def test_delete_directory_basic(self):
        """Test suppression répertoire"""
        test_dir = os.path.join(self.temp_dir, "to_delete")
        create_directory(test_dir)
        
        # Ajouter fichier dans le répertoire
        test_file_in_dir = os.path.join(test_dir, "file.txt")
        write_file(test_file_in_dir, "content")
        
        success = delete_directory(test_dir)
        
        assert success is True
        assert not os.path.exists(test_dir)
    
    def test_copy_file_basic(self):
        """Test copie fichier"""
        source = self.test_file
        dest = os.path.join(self.temp_dir, "copied.txt")
        
        # Créer fichier source
        write_file(source, self.test_content)
        
        success = copy_file(source, dest)
        
        assert success is True
        assert os.path.exists(dest)
        assert read_file(dest) == self.test_content
        assert os.path.exists(source)  # Original doit rester
    
    def test_copy_file_preserve_metadata(self):
        """Test copie avec préservation métadonnées"""
        source = self.test_file
        dest = os.path.join(self.temp_dir, "copied_meta.txt")
        
        write_file(source, "content")
        
        # Modifier permissions/timestamps
        os.chmod(source, 0o644)
        
        success = copy_file(source, dest, preserve_metadata=True)
        
        assert success is True
        
        # Vérifier métadonnées préservées
        source_stat = os.stat(source)
        dest_stat = os.stat(dest)
        
        assert source_stat.st_mode == dest_stat.st_mode
    
    def test_move_file_basic(self):
        """Test déplacement fichier"""
        source = self.test_file
        dest = os.path.join(self.temp_dir, "moved.txt")
        
        write_file(source, self.test_content)
        
        success = move_file(source, dest)
        
        assert success is True
        assert os.path.exists(dest)
        assert not os.path.exists(source)  # Original supprimé
        assert read_file(dest) == self.test_content
    
    def test_move_file_across_directories(self):
        """Test déplacement entre répertoires"""
        subdir = os.path.join(self.temp_dir, "subdir")
        create_directory(subdir)
        
        source = self.test_file
        dest = os.path.join(subdir, "moved.txt")
        
        write_file(source, "content to move")
        
        success = move_file(source, dest)
        
        assert success is True
        assert os.path.exists(dest)
        assert not os.path.exists(source)
    
    def test_get_file_size_basic(self):
        """Test obtention taille fichier"""
        content = "A" * 1000  # 1000 caractères
        write_file(self.test_file, content)
        
        size = get_file_size(self.test_file)
        
        assert size == len(content.encode('utf-8'))
    
    def test_get_file_size_empty(self):
        """Test taille fichier vide"""
        write_file(self.test_file, "")
        
        size = get_file_size(self.test_file)
        
        assert size == 0
    
    def test_get_file_size_nonexistent(self):
        """Test taille fichier inexistant"""
        nonexistent = os.path.join(self.temp_dir, "nonexistent.txt")
        
        size = get_file_size(nonexistent)
        
        assert size is None  # Ou -1 selon implémentation
    
    def test_get_file_info_basic(self):
        """Test informations fichier"""
        write_file(self.test_file, "test content")
        
        info = get_file_info(self.test_file)
        
        assert isinstance(info, dict)
        assert 'size' in info
        assert 'created' in info or 'modified' in info
        assert 'permissions' in info or 'mode' in info
        assert info['size'] > 0
    
    def test_list_directory_basic(self):
        """Test listage répertoire"""
        # Créer fichiers test
        files = ["file1.txt", "file2.txt", "file3.log"]
        for filename in files:
            write_file(os.path.join(self.temp_dir, filename), "content")
        
        # Créer sous-répertoire
        create_directory(os.path.join(self.temp_dir, "subdir"))
        
        contents = list_directory(self.temp_dir)
        
        assert isinstance(contents, list)
        assert len(contents) >= len(files) + 1  # Fichiers + subdir
        
        # Vérifier que tous les fichiers sont listés
        content_names = [item['name'] if isinstance(item, dict) else item for item in contents]
        for filename in files:
            assert filename in content_names
    
    def test_list_directory_filter(self):
        """Test listage avec filtre"""
        # Créer différents types de fichiers
        files = {
            "document.txt": "text",
            "image.jpg": "image",
            "script.py": "python",
            "data.json": "json"
        }
        
        for filename, content in files.items():
            write_file(os.path.join(self.temp_dir, filename), content)
        
        # Filtrer par extension
        txt_files = list_directory(self.temp_dir, pattern="*.txt")
        py_files = list_directory(self.temp_dir, pattern="*.py")
        
        assert len(txt_files) >= 1
        assert len(py_files) >= 1
        assert any("document.txt" in str(f) for f in txt_files)
        assert any("script.py" in str(f) for f in py_files)
    
    def test_find_files_basic(self):
        """Test recherche fichiers"""
        # Créer structure de fichiers
        structure = {
            "file1.txt": "content1",
            "subdir/file2.txt": "content2",
            "subdir/nested/file3.log": "content3",
            "other.py": "python code"
        }
        
        for filepath, content in structure.items():
            full_path = os.path.join(self.temp_dir, filepath)
            write_file(full_path, content, create_dirs=True)
        
        # Rechercher tous les .txt
        txt_files = find_files(self.temp_dir, pattern="*.txt", recursive=True)
        
        assert len(txt_files) >= 2
        assert any("file1.txt" in f for f in txt_files)
        assert any("file2.txt" in f for f in txt_files)
    
    def test_find_files_by_content(self):
        """Test recherche par contenu"""
        files_content = {
            "search1.txt": "This contains the keyword TARGET",
            "search2.txt": "This does not contain it",
            "search3.log": "Another TARGET file here"
        }
        
        for filename, content in files_content.items():
            write_file(os.path.join(self.temp_dir, filename), content)
        
        # Rechercher fichiers contenant "TARGET"
        matching_files = find_files(self.temp_dir, content_pattern="TARGET")
        
        assert len(matching_files) >= 2
        assert any("search1.txt" in f for f in matching_files)
        assert any("search3.log" in f for f in matching_files)
        assert not any("search2.txt" in f for f in matching_files)
    
    def test_compress_decompress_file(self):
        """Test compression/décompression fichier"""
        content = "Content to compress " * 100  # Contenu répétitif
        write_file(self.test_file, content)
        
        compressed_file = self.test_file + ".gz"
        
        # Compression
        success = compress_file(self.test_file, compressed_file)
        assert success is True
        assert os.path.exists(compressed_file)
        
        # Vérifier que compressé est plus petit
        original_size = get_file_size(self.test_file)
        compressed_size = get_file_size(compressed_file)
        assert compressed_size < original_size
        
        # Décompression
        decompressed_file = self.test_file + ".decompressed"
        success = decompress_file(compressed_file, decompressed_file)
        assert success is True
        
        # Vérifier contenu identique
        decompressed_content = read_file(decompressed_file)
        assert decompressed_content == content
    
    def test_calculate_file_hash_basic(self):
        """Test calcul hash fichier"""
        content = "Content for hashing"
        write_file(self.test_file, content)
        
        file_hash = calculate_file_hash(self.test_file)
        
        assert isinstance(file_hash, str)
        assert len(file_hash) > 0
        
        # Vérifier avec hash manuel
        manual_hash = hashlib.sha256(content.encode()).hexdigest()
        assert file_hash == manual_hash
    
    def test_calculate_file_hash_algorithms(self):
        """Test hash avec différents algorithmes"""
        content = "Test content"
        write_file(self.test_file, content)
        
        md5_hash = calculate_file_hash(self.test_file, algorithm='md5')
        sha1_hash = calculate_file_hash(self.test_file, algorithm='sha1')
        sha256_hash = calculate_file_hash(self.test_file, algorithm='sha256')
        
        assert len(md5_hash) == 32      # MD5 = 32 hex chars
        assert len(sha1_hash) == 40     # SHA1 = 40 hex chars
        assert len(sha256_hash) == 64   # SHA256 = 64 hex chars
        
        assert md5_hash != sha1_hash != sha256_hash
    
    def test_verify_file_integrity_valid(self):
        """Test vérification intégrité valide"""
        content = "Content for integrity check"
        write_file(self.test_file, content)
        
        # Calculer hash
        expected_hash = calculate_file_hash(self.test_file)
        
        # Vérifier intégrité
        is_valid = verify_file_integrity(self.test_file, expected_hash)
        
        assert is_valid is True
    
    def test_verify_file_integrity_corrupted(self):
        """Test vérification intégrité fichier corrompu"""
        content = "Original content"
        write_file(self.test_file, content)
        
        # Calculer hash original
        original_hash = calculate_file_hash(self.test_file)
        
        # Modifier fichier
        write_file(self.test_file, "Modified content")
        
        # Vérifier intégrité
        is_valid = verify_file_integrity(self.test_file, original_hash)
        
        assert is_valid is False
    
    @security_test
    def test_safe_file_path_basic(self):
        """Test chemin fichier sécurisé"""
        base_dir = self.temp_dir
        
        # Chemin sûr
        safe_path = safe_file_path(base_dir, "safe_file.txt")
        assert safe_path.startswith(base_dir)
        
        # Chemin avec sous-répertoire
        safe_nested = safe_file_path(base_dir, "subdir/file.txt")
        assert safe_nested.startswith(base_dir)
    
    @security_test
    def test_safe_file_path_directory_traversal(self):
        """Test protection traversée répertoire"""
        base_dir = self.temp_dir
        
        # Tentatives d'échappement
        malicious_paths = [
            "../../../etc/passwd",
            "..\\..\\windows\\system32",
            "/absolute/path/file.txt",
            "subdir/../../escape.txt"
        ]
        
        for malicious in malicious_paths:
            safe_path = safe_file_path(base_dir, malicious)
            
            # Chemin sécurisé doit rester dans base_dir
            assert safe_path is None or safe_path.startswith(base_dir)
    
    def test_get_file_extension_basic(self):
        """Test obtention extension fichier"""
        assert get_file_extension("file.txt") == ".txt"
        assert get_file_extension("document.pdf") == ".pdf"
        assert get_file_extension("archive.tar.gz") == ".gz"
        assert get_file_extension("no_extension") == ""
    
    def test_get_file_extension_edge_cases(self):
        """Test extension cas limites"""
        assert get_file_extension("") == ""
        assert get_file_extension(".hidden") == ""
        assert get_file_extension("file.") == "."
        assert get_file_extension("path/to/file.txt") == ".txt"
    
    def test_change_file_extension_basic(self):
        """Test changement extension"""
        assert change_file_extension("file.txt", ".md") == "file.md"
        assert change_file_extension("document.pdf", ".bak") == "document.bak"
        assert change_file_extension("no_ext", ".txt") == "no_ext.txt"
    
    def test_read_write_json_file(self):
        """Test lecture/écriture fichier JSON"""
        data = {
            "name": "Test User",
            "age": 30,
            "skills": ["Python", "JavaScript"],
            "active": True,
            "metadata": {
                "created": "2025-07-14",
                "version": 1.0
            }
        }
        
        json_file = os.path.join(self.temp_dir, "test.json")
        
        # Écriture
        success = write_json_file(json_file, data)
        assert success is True
        
        # Lecture
        loaded_data = read_json_file(json_file)
        assert loaded_data == data
    
    def test_read_write_csv_file(self):
        """Test lecture/écriture fichier CSV"""
        data = [
            {"name": "Alice", "age": 25, "city": "Paris"},
            {"name": "Bob", "age": 30, "city": "London"},
            {"name": "Charlie", "age": 35, "city": "Berlin"}
        ]
        
        csv_file = os.path.join(self.temp_dir, "test.csv")
        
        # Écriture
        success = write_csv_file(csv_file, data)
        assert success is True
        
        # Lecture
        loaded_data = read_csv_file(csv_file)
        assert loaded_data == data
    
    def test_backup_file_basic(self):
        """Test sauvegarde fichier"""
        content = "Important content to backup"
        write_file(self.test_file, content)
        
        backup_path = backup_file(self.test_file)
        
        assert backup_path is not None
        assert os.path.exists(backup_path)
        assert read_file(backup_path) == content
        assert backup_path != self.test_file
    
    def test_backup_file_custom_location(self):
        """Test sauvegarde emplacement personnalisé"""
        content = "Content to backup"
        write_file(self.test_file, content)
        
        backup_dir = os.path.join(self.temp_dir, "backups")
        create_directory(backup_dir)
        
        backup_path = backup_file(self.test_file, backup_dir=backup_dir)
        
        assert backup_path.startswith(backup_dir)
        assert os.path.exists(backup_path)
    
    def test_rotate_log_files_basic(self):
        """Test rotation fichiers log"""
        log_file = os.path.join(self.temp_dir, "app.log")
        
        # Créer fichier log avec contenu
        write_file(log_file, "Log entry 1\nLog entry 2\n")
        
        # Rotation
        rotated_files = rotate_log_files(log_file, max_files=3)
        
        assert isinstance(rotated_files, list)
        assert len(rotated_files) >= 1
        assert all(os.path.exists(f) for f in rotated_files)
    
    def test_clean_temp_files_basic(self):
        """Test nettoyage fichiers temporaires"""
        # Créer fichiers temporaires
        temp_files = []
        for i in range(5):
            temp_file = os.path.join(self.temp_dir, f"temp_{i}.tmp")
            write_file(temp_file, f"Temporary content {i}")
            temp_files.append(temp_file)
        
        # Ajouter fichiers non-temporaires
        regular_file = os.path.join(self.temp_dir, "important.txt")
        write_file(regular_file, "Important content")
        
        # Nettoyage
        cleaned_count = clean_temp_files(self.temp_dir, pattern="*.tmp")
        
        assert cleaned_count == 5
        assert os.path.exists(regular_file)  # Ne doit pas être supprimé
        assert not any(os.path.exists(f) for f in temp_files)
    
    @performance_test
    def test_file_operations_performance(self):
        """Test performance opérations fichiers"""
        content = "Performance test content " * 100
        
        def file_operations():
            # Créer 100 fichiers
            for i in range(100):
                test_file = os.path.join(self.temp_dir, f"perf_{i}.txt")
                write_file(test_file, content)
            
            # Lire tous les fichiers
            for i in range(100):
                test_file = os.path.join(self.temp_dir, f"perf_{i}.txt")
                read_file(test_file)
            
            return 100
        
        TestUtils.assert_performance(file_operations, max_time_ms=2000)
    
    @performance_test
    def test_large_file_handling(self):
        """Test gestion gros fichiers"""
        large_content = "Large file content " * 10000  # ~200KB
        large_file = os.path.join(self.temp_dir, "large.txt")
        
        def large_file_ops():
            write_file(large_file, large_content)
            read_content = read_file(large_file)
            file_hash = calculate_file_hash(large_file)
            return len(read_content)
        
        TestUtils.assert_performance(large_file_ops, max_time_ms=1000)
    
    @integration_test
    def test_complete_file_workflow(self):
        """Test workflow complet gestion fichiers"""
        # Scénario: Traitement pipeline de fichiers
        
        # 1. Créer structure de données
        input_data = [
            {"id": 1, "name": "Alice", "score": 95},
            {"id": 2, "name": "Bob", "score": 87},
            {"id": 3, "name": "Charlie", "score": 92}
        ]
        
        # 2. Créer répertoires de travail
        input_dir = os.path.join(self.temp_dir, "input")
        output_dir = os.path.join(self.temp_dir, "output")
        backup_dir = os.path.join(self.temp_dir, "backup")
        
        create_directory(input_dir)
        create_directory(output_dir)
        create_directory(backup_dir)
        
        # 3. Sauvegarder données en JSON
        input_file = os.path.join(input_dir, "data.json")
        write_json_file(input_file, input_data)
        
        # 4. Calculer hash pour intégrité
        input_hash = calculate_file_hash(input_file)
        
        # 5. Créer sauvegarde
        backup_path = backup_file(input_file, backup_dir=backup_dir)
        
        # 6. Traiter données (exemple: convertir en CSV)
        csv_file = os.path.join(output_dir, "processed.csv")
        write_csv_file(csv_file, input_data)
        
        # 7. Compresser fichier de sortie
        compressed_file = csv_file + ".gz"
        compress_file(csv_file, compressed_file)
        
        # 8. Générer rapport de traitement
        report = {
            "input_file": input_file,
            "output_file": csv_file,
            "compressed_file": compressed_file,
            "backup_file": backup_path,
            "input_hash": input_hash,
            "input_size": get_file_size(input_file),
            "output_size": get_file_size(csv_file),
            "compressed_size": get_file_size(compressed_file),
            "compression_ratio": get_file_size(compressed_file) / get_file_size(csv_file)
        }
        
        report_file = os.path.join(output_dir, "report.json")
        write_json_file(report_file, report)
        
        # 9. Vérifier intégrité des fichiers
        input_integrity = verify_file_integrity(input_file, input_hash)
        backup_integrity = verify_file_integrity(backup_path, input_hash)
        
        # 10. Nettoyer fichiers temporaires
        temp_pattern = "*.tmp"
        cleaned_count = clean_temp_files(self.temp_dir, pattern=temp_pattern)
        
        # === Vérifications ===
        assert os.path.exists(input_file)
        assert os.path.exists(csv_file)
        assert os.path.exists(compressed_file)
        assert os.path.exists(backup_path)
        assert os.path.exists(report_file)
        
        assert input_integrity is True
        assert backup_integrity is True
        
        assert report["compression_ratio"] < 1.0  # Fichier compressé
        
        # Vérifier données CSV
        csv_data = read_csv_file(csv_file)
        assert len(csv_data) == len(input_data)
        assert csv_data[0]["name"] == "Alice"
        
        print("✅ Workflow complet de gestion fichiers validé")


# Tests de sécurité spécialisés
class TestFileSecurityAdvanced:
    """Tests de sécurité avancés pour les fichiers"""
    
    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    @security_test
    def test_path_injection_protection(self):
        """Test protection injection de chemin"""
        base_dir = self.temp_dir
        
        # Tentatives d'injection
        malicious_inputs = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "/etc/shadow",
            "C:\\Windows\\System32\\config\\SAM",
            "file:///etc/passwd",
            "\\\\network\\share\\file.txt"
        ]
        
        for malicious in malicious_inputs:
            # safe_file_path doit nettoyer ou rejeter
            safe_path = safe_file_path(base_dir, malicious)
            
            if safe_path is not None:
                # Si un chemin est retourné, il doit être sûr
                assert safe_path.startswith(base_dir)
                assert ".." not in safe_path
    
    @security_test
    def test_file_size_limits(self):
        """Test limites de taille fichier"""
        # Créer fichier de taille limite
        max_size = 1024 * 1024  # 1MB
        large_content = "A" * (max_size + 1)
        
        large_file = os.path.join(self.temp_dir, "large.txt")
        
        # Écriture doit respecter les limites
        success = write_file(large_file, large_content, max_size=max_size)
        
        # Doit échouer ou tronquer
        assert success is False or get_file_size(large_file) <= max_size
    
    @security_test
    def test_binary_content_detection(self):
        """Test détection contenu binaire"""
        # Contenu binaire malveillant (simulé)
        binary_content = b"\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0A\x0B\x0C\x0D\x0E\x0F"
        binary_file = os.path.join(self.temp_dir, "binary.bin")
        
        write_file(binary_file, binary_content, mode='wb')
        
        # Fonction de validation doit détecter contenu binaire
        file_info = get_file_info(binary_file)
        
        if 'is_binary' in file_info:
            assert file_info['is_binary'] is True
    
    @security_test
    def test_symlink_protection(self):
        """Test protection liens symboliques"""
        if os.name != 'nt':  # Unix/Linux seulement
            target_file = os.path.join(self.temp_dir, "target.txt")
            symlink_file = os.path.join(self.temp_dir, "symlink.txt")
            
            write_file(target_file, "Target content")
            
            try:
                os.symlink(target_file, symlink_file)
                
                # Les opérations doivent gérer les symlinks correctement
                content = read_file(symlink_file, follow_symlinks=False)
                
                # Selon politique, peut être None ou contenu
                assert content is None or content == "Target content"
            except OSError:
                # Symlinks pas supportés sur cette plateforme
                pass


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
