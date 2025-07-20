"""
🎵 Spotify AI Agent - Tests String Utils Module
===============================================

Tests enterprise complets pour le module string_utils
avec validation de manipulation, sécurité et performance.

🎖️ Développé par l'équipe d'experts enterprise
"""

import pytest
import string
import secrets
from unittest.mock import patch, Mock

# Import du module à tester
from backend.app.api.utils.string_utils import (
    safe_string,
    truncate_string,
    slugify,
    extract_keywords,
    mask_sensitive_data,
    generate_random_string,
    is_valid_slug,
    normalize_whitespace,
    clean_text,
    extract_emails,
    extract_urls,
    camel_to_snake,
    snake_to_camel,
    title_case,
    remove_accents,
    escape_html,
    unescape_html,
    encode_base64,
    decode_base64,
    generate_hash,
    compare_strings_fuzzy
)

from . import TestUtils, security_test, performance_test, integration_test


class TestStringUtils:
    """Tests pour le module string_utils"""
    
    def test_safe_string_basic(self):
        """Test conversion sécurisée en string"""
        assert safe_string("hello") == "hello"
        assert safe_string(123) == "123"
        assert safe_string(None) == ""
        assert safe_string(True) == "True"
        assert safe_string([1, 2, 3]) == "[1, 2, 3]"
    
    def test_safe_string_with_default(self):
        """Test conversion avec valeur par défaut"""
        assert safe_string(None, default="default") == "default"
        assert safe_string("", default="default") == "default"
        assert safe_string("   ", default="default") == "default"
    
    def test_safe_string_encoding(self):
        """Test conversion avec encodage"""
        unicode_text = "Café à Paris 🎵"
        result = safe_string(unicode_text)
        assert result == unicode_text
        assert isinstance(result, str)
    
    def test_truncate_string_basic(self):
        """Test troncature basique"""
        text = "This is a long text that needs truncation"
        result = truncate_string(text, 20)
        
        assert len(result) <= 20
        assert result.endswith("...")
        assert "This is a long" in result
    
    def test_truncate_string_no_truncation(self):
        """Test sans troncature nécessaire"""
        text = "Short text"
        result = truncate_string(text, 20)
        
        assert result == text
        assert not result.endswith("...")
    
    def test_truncate_string_word_boundary(self):
        """Test troncature sur limite de mot"""
        text = "This is a long text that needs truncation"
        result = truncate_string(text, 20, preserve_words=True)
        
        assert len(result) <= 20
        assert not result.endswith(" ")
        assert result.endswith("...")
    
    def test_truncate_string_custom_suffix(self):
        """Test troncature avec suffixe personnalisé"""
        text = "This is a long text"
        result = truncate_string(text, 10, suffix=" [...]")
        
        assert result.endswith(" [...]")
    
    def test_slugify_basic(self):
        """Test slugification basique"""
        assert slugify("Hello World") == "hello-world"
        assert slugify("Café à Paris") == "cafe-a-paris"
        assert slugify("Test! @#$ %^& *()") == "test"
    
    def test_slugify_special_cases(self):
        """Test slugification cas spéciaux"""
        assert slugify("") == ""
        assert slugify("   ") == ""
        assert slugify("123") == "123"
        assert slugify("Hello_World") == "hello-world"
        assert slugify("a" * 100) == "a" * 50  # Limite par défaut
    
    def test_slugify_custom_separator(self):
        """Test slugification avec séparateur personnalisé"""
        assert slugify("Hello World", separator="_") == "hello_world"
        assert slugify("Test Multiple Words", separator=".") == "test.multiple.words"
    
    def test_extract_keywords_basic(self):
        """Test extraction mots-clés basique"""
        text = "Python is a great programming language for web development"
        keywords = extract_keywords(text, count=3)
        
        assert isinstance(keywords, list)
        assert len(keywords) <= 3
        assert all(isinstance(word, str) for word in keywords)
    
    def test_extract_keywords_with_stopwords(self):
        """Test extraction avec mots vides"""
        text = "The quick brown fox jumps over the lazy dog"
        keywords = extract_keywords(text, remove_stopwords=True)
        
        # Les mots vides ne doivent pas être inclus
        stopwords = {'the', 'over', 'a', 'an', 'and', 'or', 'but'}
        for word in keywords:
            assert word.lower() not in stopwords
    
    def test_extract_keywords_min_length(self):
        """Test extraction avec longueur minimale"""
        text = "I am a big fan of AI and ML"
        keywords = extract_keywords(text, min_length=3)
        
        for word in keywords:
            assert len(word) >= 3
    
    @security_test
    def test_mask_sensitive_data_email(self):
        """Test masquage données sensibles - email"""
        text = "Contact us at john.doe@example.com for support"
        result = mask_sensitive_data(text, data_type='email')
        
        assert "john.doe@example.com" not in result
        assert "j***@example.com" in result or "***@example.com" in result
    
    @security_test
    def test_mask_sensitive_data_phone(self):
        """Test masquage données sensibles - téléphone"""
        text = "Call us at +33 1 23 45 67 89"
        result = mask_sensitive_data(text, data_type='phone')
        
        assert "+33 1 23 45 67 89" not in result
        assert "***" in result
    
    @security_test
    def test_mask_sensitive_data_credit_card(self):
        """Test masquage données sensibles - carte de crédit"""
        text = "Card number: 4532-1234-5678-9012"
        result = mask_sensitive_data(text, data_type='credit_card')
        
        assert "4532-1234-5678-9012" not in result
        assert "****-****-****-9012" in result or "****" in result
    
    @security_test
    def test_mask_sensitive_data_custom_pattern(self):
        """Test masquage avec motif personnalisé"""
        text = "User ID: USR123456789"
        pattern = r'USR\d+'
        result = mask_sensitive_data(text, pattern=pattern)
        
        assert "USR123456789" not in result
        assert "USR***" in result or "***" in result
    
    def test_generate_random_string_basic(self):
        """Test génération string aléatoire basique"""
        result = generate_random_string(10)
        
        assert len(result) == 10
        assert isinstance(result, str)
    
    def test_generate_random_string_charset(self):
        """Test génération avec jeu de caractères"""
        # Lettres seulement
        result = generate_random_string(10, charset='letters')
        assert all(c.isalpha() for c in result)
        
        # Chiffres seulement
        result = generate_random_string(10, charset='digits')
        assert all(c.isdigit() for c in result)
        
        # Alphanumerique
        result = generate_random_string(10, charset='alphanumeric')
        assert all(c.isalnum() for c in result)
    
    def test_generate_random_string_secure(self):
        """Test génération sécurisée"""
        result1 = generate_random_string(20, secure=True)
        result2 = generate_random_string(20, secure=True)
        
        assert result1 != result2  # Doit être différent
        assert len(result1) == 20
        assert len(result2) == 20
    
    def test_is_valid_slug_valid(self):
        """Test validation slug valide"""
        assert is_valid_slug("hello-world") is True
        assert is_valid_slug("test123") is True
        assert is_valid_slug("a-very-long-slug-name") is True
        assert is_valid_slug("slug_with_underscores") is True
    
    def test_is_valid_slug_invalid(self):
        """Test validation slug invalide"""
        assert is_valid_slug("Hello World") is False  # Espaces
        assert is_valid_slug("test@slug") is False     # Caractères spéciaux
        assert is_valid_slug("") is False              # Vide
        assert is_valid_slug("-start") is False        # Commence par -
        assert is_valid_slug("end-") is False          # Finit par -
    
    def test_normalize_whitespace_basic(self):
        """Test normalisation espaces basique"""
        text = "  Hello    world  \n\n  test  "
        result = normalize_whitespace(text)
        
        assert result == "Hello world test"
    
    def test_normalize_whitespace_preserve_lines(self):
        """Test normalisation en préservant les lignes"""
        text = "Line 1\n\n\nLine 2\n   \nLine 3"
        result = normalize_whitespace(text, preserve_lines=True)
        
        assert "Line 1\nLine 2\nLine 3" in result
        assert "\n\n\n" not in result
    
    def test_clean_text_basic(self):
        """Test nettoyage texte basique"""
        text = "  Hello, World!  \n\n Remove extra spaces.  "
        result = clean_text(text)
        
        assert result == "Hello, World! Remove extra spaces."
    
    def test_clean_text_remove_punctuation(self):
        """Test nettoyage avec suppression ponctuation"""
        text = "Hello, World! How are you? Fine."
        result = clean_text(text, remove_punctuation=True)
        
        assert "," not in result
        assert "!" not in result
        assert "?" not in result
        assert "." not in result
        assert "Hello World How are you Fine" in result
    
    def test_clean_text_lowercase(self):
        """Test nettoyage en minuscules"""
        text = "HELLO World"
        result = clean_text(text, lowercase=True)
        
        assert result == "hello world"
    
    def test_extract_emails_basic(self):
        """Test extraction emails basique"""
        text = "Contact john@example.com or jane.doe@company.org for help"
        emails = extract_emails(text)
        
        assert "john@example.com" in emails
        assert "jane.doe@company.org" in emails
        assert len(emails) == 2
    
    def test_extract_emails_complex(self):
        """Test extraction emails complexes"""
        text = """
        Email us at support@company.com
        CEO: ceo+newsletter@startup.io
        Invalid: not.an.email
        Also: test.email@subdomain.example.co.uk
        """
        emails = extract_emails(text)
        
        assert "support@company.com" in emails
        assert "ceo+newsletter@startup.io" in emails
        assert "test.email@subdomain.example.co.uk" in emails
        assert "not.an.email" not in emails
    
    def test_extract_urls_basic(self):
        """Test extraction URLs basique"""
        text = "Visit https://example.com or http://test.org"
        urls = extract_urls(text)
        
        assert "https://example.com" in urls
        assert "http://test.org" in urls
        assert len(urls) == 2
    
    def test_extract_urls_complex(self):
        """Test extraction URLs complexes"""
        text = """
        Visit: https://subdomain.example.com/path?param=value
        FTP: ftp://files.example.com/download
        Secure: https://secure.bank.com:8443/login
        """
        urls = extract_urls(text)
        
        assert len(urls) >= 3
        assert any("subdomain.example.com" in url for url in urls)
        assert any("ftp://" in url for url in urls)
        assert any(":8443" in url for url in urls)
    
    def test_camel_to_snake_basic(self):
        """Test conversion camelCase vers snake_case"""
        assert camel_to_snake("camelCase") == "camel_case"
        assert camel_to_snake("CamelCase") == "camel_case"
        assert camel_to_snake("XMLHttpRequest") == "xmlhttp_request"
        assert camel_to_snake("iPhone") == "i_phone"
    
    def test_camel_to_snake_edge_cases(self):
        """Test conversion cas limites"""
        assert camel_to_snake("") == ""
        assert camel_to_snake("a") == "a"
        assert camel_to_snake("A") == "a"
        assert camel_to_snake("already_snake") == "already_snake"
    
    def test_snake_to_camel_basic(self):
        """Test conversion snake_case vers camelCase"""
        assert snake_to_camel("snake_case") == "snakeCase"
        assert snake_to_camel("long_variable_name") == "longVariableName"
        assert snake_to_camel("single") == "single"
    
    def test_snake_to_camel_pascal_case(self):
        """Test conversion vers PascalCase"""
        assert snake_to_camel("snake_case", pascal_case=True) == "SnakeCase"
        assert snake_to_camel("variable_name", pascal_case=True) == "VariableName"
    
    def test_title_case_basic(self):
        """Test conversion en titre"""
        assert title_case("hello world") == "Hello World"
        assert title_case("the quick brown fox") == "The Quick Brown Fox"
    
    def test_title_case_articles(self):
        """Test conversion avec articles"""
        # Avec respect des articles
        result = title_case("the lord of the rings", respect_articles=True)
        assert result == "The Lord of the Rings"
        
        # Sans respect des articles
        result = title_case("the lord of the rings", respect_articles=False)
        assert result == "The Lord Of The Rings"
    
    def test_remove_accents_basic(self):
        """Test suppression accents"""
        assert remove_accents("café") == "cafe"
        assert remove_accents("naïve") == "naive"
        assert remove_accents("résumé") == "resume"
        assert remove_accents("piñata") == "pinata"
    
    def test_remove_accents_mixed(self):
        """Test suppression accents texte mixte"""
        text = "Café à Paris, très joli endroit"
        result = remove_accents(text)
        assert result == "Cafe a Paris, tres joli endroit"
    
    @security_test
    def test_escape_html_basic(self):
        """Test échappement HTML"""
        assert escape_html("<script>") == "&lt;script&gt;"
        assert escape_html("AT&T") == "AT&amp;T"
        assert escape_html('"quoted"') == "&quot;quoted&quot;"
        assert escape_html("'single'") == "&#x27;single&#x27;"
    
    @security_test
    def test_escape_html_xss_protection(self):
        """Test protection XSS"""
        malicious = '<script>alert("XSS")</script>'
        escaped = escape_html(malicious)
        
        assert "<script>" not in escaped
        assert "&lt;script&gt;" in escaped
        assert "alert" in escaped  # Le contenu reste visible mais sécurisé
    
    def test_unescape_html_basic(self):
        """Test déséchappement HTML"""
        assert unescape_html("&lt;script&gt;") == "<script>"
        assert unescape_html("AT&amp;T") == "AT&T"
        assert unescape_html("&quot;quoted&quot;") == '"quoted"'
    
    def test_escape_unescape_roundtrip(self):
        """Test aller-retour échappement/déséchappement"""
        original = '<div class="test">Hello & "World"</div>'
        escaped = escape_html(original)
        unescaped = unescape_html(escaped)
        
        assert unescaped == original
    
    def test_encode_base64_basic(self):
        """Test encodage Base64"""
        assert encode_base64("hello") == "aGVsbG8="
        assert encode_base64("") == ""
        
        # Test avec données binaires
        data = b"binary data"
        encoded = encode_base64(data)
        assert isinstance(encoded, str)
    
    def test_decode_base64_basic(self):
        """Test décodage Base64"""
        assert decode_base64("aGVsbG8=") == "hello"
        assert decode_base64("") == ""
    
    def test_encode_decode_base64_roundtrip(self):
        """Test aller-retour Base64"""
        original = "Hello, World! 🎵"
        encoded = encode_base64(original)
        decoded = decode_base64(encoded)
        
        assert decoded == original
    
    def test_decode_base64_invalid(self):
        """Test décodage Base64 invalide"""
        result = decode_base64("invalid_base64")
        assert result is None  # Ou erreur selon implémentation
    
    @security_test
    def test_generate_hash_basic(self):
        """Test génération hash"""
        text = "password123"
        hash1 = generate_hash(text)
        hash2 = generate_hash(text)
        
        assert hash1 == hash2  # Même input, même hash
        assert len(hash1) > 0
        assert isinstance(hash1, str)
    
    @security_test
    def test_generate_hash_different_algorithms(self):
        """Test génération avec différents algorithmes"""
        text = "test"
        
        md5_hash = generate_hash(text, algorithm='md5')
        sha256_hash = generate_hash(text, algorithm='sha256')
        
        assert md5_hash != sha256_hash
        assert len(md5_hash) != len(sha256_hash)
    
    @security_test
    def test_generate_hash_with_salt(self):
        """Test génération avec salt"""
        text = "password"
        salt1 = "salt1"
        salt2 = "salt2"
        
        hash1 = generate_hash(text, salt=salt1)
        hash2 = generate_hash(text, salt=salt2)
        
        assert hash1 != hash2  # Différents salts, différents hashs
    
    def test_compare_strings_fuzzy_identical(self):
        """Test comparaison floue - identiques"""
        similarity = compare_strings_fuzzy("hello", "hello")
        assert similarity == 1.0
    
    def test_compare_strings_fuzzy_similar(self):
        """Test comparaison floue - similaires"""
        similarity = compare_strings_fuzzy("hello", "helo")
        assert 0.7 <= similarity < 1.0
        
        similarity = compare_strings_fuzzy("test", "tests")
        assert similarity > 0.7
    
    def test_compare_strings_fuzzy_different(self):
        """Test comparaison floue - différentes"""
        similarity = compare_strings_fuzzy("hello", "world")
        assert similarity < 0.5
        
        similarity = compare_strings_fuzzy("abc", "xyz")
        assert similarity < 0.3
    
    def test_compare_strings_fuzzy_case_insensitive(self):
        """Test comparaison floue insensible à la casse"""
        similarity = compare_strings_fuzzy("Hello", "hello", case_sensitive=False)
        assert similarity == 1.0
        
        similarity = compare_strings_fuzzy("Hello", "hello", case_sensitive=True)
        assert similarity < 1.0
    
    @performance_test
    def test_string_operations_performance(self):
        """Test performance opérations string"""
        large_text = "word " * 1000
        
        def process_text():
            cleaned = clean_text(large_text)
            slugified = slugify(cleaned)
            keywords = extract_keywords(cleaned, count=10)
            normalized = normalize_whitespace(cleaned)
            return len(cleaned + slugified + " ".join(keywords) + normalized)
        
        TestUtils.assert_performance(process_text, max_time_ms=100)
    
    @performance_test
    def test_regex_operations_performance(self):
        """Test performance opérations regex"""
        text_with_emails = ("Contact us at user{}@example.com " * 100).format(*range(100))
        
        def extract_all():
            emails = extract_emails(text_with_emails)
            urls = extract_urls(text_with_emails + " https://site{}.com".format(*range(50)))
            return len(emails + urls)
        
        TestUtils.assert_performance(extract_all, max_time_ms=200)
    
    @integration_test
    def test_text_processing_pipeline(self):
        """Test pipeline complet de traitement texte"""
        raw_text = """
        <script>alert('XSS')</script>
        Contact: john.doe+newsletter@example.com
        Visit: https://café-paris.com/path?param=value
        Phone: +33 1 23 45 67 89
        
        This    is    a    MESSY    text   with   
        extra spaces and UPPERCASE words.
        
        Keywords: Python, Machine Learning, AI
        """
        
        # Pipeline de nettoyage complet
        # 1. Sécurisation HTML
        safe_text = escape_html(raw_text)
        
        # 2. Extraction des données sensibles
        emails = extract_emails(raw_text)
        urls = extract_urls(raw_text)
        
        # 3. Masquage des données sensibles
        masked_text = mask_sensitive_data(raw_text, data_type='email')
        masked_text = mask_sensitive_data(masked_text, data_type='phone')
        
        # 4. Nettoyage du texte
        cleaned = clean_text(masked_text, remove_punctuation=False, lowercase=True)
        
        # 5. Normalisation des espaces
        normalized = normalize_whitespace(cleaned)
        
        # 6. Extraction de mots-clés
        keywords = extract_keywords(normalized, count=5, remove_stopwords=True)
        
        # 7. Génération d'un slug
        slug = slugify(" ".join(keywords))
        
        # Vérifications
        assert len(emails) > 0
        assert len(urls) > 0
        assert "john.doe+newsletter@example.com" in emails
        assert any("café-paris.com" in url for url in urls)
        assert "***" in normalized  # Données masquées
        assert len(keywords) > 0
        assert is_valid_slug(slug)


# Tests de cas limites
class TestStringUtilsEdgeCases:
    """Tests pour les cas limites et gestion d'erreurs"""
    
    def test_empty_strings(self):
        """Test avec chaînes vides"""
        assert slugify("") == ""
        assert clean_text("") == ""
        assert normalize_whitespace("") == ""
        assert extract_keywords("") == []
        assert extract_emails("") == []
        assert extract_urls("") == []
    
    def test_unicode_handling(self):
        """Test gestion Unicode"""
        unicode_text = "🎵 Café à Paris 中文 العربية 🚀"
        
        # Ne doit pas planter
        safe = safe_string(unicode_text)
        slugified = slugify(unicode_text)
        cleaned = clean_text(unicode_text)
        
        assert isinstance(safe, str)
        assert isinstance(slugified, str)
        assert isinstance(cleaned, str)
    
    def test_very_long_strings(self):
        """Test avec chaînes très longues"""
        long_text = "word " * 10000
        
        # Ne doit pas planter ni être trop lent
        result = truncate_string(long_text, 100)
        assert len(result) <= 103  # 100 + "..."
        
        keywords = extract_keywords(long_text, count=5)
        assert len(keywords) <= 5
    
    def test_special_characters(self):
        """Test avec caractères spéciaux"""
        special_text = "!@#$%^&*()_+-=[]{}|;:,.<>?"
        
        # Ne doit pas planter
        safe = safe_string(special_text)
        slugified = slugify(special_text)
        cleaned = clean_text(special_text, remove_punctuation=True)
        
        assert isinstance(safe, str)
        assert isinstance(slugified, str)
        assert isinstance(cleaned, str)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
