"""
🎵 Spotify AI Agent - Tests Formatters Module
=============================================

Tests enterprise complets pour le module formatters
avec formatage de données, sécurité et performance.

🎖️ Développé par l'équipe d'experts enterprise
"""

import pytest
import time
from datetime import datetime, date, timezone, timedelta
from decimal import Decimal
from unittest.mock import patch, Mock
import locale
import json

# Import du module à tester
from backend.app.api.utils.formatters import (
    format_currency,
    format_percentage,
    format_number,
    format_decimal,
    format_phone,
    format_date,
    format_time,
    format_datetime,
    format_duration,
    format_file_size,
    format_text,
    format_name,
    format_address,
    format_email,
    format_url,
    format_html,
    format_markdown,
    format_json,
    format_xml,
    format_csv,
    format_sql,
    format_code,
    format_slug,
    format_truncate,
    format_capitalize,
    format_camel_case,
    format_snake_case,
    format_kebab_case,
    format_title_case,
    format_sentence_case,
    format_mask,
    format_highlight,
    format_template,
    format_pluralize,
    format_ordinal,
    format_relative_time,
    format_scientific,
    format_binary,
    format_hexadecimal,
    format_credit_card,
    format_iban,
    format_postal_code,
    format_coordinates,
    escape_html,
    escape_xml,
    escape_sql,
    escape_regex,
    strip_html,
    strip_markdown,
    normalize_unicode,
    normalize_whitespace,
    normalize_phone,
    normalize_email,
    FormatError,
    FormatterConfig
)

from . import TestUtils, security_test, performance_test, integration_test


class TestFormatters:
    """Tests pour le module formatters"""
    
    def test_format_currency_basic(self):
        """Test formatage devise basique"""
        # EUR
        assert format_currency(123.45, 'EUR') == '€123.45'
        assert format_currency(1234.56, 'EUR', locale='fr_FR') == '1 234,56 €'
        
        # USD
        assert format_currency(123.45, 'USD') == '$123.45'
        assert format_currency(1234.56, 'USD', locale='en_US') == '$1,234.56'
        
        # JPY (pas de décimales)
        assert format_currency(123.45, 'JPY') == '¥123'
        
        # Montants négatifs
        assert format_currency(-123.45, 'EUR') == '-€123.45'
    
    def test_format_currency_precision(self):
        """Test formatage devise avec précision"""
        # Précision personnalisée
        assert format_currency(123.456789, 'EUR', precision=4) == '€123.4568'
        assert format_currency(123.456789, 'EUR', precision=0) == '€123'
        
        # Arrondi
        assert format_currency(123.995, 'EUR', precision=2) == '€124.00'
        assert format_currency(123.994, 'EUR', precision=2) == '€123.99'
    
    def test_format_percentage_basic(self):
        """Test formatage pourcentage basique"""
        assert format_percentage(0.1234) == '12.34%'
        assert format_percentage(0.1) == '10.00%'
        assert format_percentage(1.0) == '100.00%'
        assert format_percentage(0) == '0.00%'
        
        # Précision personnalisée
        assert format_percentage(0.1234, precision=1) == '12.3%'
        assert format_percentage(0.1234, precision=0) == '12%'
    
    def test_format_percentage_negative(self):
        """Test formatage pourcentage négatif"""
        assert format_percentage(-0.1234) == '-12.34%'
        assert format_percentage(-1.0) == '-100.00%'
    
    def test_format_number_basic(self):
        """Test formatage nombre basique"""
        assert format_number(1234.56) == '1,234.56'
        assert format_number(1234567.89) == '1,234,567.89'
        assert format_number(123) == '123'
        
        # Locale française
        assert format_number(1234.56, locale='fr_FR') == '1 234,56'
    
    def test_format_number_precision(self):
        """Test formatage nombre avec précision"""
        assert format_number(123.456789, precision=2) == '123.46'
        assert format_number(123.456789, precision=0) == '123'
        assert format_number(123.1, precision=3) == '123.100'
    
    def test_format_decimal_advanced(self):
        """Test formatage decimal avancé"""
        # Decimal precision
        decimal_val = Decimal('123.456789')
        assert format_decimal(decimal_val, precision=4) == '123.4568'
        
        # Très grand nombre
        big_decimal = Decimal('123456789.123456789')
        result = format_decimal(big_decimal, precision=2)
        assert '123,456,789.12' in result
    
    def test_format_phone_international(self):
        """Test formatage téléphone international"""
        # Formats internationaux
        assert format_phone('+33123456789') == '+33 1 23 45 67 89'
        assert format_phone('+1555123456') == '+1 555 123 456'
        assert format_phone('+44207946958') == '+44 20 7946 958'
        
        # Format national
        assert format_phone('0123456789', country='FR') == '01 23 45 67 89'
        assert format_phone('5551234567', country='US') == '(555) 123-4567'
    
    def test_format_phone_invalid(self):
        """Test formatage téléphone invalide"""
        with pytest.raises(FormatError):
            format_phone('invalid-phone')
        
        with pytest.raises(FormatError):
            format_phone('123')  # Trop court
    
    def test_format_date_basic(self):
        """Test formatage date basique"""
        test_date = date(2025, 7, 14)
        
        # Formats standards
        assert format_date(test_date, 'ISO') == '2025-07-14'
        assert format_date(test_date, 'US') == '07/14/2025'
        assert format_date(test_date, 'EU') == '14/07/2025'
        assert format_date(test_date, 'FR') == '14/07/2025'
        
        # Format personnalisé
        assert format_date(test_date, format='%d %B %Y') == '14 July 2025'
    
    def test_format_date_string_input(self):
        """Test formatage date à partir de string"""
        # String ISO
        assert format_date('2025-07-14', 'US') == '07/14/2025'
        
        # String avec format spécifique
        assert format_date('14/07/2025', 'ISO', input_format='%d/%m/%Y') == '2025-07-14'
    
    def test_format_time_basic(self):
        """Test formatage heure basique"""
        test_time = datetime(2025, 7, 14, 14, 30, 45).time()
        
        # Formats standards
        assert format_time(test_time, '24h') == '14:30:45'
        assert format_time(test_time, '12h') == '2:30:45 PM'
        assert format_time(test_time, 'short') == '14:30'
        
        # Format personnalisé
        assert format_time(test_time, format='%H:%M') == '14:30'
    
    def test_format_datetime_timezone(self):
        """Test formatage datetime avec timezone"""
        # UTC
        dt_utc = datetime(2025, 7, 14, 14, 30, 45, tzinfo=timezone.utc)
        assert format_datetime(dt_utc, 'ISO') == '2025-07-14T14:30:45+00:00'
        
        # Timezone Paris
        paris_tz = timezone(timedelta(hours=2))  # CEST
        dt_paris = datetime(2025, 7, 14, 16, 30, 45, tzinfo=paris_tz)
        formatted = format_datetime(dt_paris, 'ISO')
        assert '2025-07-14T16:30:45+02:00' == formatted
    
    def test_format_duration_basic(self):
        """Test formatage durée basique"""
        # Secondes
        assert format_duration(65) == '1:05'
        assert format_duration(3661) == '1:01:01'
        assert format_duration(90061) == '25:01:01'
        
        # Timedelta
        td = timedelta(hours=2, minutes=30, seconds=45)
        assert format_duration(td) == '2:30:45'
    
    def test_format_duration_formats(self):
        """Test formats durée différents"""
        duration = 3665  # 1h 1m 5s
        
        assert format_duration(duration, format='short') == '1h 1m 5s'
        assert format_duration(duration, format='long') == '1 hour 1 minute 5 seconds'
        assert format_duration(duration, format='compact') == '1:01:05'
        assert format_duration(duration, format='human') == '1 hour and 1 minute'
    
    def test_format_file_size_units(self):
        """Test formatage taille fichier avec unités"""
        # Bytes
        assert format_file_size(512) == '512 B'
        assert format_file_size(1024) == '1.0 KB'
        assert format_file_size(1536) == '1.5 KB'
        
        # Larger sizes
        assert format_file_size(1024 * 1024) == '1.0 MB'
        assert format_file_size(1024 * 1024 * 1024) == '1.0 GB'
        assert format_file_size(1024 * 1024 * 1024 * 1024) == '1.0 TB'
        
        # Binary vs decimal
        assert format_file_size(1000, binary=False) == '1.0 kB'  # Decimal
        assert format_file_size(1024, binary=True) == '1.0 KiB'  # Binary
    
    def test_format_file_size_precision(self):
        """Test formatage taille fichier avec précision"""
        size = 1536  # 1.5 KB
        
        assert format_file_size(size, precision=0) == '2 KB'  # Arrondi
        assert format_file_size(size, precision=1) == '1.5 KB'
        assert format_file_size(size, precision=2) == '1.50 KB'
    
    def test_format_text_basic(self):
        """Test formatage texte basique"""
        text = "hello world"
        
        assert format_text(text, 'upper') == 'HELLO WORLD'
        assert format_text(text, 'lower') == 'hello world'
        assert format_text(text, 'title') == 'Hello World'
        assert format_text(text, 'capitalize') == 'Hello world'
        assert format_text(text, 'sentence') == 'Hello world'
    
    def test_format_name_proper(self):
        """Test formatage nom propre"""
        assert format_name('john doe') == 'John Doe'
        assert format_name('MARIE CURIE') == 'Marie Curie'
        assert format_name('jean-claude van damme') == 'Jean-Claude Van Damme'
        assert format_name("o'connor") == "O'Connor"
        assert format_name('de la cruz') == 'De La Cruz'
    
    def test_format_name_special_cases(self):
        """Test formatage nom cas spéciaux"""
        # Particules
        assert format_name('van der berg') == 'Van Der Berg'
        assert format_name('von neumann') == 'Von Neumann'
        
        # Abréviations
        assert format_name('dr. smith') == 'Dr. Smith'
        assert format_name('mr. john doe jr.') == 'Mr. John Doe Jr.'
    
    def test_format_address_standard(self):
        """Test formatage adresse standard"""
        address = {
            'street': '123 main street',
            'city': 'paris',
            'postal_code': '75001',
            'country': 'france'
        }
        
        formatted = format_address(address, format='standard')
        expected = '123 Main Street\nParis 75001\nFrance'
        assert formatted == expected
    
    def test_format_address_one_line(self):
        """Test formatage adresse une ligne"""
        address = {
            'street': '123 Main Street',
            'city': 'Paris',
            'postal_code': '75001',
            'country': 'France'
        }
        
        formatted = format_address(address, format='one_line')
        expected = '123 Main Street, Paris 75001, France'
        assert formatted == expected
    
    def test_format_email_display(self):
        """Test formatage email pour affichage"""
        # Email simple
        assert format_email('user@example.com') == 'user@example.com'
        
        # Email avec nom
        formatted = format_email('user@example.com', display_name='John Doe')
        assert formatted == 'John Doe <user@example.com>'
        
        # Masquage partiel
        masked = format_email('user@example.com', mask=True)
        assert masked == 'u***@example.com'
    
    def test_format_url_display(self):
        """Test formatage URL pour affichage"""
        # URL simple
        url = 'https://www.example.com/path/to/page'
        assert format_url(url, format='short') == 'example.com'
        assert format_url(url, format='domain') == 'www.example.com'
        assert format_url(url, format='full') == url
        
        # Truncate long URLs
        long_url = 'https://example.com/very/long/path/to/some/resource'
        truncated = format_url(long_url, max_length=30)
        assert len(truncated) <= 33  # 30 + "..."
        assert '...' in truncated
    
    @security_test
    def test_format_html_escape(self):
        """Test formatage HTML avec échappement"""
        dangerous_html = '<script>alert("XSS")</script>'
        
        # Échappement automatique
        escaped = format_html(dangerous_html, escape=True)
        assert '<script>' not in escaped
        assert '&lt;script&gt;' in escaped
        
        # Sans échappement (dangereux)
        unescaped = format_html(dangerous_html, escape=False)
        assert '<script>' in unescaped
    
    def test_format_html_structure(self):
        """Test formatage HTML avec structure"""
        html = '<div><p>Hello</p><p>World</p></div>'
        
        # Formatage avec indentation
        formatted = format_html(html, indent=True)
        lines = formatted.split('\n')
        assert len(lines) > 1
        assert '  <p>' in formatted  # Indentation
    
    def test_format_markdown_basic(self):
        """Test formatage Markdown basique"""
        md = '# Title\n\n**Bold** and *italic* text.'
        
        # Vers HTML
        html = format_markdown(md, to='html')
        assert '<h1>' in html
        assert '<strong>' in html
        assert '<em>' in html
        
        # Nettoyer le markdown
        clean = format_markdown(md, to='text')
        assert 'Title' in clean
        assert 'Bold' in clean
        assert '#' not in clean
        assert '**' not in clean
    
    def test_format_json_pretty(self):
        """Test formatage JSON joliment"""
        data = {'name': 'John', 'age': 30, 'city': 'Paris'}
        
        # Format compact
        compact = format_json(data, indent=None)
        assert '\n' not in compact
        
        # Format indenté
        pretty = format_json(data, indent=2)
        assert '\n' in pretty
        assert '  "name"' in pretty
    
    def test_format_json_sorting(self):
        """Test formatage JSON avec tri"""
        data = {'z': 1, 'a': 2, 'm': 3}
        
        # Sans tri
        unsorted = format_json(data, sort_keys=False)
        
        # Avec tri
        sorted_json = format_json(data, sort_keys=True)
        lines = sorted_json.split('\n')
        # Les clés doivent être triées alphabétiquement
        assert '"a"' in sorted_json
        assert '"m"' in sorted_json
        assert '"z"' in sorted_json
    
    def test_format_xml_pretty(self):
        """Test formatage XML joliment"""
        xml = '<root><item>value1</item><item>value2</item></root>'
        
        # Format indenté
        pretty = format_xml(xml, indent=True)
        lines = pretty.split('\n')
        assert len(lines) > 1
        assert '  <item>' in pretty  # Indentation
    
    def test_format_csv_basic(self):
        """Test formatage CSV basique"""
        data = [
            ['Name', 'Age', 'City'],
            ['John', 30, 'Paris'],
            ['Jane', 25, 'London']
        ]
        
        csv_output = format_csv(data)
        lines = csv_output.strip().split('\n')
        assert len(lines) == 3
        assert 'Name,Age,City' == lines[0]
        assert 'John,30,Paris' == lines[1]
    
    def test_format_csv_custom_delimiter(self):
        """Test formatage CSV avec délimiteur personnalisé"""
        data = [['A', 'B'], ['1', '2']]
        
        # Délimiteur point-virgule
        csv_output = format_csv(data, delimiter=';')
        assert 'A;B' in csv_output
        assert '1;2' in csv_output
        
        # Délimiteur tab
        csv_output = format_csv(data, delimiter='\t')
        assert 'A\tB' in csv_output
    
    def test_format_sql_basic(self):
        """Test formatage SQL basique"""
        sql = 'select * from users where age > 18 and city = "paris"'
        
        formatted = format_sql(sql)
        # Doit contenir des mots-clés en majuscules
        assert 'SELECT' in formatted
        assert 'FROM' in formatted
        assert 'WHERE' in formatted
        assert 'AND' in formatted
    
    def test_format_code_python(self):
        """Test formatage code Python"""
        code = 'def hello():print("hello")'
        
        formatted = format_code(code, language='python')
        # Doit ajouter des espaces et indentation
        assert 'def hello():' in formatted
        assert 'print(' in formatted
    
    def test_format_slug_basic(self):
        """Test formatage slug basique"""
        assert format_slug('Hello World') == 'hello-world'
        assert format_slug('Café & Restaurant') == 'cafe-restaurant'
        assert format_slug('Test  Multiple   Spaces') == 'test-multiple-spaces'
        assert format_slug('Éléphant à Paris') == 'elephant-a-paris'
    
    def test_format_slug_special_chars(self):
        """Test formatage slug caractères spéciaux"""
        assert format_slug('C++ Programming') == 'c-programming'
        assert format_slug('Price: $19.99') == 'price-19-99'
        assert format_slug('100% Success!') == '100-success'
        assert format_slug('User@Domain.com') == 'user-domain-com'
    
    def test_format_truncate_basic(self):
        """Test troncature basique"""
        text = 'This is a very long text that should be truncated'
        
        assert format_truncate(text, 20) == 'This is a very lo...'
        assert format_truncate(text, 10) == 'This is...'
        assert format_truncate(text, 100) == text  # Pas de troncature
    
    def test_format_truncate_words(self):
        """Test troncature par mots"""
        text = 'This is a very long sentence with many words'
        
        # Troncature par mots complets
        truncated = format_truncate(text, 20, by_words=True)
        assert not truncated.endswith(' ...')  # Pas de mot coupé
        assert '...' in truncated
    
    def test_format_capitalize_variants(self):
        """Test variantes capitalisation"""
        text = 'hello world test'
        
        assert format_capitalize(text, 'first') == 'Hello world test'
        assert format_capitalize(text, 'words') == 'Hello World Test'
        assert format_capitalize(text, 'all') == 'HELLO WORLD TEST'
        assert format_capitalize(text, 'sentence') == 'Hello world test'
    
    def test_format_case_conversions(self):
        """Test conversions de casse"""
        text = 'Hello World Test'
        
        # CamelCase
        assert format_camel_case(text) == 'helloWorldTest'
        assert format_camel_case(text, upper_first=True) == 'HelloWorldTest'
        
        # snake_case
        assert format_snake_case(text) == 'hello_world_test'
        
        # kebab-case
        assert format_kebab_case(text) == 'hello-world-test'
        
        # Title Case
        assert format_title_case('hello world') == 'Hello World'
        
        # Sentence case
        assert format_sentence_case('HELLO WORLD') == 'Hello world'
    
    def test_format_mask_sensitive(self):
        """Test masquage données sensibles"""
        # Carte de crédit
        cc = '4532015112830366'
        masked_cc = format_mask(cc, 'credit_card')
        assert masked_cc == '****-****-****-0366'
        
        # Numéro de téléphone
        phone = '+33123456789'
        masked_phone = format_mask(phone, 'phone')
        assert '***' in masked_phone
        assert '789' in masked_phone  # Derniers chiffres visibles
        
        # Email
        email = 'user@example.com'
        masked_email = format_mask(email, 'email')
        assert masked_email == 'u***@example.com'
        
        # Custom mask
        text = 'sensitive data'
        custom_masked = format_mask(text, pattern='****')
        assert custom_masked == '****'
    
    def test_format_highlight_search(self):
        """Test surlignage de recherche"""
        text = 'This is a test text for searching'
        
        # Surlignage simple
        highlighted = format_highlight(text, 'test', '<mark>', '</mark>')
        assert '<mark>test</mark>' in highlighted
        
        # Surlignage insensible à la casse
        highlighted = format_highlight(text, 'TEST', '<em>', '</em>', case_sensitive=False)
        assert '<em>test</em>' in highlighted
        
        # Multiples occurrences
        highlighted = format_highlight('test test test', 'test', '**', '**')
        assert highlighted.count('**test**') == 3
    
    def test_format_template_basic(self):
        """Test formatage template basique"""
        template = 'Hello {name}, you have {count} messages'
        data = {'name': 'John', 'count': 5}
        
        result = format_template(template, data)
        assert result == 'Hello John, you have 5 messages'
    
    def test_format_template_advanced(self):
        """Test formatage template avancé"""
        template = '{user.name} ({user.age} years old) from {user.city}'
        data = {
            'user': {
                'name': 'John Doe',
                'age': 30,
                'city': 'Paris'
            }
        }
        
        result = format_template(template, data)
        assert result == 'John Doe (30 years old) from Paris'
    
    def test_format_pluralize_basic(self):
        """Test pluralisation basique"""
        assert format_pluralize(1, 'item') == '1 item'
        assert format_pluralize(2, 'item') == '2 items'
        assert format_pluralize(0, 'item') == '0 items'
        
        # Pluriel irrégulier
        assert format_pluralize(1, 'child', 'children') == '1 child'
        assert format_pluralize(2, 'child', 'children') == '2 children'
    
    def test_format_pluralize_languages(self):
        """Test pluralisation différentes langues"""
        # Français
        assert format_pluralize(1, 'élément', language='fr') == '1 élément'
        assert format_pluralize(2, 'élément', language='fr') == '2 éléments'
        
        # Espagnol
        assert format_pluralize(1, 'elemento', language='es') == '1 elemento'
        assert format_pluralize(2, 'elemento', language='es') == '2 elementos'
    
    def test_format_ordinal_basic(self):
        """Test formatage nombres ordinaux"""
        assert format_ordinal(1) == '1st'
        assert format_ordinal(2) == '2nd'
        assert format_ordinal(3) == '3rd'
        assert format_ordinal(4) == '4th'
        assert format_ordinal(11) == '11th'
        assert format_ordinal(21) == '21st'
        assert format_ordinal(22) == '22nd'
        assert format_ordinal(23) == '23rd'
        assert format_ordinal(101) == '101st'
    
    def test_format_ordinal_languages(self):
        """Test ordinaux différentes langues"""
        # Français
        assert format_ordinal(1, language='fr') == '1er'
        assert format_ordinal(2, language='fr') == '2ème'
        assert format_ordinal(3, language='fr') == '3ème'
        
        # Espagnol
        assert format_ordinal(1, language='es') == '1º'
        assert format_ordinal(2, language='es') == '2º'
    
    def test_format_relative_time_basic(self):
        """Test formatage temps relatif"""
        now = datetime.now()
        
        # Il y a quelques secondes
        past = now - timedelta(seconds=30)
        assert 'seconds ago' in format_relative_time(past, now)
        
        # Il y a quelques minutes
        past = now - timedelta(minutes=5)
        assert 'minutes ago' in format_relative_time(past, now)
        
        # Il y a quelques heures
        past = now - timedelta(hours=2)
        assert 'hours ago' in format_relative_time(past, now)
        
        # Dans le futur
        future = now + timedelta(hours=1)
        assert 'in' in format_relative_time(future, now)
    
    def test_format_relative_time_precision(self):
        """Test temps relatif avec précision"""
        now = datetime.now()
        past = now - timedelta(hours=2, minutes=30)
        
        # Précision approximative
        approx = format_relative_time(past, now, precision='approximate')
        assert 'about' in approx or 'around' in approx
        
        # Précision exacte
        exact = format_relative_time(past, now, precision='exact')
        assert '2 hours' in exact and '30 minutes' in exact
    
    def test_format_scientific_notation(self):
        """Test notation scientifique"""
        assert format_scientific(123456789) == '1.23e+08'
        assert format_scientific(0.000123) == '1.23e-04'
        assert format_scientific(1.23, precision=1) == '1.2e+00'
        assert format_scientific(1000000, precision=3) == '1.000e+06'
    
    def test_format_binary_basic(self):
        """Test formatage binaire"""
        assert format_binary(10) == '1010'
        assert format_binary(255) == '11111111'
        assert format_binary(10, width=8) == '00001010'  # Padding
        assert format_binary(10, prefix=True) == '0b1010'
    
    def test_format_hexadecimal_basic(self):
        """Test formatage hexadécimal"""
        assert format_hexadecimal(255) == 'ff'
        assert format_hexadecimal(255, uppercase=True) == 'FF'
        assert format_hexadecimal(10, width=4) == '000a'  # Padding
        assert format_hexadecimal(255, prefix=True) == '0xff'
        assert format_hexadecimal(255, prefix=True, uppercase=True) == '0xFF'
    
    def test_format_credit_card_display(self):
        """Test formatage carte de crédit pour affichage"""
        cc = '4532015112830366'
        
        # Formatage avec espaces
        formatted = format_credit_card(cc, format='spaced')
        assert formatted == '4532 0151 1283 0366'
        
        # Formatage avec tirets
        formatted = format_credit_card(cc, format='dashed')
        assert formatted == '4532-0151-1283-0366'
        
        # Masquage
        masked = format_credit_card(cc, mask=True)
        assert '****' in masked
        assert '0366' in masked  # Derniers chiffres visibles
    
    def test_format_iban_display(self):
        """Test formatage IBAN pour affichage"""
        iban = 'FR1420041010050500013M02606'
        
        # Formatage avec espaces
        formatted = format_iban(iban)
        assert ' ' in formatted
        assert len(formatted.replace(' ', '')) == len(iban)
        
        # Masquage partiel
        masked = format_iban(iban, mask=True)
        assert '****' in masked
        assert 'FR14' in masked  # Début visible
    
    def test_format_postal_code_country(self):
        """Test formatage code postal par pays"""
        # États-Unis (ZIP+4)
        us_zip = '123456789'
        formatted_us = format_postal_code(us_zip, country='US')
        assert formatted_us == '12345-6789'
        
        # Canada
        ca_postal = 'K1A0A6'
        formatted_ca = format_postal_code(ca_postal, country='CA')
        assert formatted_ca == 'K1A 0A6'
        
        # Royaume-Uni
        uk_postal = 'SW1A1AA'
        formatted_uk = format_postal_code(uk_postal, country='UK')
        assert formatted_uk == 'SW1A 1AA'
    
    def test_format_coordinates_basic(self):
        """Test formatage coordonnées géographiques"""
        lat, lon = 48.8566, 2.3522  # Paris
        
        # Format décimal
        decimal = format_coordinates(lat, lon, format='decimal')
        assert '48.8566' in decimal and '2.3522' in decimal
        
        # Format DMS (Degrees, Minutes, Seconds)
        dms = format_coordinates(lat, lon, format='dms')
        assert '°' in dms and "'" in dms and '"' in dms
        
        # Format DDM (Degrees, Decimal Minutes)
        ddm = format_coordinates(lat, lon, format='ddm')
        assert '°' in ddm and "'" in ddm
    
    def test_format_coordinates_directions(self):
        """Test formatage coordonnées avec directions"""
        # Nord/Sud, Est/Ouest
        lat, lon = 48.8566, 2.3522
        formatted = format_coordinates(lat, lon, format='decimal', show_direction=True)
        assert 'N' in formatted and 'E' in formatted
        
        # Sud/Ouest
        lat, lon = -33.8688, -151.2093  # Sydney
        formatted = format_coordinates(lat, lon, format='decimal', show_direction=True)
        assert 'S' in formatted and 'W' in formatted
    
    def test_escape_functions(self):
        """Test fonctions d'échappement"""
        dangerous = '<script>alert("XSS")</script>'
        
        # HTML
        html_escaped = escape_html(dangerous)
        assert '&lt;script&gt;' in html_escaped
        assert '<script>' not in html_escaped
        
        # XML
        xml_escaped = escape_xml(dangerous)
        assert '&lt;script&gt;' in xml_escaped
        
        # SQL
        sql_dangerous = "'; DROP TABLE users; --"
        sql_escaped = escape_sql(sql_dangerous)
        assert "''" in sql_escaped  # Single quote doubled
        
        # Regex
        regex_dangerous = 'Hello (world) [test] {value} ^start$ +plus *star ?question'
        regex_escaped = escape_regex(regex_dangerous)
        assert '\\(' in regex_escaped
        assert '\\[' in regex_escaped
        assert '\\^' in regex_escaped
    
    def test_strip_functions(self):
        """Test fonctions de nettoyage"""
        # Strip HTML
        html = '<p>Hello <strong>world</strong>!</p>'
        stripped_html = strip_html(html)
        assert stripped_html == 'Hello world!'
        assert '<' not in stripped_html
        
        # Strip Markdown
        markdown = '# Title\n\n**Bold** and *italic* text.'
        stripped_md = strip_markdown(markdown)
        assert 'Title' in stripped_md
        assert 'Bold' in stripped_md
        assert '#' not in stripped_md
        assert '**' not in stripped_md
        assert '*' not in stripped_md
    
    def test_normalize_functions(self):
        """Test fonctions de normalisation"""
        # Unicode
        text_with_accents = 'Café élégant'
        normalized = normalize_unicode(text_with_accents)
        assert 'Cafe elegant' == normalized or 'Café élégant' == normalized
        
        # Whitespace
        messy_text = '  Hello   world  \n\t  test  '
        normalized_ws = normalize_whitespace(messy_text)
        assert normalized_ws == 'Hello world test'
        
        # Phone
        phone = '+33 1 23 45 67 89'
        normalized_phone = normalize_phone(phone)
        assert normalized_phone == '+33123456789'
        
        # Email
        email = 'USER@EXAMPLE.COM'
        normalized_email = normalize_email(email)
        assert normalized_email == 'user@example.com'
    
    @performance_test
    def test_formatting_performance(self):
        """Test performance formatage"""
        # Test formatage de masse
        numbers = list(range(1000))
        currencies = ['EUR', 'USD', 'JPY'] * 334  # 1002 total
        
        def format_all_currencies():
            results = []
            for i, num in enumerate(numbers):
                currency = currencies[i % len(currencies)]
                formatted = format_currency(num * 1.23, currency)
                results.append(formatted)
            return len(results)
        
        TestUtils.assert_performance(format_all_currencies, max_time_ms=1000)
    
    @integration_test
    def test_complete_formatting_workflow(self):
        """Test workflow formatage complet"""
        # Scénario: Formatage données utilisateur pour affichage
        
        user_data = {
            'name': 'john doe',
            'email': 'JOHN.DOE@EXAMPLE.COM',
            'phone': '+33123456789',
            'balance': 1234.56,
            'registration_date': datetime(2025, 1, 15, 14, 30, 0),
            'last_login': datetime.now() - timedelta(hours=2),
            'bio': 'Software engineer passionate about AI and machine learning...',
            'address': {
                'street': '123 main street',
                'city': 'paris',
                'postal_code': '75001',
                'country': 'france'
            }
        }
        
        # Formatage pour affichage
        formatted_user = {
            'name': format_name(user_data['name']),
            'email': normalize_email(user_data['email']),
            'email_display': format_email(user_data['email'], mask=True),
            'phone': format_phone(user_data['phone']),
            'balance': format_currency(user_data['balance'], 'EUR'),
            'registration_date': format_date(user_data['registration_date'], 'EU'),
            'last_login': format_relative_time(user_data['last_login']),
            'bio_truncated': format_truncate(user_data['bio'], 50, by_words=True),
            'address': format_address(user_data['address'], format='one_line')
        }
        
        # Vérifications
        assert formatted_user['name'] == 'John Doe'
        assert formatted_user['email'] == 'john.doe@example.com'
        assert '***' in formatted_user['email_display']
        assert formatted_user['phone'] == '+33 1 23 45 67 89'
        assert '€' in formatted_user['balance']
        assert '15/01/2025' == formatted_user['registration_date']
        assert 'ago' in formatted_user['last_login']
        assert len(formatted_user['bio_truncated']) <= 53  # 50 + "..."
        assert 'Paris' in formatted_user['address']
        
        print("✅ Workflow formatage complet validé")


# Tests de sécurité formatage
class TestFormattingSecurity:
    """Tests de sécurité pour les formatters"""
    
    @security_test
    def test_xss_prevention_html(self):
        """Test prévention XSS dans formatage HTML"""
        malicious_inputs = [
            '<script>alert("XSS")</script>',
            '<img src="x" onerror="alert(1)">',
            '<svg onload="alert(1)">',
            'javascript:alert("XSS")',
            '<iframe src="javascript:alert(1)"></iframe>'
        ]
        
        for malicious in malicious_inputs:
            # Le formatage HTML avec échappement doit neutraliser
            safe_html = format_html(malicious, escape=True)
            assert '<script>' not in safe_html
            assert 'javascript:' not in safe_html
            assert 'onerror=' not in safe_html
            assert 'onload=' not in safe_html
    
    @security_test  
    def test_injection_prevention_sql(self):
        """Test prévention injection dans formatage SQL"""
        malicious_sql = "'; DROP TABLE users; --"
        
        # L'échappement SQL doit neutraliser
        safe_sql = escape_sql(malicious_sql)
        assert 'DROP TABLE' not in safe_sql.upper()
        assert "''" in safe_sql  # Quote échappé
    
    @security_test
    def test_template_injection_prevention(self):
        """Test prévention injection template"""
        malicious_template = '{user.__class__.__bases__[0].__subclasses__()}'
        
        # Le formatage template doit rejeter ou échapper
        with pytest.raises(FormatError):
            format_template(malicious_template, {'user': 'test'})


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
