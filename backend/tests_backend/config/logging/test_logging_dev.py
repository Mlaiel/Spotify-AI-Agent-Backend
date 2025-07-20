from unittest.mock import Mock

import configparser
import os
import pytest
from pathlib import Path

CONF_PATH = Path(__file__).parent.parent.parent.parent / 'config' / 'logging' / 'logging_dev.conf'

def test_logging_dev_conf_exists():
    assert CONF_PATH.exists(), f"Fichier de config logging.dev.conf introuvable: {CONF_PATH}"

def test_logging_dev_conf_parseable():
    parser = configparser.ConfigParser()
    parser.read(CONF_PATH)
    assert parser.sections(), "Le fichier logging.dev.conf doit contenir des sections."

def test_logging_dev_loggers():
    parser = configparser.ConfigParser()
    parser.read(CONF_PATH)
    loggers = parser.get('loggers', 'keys').split(',')
    for logger in ['root', 'app', 'security', 'ml']:
        assert logger in loggers, f"Logger manquant: {logger}"
        section = f'logger_{logger}'
        assert parser.has_section(section), f"Section {section} absente"
        level = parser.get(section, 'level')
        assert level in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']

def test_logging_dev_handlers():
    parser = configparser.ConfigParser()
    parser.read(CONF_PATH)
    handlers = parser.get('handlers', 'keys').split(',')
    for handler in ['consoleHandler', 'fileHandler']:
        assert handler in handlers, f"Handler manquant: {handler}"
        section = f'handler_{handler}'
        assert parser.has_section(section), f"Section {section} absente"

def test_logging_dev_formatters():
    parser = configparser.ConfigParser()
    parser.read(CONF_PATH)
    formatters = parser.get('formatters', 'keys').split(',')
    assert 'devFormatter' in formatters, "Formatter devFormatter manquant"

def test_logging_dev_handler_classes():
    parser = configparser.ConfigParser()
    parser.read(CONF_PATH)
    assert parser.get('handler_consoleHandler', 'class') == 'StreamHandler'
    assert parser.get('handler_fileHandler', 'class') == 'FileHandler' or 'class' in parser.options('handler_fileHandler')
