"""
🎨 Advanced Message Formatters - Production-Ready System
======================================================

Formateurs ultra-avancés pour conversion et optimisation de messages
vers différents canaux avec support rich content et personnalisation.
"""

import json
import re
import html
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from enum import Enum
from abc import ABC, abstractmethod
import logging

from markdown import markdown
from bs4 import BeautifulSoup
import emoji

logger = logging.getLogger(__name__)


class FormatterType(Enum):
    """Types de formateurs disponibles"""
    TEXT = "text"
    MARKDOWN = "markdown"
    HTML = "html"
    SLACK_BLOCKS = "slack_blocks"
    SLACK_MARKDOWN = "slack_markdown"
    TEAMS_CARD = "teams_card"
    DISCORD_EMBED = "discord_embed"
    EMAIL_HTML = "email_html"
    EMAIL_TEXT = "email_text"
    SMS = "sms"
    PUSH_NOTIFICATION = "push_notification"


class MessageElement(Enum):
    """Éléments de message supportés"""
    HEADER = "header"
    BODY = "body"
    FOOTER = "footer"
    BUTTON = "button"
    LINK = "link"
    IMAGE = "image"
    EMOJI = "emoji"
    MENTION = "mention"
    CODE_BLOCK = "code_block"
    QUOTE = "quote"
    LIST = "list"
    TABLE = "table"


class BaseFormatter(ABC):
    """Classe de base pour les formateurs"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Paramètres de formatage
        self.max_length = config.get('max_length', 4096)
        self.emoji_enabled = config.get('emoji_enabled', True)
        self.rich_content = config.get('rich_content', True)
        
        # Cache des éléments formatés
        self.element_cache: Dict[str, str] = {}
    
    @abstractmethod
    async def format_message(
        self, 
        content: str, 
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Formatage principal du message"""
        pass
    
    def _extract_elements(self, content: str) -> Dict[MessageElement, List[str]]:
        """Extraction des éléments du message"""
        
        elements = {element: [] for element in MessageElement}
        
        # Headers (# ## ###)
        headers = re.findall(r'^(#{1,6})\s+(.+)$', content, re.MULTILINE)
        elements[MessageElement.HEADER] = [header[1] for header in headers]
        
        # Links [text](url)
        links = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', content)
        elements[MessageElement.LINK] = [{'text': link[0], 'url': link[1]} for link in links]
        
        # Code blocks ```code```
        code_blocks = re.findall(r'```([^`]+)```', content, re.DOTALL)
        elements[MessageElement.CODE_BLOCK] = code_blocks
        
        # Quotes > text
        quotes = re.findall(r'^>\s+(.+)$', content, re.MULTILINE)
        elements[MessageElement.QUOTE] = quotes
        
        # Lists - item
        list_items = re.findall(r'^[-*]\s+(.+)$', content, re.MULTILINE)
        elements[MessageElement.LIST] = list_items
        
        # Emojis
        if self.emoji_enabled:
            emojis = emoji.emoji_list(content)
            elements[MessageElement.EMOJI] = [e['emoji'] for e in emojis]
        
        # Mentions @user
        mentions = re.findall(r'@(\w+)', content)
        elements[MessageElement.MENTION] = mentions
        
        return elements
    
    def _optimize_length(self, content: str) -> str:
        """Optimisation de la longueur du message"""
        
        if len(content) <= self.max_length:
            return content
        
        # Troncature intelligente
        truncated = content[:self.max_length - 3]
        
        # Chercher le dernier point ou espace pour une coupure propre
        last_sentence = truncated.rfind('.')
        last_space = truncated.rfind(' ')
        
        cut_point = max(last_sentence, last_space)
        if cut_point > self.max_length * 0.8:  # Au moins 80% du contenu
            truncated = truncated[:cut_point]
        
        return truncated + "..."
    
    def _sanitize_content(self, content: str) -> str:
        """Nettoyage et sanitisation du contenu"""
        
        # Suppression des caractères de contrôle
        content = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', content)
        
        # Normalisation des espaces
        content = re.sub(r'\s+', ' ', content)
        
        # Suppression des espaces en début/fin
        content = content.strip()
        
        return content


class TextFormatter(BaseFormatter):
    """Formateur pour texte brut"""
    
    async def format_message(
        self, 
        content: str, 
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Formatage en texte brut"""
        
        # Nettoyage du contenu
        clean_content = self._sanitize_content(content)
        
        # Suppression du formatage Markdown
        clean_content = self._strip_markdown(clean_content)
        
        # Optimisation de la longueur
        clean_content = self._optimize_length(clean_content)
        
        return {
            'content': clean_content,
            'content_type': 'text/plain',
            'length': len(clean_content),
            'metadata': metadata
        }
    
    def _strip_markdown(self, content: str) -> str:
        """Suppression du formatage Markdown"""
        
        # Suppression des headers
        content = re.sub(r'^#{1,6}\s+', '', content, flags=re.MULTILINE)
        
        # Suppression du formatage gras/italique
        content = re.sub(r'\*\*([^*]+)\*\*', r'\1', content)
        content = re.sub(r'\*([^*]+)\*', r'\1', content)
        
        # Suppression des liens
        content = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', content)
        
        # Suppression des code blocks
        content = re.sub(r'```[^`]*```', '', content, flags=re.DOTALL)
        content = re.sub(r'`([^`]+)`', r'\1', content)
        
        # Suppression des quotes
        content = re.sub(r'^>\s+', '', content, flags=re.MULTILINE)
        
        return content


class MarkdownFormatter(BaseFormatter):
    """Formateur pour Markdown"""
    
    async def format_message(
        self, 
        content: str, 
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Formatage en Markdown"""
        
        # Nettoyage du contenu
        clean_content = self._sanitize_content(content)
        
        # Enhancement du Markdown
        enhanced_content = await self._enhance_markdown(clean_content, metadata)
        
        # Optimisation de la longueur
        enhanced_content = self._optimize_length(enhanced_content)
        
        # Validation du Markdown
        is_valid = self._validate_markdown(enhanced_content)
        
        return {
            'content': enhanced_content,
            'content_type': 'text/markdown',
            'length': len(enhanced_content),
            'is_valid': is_valid,
            'metadata': metadata
        }
    
    async def _enhance_markdown(self, content: str, metadata: Dict[str, Any]) -> str:
        """Enhancement du contenu Markdown"""
        
        enhanced = content
        
        # Ajout d'emphasis basé sur la priorité
        priority = metadata.get('priority', 'normal')
        if priority in ['high', 'critical', 'urgent']:
            # Mise en gras des éléments importants
            enhanced = re.sub(r'\b(alert|urgent|critical|important)\b', r'**\1**', enhanced, flags=re.IGNORECASE)
        
        # Amélioration des liens
        enhanced = self._enhance_links(enhanced)
        
        # Ajout d'éléments visuels
        if self.emoji_enabled:
            enhanced = self._add_contextual_emojis(enhanced, metadata)
        
        # Structuration du contenu
        enhanced = self._structure_content(enhanced)
        
        return enhanced
    
    def _enhance_links(self, content: str) -> str:
        """Amélioration des liens"""
        
        # Conversion des URLs brutes en liens Markdown
        url_pattern = r'(https?://[^\s]+)'
        content = re.sub(url_pattern, r'[\1](\1)', content)
        
        return content
    
    def _add_contextual_emojis(self, content: str, metadata: Dict[str, Any]) -> str:
        """Ajout d'emojis contextuels"""
        
        emoji_map = {
            'welcome': '👋',
            'alert': '🚨',
            'warning': '⚠️',
            'info': 'ℹ️',
            'success': '✅',
            'error': '❌',
            'recommendation': '💡',
            'performance': '📊',
            'security': '🔒',
            'billing': '💰'
        }
        
        template_type = metadata.get('template_type', '')
        if template_type in emoji_map and not content.startswith(emoji_map[template_type]):
            content = f"{emoji_map[template_type]} {content}"
        
        return content
    
    def _structure_content(self, content: str) -> str:
        """Structuration du contenu"""
        
        lines = content.split('\n')
        structured_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Si la ligne ne commence pas par un élément Markdown, l'ajouter comme paragraphe
            if not re.match(r'^(#{1,6}|[-*]|\d+\.|\||>)', line):
                if len(structured_lines) > 0 and not structured_lines[-1].startswith('#'):
                    structured_lines.append('')  # Ligne vide entre paragraphes
            
            structured_lines.append(line)
        
        return '\n'.join(structured_lines)
    
    def _validate_markdown(self, content: str) -> bool:
        """Validation du Markdown"""
        
        try:
            # Test de parsing avec markdown
            html_output = markdown(content)
            return len(html_output) > 0
        except Exception:
            return False


class SlackBlocksFormatter(BaseFormatter):
    """Formateur pour Slack Blocks"""
    
    async def format_message(
        self, 
        content: str, 
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Formatage en Slack Blocks"""
        
        # Extraction des éléments
        elements = self._extract_elements(content)
        
        # Construction des blocks
        blocks = await self._build_slack_blocks(content, elements, metadata)
        
        # Validation des blocks
        is_valid = self._validate_slack_blocks(blocks)
        
        return {
            'blocks': blocks,
            'content_type': 'application/json',
            'is_valid': is_valid,
            'metadata': metadata
        }
    
    async def _build_slack_blocks(
        self, 
        content: str, 
        elements: Dict[MessageElement, List], 
        metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Construction des Slack Blocks"""
        
        blocks = []
        
        # Header block si présent
        if elements[MessageElement.HEADER]:
            header_text = elements[MessageElement.HEADER][0]
            blocks.append({
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": header_text[:150]  # Limite Slack
                }
            })
        
        # Section principale
        main_content = self._clean_content_for_slack(content)
        if main_content:
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": main_content[:3000]  # Limite Slack
                }
            })
        
        # Boutons d'action si spécifiés
        actions = metadata.get('actions', [])
        if actions:
            action_block = self._build_action_block(actions)
            blocks.append(action_block)
        
        # Footer avec métadonnées
        footer = self._build_footer_block(metadata)
        if footer:
            blocks.append(footer)
        
        return blocks
    
    def _clean_content_for_slack(self, content: str) -> str:
        """Nettoyage du contenu pour Slack"""
        
        # Conversion des headers Markdown en gras
        content = re.sub(r'^#{1,6}\s+(.+)$', r'*\1*', content, flags=re.MULTILINE)
        
        # Optimisation de la longueur
        content = self._optimize_length(content)
        
        return content
    
    def _build_action_block(self, actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Construction du block d'actions"""
        
        elements = []
        
        for action in actions[:5]:  # Limite de 5 actions Slack
            element = {
                "type": "button",
                "text": {
                    "type": "plain_text",
                    "text": action.get('text', 'Action')[:75]
                },
                "action_id": action.get('action_id', f"action_{len(elements)}")
            }
            
            if 'url' in action:
                element['url'] = action['url']
            
            elements.append(element)
        
        return {
            "type": "actions",
            "elements": elements
        }
    
    def _build_footer_block(self, metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Construction du block footer"""
        
        footer_text = metadata.get('footer')
        if not footer_text:
            # Footer par défaut avec timestamp
            footer_text = f"Generated at {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}"
        
        return {
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": footer_text[:75]
                }
            ]
        }
    
    def _validate_slack_blocks(self, blocks: List[Dict[str, Any]]) -> bool:
        """Validation des Slack Blocks"""
        
        if not blocks or len(blocks) > 50:  # Limite Slack
            return False
        
        required_fields = {
            'header': ['type', 'text'],
            'section': ['type', 'text'],
            'actions': ['type', 'elements'],
            'context': ['type', 'elements']
        }
        
        for block in blocks:
            block_type = block.get('type')
            if block_type in required_fields:
                for field in required_fields[block_type]:
                    if field not in block:
                        return False
        
        return True


class TeamsCardFormatter(BaseFormatter):
    """Formateur pour Microsoft Teams Cards"""
    
    async def format_message(
        self, 
        content: str, 
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Formatage en Teams Adaptive Card"""
        
        # Extraction des éléments
        elements = self._extract_elements(content)
        
        # Construction de l'Adaptive Card
        card = await self._build_teams_card(content, elements, metadata)
        
        # Validation de la card
        is_valid = self._validate_teams_card(card)
        
        return {
            'card': card,
            'content_type': 'application/json',
            'is_valid': is_valid,
            'metadata': metadata
        }
    
    async def _build_teams_card(
        self, 
        content: str, 
        elements: Dict[MessageElement, List], 
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Construction de l'Adaptive Card Teams"""
        
        card = {
            "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
            "type": "AdaptiveCard",
            "version": "1.3",
            "body": []
        }
        
        # Header
        if elements[MessageElement.HEADER]:
            card['body'].append({
                "type": "TextBlock",
                "text": elements[MessageElement.HEADER][0],
                "size": "Large",
                "weight": "Bolder",
                "color": "Accent"
            })
        
        # Contenu principal
        main_content = self._clean_content_for_teams(content)
        if main_content:
            card['body'].append({
                "type": "TextBlock",
                "text": main_content,
                "wrap": True,
                "markdown": True
            })
        
        # Actions
        actions = metadata.get('actions', [])
        if actions:
            card['actions'] = self._build_teams_actions(actions)
        
        return card
    
    def _clean_content_for_teams(self, content: str) -> str:
        """Nettoyage du contenu pour Teams"""
        
        # Teams supporte le Markdown, on garde le formatage
        cleaned = self._sanitize_content(content)
        cleaned = self._optimize_length(cleaned)
        
        return cleaned
    
    def _build_teams_actions(self, actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Construction des actions Teams"""
        
        teams_actions = []
        
        for action in actions:
            if 'url' in action:
                teams_actions.append({
                    "type": "Action.OpenUrl",
                    "title": action.get('text', 'Open'),
                    "url": action['url']
                })
            else:
                teams_actions.append({
                    "type": "Action.Submit",
                    "title": action.get('text', 'Submit'),
                    "data": action.get('data', {})
                })
        
        return teams_actions
    
    def _validate_teams_card(self, card: Dict[str, Any]) -> bool:
        """Validation de l'Adaptive Card"""
        
        required_fields = ['$schema', 'type', 'version', 'body']
        
        for field in required_fields:
            if field not in card:
                return False
        
        return card['type'] == 'AdaptiveCard'


class MessageFormatter:
    """Formateur principal avec support multi-canal"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Initialisation des formateurs
        self.formatters = {
            FormatterType.TEXT: TextFormatter(config),
            FormatterType.MARKDOWN: MarkdownFormatter(config),
            FormatterType.SLACK_BLOCKS: SlackBlocksFormatter(config),
            FormatterType.TEAMS_CARD: TeamsCardFormatter(config)
        }
        
        # Cache des messages formatés
        self.format_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_ttl = config.get('cache_ttl', 3600)
    
    async def format_for_channel(
        self, 
        content: str, 
        channel: str, 
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Formatage pour un canal spécifique"""
        
        # Mapping canal -> formateur
        channel_formatters = {
            'slack': FormatterType.SLACK_BLOCKS,
            'teams': FormatterType.TEAMS_CARD,
            'discord': FormatterType.MARKDOWN,
            'email': FormatterType.HTML,
            'sms': FormatterType.TEXT,
            'web': FormatterType.MARKDOWN,
            'mobile': FormatterType.TEXT
        }
        
        formatter_type = channel_formatters.get(channel, FormatterType.TEXT)
        
        # Vérification du cache
        cache_key = f"{hash(content)}_{formatter_type.value}_{hash(str(metadata))}"
        if cache_key in self.format_cache:
            self.logger.debug(f"Cache hit for formatter {formatter_type.value}")
            return self.format_cache[cache_key]
        
        # Formatage
        formatter = self.formatters[formatter_type]
        result = await formatter.format_message(content, metadata)
        
        # Ajout au cache
        self.format_cache[cache_key] = result
        
        # Nettoyage du cache si nécessaire
        if len(self.format_cache) > 1000:
            await self._cleanup_cache()
        
        return result
    
    async def format_multiple_channels(
        self, 
        content: str, 
        channels: List[str], 
        metadata: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """Formatage pour plusieurs canaux simultanément"""
        
        results = {}
        
        # Formatage parallèle
        tasks = []
        for channel in channels:
            task = self.format_for_channel(content, channel, metadata)
            tasks.append((channel, task))
        
        # Attente des résultats
        for channel, task in tasks:
            try:
                result = await task
                results[channel] = result
            except Exception as e:
                self.logger.error(f"Formatting failed for channel {channel}: {str(e)}")
                results[channel] = {
                    'error': str(e),
                    'content': content,
                    'content_type': 'text/plain'
                }
        
        return results
    
    async def _cleanup_cache(self):
        """Nettoyage du cache"""
        
        # Suppression de la moitié des entrées les plus anciennes
        # (implémentation simplifiée)
        cache_size = len(self.format_cache)
        items_to_remove = cache_size // 2
        
        keys_to_remove = list(self.format_cache.keys())[:items_to_remove]
        for key in keys_to_remove:
            del self.format_cache[key]
        
        self.logger.info(f"Cache cleaned up: removed {items_to_remove} entries")


class RichContentFormatter:
    """Formateur pour contenu riche avec médias"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Support des médias
        self.media_enabled = config.get('media_enabled', True)
        self.max_media_size = config.get('max_media_size', 10 * 1024 * 1024)  # 10MB
        
        # Templates pour contenu riche
        self.rich_templates = {
            'card': self._build_card_template,
            'carousel': self._build_carousel_template,
            'gallery': self._build_gallery_template,
            'timeline': self._build_timeline_template
        }
    
    async def format_rich_content(
        self, 
        content: str, 
        media_items: List[Dict[str, Any]], 
        template_type: str = 'card'
    ) -> Dict[str, Any]:
        """Formatage de contenu riche avec médias"""
        
        if not self.media_enabled:
            return {'content': content, 'type': 'text'}
        
        # Sélection du template
        template_builder = self.rich_templates.get(
            template_type, 
            self.rich_templates['card']
        )
        
        # Construction du contenu riche
        rich_content = await template_builder(content, media_items)
        
        return {
            'content': rich_content,
            'type': 'rich_content',
            'template_type': template_type,
            'media_count': len(media_items)
        }
    
    async def _build_card_template(
        self, 
        content: str, 
        media_items: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Construction d'un template card"""
        
        card = {
            'type': 'card',
            'content': content,
            'media': []
        }
        
        # Ajout des médias
        for item in media_items[:3]:  # Limite à 3 médias par card
            if self._validate_media_item(item):
                card['media'].append({
                    'type': item.get('type', 'image'),
                    'url': item.get('url'),
                    'alt_text': item.get('alt_text', ''),
                    'caption': item.get('caption', '')
                })
        
        return card
    
    async def _build_carousel_template(
        self, 
        content: str, 
        media_items: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Construction d'un template carousel"""
        
        carousel = {
            'type': 'carousel',
            'content': content,
            'items': []
        }
        
        # Construction des éléments du carousel
        for item in media_items[:10]:  # Limite à 10 éléments
            if self._validate_media_item(item):
                carousel['items'].append({
                    'media': {
                        'type': item.get('type', 'image'),
                        'url': item.get('url'),
                        'alt_text': item.get('alt_text', '')
                    },
                    'title': item.get('title', ''),
                    'description': item.get('description', ''),
                    'action': item.get('action', {})
                })
        
        return carousel
    
    async def _build_gallery_template(
        self, 
        content: str, 
        media_items: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Construction d'un template gallery"""
        
        gallery = {
            'type': 'gallery',
            'content': content,
            'layout': 'grid',
            'items': []
        }
        
        # Ajout des éléments de la galerie
        for item in media_items:
            if self._validate_media_item(item):
                gallery['items'].append({
                    'type': item.get('type', 'image'),
                    'url': item.get('url'),
                    'thumbnail_url': item.get('thumbnail_url', item.get('url')),
                    'alt_text': item.get('alt_text', ''),
                    'metadata': item.get('metadata', {})
                })
        
        return gallery
    
    async def _build_timeline_template(
        self, 
        content: str, 
        media_items: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Construction d'un template timeline"""
        
        timeline = {
            'type': 'timeline',
            'content': content,
            'events': []
        }
        
        # Construction des événements de la timeline
        for item in media_items:
            if 'timestamp' in item and self._validate_media_item(item):
                timeline['events'].append({
                    'timestamp': item['timestamp'],
                    'title': item.get('title', ''),
                    'description': item.get('description', ''),
                    'media': {
                        'type': item.get('type', 'image'),
                        'url': item.get('url'),
                        'alt_text': item.get('alt_text', '')
                    }
                })
        
        # Tri par timestamp
        timeline['events'].sort(key=lambda x: x['timestamp'])
        
        return timeline
    
    def _validate_media_item(self, item: Dict[str, Any]) -> bool:
        """Validation d'un élément média"""
        
        # Vérification des champs requis
        if 'url' not in item:
            return False
        
        # Vérification du type de média
        supported_types = ['image', 'video', 'audio', 'document']
        if item.get('type') not in supported_types:
            return False
        
        # Vérification de la taille (simulation)
        size = item.get('size', 0)
        if size > self.max_media_size:
            return False
        
        return True
