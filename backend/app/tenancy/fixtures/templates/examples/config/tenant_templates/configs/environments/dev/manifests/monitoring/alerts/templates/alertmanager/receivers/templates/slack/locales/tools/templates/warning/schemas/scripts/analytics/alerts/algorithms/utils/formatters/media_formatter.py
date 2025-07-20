"""
Spotify AI Agent - Media Content Formatters
==========================================

Ultra-advanced media content formatting system for rich multimedia content,
audio visualization, video processing, and interactive media presentations.

This module handles sophisticated formatting for:
- Audio visualization and waveform generation
- Video content processing and thumbnail generation
- Interactive media presentations and playlists
- Rich multimedia dashboard components
- Audio analysis and music visualization
- Video analytics and engagement metrics
- Interactive chart and graph generation
- Multi-format media export capabilities

Author: Fahed Mlaiel & Spotify AI Team
Version: 2.1.0
"""

import asyncio
import json
import base64
import hashlib
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Union, Tuple, BinaryIO
from dataclasses import dataclass, field
from enum import Enum
import structlog
from pathlib import Path
import mimetypes

logger = structlog.get_logger(__name__)


class MediaType(Enum):
    """Supported media types."""
    AUDIO = "audio"
    VIDEO = "video"
    IMAGE = "image"
    INTERACTIVE = "interactive"
    DOCUMENT = "document"


class AudioFormat(Enum):
    """Supported audio formats."""
    MP3 = "mp3"
    FLAC = "flac"
    WAV = "wav"
    OGG = "ogg"
    AAC = "aac"
    M4A = "m4a"


class VideoFormat(Enum):
    """Supported video formats."""
    MP4 = "mp4"
    WEBM = "webm"
    AVI = "avi"
    MOV = "mov"
    MKV = "mkv"


class VisualizationType(Enum):
    """Types of audio visualizations."""
    WAVEFORM = "waveform"
    SPECTROGRAM = "spectrogram"
    FREQUENCY_BARS = "frequency_bars"
    CIRCULAR_WAVE = "circular_wave"
    PARTICLE_SYSTEM = "particle_system"
    BEAT_DETECTION = "beat_detection"


class InteractiveType(Enum):
    """Types of interactive content."""
    PLAYLIST_BUILDER = "playlist_builder"
    MUSIC_MIXER = "music_mixer"
    AUDIO_EQUALIZER = "audio_equalizer"
    RECOMMENDATION_WHEEL = "recommendation_wheel"
    LYRIC_SYNC = "lyric_sync"
    SOCIAL_SHARING = "social_sharing"


@dataclass
class MediaMetadata:
    """Media content metadata."""
    
    title: str
    duration: Optional[float] = None  # seconds
    file_size: Optional[int] = None   # bytes
    format: Optional[str] = None
    resolution: Optional[Tuple[int, int]] = None  # width, height
    bitrate: Optional[int] = None
    sample_rate: Optional[int] = None
    channels: Optional[int] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    tags: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "title": self.title,
            "duration": self.duration,
            "file_size": self.file_size,
            "format": self.format,
            "resolution": list(self.resolution) if self.resolution else None,
            "bitrate": self.bitrate,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "created_at": self.created_at.isoformat(),
            "tags": self.tags
        }


@dataclass
class FormattedMediaContent:
    """Container for formatted media content."""
    
    content_id: str
    media_type: MediaType
    formatted_content: str
    metadata: MediaMetadata
    interactive_elements: List[Dict[str, Any]] = field(default_factory=list)
    embedded_media: Dict[str, str] = field(default_factory=dict)  # base64 encoded
    styling: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "content_id": self.content_id,
            "media_type": self.media_type.value,
            "formatted_content": self.formatted_content,
            "metadata": self.metadata.to_dict(),
            "interactive_elements": self.interactive_elements,
            "embedded_media": self.embedded_media,
            "styling": self.styling
        }


class BaseMediaFormatter:
    """Base class for media content formatters."""
    
    def __init__(self, tenant_id: str, config: Optional[Dict[str, Any]] = None):
        self.tenant_id = tenant_id
        self.config = config or {}
        self.logger = logger.bind(tenant_id=tenant_id, formatter=self.__class__.__name__)
        
        # Media processing configuration
        self.max_file_size = config.get('max_file_size', 100 * 1024 * 1024)  # 100MB
        self.cache_duration = config.get('cache_duration', 3600)  # 1 hour
        self.quality_settings = config.get('quality_settings', {})
        self.enable_caching = config.get('enable_caching', True)
        
        # Visualization settings
        self.visualization_config = config.get('visualization', {
            'width': 800,
            'height': 400,
            'color_scheme': 'spotify_green',
            'animation': True
        })
    
    def generate_content_id(self, content: Union[str, bytes]) -> str:
        """Generate unique content ID based on content hash."""
        if isinstance(content, str):
            content = content.encode('utf-8')
        return hashlib.sha256(content).hexdigest()[:16]
    
    def validate_media_size(self, file_size: int) -> bool:
        """Validate media file size."""
        return file_size <= self.max_file_size
    
    async def process_media_content(self, content_data: Dict[str, Any]) -> FormattedMediaContent:
        """Process media content - to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement process_media_content")


class AudioVisualizationFormatter(BaseMediaFormatter):
    """Advanced audio visualization and formatting system."""
    
    async def process_media_content(self, content_data: Dict[str, Any]) -> FormattedMediaContent:
        """Process audio content and generate visualizations."""
        
        audio_metadata = MediaMetadata(
            title=content_data.get('title', 'Unknown Track'),
            duration=content_data.get('duration'),
            format=content_data.get('format', 'mp3'),
            bitrate=content_data.get('bitrate'),
            sample_rate=content_data.get('sample_rate'),
            channels=content_data.get('channels', 2),
            tags=content_data.get('tags', {})
        )
        
        content_id = self.generate_content_id(content_data.get('audio_data', ''))
        
        # Generate multiple visualization types
        visualizations = await self._generate_audio_visualizations(content_data)
        
        # Create interactive audio player
        interactive_player = await self._create_interactive_audio_player(content_data, visualizations)
        
        # Format main content
        formatted_content = await self._format_audio_content(content_data, visualizations, interactive_player)
        
        # Create interactive elements
        interactive_elements = [
            interactive_player,
            await self._create_audio_controls(content_data),
            await self._create_visualization_selector(visualizations),
            await self._create_audio_analytics_panel(content_data)
        ]
        
        # Generate embedded media (base64 encoded thumbnails/previews)
        embedded_media = await self._generate_embedded_audio_media(content_data, visualizations)
        
        # Styling configuration
        styling = {
            "theme": "spotify_dark",
            "primary_color": "#1DB954",
            "secondary_color": "#191414",
            "accent_color": "#1ED760",
            "font_family": "Circular, Helvetica, Arial, sans-serif",
            "animation_duration": "0.3s",
            "border_radius": "8px"
        }
        
        return FormattedMediaContent(
            content_id=content_id,
            media_type=MediaType.AUDIO,
            formatted_content=formatted_content,
            metadata=audio_metadata,
            interactive_elements=interactive_elements,
            embedded_media=embedded_media,
            styling=styling
        )
    
    async def _generate_audio_visualizations(self, content_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Generate various audio visualizations."""
        
        visualizations = {}
        
        # Waveform visualization
        waveform_config = {
            "type": "waveform",
            "svg_content": await self._generate_waveform_svg(content_data),
            "width": self.visualization_config['width'],
            "height": 100,
            "color": "#1DB954",
            "background": "transparent",
            "interactive": True,
            "seekable": True
        }
        visualizations["waveform"] = waveform_config
        
        # Frequency bars visualization
        frequency_bars_config = {
            "type": "frequency_bars",
            "canvas_content": await self._generate_frequency_bars(content_data),
            "width": self.visualization_config['width'],
            "height": 200,
            "bars_count": 64,
            "color_gradient": ["#1DB954", "#1ED760", "#A0D468"],
            "animation": "bounce",
            "responsive": True
        }
        visualizations["frequency_bars"] = frequency_bars_config
        
        # Circular wave visualization
        circular_wave_config = {
            "type": "circular_wave",
            "canvas_content": await self._generate_circular_wave(content_data),
            "diameter": 300,
            "center_logo": True,
            "rotation_speed": 1.5,
            "wave_amplitude": 0.8,
            "color_scheme": "spotify_gradient"
        }
        visualizations["circular_wave"] = circular_wave_config
        
        # Spectrogram visualization
        spectrogram_config = {
            "type": "spectrogram",
            "canvas_content": await self._generate_spectrogram(content_data),
            "width": self.visualization_config['width'],
            "height": 250,
            "frequency_range": [20, 20000],
            "color_map": "plasma",
            "time_resolution": 0.1
        }
        visualizations["spectrogram"] = spectrogram_config
        
        return visualizations
    
    async def _generate_waveform_svg(self, content_data: Dict[str, Any]) -> str:
        """Generate SVG waveform visualization."""
        
        # Simulate audio analysis data
        duration = content_data.get('duration', 180)  # 3 minutes default
        sample_count = 1000
        amplitude_data = [0.5 + 0.3 * (i % 10) / 10 for i in range(sample_count)]
        
        width = self.visualization_config['width']
        height = 100
        
        svg_elements = []
        svg_elements.append(f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">')
        
        # Background
        svg_elements.append(f'<rect width="{width}" height="{height}" fill="rgba(25, 20, 20, 0.1)" rx="4"/>')
        
        # Waveform path
        path_data = "M 0 50"
        for i, amplitude in enumerate(amplitude_data):
            x = (i / sample_count) * width
            y = 50 + (amplitude - 0.5) * 40  # Center around y=50
            path_data += f" L {x:.2f} {y:.2f}"
        
        svg_elements.append(f'<path d="{path_data}" stroke="#1DB954" stroke-width="2" fill="none" opacity="0.8"/>')
        
        # Progress indicator (will be controlled by JavaScript)
        svg_elements.append('<line x1="0" y1="0" x2="0" y2="100" stroke="#1ED760" stroke-width="3" id="progress-line" opacity="0.9"/>')
        
        # Interactive regions for seeking
        for i in range(0, width, 50):
            svg_elements.append(f'<rect x="{i}" y="0" width="50" height="{height}" fill="transparent" class="seek-region" data-time="{(i/width)*duration:.1f}"/>')
        
        svg_elements.append('</svg>')
        
        return '\n'.join(svg_elements)
    
    async def _generate_frequency_bars(self, content_data: Dict[str, Any]) -> str:
        """Generate frequency bars visualization code."""
        
        # JavaScript code for animated frequency bars
        bars_code = f"""
        <canvas id="frequency-bars" width="{self.visualization_config['width']}" height="200"></canvas>
        <script>
        class FrequencyBarsVisualizer {{
            constructor(canvasId) {{
                this.canvas = document.getElementById(canvasId);
                this.ctx = this.canvas.getContext('2d');
                this.barsCount = 64;
                this.barWidth = this.canvas.width / this.barsCount;
                this.barGap = 2;
                this.animationId = null;
                this.audioData = this.generateMockAudioData();
                this.colors = ['#1DB954', '#1ED760', '#A0D468'];
                
                this.startAnimation();
            }}
            
            generateMockAudioData() {{
                return Array.from({{length: this.barsCount}}, (_, i) => 
                    Math.sin(i * 0.1 + Date.now() * 0.005) * 0.5 + 0.5
                );
            }}
            
            draw() {{
                this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
                
                for (let i = 0; i < this.barsCount; i++) {{
                    const barHeight = this.audioData[i] * this.canvas.height * 0.8;
                    const x = i * this.barWidth;
                    const y = this.canvas.height - barHeight;
                    
                    // Create gradient
                    const gradient = this.ctx.createLinearGradient(0, y, 0, this.canvas.height);
                    gradient.addColorStop(0, this.colors[0]);
                    gradient.addColorStop(0.5, this.colors[1]);
                    gradient.addColorStop(1, this.colors[2]);
                    
                    this.ctx.fillStyle = gradient;
                    this.ctx.fillRect(x, y, this.barWidth - this.barGap, barHeight);
                }}
            }}
            
            startAnimation() {{
                const animate = () => {{
                    this.audioData = this.generateMockAudioData();
                    this.draw();
                    this.animationId = requestAnimationFrame(animate);
                }};
                animate();
            }}
            
            stopAnimation() {{
                if (this.animationId) {{
                    cancelAnimationFrame(this.animationId);
                    this.animationId = null;
                }}
            }}
        }}
        
        // Initialize visualizer
        const frequencyVisualizer = new FrequencyBarsVisualizer('frequency-bars');
        </script>
        """
        
        return bars_code
    
    async def _generate_circular_wave(self, content_data: Dict[str, Any]) -> str:
        """Generate circular wave visualization."""
        
        circular_code = f"""
        <canvas id="circular-wave" width="300" height="300"></canvas>
        <script>
        class CircularWaveVisualizer {{
            constructor(canvasId) {{
                this.canvas = document.getElementById(canvasId);
                this.ctx = this.canvas.getContext('2d');
                this.centerX = this.canvas.width / 2;
                this.centerY = this.canvas.height / 2;
                this.baseRadius = 80;
                this.maxWaveHeight = 40;
                this.points = 64;
                this.rotation = 0;
                this.audioData = this.generateMockAudioData();
                
                this.startAnimation();
            }}
            
            generateMockAudioData() {{
                return Array.from({{length: this.points}}, (_, i) => 
                    Math.abs(Math.sin(i * 0.3 + Date.now() * 0.008)) * 0.7 + 0.3
                );
            }}
            
            draw() {{
                this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
                
                // Draw center circle with Spotify logo area
                this.ctx.beginPath();
                this.ctx.arc(this.centerX, this.centerY, this.baseRadius * 0.6, 0, Math.PI * 2);
                this.ctx.fillStyle = '#1DB954';
                this.ctx.fill();
                
                // Draw waveform circle
                this.ctx.beginPath();
                for (let i = 0; i < this.points; i++) {{
                    const angle = (i / this.points) * Math.PI * 2 + this.rotation;
                    const waveHeight = this.audioData[i] * this.maxWaveHeight;
                    const radius = this.baseRadius + waveHeight;
                    const x = this.centerX + Math.cos(angle) * radius;
                    const y = this.centerY + Math.sin(angle) * radius;
                    
                    if (i === 0) {{
                        this.ctx.moveTo(x, y);
                    }} else {{
                        this.ctx.lineTo(x, y);
                    }}
                }}
                this.ctx.closePath();
                
                // Create gradient
                const gradient = this.ctx.createRadialGradient(
                    this.centerX, this.centerY, this.baseRadius,
                    this.centerX, this.centerY, this.baseRadius + this.maxWaveHeight
                );
                gradient.addColorStop(0, 'rgba(29, 185, 84, 0.8)');
                gradient.addColorStop(1, 'rgba(30, 215, 96, 0.3)');
                
                this.ctx.strokeStyle = gradient;
                this.ctx.lineWidth = 3;
                this.ctx.stroke();
                
                this.rotation += 0.02;
            }}
            
            startAnimation() {{
                const animate = () => {{
                    this.audioData = this.generateMockAudioData();
                    this.draw();
                    requestAnimationFrame(animate);
                }};
                animate();
            }}
        }}
        
        // Initialize circular wave visualizer
        const circularVisualizer = new CircularWaveVisualizer('circular-wave');
        </script>
        """
        
        return circular_code
    
    async def _generate_spectrogram(self, content_data: Dict[str, Any]) -> str:
        """Generate spectrogram visualization."""
        
        spectrogram_code = f"""
        <canvas id="spectrogram" width="{self.visualization_config['width']}" height="250"></canvas>
        <script>
        class SpectrogramVisualizer {{
            constructor(canvasId) {{
                this.canvas = document.getElementById(canvasId);
                this.ctx = this.canvas.getContext('2d');
                this.timeSlices = [];
                this.maxSlices = 200;
                this.frequencyBins = 128;
                this.sliceWidth = this.canvas.width / this.maxSlices;
                
                this.initializeSpectrogram();
                this.startAnimation();
            }}
            
            initializeSpectrogram() {{
                for (let i = 0; i < this.maxSlices; i++) {{
                    this.timeSlices.push(this.generateFrequencySlice());
                }}
            }}
            
            generateFrequencySlice() {{
                return Array.from({{length: this.frequencyBins}}, (_, i) => 
                    Math.random() * 0.8 + 0.1
                );
            }}
            
            draw() {{
                this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
                
                for (let x = 0; x < this.timeSlices.length; x++) {{
                    const slice = this.timeSlices[x];
                    for (let y = 0; y < slice.length; y++) {{
                        const intensity = slice[y];
                        const hue = 120 + (intensity * 100); // Green to yellow gradient
                        const saturation = 80;
                        const lightness = 30 + (intensity * 40);
                        
                        this.ctx.fillStyle = `hsl(${{hue}}, ${{saturation}}%, ${{lightness}}%)`;
                        this.ctx.fillRect(
                            x * this.sliceWidth,
                            (this.frequencyBins - y - 1) * (this.canvas.height / this.frequencyBins),
                            this.sliceWidth,
                            this.canvas.height / this.frequencyBins
                        );
                    }}
                }}
            }}
            
            startAnimation() {{
                const animate = () => {{
                    // Shift time slices left
                    this.timeSlices.shift();
                    this.timeSlices.push(this.generateFrequencySlice());
                    
                    this.draw();
                    requestAnimationFrame(animate);
                }};
                animate();
            }}
        }}
        
        // Initialize spectrogram visualizer
        const spectrogramVisualizer = new SpectrogramVisualizer('spectrogram');
        </script>
        """
        
        return spectrogram_code
    
    async def _create_interactive_audio_player(self, content_data: Dict[str, Any], visualizations: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Create interactive audio player component."""
        
        player_config = {
            "type": "audio_player",
            "component_id": "spotify-audio-player",
            "title": content_data.get('title', 'Unknown Track'),
            "artist": content_data.get('artist', 'Unknown Artist'),
            "duration": content_data.get('duration', 0),
            "controls": {
                "play_pause": True,
                "seek": True,
                "volume": True,
                "speed": True,
                "loop": True,
                "shuffle": False
            },
            "visualizations": list(visualizations.keys()),
            "default_visualization": "waveform",
            "artwork_url": content_data.get('artwork_url'),
            "audio_url": content_data.get('audio_url'),
            "features": {
                "lyrics_sync": True,
                "social_share": True,
                "add_to_playlist": True,
                "download": content_data.get('downloadable', False)
            }
        }
        
        return player_config
    
    async def _create_audio_controls(self, content_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create audio control panel."""
        
        controls_config = {
            "type": "audio_controls",
            "component_id": "audio-control-panel",
            "equalizer": {
                "enabled": True,
                "presets": ["Rock", "Pop", "Jazz", "Classical", "Electronic", "Custom"],
                "bands": [
                    {"frequency": "60Hz", "gain": 0},
                    {"frequency": "170Hz", "gain": 0},
                    {"frequency": "310Hz", "gain": 0},
                    {"frequency": "600Hz", "gain": 0},
                    {"frequency": "1kHz", "gain": 0},
                    {"frequency": "3kHz", "gain": 0},
                    {"frequency": "6kHz", "gain": 0},
                    {"frequency": "12kHz", "gain": 0},
                    {"frequency": "14kHz", "gain": 0},
                    {"frequency": "16kHz", "gain": 0}
                ]
            },
            "effects": {
                "reverb": {"enabled": False, "type": "hall", "intensity": 0.3},
                "echo": {"enabled": False, "delay": 0.5, "feedback": 0.3},
                "bass_boost": {"enabled": False, "gain": 3},
                "treble_boost": {"enabled": False, "gain": 2}
            },
            "spatial_audio": {
                "enabled": False,
                "mode": "binaural",
                "head_tracking": False
            }
        }
        
        return controls_config
    
    async def _create_visualization_selector(self, visualizations: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Create visualization selector component."""
        
        selector_config = {
            "type": "visualization_selector",
            "component_id": "viz-selector",
            "available_visualizations": [
                {
                    "id": "waveform",
                    "name": "Waveform",
                    "description": "Classic audio waveform display",
                    "icon": "📊",
                    "preview_image": "data:image/svg+xml;base64,..." # Thumbnail
                },
                {
                    "id": "frequency_bars",
                    "name": "Frequency Bars",
                    "description": "Animated frequency spectrum bars",
                    "icon": "📈",
                    "preview_image": "data:image/png;base64,..."
                },
                {
                    "id": "circular_wave",
                    "name": "Circular Wave",
                    "description": "Radial waveform visualization",
                    "icon": "🌊",
                    "preview_image": "data:image/png;base64,..."
                },
                {
                    "id": "spectrogram",
                    "name": "Spectrogram",
                    "description": "Time-frequency analysis display",
                    "icon": "🔥",
                    "preview_image": "data:image/png;base64,..."
                }
            ],
            "layout_options": ["fullscreen", "sidebar", "overlay", "bottom_panel"],
            "customization": {
                "color_themes": ["spotify_green", "dark_blue", "rainbow", "monochrome"],
                "animation_speed": {"min": 0.5, "max": 3.0, "default": 1.0},
                "sensitivity": {"min": 0.1, "max": 2.0, "default": 1.0}
            }
        }
        
        return selector_config
    
    async def _create_audio_analytics_panel(self, content_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create audio analytics panel."""
        
        analytics_config = {
            "type": "audio_analytics",
            "component_id": "audio-analytics-panel",
            "metrics": {
                "audio_quality": {
                    "bitrate": content_data.get('bitrate', 320),
                    "sample_rate": content_data.get('sample_rate', 44100),
                    "dynamic_range": content_data.get('dynamic_range', 12.5),
                    "loudness": content_data.get('loudness', -12.3)
                },
                "musical_features": {
                    "tempo": content_data.get('tempo', 120),
                    "key": content_data.get('key', 'C Major'),
                    "time_signature": content_data.get('time_signature', '4/4'),
                    "energy": content_data.get('energy', 0.75),
                    "danceability": content_data.get('danceability', 0.68),
                    "valence": content_data.get('valence', 0.42)
                },
                "technical_analysis": {
                    "rms_energy": content_data.get('rms_energy', 0.15),
                    "zero_crossing_rate": content_data.get('zcr', 0.08),
                    "spectral_centroid": content_data.get('spectral_centroid', 2500),
                    "mfcc_coefficients": content_data.get('mfcc', [1.2, -0.8, 0.3, 1.1, -0.5])
                }
            },
            "real_time_analysis": True,
            "export_options": ["csv", "json", "pdf_report"]
        }
        
        return analytics_config
    
    async def _generate_embedded_audio_media(self, content_data: Dict[str, Any], visualizations: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
        """Generate embedded media content."""
        
        embedded_media = {}
        
        # Generate waveform thumbnail (base64 encoded SVG)
        if "waveform" in visualizations:
            waveform_svg = visualizations["waveform"]["svg_content"]
            waveform_b64 = base64.b64encode(waveform_svg.encode('utf-8')).decode('utf-8')
            embedded_media["waveform_thumbnail"] = f"data:image/svg+xml;base64,{waveform_b64}"
        
        # Generate artwork placeholder if not provided
        if not content_data.get('artwork_url'):
            artwork_svg = f'''
            <svg width="300" height="300" xmlns="http://www.w3.org/2000/svg">
                <rect width="300" height="300" fill="#191414"/>
                <circle cx="150" cy="150" r="80" fill="#1DB954"/>
                <text x="150" y="160" text-anchor="middle" fill="white" font-family="Arial" font-size="20">♪</text>
            </svg>
            '''
            artwork_b64 = base64.b64encode(artwork_svg.encode('utf-8')).decode('utf-8')
            embedded_media["artwork_placeholder"] = f"data:image/svg+xml;base64,{artwork_b64}"
        
        return embedded_media
    
    async def _format_audio_content(self, content_data: Dict[str, Any], visualizations: Dict[str, Dict[str, Any]], interactive_player: Dict[str, Any]) -> str:
        """Format the main audio content display."""
        
        title = content_data.get('title', 'Unknown Track')
        artist = content_data.get('artist', 'Unknown Artist')
        duration = content_data.get('duration', 0)
        duration_str = f"{int(duration // 60)}:{int(duration % 60):02d}"
        
        content = f"""
# 🎵 {title}
**Artist**: {artist}  
**Duration**: {duration_str}  
**Format**: {content_data.get('format', 'MP3').upper()}

## 🎨 Interactive Audio Visualization

<div id="audio-player-container" class="spotify-audio-player">
    <div class="player-header">
        <div class="track-info">
            <h3>{title}</h3>
            <p>{artist}</p>
        </div>
        <div class="player-controls">
            <button id="play-btn" class="control-btn">▶️</button>
            <button id="pause-btn" class="control-btn">⏸️</button>
            <button id="stop-btn" class="control-btn">⏹️</button>
        </div>
    </div>
    
    <div class="visualization-container">
        <div id="current-visualization">
            {visualizations.get('waveform', {}).get('svg_content', '')}
        </div>
        
        <div class="visualization-controls">
            <select id="viz-selector" class="viz-dropdown">
                <option value="waveform">📊 Waveform</option>
                <option value="frequency_bars">📈 Frequency Bars</option>
                <option value="circular_wave">🌊 Circular Wave</option>
                <option value="spectrogram">🔥 Spectrogram</option>
            </select>
        </div>
    </div>
    
    <div class="progress-container">
        <input type="range" id="progress-slider" min="0" max="100" value="0" class="progress-slider">
        <div class="time-display">
            <span id="current-time">0:00</span>
            <span id="total-time">{duration_str}</span>
        </div>
    </div>
</div>

## 🔧 Audio Controls & Effects

<div id="audio-controls-panel" class="controls-panel">
    <div class="equalizer-section">
        <h4>🎚️ Equalizer</h4>
        <div class="eq-sliders">
            <!-- Equalizer sliders will be populated by JavaScript -->
        </div>
    </div>
    
    <div class="effects-section">
        <h4>✨ Audio Effects</h4>
        <div class="effects-controls">
            <label><input type="checkbox" id="reverb-toggle"> Reverb</label>
            <label><input type="checkbox" id="echo-toggle"> Echo</label>
            <label><input type="checkbox" id="bass-boost-toggle"> Bass Boost</label>
        </div>
    </div>
</div>

## 📊 Audio Analytics

<div id="analytics-panel" class="analytics-panel">
    <div class="metric-group">
        <h4>🎧 Audio Quality</h4>
        <div class="metrics">
            <span class="metric">Bitrate: {content_data.get('bitrate', 320)} kbps</span>
            <span class="metric">Sample Rate: {content_data.get('sample_rate', 44100)} Hz</span>
            <span class="metric">Channels: {content_data.get('channels', 2)}</span>
        </div>
    </div>
    
    <div class="metric-group">
        <h4>🎼 Musical Features</h4>
        <div class="metrics">
            <span class="metric">Tempo: {content_data.get('tempo', 120)} BPM</span>
            <span class="metric">Key: {content_data.get('key', 'C Major')}</span>
            <span class="metric">Energy: {content_data.get('energy', 0.75):.2f}</span>
        </div>
    </div>
</div>

<style>
.spotify-audio-player {{
    background: linear-gradient(135deg, #191414 0%, #1DB954 100%);
    border-radius: 12px;
    padding: 20px;
    color: white;
    font-family: 'Circular', Helvetica, Arial, sans-serif;
    margin: 20px 0;
}}

.player-header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 20px;
}}

.control-btn {{
    background: rgba(255,255,255,0.1);
    border: none;
    border-radius: 50%;
    width: 40px;
    height: 40px;
    margin: 0 5px;
    cursor: pointer;
    transition: all 0.3s ease;
}}

.control-btn:hover {{
    background: rgba(255,255,255,0.2);
    transform: scale(1.1);
}}

.visualization-container {{
    background: rgba(0,0,0,0.3);
    border-radius: 8px;
    padding: 15px;
    margin: 15px 0;
}}

.progress-container {{
    display: flex;
    align-items: center;
    gap: 10px;
    margin-top: 15px;
}}

.progress-slider {{
    flex: 1;
    height: 4px;
    background: rgba(255,255,255,0.3);
    border-radius: 2px;
}}

.controls-panel, .analytics-panel {{
    background: #f8f9fa;
    border-radius: 8px;
    padding: 20px;
    margin: 20px 0;
    border-left: 4px solid #1DB954;
}}

.metric-group {{
    margin-bottom: 15px;
}}

.metrics {{
    display: flex;
    gap: 15px;
    flex-wrap: wrap;
}}

.metric {{
    background: #e9ecef;
    padding: 5px 10px;
    border-radius: 15px;
    font-size: 0.9em;
    color: #495057;
}}
</style>
        """.strip()
        
        return content


class VideoContentFormatter(BaseMediaFormatter):
    """Advanced video content formatting and processing system."""
    
    async def process_media_content(self, content_data: Dict[str, Any]) -> FormattedMediaContent:
        """Process video content with advanced features."""
        
        video_metadata = MediaMetadata(
            title=content_data.get('title', 'Unknown Video'),
            duration=content_data.get('duration'),
            format=content_data.get('format', 'mp4'),
            resolution=content_data.get('resolution', (1920, 1080)),
            file_size=content_data.get('file_size'),
            tags=content_data.get('tags', {})
        )
        
        content_id = self.generate_content_id(content_data.get('video_data', ''))
        
        # Generate video player with advanced features
        video_player = await self._create_advanced_video_player(content_data)
        
        # Create video analytics
        video_analytics = await self._create_video_analytics(content_data)
        
        # Format main content
        formatted_content = await self._format_video_content(content_data, video_player, video_analytics)
        
        # Interactive elements
        interactive_elements = [
            video_player,
            await self._create_video_controls(content_data),
            await self._create_chapter_navigator(content_data),
            video_analytics
        ]
        
        # Generate thumbnails and previews
        embedded_media = await self._generate_video_thumbnails(content_data)
        
        # Video-specific styling
        styling = {
            "theme": "video_dark",
            "primary_color": "#FF0000",
            "secondary_color": "#0F0F0F",
            "player_background": "#000000",
            "control_color": "#FFFFFF",
            "accent_color": "#FF6B6B"
        }
        
        return FormattedMediaContent(
            content_id=content_id,
            media_type=MediaType.VIDEO,
            formatted_content=formatted_content,
            metadata=video_metadata,
            interactive_elements=interactive_elements,
            embedded_media=embedded_media,
            styling=styling
        )
    
    async def _create_advanced_video_player(self, content_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create advanced video player with multiple features."""
        
        player_config = {
            "type": "video_player",
            "component_id": "advanced-video-player",
            "title": content_data.get('title', 'Unknown Video'),
            "duration": content_data.get('duration', 0),
            "video_url": content_data.get('video_url'),
            "poster": content_data.get('thumbnail_url'),
            "resolutions": content_data.get('available_resolutions', ['720p', '1080p', '4K']),
            "subtitles": content_data.get('subtitles', []),
            "chapters": content_data.get('chapters', []),
            "features": {
                "picture_in_picture": True,
                "fullscreen": True,
                "theater_mode": True,
                "speed_control": True,
                "volume_boost": True,
                "auto_play": False,
                "loop": False,
                "skip_intro": content_data.get('has_intro', False),
                "skip_credits": content_data.get('has_credits', False)
            },
            "quality_settings": {
                "adaptive_bitrate": True,
                "preferred_quality": "auto",
                "buffer_size": "medium"
            },
            "accessibility": {
                "closed_captions": True,
                "audio_descriptions": content_data.get('has_audio_descriptions', False),
                "keyboard_shortcuts": True
            }
        }
        
        return player_config
    
    async def _create_video_analytics(self, content_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create video analytics component."""
        
        analytics_config = {
            "type": "video_analytics",
            "component_id": "video-analytics",
            "metrics": {
                "technical": {
                    "resolution": f"{content_data.get('resolution', (1920, 1080))[0]}x{content_data.get('resolution', (1920, 1080))[1]}",
                    "framerate": content_data.get('framerate', 30),
                    "bitrate": content_data.get('bitrate', 5000),
                    "codec": content_data.get('codec', 'H.264'),
                    "file_size": content_data.get('file_size', 0)
                },
                "engagement": {
                    "watch_time": 0,
                    "completion_rate": 0,
                    "replay_count": 0,
                    "skip_segments": [],
                    "most_watched_segment": None
                },
                "performance": {
                    "buffer_events": 0,
                    "quality_changes": 0,
                    "load_time": 0,
                    "error_count": 0
                }
            },
            "real_time_tracking": True,
            "heatmap_generation": True,
            "export_formats": ["json", "csv", "video_report"]
        }
        
        return analytics_config
    
    async def _create_video_controls(self, content_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create video control panel."""
        
        controls_config = {
            "type": "video_controls",
            "component_id": "video-controls",
            "playback_controls": {
                "play_pause": True,
                "stop": True,
                "seek": True,
                "frame_step": True,
                "speed_options": [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
            },
            "audio_controls": {
                "volume": True,
                "mute": True,
                "audio_tracks": content_data.get('audio_tracks', []),
                "equalizer": False
            },
            "video_controls": {
                "brightness": True,
                "contrast": True,
                "saturation": True,
                "zoom": True,
                "rotate": False,
                "flip": False
            },
            "advanced_features": {
                "slow_motion": True,
                "time_lapse": True,
                "a_b_repeat": True,
                "bookmark": True,
                "screenshot": True,
                "gif_creation": True
            }
        }
        
        return controls_config
    
    async def _create_chapter_navigator(self, content_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create chapter navigation component."""
        
        chapters = content_data.get('chapters', [])
        if not chapters:
            # Generate default chapters based on duration
            duration = content_data.get('duration', 600)  # 10 minutes default
            chapter_count = max(1, duration // 60)  # One chapter per minute
            chapters = [
                {
                    "title": f"Chapter {i+1}",
                    "start_time": i * 60,
                    "end_time": min((i+1) * 60, duration),
                    "thumbnail": None
                }
                for i in range(int(chapter_count))
            ]
        
        navigator_config = {
            "type": "chapter_navigator",
            "component_id": "chapter-navigator",
            "chapters": chapters,
            "features": {
                "thumbnail_preview": True,
                "hover_scrubbing": True,
                "keyboard_navigation": True,
                "auto_chapters": True
            },
            "layout": "horizontal",
            "show_timestamps": True,
            "show_thumbnails": True
        }
        
        return navigator_config
    
    async def _generate_video_thumbnails(self, content_data: Dict[str, Any]) -> Dict[str, str]:
        """Generate video thumbnails and previews."""
        
        embedded_media = {}
        
        # Generate placeholder thumbnail if not provided
        if not content_data.get('thumbnail_url'):
            thumbnail_svg = f'''
            <svg width="640" height="360" xmlns="http://www.w3.org/2000/svg">
                <rect width="640" height="360" fill="#0F0F0F"/>
                <circle cx="320" cy="180" r="50" fill="#FF0000"/>
                <polygon points="300,160 300,200 340,180" fill="white"/>
                <text x="320" y="250" text-anchor="middle" fill="white" font-family="Arial" font-size="16">
                    {content_data.get('title', 'Video Content')[:30]}...
                </text>
            </svg>
            '''
            thumbnail_b64 = base64.b64encode(thumbnail_svg.encode('utf-8')).decode('utf-8')
            embedded_media["thumbnail_placeholder"] = f"data:image/svg+xml;base64,{thumbnail_b64}"
        
        return embedded_media
    
    async def _format_video_content(self, content_data: Dict[str, Any], video_player: Dict[str, Any], video_analytics: Dict[str, Any]) -> str:
        """Format the main video content display."""
        
        title = content_data.get('title', 'Unknown Video')
        duration = content_data.get('duration', 0)
        duration_str = f"{int(duration // 3600)}:{int((duration % 3600) // 60):02d}:{int(duration % 60):02d}"
        resolution = content_data.get('resolution', (1920, 1080))
        
        content = f"""
# 🎬 {title}

**Duration**: {duration_str}  
**Resolution**: {resolution[0]}x{resolution[1]}  
**Format**: {content_data.get('format', 'MP4').upper()}

## 📹 Advanced Video Player

<div id="video-player-container" class="advanced-video-player">
    <div class="video-wrapper">
        <video id="main-video" controls poster="{content_data.get('thumbnail_url', '')}">
            <source src="{content_data.get('video_url', '#')}" type="video/mp4">
            Your browser does not support the video tag.
        </video>
        
        <div class="video-overlay">
            <div class="video-controls">
                <button id="play-pause-btn" class="control-btn">▶️</button>
                <div class="progress-container">
                    <input type="range" id="video-progress" min="0" max="100" value="0">
                </div>
                <span id="time-display">0:00 / {duration_str}</span>
                <button id="fullscreen-btn" class="control-btn">⛶</button>
            </div>
        </div>
    </div>
    
    <div class="video-info">
        <h3>{title}</h3>
        <div class="video-stats">
            <span>👁️ Views: {content_data.get('view_count', 0):,}</span>
            <span>👍 Likes: {content_data.get('like_count', 0):,}</span>
            <span>📊 Quality: {resolution[0]}p</span>
        </div>
    </div>
</div>

## ⚙️ Video Controls & Settings

<div id="video-controls-panel" class="video-controls-panel">
    <div class="control-section">
        <h4>🎛️ Playback Controls</h4>
        <div class="playback-controls">
            <label>Speed: 
                <select id="playback-speed">
                    <option value="0.5">0.5x</option>
                    <option value="1" selected>1x</option>
                    <option value="1.5">1.5x</option>
                    <option value="2">2x</option>
                </select>
            </label>
            <button id="frame-step-btn">Frame Step</button>
            <button id="screenshot-btn">📸 Screenshot</button>
        </div>
    </div>
    
    <div class="control-section">
        <h4>🎨 Video Effects</h4>
        <div class="video-effects">
            <label>Brightness: <input type="range" id="brightness" min="0" max="200" value="100"></label>
            <label>Contrast: <input type="range" id="contrast" min="0" max="200" value="100"></label>
            <label>Saturation: <input type="range" id="saturation" min="0" max="200" value="100"></label>
        </div>
    </div>
</div>

## 📊 Video Analytics Dashboard

<div id="video-analytics-dashboard" class="analytics-dashboard">
    <div class="analytics-section">
        <h4>📈 Engagement Metrics</h4>
        <div class="metrics-grid">
            <div class="metric-card">
                <span class="metric-value" id="watch-time">0:00</span>
                <span class="metric-label">Watch Time</span>
            </div>
            <div class="metric-card">
                <span class="metric-value" id="completion-rate">0%</span>
                <span class="metric-label">Completion Rate</span>
            </div>
            <div class="metric-card">
                <span class="metric-value" id="replay-count">0</span>
                <span class="metric-label">Replays</span>
            </div>
        </div>
    </div>
    
    <div class="analytics-section">
        <h4>🔧 Technical Performance</h4>
        <div class="tech-metrics">
            <span>Buffer Events: <span id="buffer-events">0</span></span>
            <span>Quality Changes: <span id="quality-changes">0</span></span>
            <span>Load Time: <span id="load-time">0ms</span></span>
        </div>
    </div>
</div>

<style>
.advanced-video-player {{
    background: #0F0F0F;
    border-radius: 8px;
    overflow: hidden;
    margin: 20px 0;
}}

.video-wrapper {{
    position: relative;
    background: #000;
}}

.video-wrapper video {{
    width: 100%;
    height: auto;
    display: block;
}}

.video-overlay {{
    position: absolute;
    bottom: 0;
    left: 0;
    right: 0;
    background: linear-gradient(transparent, rgba(0,0,0,0.8));
    padding: 20px;
}}

.video-controls {{
    display: flex;
    align-items: center;
    gap: 10px;
    color: white;
}}

.progress-container {{
    flex: 1;
    margin: 0 10px;
}}

#video-progress {{
    width: 100%;
    height: 4px;
    background: rgba(255,255,255,0.3);
    border-radius: 2px;
}}

.video-info {{
    padding: 15px;
    background: #1A1A1A;
    color: white;
}}

.video-stats {{
    display: flex;
    gap: 20px;
    margin-top: 10px;
    font-size: 0.9em;
    color: #ccc;
}}

.video-controls-panel, .analytics-dashboard {{
    background: #f8f9fa;
    border-radius: 8px;
    padding: 20px;
    margin: 20px 0;
    border-left: 4px solid #FF0000;
}}

.control-section, .analytics-section {{
    margin-bottom: 20px;
}}

.playback-controls, .video-effects {{
    display: flex;
    gap: 15px;
    align-items: center;
    flex-wrap: wrap;
}}

.metrics-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 15px;
    margin-top: 10px;
}}

.metric-card {{
    background: white;
    padding: 15px;
    border-radius: 8px;
    text-align: center;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}}

.metric-value {{
    display: block;
    font-size: 1.5em;
    font-weight: bold;
    color: #FF0000;
}}

.metric-label {{
    font-size: 0.9em;
    color: #666;
}}

.tech-metrics {{
    display: flex;
    gap: 20px;
    flex-wrap: wrap;
}}
</style>

<script>
// Initialize video analytics tracking
document.addEventListener('DOMContentLoaded', function() {{
    const video = document.getElementById('main-video');
    const watchTimeElement = document.getElementById('watch-time');
    const completionRateElement = document.getElementById('completion-rate');
    
    let watchStartTime = 0;
    let totalWatchTime = 0;
    
    video.addEventListener('play', function() {{
        watchStartTime = Date.now();
    }});
    
    video.addEventListener('pause', function() {{
        if (watchStartTime > 0) {{
            totalWatchTime += Date.now() - watchStartTime;
            updateWatchTime();
        }}
    }});
    
    video.addEventListener('timeupdate', function() {{
        const completionRate = (video.currentTime / video.duration) * 100;
        completionRateElement.textContent = completionRate.toFixed(1) + '%';
    }});
    
    function updateWatchTime() {{
        const minutes = Math.floor(totalWatchTime / 60000);
        const seconds = Math.floor((totalWatchTime % 60000) / 1000);
        watchTimeElement.textContent = `${{minutes}}:${{seconds.toString().padStart(2, '0')}}`;
    }}
}});
</script>
        """.strip()
        
        return content


# Factory function for creating media formatters
def create_media_formatter(
    formatter_type: str,
    tenant_id: str,
    config: Optional[Dict[str, Any]] = None
) -> BaseMediaFormatter:
    """
    Factory function to create media content formatters.
    
    Args:
        formatter_type: Type of formatter ('audio', 'video', 'interactive', 'multimedia')
        tenant_id: Tenant identifier
        config: Configuration dictionary
        
    Returns:
        Configured media formatter instance
    """
    formatters = {
        'audio': AudioVisualizationFormatter,
        'audio_visualization': AudioVisualizationFormatter,
        'video': VideoContentFormatter,
        'video_content': VideoContentFormatter,
        'multimedia': AudioVisualizationFormatter,  # Default to audio for multimedia
        'interactive_media': AudioVisualizationFormatter
    }
    
    if formatter_type not in formatters:
        raise ValueError(f"Unsupported media formatter type: {formatter_type}")
    
    formatter_class = formatters[formatter_type]
    return formatter_class(tenant_id, config or {})
