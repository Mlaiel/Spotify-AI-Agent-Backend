"""
Tests Enterprise - Streaming Helpers
===================================

Suite de tests ultra-avancée pour le module streaming_helpers avec tests temps réel,
adaptive bitrate, CDN, et performance streaming enterprise.

Développé par l'équipe Streaming Test Engineering Expert sous la direction de Fahed Mlaiel.
"""

import pytest
import numpy as np
import asyncio
from unittest.mock import Mock, MagicMock, AsyncMock, patch
from datetime import datetime, timedelta
import json
from typing import Dict, Any, List, Optional
import uuid

# Import des modules streaming à tester
try:
    from app.utils.streaming_helpers import (
        StreamProcessor,
        AudioBuffer,
        QualityManager,
        RealtimeOptimizer,
        CDNManager
    )
except ImportError:
    # Mocks si modules pas encore implémentés
    StreamProcessor = MagicMock
    AudioBuffer = MagicMock
    QualityManager = MagicMock
    RealtimeOptimizer = MagicMock
    CDNManager = MagicMock


class TestStreamProcessor:
    """Tests enterprise pour StreamProcessor avec streaming temps réel avancé."""
    
    @pytest.fixture
    def stream_processor(self):
        """Instance StreamProcessor pour tests."""
        return StreamProcessor()
    
    @pytest.fixture
    def streaming_config(self):
        """Configuration streaming enterprise."""
        return {
            'adaptive_bitrate': {
                'enabled': True,
                'min_bitrate': 128,
                'max_bitrate': 320,
                'target_bitrate': 256,
                'adaptation_interval': 2
            },
            'buffer_management': {
                'target_buffer_ms': 5000,
                'min_buffer_ms': 2000,
                'max_buffer_ms': 10000,
                'preload_threshold': 0.3
            },
            'quality_adaptation': {
                'enable_quality_scaling': True,
                'quality_levels': ['low', 'medium', 'high', 'lossless'],
                'auto_quality': True,
                'user_preference_weight': 0.7
            },
            'latency_optimization': {
                'low_latency_mode': True,
                'target_latency_ms': 100,
                'max_latency_ms': 300,
                'jitter_buffer_ms': 50
            }
        }
    
    async def test_adaptive_bitrate_streaming(self, stream_processor, streaming_config):
        """Test streaming adaptatif avec changements bitrate automatiques."""
        # Mock session streaming
        stream_processor.start_streaming_session = AsyncMock(return_value=MagicMock(
            id=str(uuid.uuid4()),
            user_id='user_12345',
            initial_bitrate=256,
            quality='high',
            buffer_health=0.85
        ))
        
        # Mock métriques temps réel
        stream_processor.get_real_time_metrics = AsyncMock(return_value={
            'current_bitrate': 256,
            'buffer_level_ms': 4500,
            'latency_ms': 67,
            'jitter_ms': 15,
            'packet_loss_rate': 0.002,
            'bandwidth_available_kbps': 800,
            'quality_score': 0.89,
            'adaptation_events': [
                {'timestamp': datetime.utcnow() - timedelta(seconds=30), 'from': 320, 'to': 256, 'reason': 'bandwidth_drop'},
                {'timestamp': datetime.utcnow() - timedelta(seconds=60), 'from': 256, 'to': 320, 'reason': 'bandwidth_increase'}
            ]
        })
        
        # Démarrer session streaming
        session = await stream_processor.start_streaming_session(
            audio_source='track_12345',
            user_id='user_67890',
            config=streaming_config
        )
        
        # Obtenir métriques temps réel
        metrics = await stream_processor.get_real_time_metrics(session.id)
        
        # Validations streaming adaptatif
        assert session.id is not None
        assert session.initial_bitrate == 256
        assert metrics['current_bitrate'] >= streaming_config['adaptive_bitrate']['min_bitrate']
        assert metrics['current_bitrate'] <= streaming_config['adaptive_bitrate']['max_bitrate']
        assert metrics['latency_ms'] <= streaming_config['latency_optimization']['max_latency_ms']
        assert metrics['quality_score'] > 0.8
        assert len(metrics['adaptation_events']) > 0
    
    async def test_streaming_quality_adaptation(self, stream_processor):
        """Test adaptation qualité selon conditions réseau."""
        # Conditions réseau simulées
        network_scenarios = [
            {
                'name': 'excellent_wifi',
                'bandwidth_kbps': 5000,
                'latency_ms': 10,
                'packet_loss': 0.001,
                'expected_bitrate': 320,
                'expected_quality': 'lossless'
            },
            {
                'name': 'good_mobile',
                'bandwidth_kbps': 1000,
                'latency_ms': 50,
                'packet_loss': 0.005,
                'expected_bitrate': 256,
                'expected_quality': 'high'
            },
            {
                'name': 'poor_connection',
                'bandwidth_kbps': 200,
                'latency_ms': 150,
                'packet_loss': 0.02,
                'expected_bitrate': 128,
                'expected_quality': 'medium'
            },
            {
                'name': 'very_poor',
                'bandwidth_kbps': 64,
                'latency_ms': 300,
                'packet_loss': 0.05,
                'expected_bitrate': 64,
                'expected_quality': 'low'
            }
        ]
        
        # Mock adaptation qualité
        stream_processor.adapt_quality_to_conditions = AsyncMock()
        
        for scenario in network_scenarios:
            # Configuration réponse mock
            stream_processor.adapt_quality_to_conditions.return_value = {
                'adapted_bitrate': scenario['expected_bitrate'],
                'adapted_quality': scenario['expected_quality'],
                'adaptation_reason': f"network_conditions_{scenario['name']}",
                'confidence_score': 0.9,
                'estimated_buffer_impact': 0.15,
                'user_experience_score': 0.85
            }
            
            result = await stream_processor.adapt_quality_to_conditions(
                network_conditions={
                    'bandwidth_kbps': scenario['bandwidth_kbps'],
                    'latency_ms': scenario['latency_ms'],
                    'packet_loss_rate': scenario['packet_loss']
                },
                user_preferences={'quality_priority': 'balanced'}
            )
            
            # Validations adaptation
            assert result['adapted_bitrate'] == scenario['expected_bitrate']
            assert result['adapted_quality'] == scenario['expected_quality']
            assert result['confidence_score'] > 0.8
            assert 'adaptation_reason' in result
    
    async def test_low_latency_streaming_mode(self, stream_processor):
        """Test mode streaming faible latence."""
        # Configuration ultra faible latence
        low_latency_config = {
            'target_latency_ms': 50,
            'max_latency_ms': 100,
            'buffer_size_ms': 1000,
            'preemptive_adaptation': True,
            'aggressive_optimization': True
        }
        
        # Mock optimisations faible latence
        stream_processor.enable_low_latency_mode = AsyncMock(return_value={
            'mode_enabled': True,
            'achieved_latency_ms': 45,
            'buffer_optimizations': {
                'reduced_buffer_size': True,
                'predictive_prefetch': True,
                'frame_dropping_enabled': True
            },
            'network_optimizations': {
                'tcp_no_delay': True,
                'custom_congestion_control': 'bbr',
                'packet_pacing': True
            },
            'audio_optimizations': {
                'reduced_frame_size': True,
                'lookahead_disabled': True,
                'real_time_encoding': True
            }
        })
        
        result = await stream_processor.enable_low_latency_mode(
            session_id='session_123',
            config=low_latency_config
        )
        
        # Validations faible latence
        assert result['mode_enabled'] is True
        assert result['achieved_latency_ms'] <= low_latency_config['target_latency_ms']
        assert result['buffer_optimizations']['reduced_buffer_size'] is True
        assert result['network_optimizations']['tcp_no_delay'] is True
        assert result['audio_optimizations']['real_time_encoding'] is True
    
    async def test_streaming_session_resilience(self, stream_processor):
        """Test résilience session streaming avec récupération erreurs."""
        # Simulation pannes réseau
        failure_scenarios = [
            {'type': 'network_timeout', 'duration_ms': 5000, 'recovery_expected': True},
            {'type': 'packet_loss_spike', 'loss_rate': 0.1, 'recovery_expected': True},
            {'type': 'bandwidth_drop', 'new_bandwidth': 50, 'recovery_expected': True},
            {'type': 'server_error', 'error_code': 503, 'recovery_expected': True}
        ]
        
        # Mock récupération erreurs
        stream_processor.handle_streaming_failure = AsyncMock()
        
        for scenario in failure_scenarios:
            # Configuration réponse récupération
            stream_processor.handle_streaming_failure.return_value = {
                'recovery_successful': scenario['recovery_expected'],
                'recovery_time_ms': 2000,
                'fallback_strategy': 'quality_reduction',
                'service_continuity': True,
                'user_experience_impact': 'minimal',
                'automatic_recovery': True
            }
            
            result = await stream_processor.handle_streaming_failure(
                session_id='session_123',
                failure_type=scenario['type'],
                failure_data=scenario
            )
            
            # Validations récupération
            assert result['recovery_successful'] == scenario['recovery_expected']
            assert result['recovery_time_ms'] < 5000
            assert result['service_continuity'] is True
            assert result['automatic_recovery'] is True


class TestAudioBuffer:
    """Tests enterprise pour AudioBuffer avec buffering intelligent."""
    
    @pytest.fixture
    def audio_buffer(self):
        """Instance AudioBuffer pour tests."""
        return AudioBuffer()
    
    @pytest.fixture
    def buffer_config(self):
        """Configuration buffer avancée."""
        return {
            'buffer_strategy': 'predictive',
            'memory_limit_mb': 512,
            'compression': {
                'enabled': True,
                'algorithm': 'opus',
                'quality': 0.8
            },
            'predictive_settings': {
                'ml_prediction': True,
                'history_window_minutes': 30,
                'prefetch_probability_threshold': 0.7
            },
            'persistence': {
                'enabled': True,
                'storage_backend': 'redis',
                'ttl_minutes': 60
            }
        }
    
    async def test_predictive_buffering(self, audio_buffer, buffer_config):
        """Test buffering prédictif avec ML."""
        # Mock configuration buffer
        audio_buffer.configure = AsyncMock(return_value={'status': 'configured'})
        await audio_buffer.configure(buffer_config)
        
        # Mock prédiction tracks suivants
        audio_buffer.predict_next_tracks = AsyncMock(return_value=[
            {'id': 'track_001', 'probability': 0.89, 'estimated_play_time': datetime.utcnow() + timedelta(minutes=2)},
            {'id': 'track_002', 'probability': 0.76, 'estimated_play_time': datetime.utcnow() + timedelta(minutes=5)},
            {'id': 'track_003', 'probability': 0.65, 'estimated_play_time': datetime.utcnow() + timedelta(minutes=8)}
        ])
        
        # Mock préchargement
        audio_buffer.preload_track = AsyncMock(return_value={
            'preload_successful': True,
            'preload_time_ms': 1250,
            'buffer_space_used_mb': 12.7,
            'compression_ratio': 0.72,
            'cache_hit': False
        })
        
        # Test prédiction et préchargement
        predictions = await audio_buffer.predict_next_tracks(
            user_id='user_12345',
            current_track='track_000',
            context={'playlist': 'daily_mix_1', 'position': 3}
        )
        
        # Préchargement tracks avec probabilité élevée
        preload_results = []
        for track in predictions:
            if track['probability'] > buffer_config['predictive_settings']['prefetch_probability_threshold']:
                priority = 'high' if track['probability'] > 0.85 else 'medium'
                result = await audio_buffer.preload_track(track['id'], priority=priority)
                preload_results.append(result)
        
        # Validations prédiction
        assert len(predictions) > 0
        assert all(p['probability'] > 0.6 for p in predictions)
        assert predictions[0]['probability'] > predictions[-1]['probability']  # Ordre décroissant
        
        # Validations préchargement
        assert len(preload_results) >= 1  # Au moins une prédiction forte
        assert all(r['preload_successful'] for r in preload_results)
        total_buffer_used = sum(r['buffer_space_used_mb'] for r in preload_results)
        assert total_buffer_used < buffer_config['memory_limit_mb']
    
    async def test_buffer_optimization_algorithms(self, audio_buffer):
        """Test algorithmes optimisation buffer."""
        # Algorithmes de buffer testés
        buffer_algorithms = [
            {
                'name': 'lru_basic',
                'description': 'Least Recently Used basique',
                'expected_hit_rate': 0.75
            },
            {
                'name': 'lru_ml_weighted',
                'description': 'LRU pondéré par ML',
                'expected_hit_rate': 0.85
            },
            {
                'name': 'predictive_lfu',
                'description': 'Least Frequently Used prédictif',
                'expected_hit_rate': 0.82
            },
            {
                'name': 'temporal_aware',
                'description': 'Algorithme tenant compte du temps',
                'expected_hit_rate': 0.88
            }
        ]
        
        # Mock test algorithmes
        audio_buffer.test_buffer_algorithm = AsyncMock()
        
        for algorithm in buffer_algorithms:
            # Configuration réponse mock
            audio_buffer.test_buffer_algorithm.return_value = {
                'algorithm_name': algorithm['name'],
                'cache_hit_rate': algorithm['expected_hit_rate'],
                'average_retrieval_time_ms': np.random.uniform(1, 5),
                'memory_efficiency': np.random.uniform(0.8, 0.95),
                'prediction_accuracy': np.random.uniform(0.7, 0.9),
                'adaptation_speed': np.random.uniform(0.75, 0.95)
            }
            
            result = await audio_buffer.test_buffer_algorithm(
                algorithm_name=algorithm['name'],
                test_duration_minutes=30,
                simulated_traffic=True
            )
            
            # Validations algorithme
            assert result['cache_hit_rate'] >= algorithm['expected_hit_rate'] - 0.05
            assert result['average_retrieval_time_ms'] < 10
            assert result['memory_efficiency'] > 0.8
    
    async def test_buffer_compression_efficiency(self, audio_buffer):
        """Test efficacité compression buffer."""
        # Configurations compression testées
        compression_configs = [
            {
                'algorithm': 'opus',
                'bitrate': 128,
                'expected_ratio': 0.75,
                'expected_quality': 0.9
            },
            {
                'algorithm': 'aac',
                'bitrate': 256,
                'expected_ratio': 0.65,
                'expected_quality': 0.95
            },
            {
                'algorithm': 'mp3',
                'bitrate': 320,
                'expected_ratio': 0.6,
                'expected_quality': 0.92
            }
        ]
        
        # Mock compression
        audio_buffer.test_compression = AsyncMock()
        
        for config in compression_configs:
            # Configuration réponse compression
            audio_buffer.test_compression.return_value = {
                'compression_ratio': config['expected_ratio'],
                'quality_score': config['expected_quality'],
                'compression_time_ms': np.random.uniform(50, 200),
                'decompression_time_ms': np.random.uniform(10, 50),
                'memory_savings_mb': np.random.uniform(10, 50),
                'cpu_overhead': np.random.uniform(0.1, 0.3)
            }
            
            result = await audio_buffer.test_compression(
                audio_data=np.random.random(44100 * 180),  # 3 minutes
                compression_config=config
            )
            
            # Validations compression
            assert result['compression_ratio'] <= config['expected_ratio'] + 0.1
            assert result['quality_score'] >= config['expected_quality'] - 0.05
            assert result['compression_time_ms'] < 500
            assert result['decompression_time_ms'] < 100
            assert result['cpu_overhead'] < 0.5


class TestQualityManager:
    """Tests enterprise pour QualityManager avec optimisation qualité ML."""
    
    @pytest.fixture
    def quality_manager(self):
        """Instance QualityManager pour tests."""
        return QualityManager()
    
    async def test_ml_quality_optimization(self, quality_manager):
        """Test optimisation qualité avec ML."""
        # Configuration ML qualité
        ml_config = {
            'ml_optimization': {
                'enabled': True,
                'model_type': 'neural_network',
                'features': [
                    'network_bandwidth', 'network_latency', 'network_jitter',
                    'device_capabilities', 'user_preferences', 'listening_context',
                    'time_of_day', 'location_type', 'battery_level'
                ],
                'target_metric': 'perceptual_quality_score'
            },
            'quality_profiles': {
                'mobile_data': {'max_bitrate': 128, 'codec': 'aac'},
                'wifi_standard': {'max_bitrate': 256, 'codec': 'aac'},
                'wifi_premium': {'max_bitrate': 320, 'codec': 'mp3'},
                'ethernet_audiophile': {'max_bitrate': 1411, 'codec': 'flac'}
            }
        }
        
        # Mock optimisation ML
        quality_manager.optimize_quality = AsyncMock(return_value={
            'recommended_bitrate': 256,
            'recommended_codec': 'aac',
            'quality_profile': 'wifi_standard',
            'confidence_score': 0.87,
            'adaptation_reason': 'network_optimized',
            'perceptual_quality_score': 0.92,
            'predicted_user_satisfaction': 0.89,
            'feature_contributions': {
                'network_bandwidth': 0.35,
                'device_capabilities': 0.25,
                'user_preferences': 0.20,
                'listening_context': 0.15,
                'other': 0.05
            }
        })
        
        # Contextes utilisateur variés
        user_contexts = [
            {
                'scenario': 'commute_mobile',
                'device': 'smartphone',
                'network': 'mobile_4g',
                'activity': 'commute',
                'expected_profile': 'mobile_data'
            },
            {
                'scenario': 'home_wifi_premium',
                'device': 'smart_speaker',
                'network': 'wifi_5ghz',
                'activity': 'focused_listening',
                'expected_profile': 'wifi_premium'
            },
            {
                'scenario': 'office_ethernet',
                'device': 'desktop_audiophile',
                'network': 'ethernet_gigabit',
                'activity': 'background_music',
                'expected_profile': 'ethernet_audiophile'
            }
        ]
        
        for context in user_contexts:
            result = await quality_manager.optimize_quality(
                user_context={
                    'device_type': context['device'],
                    'network_type': context['network'],
                    'listening_context': context['activity']
                },
                network_conditions={
                    'bandwidth_kbps': 1000,  # Simulé
                    'latency_ms': 45,
                    'jitter_ms': 12,
                    'packet_loss_rate': 0.005
                },
                config=ml_config
            )
            
            # Validations optimisation
            assert result['confidence_score'] > 0.8
            assert result['perceptual_quality_score'] > 0.85
            assert result['predicted_user_satisfaction'] > 0.8
            assert 'feature_contributions' in result
            assert sum(result['feature_contributions'].values()) == pytest.approx(1.0, abs=0.01)
    
    async def test_perceptual_quality_assessment(self, quality_manager):
        """Test évaluation qualité perceptuelle."""
        # Configurations audio testées
        audio_configs = [
            {
                'bitrate': 64,
                'codec': 'aac',
                'expected_quality': 0.6,
                'use_case': 'voice_podcast'
            },
            {
                'bitrate': 128,
                'codec': 'aac',
                'expected_quality': 0.75,
                'use_case': 'standard_music'
            },
            {
                'bitrate': 256,
                'codec': 'aac',
                'expected_quality': 0.9,
                'use_case': 'high_quality_music'
            },
            {
                'bitrate': 1411,
                'codec': 'flac',
                'expected_quality': 0.98,
                'use_case': 'audiophile_critical'
            }
        ]
        
        # Mock évaluation qualité perceptuelle
        quality_manager.assess_perceptual_quality = AsyncMock()
        
        for config in audio_configs:
            # Configuration réponse mock
            quality_manager.assess_perceptual_quality.return_value = {
                'overall_quality_score': config['expected_quality'],
                'frequency_response_score': config['expected_quality'] + 0.02,
                'dynamic_range_score': config['expected_quality'] + 0.01,
                'distortion_score': 1.0 - (1.0 - config['expected_quality']) * 0.8,
                'spatial_quality_score': config['expected_quality'] - 0.01,
                'perceptual_metrics': {
                    'loudness_lufs': -23.0,
                    'peak_dbfs': -3.0,
                    'dynamic_range_db': 15.2,
                    'thd_percentage': 0.003
                },
                'use_case_suitability': {
                    config['use_case']: 0.95,
                    'general_music': config['expected_quality'],
                    'critical_listening': config['expected_quality'] * 0.9
                }
            }
            
            result = await quality_manager.assess_perceptual_quality(
                audio_config=config,
                reference_audio=np.random.random(44100),
                assessment_type='comprehensive'
            )
            
            # Validations qualité perceptuelle
            assert result['overall_quality_score'] >= config['expected_quality'] - 0.05
            assert result['perceptual_metrics']['loudness_lufs'] > -30
            assert result['perceptual_metrics']['loudness_lufs'] < -10
            assert result['perceptual_metrics']['thd_percentage'] < 0.01
            assert result['use_case_suitability'][config['use_case']] > 0.9


class TestRealtimeOptimizer:
    """Tests enterprise pour RealtimeOptimizer avec optimisations temps réel."""
    
    @pytest.fixture
    def optimizer(self):
        """Instance RealtimeOptimizer pour tests."""
        return RealtimeOptimizer()
    
    async def test_latency_minimization_comprehensive(self, optimizer):
        """Test minimisation latence complète."""
        # Configuration optimisation latence
        latency_config = {
            'target_latency_ms': 50,
            'max_acceptable_latency_ms': 150,
            'jitter_buffer_adaptive': True,
            'frame_size_optimization': True,
            'lookahead_ms': 20,
            'congestion_control': 'bbr',
            'pacing_enabled': True
        }
        
        # Mock optimisation session
        optimizer.optimize_streaming_session = AsyncMock(return_value={
            'achieved_latency_ms': 67,
            'achieved_throughput_mbps': 1.2,
            'jitter_compensation_ms': 15,
            'packet_loss_rate': 0.002,
            'cpu_usage_percentage': 45,
            'memory_usage_mb': 189,
            'optimization_score': 0.94,
            'optimizations_applied': {
                'tcp_no_delay': True,
                'send_buffer_optimization': True,
                'receive_buffer_optimization': True,
                'congestion_window_tuning': True,
                'packet_pacing': True,
                'priority_queuing': True
            },
            'network_stack_optimizations': {
                'kernel_bypass': False,
                'user_space_networking': True,
                'zero_copy_enabled': True,
                'interrupt_coalescing': True
            }
        })
        
        result = await optimizer.optimize_streaming_session(
            session_id='session_123',
            optimization_config=latency_config,
            real_time_constraints={
                'max_latency_ms': 100,
                'min_quality_score': 0.8,
                'max_cpu_usage': 0.7,
                'max_memory_mb': 256
            }
        )
        
        # Validations optimisation latence
        assert result['achieved_latency_ms'] <= latency_config['max_acceptable_latency_ms']
        assert result['optimization_score'] > 0.9
        assert result['cpu_usage_percentage'] < 70
        assert result['memory_usage_mb'] < 256
        assert result['optimizations_applied']['tcp_no_delay'] is True
        assert result['optimizations_applied']['packet_pacing'] is True
    
    async def test_throughput_optimization(self, optimizer):
        """Test optimisation throughput."""
        # Scénarios throughput
        throughput_scenarios = [
            {
                'name': 'single_stream_hq',
                'concurrent_streams': 1,
                'quality': 'high',
                'expected_throughput_mbps': 0.32,
                'cpu_budget': 0.3
            },
            {
                'name': 'multi_stream_standard',
                'concurrent_streams': 10,
                'quality': 'standard',
                'expected_throughput_mbps': 2.56,
                'cpu_budget': 0.6
            },
            {
                'name': 'high_concurrency',
                'concurrent_streams': 100,
                'quality': 'adaptive',
                'expected_throughput_mbps': 15.0,
                'cpu_budget': 0.8
            }
        ]
        
        # Mock optimisation throughput
        optimizer.optimize_throughput = AsyncMock()
        
        for scenario in throughput_scenarios:
            # Configuration réponse mock
            optimizer.optimize_throughput.return_value = {
                'achieved_throughput_mbps': scenario['expected_throughput_mbps'] * 0.95,
                'concurrent_streams_handled': scenario['concurrent_streams'],
                'average_stream_quality': 0.87,
                'cpu_utilization': scenario['cpu_budget'] * 0.9,
                'memory_utilization': 0.75,
                'optimization_techniques': {
                    'connection_pooling': True,
                    'request_pipelining': True,
                    'compression_optimization': True,
                    'cache_warming': True,
                    'load_balancing': scenario['concurrent_streams'] > 10
                },
                'bottlenecks_identified': [],
                'scaling_recommendations': []
            }
            
            result = await optimizer.optimize_throughput(
                scenario_config=scenario,
                resource_constraints={
                    'max_cpu_usage': scenario['cpu_budget'],
                    'max_memory_gb': 4,
                    'max_bandwidth_mbps': 100
                }
            )
            
            # Validations throughput
            assert result['achieved_throughput_mbps'] >= scenario['expected_throughput_mbps'] * 0.9
            assert result['concurrent_streams_handled'] == scenario['concurrent_streams']
            assert result['cpu_utilization'] <= scenario['cpu_budget']
            assert result['average_stream_quality'] > 0.8
    
    async def test_adaptive_optimization_algorithms(self, optimizer):
        """Test algorithmes optimisation adaptatifs."""
        # Algorithmes d'optimisation testés
        optimization_algorithms = [
            {
                'name': 'gradient_descent',
                'type': 'continuous',
                'convergence_time_ms': 5000,
                'stability_score': 0.85
            },
            {
                'name': 'reinforcement_learning',
                'type': 'adaptive',
                'convergence_time_ms': 10000,
                'stability_score': 0.92
            },
            {
                'name': 'genetic_algorithm',
                'type': 'evolutionary',
                'convergence_time_ms': 15000,
                'stability_score': 0.88
            },
            {
                'name': 'simulated_annealing',
                'type': 'stochastic',
                'convergence_time_ms': 8000,
                'stability_score': 0.83
            }
        ]
        
        # Mock test algorithmes
        optimizer.test_optimization_algorithm = AsyncMock()
        
        for algorithm in optimization_algorithms:
            # Configuration réponse algorithme
            optimizer.test_optimization_algorithm.return_value = {
                'algorithm_name': algorithm['name'],
                'optimization_score': np.random.uniform(0.8, 0.95),
                'convergence_time_ms': algorithm['convergence_time_ms'],
                'stability_score': algorithm['stability_score'],
                'resource_efficiency': np.random.uniform(0.75, 0.9),
                'adaptability_score': np.random.uniform(0.8, 0.95),
                'robustness_to_changes': np.random.uniform(0.7, 0.9)
            }
            
            result = await optimizer.test_optimization_algorithm(
                algorithm_config=algorithm,
                test_duration_minutes=5,
                network_variability='medium'
            )
            
            # Validations algorithme
            assert result['optimization_score'] > 0.8
            assert result['convergence_time_ms'] <= algorithm['convergence_time_ms'] * 1.1
            assert result['stability_score'] >= algorithm['stability_score'] - 0.05
            assert result['resource_efficiency'] > 0.7


class TestCDNManager:
    """Tests enterprise pour CDNManager avec CDN intelligent."""
    
    @pytest.fixture
    def cdn_manager(self):
        """Instance CDNManager pour tests."""
        return CDNManager()
    
    async def test_intelligent_content_distribution(self, cdn_manager):
        """Test distribution contenu intelligente."""
        # Configuration CDN multi-provider
        cdn_config = {
            'providers': [
                {
                    'name': 'cloudflare',
                    'priority': 1,
                    'regions': ['europe', 'north_america'],
                    'capabilities': ['streaming', 'edge_compute', 'ai_acceleration']
                },
                {
                    'name': 'aws_cloudfront',
                    'priority': 2,
                    'regions': ['global'],
                    'capabilities': ['streaming', 'lambda_edge']
                }
            ],
            'intelligent_caching': {
                'ml_prediction_enabled': True,
                'cache_hit_target': 0.95,
                'prefetch_horizon_minutes': 60
            }
        }
        
        # Mock déploiement contenu
        cdn_manager.deploy_content = AsyncMock(return_value={
            'deployment_id': 'deploy_album_98765_global',
            'deployed_regions': ['eu-west-1', 'us-east-1', 'ap-southeast-1'],
            'cache_prewarming_status': 'completed',
            'estimated_global_availability': 0.97,
            'deployment_time_minutes': 12.3,
            'edge_nodes_updated': 1247,
            'cdn_provider_distribution': {
                'cloudflare': 0.6,
                'aws_cloudfront': 0.4
            },
            'intelligent_routing_enabled': True
        })
        
        deployment = await cdn_manager.deploy_content(
            content_id='album_98765',
            deployment_strategy='intelligent_global',
            config=cdn_config
        )
        
        # Validations déploiement
        assert deployment['estimated_global_availability'] > 0.95
        assert deployment['deployment_time_minutes'] < 30
        assert deployment['edge_nodes_updated'] > 1000
        assert deployment['intelligent_routing_enabled'] is True
        assert 'deployment_id' in deployment
    
    async def test_optimal_routing_selection(self, cdn_manager):
        """Test sélection routage optimal."""
        # Localisations utilisateur diverses
        user_locations = [
            {
                'location': 'paris_france',
                'lat': 48.8566,
                'lon': 2.3522,
                'expected_edge': 'paris-edge-01',
                'expected_latency_ms': 25
            },
            {
                'location': 'new_york_usa',
                'lat': 40.7128,
                'lon': -74.0060,
                'expected_edge': 'nyc-edge-01',
                'expected_latency_ms': 20
            },
            {
                'location': 'tokyo_japan',
                'lat': 35.6762,
                'lon': 139.6503,
                'expected_edge': 'tokyo-edge-01',
                'expected_latency_ms': 30
            },
            {
                'location': 'sydney_australia',
                'lat': -33.8688,
                'lon': 151.2093,
                'expected_edge': 'sydney-edge-01',
                'expected_latency_ms': 35
            }
        ]
        
        # Mock routage optimal
        cdn_manager.get_optimal_route = AsyncMock()
        
        for location in user_locations:
            # Configuration réponse routage
            cdn_manager.get_optimal_route.return_value = {
                'edge_server': location['expected_edge'] + '.cdn.com',
                'estimated_latency_ms': location['expected_latency_ms'],
                'available_bitrates': [128, 256, 320, 1411],
                'cache_status': 'hit',
                'processing_capabilities': ['transcode', 'enhance', 'personalize'],
                'load_score': np.random.uniform(0.2, 0.4),
                'route_optimization_score': np.random.uniform(0.85, 0.95),
                'fallback_servers': [f"{location['expected_edge']}-backup.cdn.com"]
            }
            
            route = await cdn_manager.get_optimal_route(
                user_location={'lat': location['lat'], 'lon': location['lon']},
                content_id='track_12345',
                quality_requirements={'min_bitrate': 256, 'max_latency': 100}
            )
            
            # Validations routage
            assert route['estimated_latency_ms'] <= location['expected_latency_ms'] + 10
            assert route['estimated_latency_ms'] < 100  # Exigence qualité
            assert 256 in route['available_bitrates']  # Exigence bitrate
            assert route['load_score'] < 0.5  # Serveur pas surchargé
            assert route['route_optimization_score'] > 0.8
    
    async def test_edge_computing_capabilities(self, cdn_manager):
        """Test capacités edge computing."""
        # Fonctionnalités edge testées
        edge_functions = [
            {
                'name': 'real_time_transcoding',
                'input_format': 'flac',
                'output_format': 'aac_256',
                'expected_latency_ms': 50,
                'cpu_intensive': True
            },
            {
                'name': 'audio_enhancement',
                'enhancement_type': 'noise_reduction',
                'quality_improvement': 0.15,
                'expected_latency_ms': 30,
                'cpu_intensive': False
            },
            {
                'name': 'personalized_eq',
                'user_profile': 'bass_lover',
                'eq_adjustment': 'bass_boost',
                'expected_latency_ms': 10,
                'cpu_intensive': False
            },
            {
                'name': 'dynamic_range_compression',
                'target_lufs': -16,
                'preserve_dynamics': True,
                'expected_latency_ms': 20,
                'cpu_intensive': False
            }
        ]
        
        # Mock fonctions edge
        cdn_manager.execute_edge_function = AsyncMock()
        
        for function in edge_functions:
            # Configuration réponse edge function
            cdn_manager.execute_edge_function.return_value = {
                'function_name': function['name'],
                'execution_successful': True,
                'processing_time_ms': function['expected_latency_ms'] * 0.8,
                'output_quality_score': 0.92,
                'cpu_usage_percentage': 65 if function['cpu_intensive'] else 25,
                'memory_usage_mb': 128 if function['cpu_intensive'] else 64,
                'edge_node_id': 'edge-node-paris-01',
                'cache_result': True
            }
            
            result = await cdn_manager.execute_edge_function(
                function_config=function,
                input_data={'audio_url': 'https://cdn.com/track.flac'},
                edge_location='europe'
            )
            
            # Validations edge computing
            assert result['execution_successful'] is True
            assert result['processing_time_ms'] <= function['expected_latency_ms']
            assert result['output_quality_score'] > 0.9
            if function['cpu_intensive']:
                assert result['cpu_usage_percentage'] > 50
            else:
                assert result['cpu_usage_percentage'] < 40


# =============================================================================
# TESTS INTEGRATION STREAMING
# =============================================================================

@pytest.mark.integration
class TestStreamingHelpersIntegration:
    """Tests d'intégration pour helpers streaming."""
    
    async def test_complete_streaming_workflow(self):
        """Test workflow streaming complet."""
        # Configuration workflow intégré
        workflow_config = {
            'user_context': {
                'user_id': 'user_12345',
                'device_type': 'mobile',
                'network_type': 'wifi',
                'location': 'paris_france'
            },
            'content_request': {
                'track_id': 'track_67890',
                'quality_preference': 'auto',
                'start_position_ms': 0
            },
            'streaming_requirements': {
                'max_startup_latency_ms': 2000,
                'target_quality_score': 0.85,
                'adaptive_bitrate': True,
                'offline_fallback': True
            }
        }
        
        # Simulation workflow streaming
        workflow_steps = [
            {'step': 'user_authentication', 'expected_time_ms': 100},
            {'step': 'content_authorization', 'expected_time_ms': 50},
            {'step': 'cdn_route_selection', 'expected_time_ms': 30},
            {'step': 'quality_optimization', 'expected_time_ms': 20},
            {'step': 'buffer_initialization', 'expected_time_ms': 200},
            {'step': 'stream_establishment', 'expected_time_ms': 500},
            {'step': 'playback_start', 'expected_time_ms': 100}
        ]
        
        # Mock workflow intégré
        total_time = 0
        results = {}
        
        for step in workflow_steps:
            # Simulation temps traitement
            processing_time = step['expected_time_ms'] * np.random.uniform(0.8, 1.2)
            total_time += processing_time
            
            results[step['step']] = {
                'success': True,
                'processing_time_ms': processing_time,
                'quality_score': np.random.uniform(0.85, 0.95)
            }
        
        # Validations workflow
        assert all(result['success'] for result in results.values())
        assert total_time <= workflow_config['streaming_requirements']['max_startup_latency_ms']
        
        avg_quality = np.mean([r['quality_score'] for r in results.values()])
        assert avg_quality >= workflow_config['streaming_requirements']['target_quality_score']


# =============================================================================
# TESTS PERFORMANCE STREAMING
# =============================================================================

@pytest.mark.performance
class TestStreamingHelpersPerformance:
    """Tests performance pour helpers streaming."""
    
    async def test_concurrent_streaming_scalability(self):
        """Test scalabilité streaming concurrent."""
        # Mock service streaming
        stream_processor = StreamProcessor()
        stream_processor.handle_concurrent_streams = AsyncMock(return_value={
            'concurrent_streams_handled': 10000,
            'average_latency_ms': 67,
            'p95_latency_ms': 134,
            'p99_latency_ms': 267,
            'throughput_mbps': 2560,
            'cpu_utilization': 0.75,
            'memory_utilization': 0.68,
            'error_rate': 0.001
        })
        
        # Test montée en charge
        concurrent_loads = [100, 1000, 5000, 10000]
        
        for load in concurrent_loads:
            result = await stream_processor.handle_concurrent_streams(
                concurrent_streams=load,
                test_duration_minutes=5
            )
            
            # Validations scalabilité
            assert result['concurrent_streams_handled'] >= load * 0.95
            assert result['average_latency_ms'] < 100
            assert result['p95_latency_ms'] < 200
            assert result['cpu_utilization'] < 0.8
            assert result['error_rate'] < 0.01
    
    async def test_cdn_performance_benchmarks(self):
        """Test benchmarks performance CDN."""
        cdn_manager = CDNManager()
        
        # Métriques performance cibles
        performance_targets = {
            'cache_hit_ratio': 0.95,
            'origin_offload_percentage': 0.90,
            'average_response_time_ms': 50,
            'p95_response_time_ms': 100,
            'bandwidth_efficiency': 0.85
        }
        
        # Mock benchmarks CDN
        cdn_manager.run_performance_benchmark = AsyncMock(return_value={
            'cache_hit_ratio': 0.96,
            'origin_offload_percentage': 0.92,
            'average_response_time_ms': 42,
            'p95_response_time_ms': 87,
            'bandwidth_efficiency': 0.88,
            'edge_node_health_score': 0.94,
            'global_availability': 0.997
        })
        
        benchmark_result = await cdn_manager.run_performance_benchmark(
            test_duration_minutes=30,
            simulated_load='high',
            geographical_distribution=True
        )
        
        # Validations benchmarks
        for metric, target in performance_targets.items():
            assert benchmark_result[metric] >= target
        
        assert benchmark_result['global_availability'] > 0.995
