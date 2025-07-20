#!/usr/bin/env python3
"""
🚀 Context Performance Benchmark Script
=====================================

Script d'évaluation des performances du système de contexte tenant
avec métriques détaillées et reporting automatique.

Author: Lead Dev + Architecte IA - Fahed Mlaiel
"""

import asyncio
import time
import statistics
import json
from datetime import datetime, timezone
from typing import Dict, List, Any
import argparse
import sys
import os

# Ajouter le chemin du module au PYTHONPATH
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from app.tenancy.data_isolation.core import (
    ContextManager,
    TenantContext,
    TenantType,
    IsolationLevel,
    PerformanceOptimizer
)


class PerformanceBenchmark:
    """Benchmark complet des performances du système de contexte"""
    
    def __init__(self):
        self.context_manager = ContextManager()
        self.optimizer = PerformanceOptimizer()
        self.results = {}
        
        # Contextes de test
        self.test_contexts = self._create_test_contexts()
    
    def _create_test_contexts(self) -> List[TenantContext]:
        """Crée des contextes de test variés"""
        contexts = []
        
        tenant_types = [
            TenantType.SPOTIFY_ARTIST,
            TenantType.RECORD_LABEL,
            TenantType.MUSIC_PRODUCER,
            TenantType.ENTERPRISE
        ]
        
        isolation_levels = [
            IsolationLevel.BASIC,
            IsolationLevel.STRICT,
            IsolationLevel.PARANOID
        ]
        
        for i in range(50):  # 50 contextes de test
            tenant_type = tenant_types[i % len(tenant_types)]
            isolation_level = isolation_levels[i % len(isolation_levels)]
            
            context = TenantContext(
                tenant_id=f"test_tenant_{i:03d}",
                tenant_type=tenant_type,
                isolation_level=isolation_level
            )
            contexts.append(context)
        
        return contexts
    
    async def benchmark_context_switching(self, iterations: int = 1000) -> Dict[str, Any]:
        """Benchmark du changement de contexte"""
        print(f"🔄 Benchmarking context switching ({iterations} iterations)...")
        
        times = []
        successful_switches = 0
        failed_switches = 0
        
        for i in range(iterations):
            context = self.test_contexts[i % len(self.test_contexts)]
            
            start_time = time.perf_counter()
            try:
                result = await self.context_manager.set_context(context)
                end_time = time.perf_counter()
                
                switch_time = (end_time - start_time) * 1000  # ms
                times.append(switch_time)
                
                if result['success']:
                    successful_switches += 1
                else:
                    failed_switches += 1
                    
            except Exception as e:
                failed_switches += 1
                print(f"   ❌ Switch failed for {context.tenant_id}: {e}")
        
        if times:
            results = {
                'total_iterations': iterations,
                'successful_switches': successful_switches,
                'failed_switches': failed_switches,
                'success_rate': (successful_switches / iterations) * 100,
                'avg_time_ms': statistics.mean(times),
                'median_time_ms': statistics.median(times),
                'min_time_ms': min(times),
                'max_time_ms': max(times),
                'std_dev_ms': statistics.stdev(times) if len(times) > 1 else 0,
                'p95_time_ms': self._percentile(times, 95),
                'p99_time_ms': self._percentile(times, 99)
            }
        else:
            results = {'error': 'No successful context switches'}
        
        print(f"   ✅ Context switching benchmark completed")
        print(f"   📊 Success rate: {results.get('success_rate', 0):.1f}%")
        print(f"   ⏱️  Average time: {results.get('avg_time_ms', 0):.2f}ms")
        
        return results
    
    async def benchmark_concurrent_contexts(self, concurrent_count: int = 100) -> Dict[str, Any]:
        """Benchmark des contextes concurrents"""
        print(f"🔀 Benchmarking concurrent contexts ({concurrent_count} concurrent)...")
        
        async def switch_context_task(context_index: int) -> Dict[str, Any]:
            context = self.test_contexts[context_index % len(self.test_contexts)]
            
            start_time = time.perf_counter()
            try:
                result = await self.context_manager.set_context(context)
                end_time = time.perf_counter()
                
                return {
                    'success': result['success'],
                    'time_ms': (end_time - start_time) * 1000,
                    'context_id': context.tenant_id
                }
            except Exception as e:
                return {
                    'success': False,
                    'error': str(e),
                    'time_ms': 0,
                    'context_id': context.tenant_id
                }
        
        # Exécution concurrente
        start_time = time.perf_counter()
        tasks = [switch_context_task(i) for i in range(concurrent_count)]
        task_results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = time.perf_counter()
        
        # Analyse des résultats
        successful_tasks = [r for r in task_results if isinstance(r, dict) and r.get('success')]
        failed_tasks = [r for r in task_results if not isinstance(r, dict) or not r.get('success')]
        
        times = [r['time_ms'] for r in successful_tasks]
        
        results = {
            'concurrent_count': concurrent_count,
            'successful_tasks': len(successful_tasks),
            'failed_tasks': len(failed_tasks),
            'success_rate': (len(successful_tasks) / concurrent_count) * 100,
            'total_execution_time_ms': (end_time - start_time) * 1000,
            'avg_task_time_ms': statistics.mean(times) if times else 0,
            'max_task_time_ms': max(times) if times else 0,
            'throughput_ops_per_sec': concurrent_count / (end_time - start_time)
        }
        
        print(f"   ✅ Concurrent contexts benchmark completed")
        print(f"   📊 Success rate: {results['success_rate']:.1f}%")
        print(f"   🚀 Throughput: {results['throughput_ops_per_sec']:.1f} ops/sec")
        
        return results
    
    async def benchmark_optimization_performance(self, iterations: int = 500) -> Dict[str, Any]:
        """Benchmark des optimisations de performance"""
        print(f"⚡ Benchmarking performance optimization ({iterations} iterations)...")
        
        optimization_times = []
        cache_hits = []
        optimization_gains = []
        
        for i in range(iterations):
            context = self.test_contexts[i % len(self.test_contexts)]
            
            # Données de test simulées
            test_data = {
                'operation_type': 'data_query',
                'query': f'SELECT * FROM tracks WHERE tenant_id = "{context.tenant_id}"',
                'data_size': 1024 * (i % 10 + 1)  # Taille variable
            }
            
            start_time = time.perf_counter()
            try:
                result = await self.optimizer.optimize_operation(
                    operation_type=test_data['operation_type'],
                    context=context,
                    data=test_data,
                    query=test_data['query']
                )
                end_time = time.perf_counter()
                
                optimization_time = (end_time - start_time) * 1000
                optimization_times.append(optimization_time)
                
                cache_hits.append(1 if result.get('from_cache') else 0)
                optimization_gains.append(result.get('performance_gain_percent', 0))
                
            except Exception as e:
                print(f"   ❌ Optimization failed: {e}")
        
        if optimization_times:
            results = {
                'total_iterations': iterations,
                'avg_optimization_time_ms': statistics.mean(optimization_times),
                'cache_hit_rate': (sum(cache_hits) / len(cache_hits)) * 100,
                'avg_performance_gain': statistics.mean(optimization_gains),
                'max_performance_gain': max(optimization_gains),
                'optimization_overhead_ms': statistics.mean(optimization_times)
            }
        else:
            results = {'error': 'No successful optimizations'}
        
        print(f"   ✅ Optimization benchmark completed")
        print(f"   📈 Cache hit rate: {results.get('cache_hit_rate', 0):.1f}%")
        print(f"   🎯 Avg performance gain: {results.get('avg_performance_gain', 0):.1f}%")
        
        return results
    
    async def benchmark_memory_usage(self) -> Dict[str, Any]:
        """Benchmark de l'utilisation mémoire"""
        print("🧠 Benchmarking memory usage...")
        
        import psutil
        import gc
        
        process = psutil.Process()
        
        # Mesure initiale
        gc.collect()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Créer de nombreux contextes
        contexts_created = []
        for i in range(1000):
            context = TenantContext(
                tenant_id=f"memory_test_{i}",
                tenant_type=TenantType.SPOTIFY_ARTIST,
                isolation_level=IsolationLevel.STRICT
            )
            contexts_created.append(context)
            
            # Activer le contexte
            await self.context_manager.set_context(context)
        
        # Mesure après création
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Nettoyage
        contexts_created.clear()
        gc.collect()
        
        # Mesure après nettoyage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        results = {
            'initial_memory_mb': initial_memory,
            'peak_memory_mb': peak_memory,
            'final_memory_mb': final_memory,
            'memory_increase_mb': peak_memory - initial_memory,
            'memory_per_context_kb': ((peak_memory - initial_memory) * 1024) / 1000,
            'memory_leak_mb': final_memory - initial_memory,
            'contexts_created': 1000
        }
        
        print(f"   ✅ Memory benchmark completed")
        print(f"   📊 Memory per context: {results['memory_per_context_kb']:.2f}KB")
        print(f"   🔍 Memory leak: {results['memory_leak_mb']:.2f}MB")
        
        return results
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calcule un percentile"""
        if not data:
            return 0.0
        
        sorted_data = sorted(data)
        index = (percentile / 100) * (len(sorted_data) - 1)
        
        if index.is_integer():
            return sorted_data[int(index)]
        else:
            lower = sorted_data[int(index)]
            upper = sorted_data[int(index) + 1]
            fraction = index - int(index)
            return lower + fraction * (upper - lower)
    
    async def run_full_benchmark(self) -> Dict[str, Any]:
        """Exécute le benchmark complet"""
        print("🚀 Starting comprehensive performance benchmark...")
        print("=" * 60)
        
        start_time = time.perf_counter()
        
        # Exécution des benchmarks
        self.results['context_switching'] = await self.benchmark_context_switching()
        self.results['concurrent_contexts'] = await self.benchmark_concurrent_contexts()
        self.results['optimization_performance'] = await self.benchmark_optimization_performance()
        self.results['memory_usage'] = await self.benchmark_memory_usage()
        
        end_time = time.perf_counter()
        
        # Méta-informations
        self.results['benchmark_metadata'] = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'total_duration_seconds': end_time - start_time,
            'test_contexts_count': len(self.test_contexts),
            'python_version': sys.version,
            'benchmark_version': '2.0.0'
        }
        
        print("=" * 60)
        print("🏁 Benchmark completed!")
        
        return self.results
    
    def generate_report(self, output_file: str = None) -> str:
        """Génère un rapport de benchmark"""
        if not self.results:
            return "No benchmark results available"
        
        report_lines = [
            "# 📊 Context Performance Benchmark Report",
            "",
            f"**Generated:** {self.results['benchmark_metadata']['timestamp']}",
            f"**Duration:** {self.results['benchmark_metadata']['total_duration_seconds']:.2f}s",
            "",
            "## 🔄 Context Switching Performance",
            ""
        ]
        
        # Context Switching
        if 'context_switching' in self.results:
            cs = self.results['context_switching']
            report_lines.extend([
                f"- **Success Rate:** {cs.get('success_rate', 0):.1f}%",
                f"- **Average Time:** {cs.get('avg_time_ms', 0):.2f}ms",
                f"- **Median Time:** {cs.get('median_time_ms', 0):.2f}ms",
                f"- **95th Percentile:** {cs.get('p95_time_ms', 0):.2f}ms",
                f"- **99th Percentile:** {cs.get('p99_time_ms', 0):.2f}ms",
                ""
            ])
        
        # Concurrent Contexts
        if 'concurrent_contexts' in self.results:
            cc = self.results['concurrent_contexts']
            report_lines.extend([
                "## 🔀 Concurrent Contexts Performance",
                "",
                f"- **Success Rate:** {cc.get('success_rate', 0):.1f}%",
                f"- **Throughput:** {cc.get('throughput_ops_per_sec', 0):.1f} ops/sec",
                f"- **Average Task Time:** {cc.get('avg_task_time_ms', 0):.2f}ms",
                ""
            ])
        
        # Optimization Performance
        if 'optimization_performance' in self.results:
            op = self.results['optimization_performance']
            report_lines.extend([
                "## ⚡ Optimization Performance",
                "",
                f"- **Cache Hit Rate:** {op.get('cache_hit_rate', 0):.1f}%",
                f"- **Average Performance Gain:** {op.get('avg_performance_gain', 0):.1f}%",
                f"- **Optimization Overhead:** {op.get('optimization_overhead_ms', 0):.2f}ms",
                ""
            ])
        
        # Memory Usage
        if 'memory_usage' in self.results:
            mu = self.results['memory_usage']
            report_lines.extend([
                "## 🧠 Memory Usage",
                "",
                f"- **Memory per Context:** {mu.get('memory_per_context_kb', 0):.2f}KB",
                f"- **Peak Memory:** {mu.get('peak_memory_mb', 0):.1f}MB",
                f"- **Memory Leak:** {mu.get('memory_leak_mb', 0):.2f}MB",
                ""
            ])
        
        report_lines.extend([
            "## 📋 Summary",
            "",
            "### ✅ Performance Targets",
            "- Context switching: < 1ms ✓" if cs.get('avg_time_ms', 999) < 1 else "- Context switching: < 1ms ❌",
            "- Cache hit rate: > 80% ✓" if op.get('cache_hit_rate', 0) > 80 else "- Cache hit rate: > 80% ❌",
            "- Memory per context: < 50KB ✓" if mu.get('memory_per_context_kb', 999) < 50 else "- Memory per context: < 50KB ❌",
            ""
        ])
        
        report = "\n".join(report_lines)
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"📝 Report saved to: {output_file}")
        
        return report


async def main():
    """Fonction principale"""
    parser = argparse.ArgumentParser(description='Context Performance Benchmark')
    parser.add_argument('--output', '-o', help='Output file for report')
    parser.add_argument('--json', action='store_true', help='Output results as JSON')
    parser.add_argument('--quick', action='store_true', help='Quick benchmark (fewer iterations)')
    
    args = parser.parse_args()
    
    # Initialisation du benchmark
    benchmark = PerformanceBenchmark()
    
    try:
        # Exécution du benchmark
        if args.quick:
            print("🚀 Running quick benchmark...")
            benchmark.results['context_switching'] = await benchmark.benchmark_context_switching(100)
            benchmark.results['concurrent_contexts'] = await benchmark.benchmark_concurrent_contexts(20)
        else:
            await benchmark.run_full_benchmark()
        
        # Génération du rapport
        if args.json:
            output = json.dumps(benchmark.results, indent=2)
            print(output)
            
            if args.output:
                with open(args.output, 'w') as f:
                    f.write(output)
        else:
            report = benchmark.generate_report(args.output)
            if not args.output:
                print(report)
        
        # Nettoyage
        await benchmark.context_manager.shutdown()
        
    except KeyboardInterrupt:
        print("\n🛑 Benchmark interrupted by user")
        await benchmark.context_manager.shutdown()
        sys.exit(1)
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        await benchmark.context_manager.shutdown()
        sys.exit(1)


if __name__ == '__main__':
    asyncio.run(main())
