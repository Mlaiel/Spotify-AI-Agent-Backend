# 🔬 Test Runner & Configuration Script
# ====================================
# 
# Script principal pour l'exécution et la configuration
# des tests du module temps réel avec toutes les options
# et configurations d'entreprise
#
# 🎖️ Expert: Test Automation Engineer + DevOps Specialist
#
# 👨‍💻 Développé par: Fahed Mlaiel
# ====================================

"""
🚀 Realtime Module Test Runner
==============================

Comprehensive test execution script for the Spotify AI Agent
realtime infrastructure with enterprise-grade configurations,
monitoring, and reporting capabilities.

Features:
- Multi-environment test execution
- Performance benchmarking
- Coverage reporting with thresholds
- Parallel test execution
- CI/CD integration
- Test result analytics
- Automated environment setup
"""

import asyncio
import argparse
import json
import logging
import os
import sys
import time
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import yaml

# Configuration des couleurs pour la sortie console
class Colors:
    """Console color codes"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class TestConfig:
    """Configuration pour l'exécution des tests"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.test_dir = self.base_dir
        self.backend_dir = self.base_dir.parent.parent.parent.parent / "backend"
        self.coverage_threshold = 95
        self.parallel_workers = 4
        self.timeout = 300  # 5 minutes
        
        # Configuration des environnements
        self.environments = {
            "development": {
                "redis_url": "redis://localhost:6379/15",
                "kafka_servers": "localhost:9092",
                "log_level": "DEBUG"
            },
            "testing": {
                "redis_url": "redis://localhost:6379/14",
                "kafka_servers": "localhost:9092",
                "log_level": "INFO"
            },
            "ci": {
                "redis_url": "redis://redis:6379/0",
                "kafka_servers": "kafka:9092",
                "log_level": "WARNING"
            }
        }


class TestRunner:
    """Gestionnaire principal pour l'exécution des tests"""
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.logger = self._setup_logging()
        self.test_results = {}
        self.start_time = None
        self.end_time = None
        
    def _setup_logging(self) -> logging.Logger:
        """Configuration du logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(self.config.base_dir / "test_runner.log")
            ]
        )
        return logging.getLogger(__name__)
    
    def print_colored(self, message: str, color: str = Colors.ENDC):
        """Affichage coloré dans la console"""
        print(f"{color}{message}{Colors.ENDC}")
    
    def print_header(self, title: str):
        """Affichage d'en-tête"""
        self.print_colored("=" * 80, Colors.HEADER)
        self.print_colored(f"🚀 {title}", Colors.HEADER + Colors.BOLD)
        self.print_colored("=" * 80, Colors.HEADER)
    
    def setup_environment(self, env_name: str = "development"):
        """Configuration de l'environnement de test"""
        self.print_header(f"CONFIGURATION ENVIRONNEMENT: {env_name.upper()}")
        
        if env_name not in self.config.environments:
            raise ValueError(f"Environnement inconnu: {env_name}")
        
        env_config = self.config.environments[env_name]
        
        # Définir les variables d'environnement
        env_vars = {
            "REDIS_TEST_URL": env_config["redis_url"],
            "KAFKA_BOOTSTRAP_SERVERS": env_config["kafka_servers"],
            "LOG_LEVEL": env_config["log_level"],
            "JWT_SECRET_KEY": "test-secret-key-ultra-secure-for-testing",
            "SPOTIFY_CLIENT_ID": "test-client-id",
            "SPOTIFY_CLIENT_SECRET": "test-client-secret",
            "ENVIRONMENT": env_name,
            "PYTHONPATH": str(self.config.backend_dir)
        }
        
        for key, value in env_vars.items():
            os.environ[key] = value
            self.print_colored(f"✅ {key} = {value}", Colors.OKGREEN)
        
        self.logger.info(f"Environnement configuré: {env_name}")
    
    def check_dependencies(self) -> bool:
        """Vérification des dépendances"""
        self.print_header("VÉRIFICATION DES DÉPENDANCES")
        
        dependencies = [
            ("python", ["python", "--version"]),
            ("pytest", ["pytest", "--version"]),
            ("redis", ["redis-cli", "ping"]),
        ]
        
        all_ok = True
        
        for name, command in dependencies:
            try:
                result = subprocess.run(
                    command, 
                    capture_output=True, 
                    text=True, 
                    timeout=10
                )
                if result.returncode == 0:
                    self.print_colored(f"✅ {name}: OK", Colors.OKGREEN)
                else:
                    self.print_colored(f"❌ {name}: ERREUR", Colors.FAIL)
                    all_ok = False
            except (subprocess.TimeoutExpired, FileNotFoundError):
                self.print_colored(f"❌ {name}: NON TROUVÉ", Colors.FAIL)
                all_ok = False
        
        return all_ok
    
    def run_test_category(self, category: str, extra_args: List[str] = None) -> Dict[str, Any]:
        """Exécution d'une catégorie de tests"""
        extra_args = extra_args or []
        
        self.print_header(f"EXÉCUTION TESTS: {category.upper()}")
        
        # Construction de la commande pytest
        cmd = [
            "pytest",
            str(self.config.test_dir),
            "-v",
            "--tb=short",
            f"--timeout={self.config.timeout}",
            f"-m", category
        ] + extra_args
        
        # Ajout de coverage pour les tests unitaires et d'intégration
        if category in ["unit", "integration"]:
            cmd.extend([
                "--cov=app.realtime",
                "--cov-report=term-missing",
                f"--cov-fail-under={self.config.coverage_threshold}"
            ])
        
        # Exécution parallèle pour les tests de performance
        if category == "performance":
            cmd.extend(["-n", str(self.config.parallel_workers)])
        
        self.logger.info(f"Commande: {' '.join(cmd)}")
        
        start_time = time.time()
        
        try:
            result = subprocess.run(
                cmd,
                cwd=self.config.backend_dir,
                capture_output=True,
                text=True,
                timeout=self.config.timeout
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Analyse des résultats
            test_result = {
                "category": category,
                "duration": round(duration, 2),
                "return_code": result.returncode,
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
            
            # Extraction des métriques depuis la sortie pytest
            test_result.update(self._parse_pytest_output(result.stdout))
            
            if test_result["success"]:
                self.print_colored(
                    f"✅ Tests {category} réussis en {duration:.2f}s", 
                    Colors.OKGREEN
                )
            else:
                self.print_colored(
                    f"❌ Tests {category} échoués en {duration:.2f}s", 
                    Colors.FAIL
                )
                self.print_colored("STDERR:", Colors.WARNING)
                print(result.stderr)
            
            return test_result
            
        except subprocess.TimeoutExpired:
            self.print_colored(
                f"⏰ Timeout pour les tests {category} après {self.config.timeout}s", 
                Colors.WARNING
            )
            return {
                "category": category,
                "success": False,
                "error": "timeout",
                "duration": self.config.timeout
            }
    
    def _parse_pytest_output(self, output: str) -> Dict[str, Any]:
        """Parse la sortie pytest pour extraire les métriques"""
        metrics = {
            "tests_run": 0,
            "tests_passed": 0,
            "tests_failed": 0,
            "tests_skipped": 0,
            "coverage": 0.0
        }
        
        lines = output.split('\n')
        
        for line in lines:
            # Extraction du résumé des tests
            if "passed" in line and ("failed" in line or "error" in line):
                # Ex: "5 failed, 10 passed in 2.34s"
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == "passed" and i > 0:
                        metrics["tests_passed"] = int(parts[i-1])
                    elif part == "failed" and i > 0:
                        metrics["tests_failed"] = int(parts[i-1])
                    elif part == "skipped" and i > 0:
                        metrics["tests_skipped"] = int(parts[i-1])
            
            # Extraction de la couverture
            elif "TOTAL" in line and "%" in line:
                # Ex: "TOTAL 1234 567 54% 89%"
                parts = line.split()
                for part in parts:
                    if part.endswith('%'):
                        try:
                            metrics["coverage"] = float(part[:-1])
                            break
                        except ValueError:
                            pass
        
        metrics["tests_run"] = (
            metrics["tests_passed"] + 
            metrics["tests_failed"] + 
            metrics["tests_skipped"]
        )
        
        return metrics
    
    def run_full_suite(self, categories: List[str] = None) -> Dict[str, Any]:
        """Exécution complète de la suite de tests"""
        categories = categories or ["unit", "integration", "performance", "security"]
        
        self.start_time = time.time()
        self.print_header("SUITE COMPLÈTE DE TESTS TEMPS RÉEL")
        
        results = {
            "start_time": datetime.now().isoformat(),
            "categories": {},
            "summary": {}
        }
        
        total_tests = 0
        total_passed = 0
        total_failed = 0
        total_duration = 0.0
        
        for category in categories:
            category_result = self.run_test_category(category)
            results["categories"][category] = category_result
            
            if "tests_run" in category_result:
                total_tests += category_result["tests_run"]
                total_passed += category_result["tests_passed"]
                total_failed += category_result["tests_failed"]
            
            if "duration" in category_result:
                total_duration += category_result["duration"]
        
        self.end_time = time.time()
        
        # Résumé global
        results["summary"] = {
            "total_duration": round(total_duration, 2),
            "total_tests": total_tests,
            "total_passed": total_passed,
            "total_failed": total_failed,
            "success_rate": round((total_passed / total_tests * 100) if total_tests > 0 else 0, 2),
            "overall_success": total_failed == 0
        }
        
        results["end_time"] = datetime.now().isoformat()
        
        self._print_summary(results)
        self._save_results(results)
        
        return results
    
    def _print_summary(self, results: Dict[str, Any]):
        """Affichage du résumé des résultats"""
        self.print_header("RÉSUMÉ DES TESTS")
        
        summary = results["summary"]
        
        # Statistiques globales
        self.print_colored(f"📊 Total des tests: {summary['total_tests']}", Colors.OKBLUE)
        self.print_colored(f"✅ Tests réussis: {summary['total_passed']}", Colors.OKGREEN)
        self.print_colored(f"❌ Tests échoués: {summary['total_failed']}", Colors.FAIL if summary['total_failed'] > 0 else Colors.OKGREEN)
        self.print_colored(f"⏱️  Durée totale: {summary['total_duration']}s", Colors.OKBLUE)
        self.print_colored(f"📈 Taux de succès: {summary['success_rate']}%", Colors.OKGREEN if summary['success_rate'] >= 95 else Colors.WARNING)
        
        # Résultats par catégorie
        self.print_colored("\n📋 Résultats par catégorie:", Colors.OKBLUE)
        for category, result in results["categories"].items():
            status = "✅" if result.get("success", False) else "❌"
            duration = result.get("duration", 0)
            tests_run = result.get("tests_run", 0)
            coverage = result.get("coverage", 0)
            
            self.print_colored(
                f"  {status} {category.upper()}: {tests_run} tests en {duration}s (couverture: {coverage}%)",
                Colors.OKGREEN if result.get("success", False) else Colors.FAIL
            )
        
        # Statut final
        if summary["overall_success"]:
            self.print_colored("\n🎉 TOUS LES TESTS SONT PASSÉS!", Colors.OKGREEN + Colors.BOLD)
        else:
            self.print_colored("\n💥 CERTAINS TESTS ONT ÉCHOUÉ!", Colors.FAIL + Colors.BOLD)
    
    def _save_results(self, results: Dict[str, Any]):
        """Sauvegarde des résultats"""
        results_file = self.config.base_dir / f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        self.print_colored(f"📄 Résultats sauvegardés: {results_file}", Colors.OKBLUE)
    
    def run_specific_test(self, test_file: str, test_name: str = None):
        """Exécution d'un test spécifique"""
        test_path = f"{test_file}"
        if test_name:
            test_path += f"::{test_name}"
        
        self.print_header(f"TEST SPÉCIFIQUE: {test_path}")
        
        cmd = [
            "pytest",
            test_path,
            "-v",
            "-s",  # Pas de capture pour voir les prints
            "--tb=long"
        ]
        
        subprocess.run(cmd, cwd=self.config.backend_dir)
    
    def generate_coverage_report(self):
        """Génération du rapport de couverture HTML"""
        self.print_header("GÉNÉRATION RAPPORT DE COUVERTURE")
        
        cmd = [
            "pytest",
            str(self.config.test_dir),
            "--cov=app.realtime",
            "--cov-report=html",
            "--cov-report=xml",
            "-q"  # Mode silencieux
        ]
        
        result = subprocess.run(cmd, cwd=self.config.backend_dir)
        
        if result.returncode == 0:
            coverage_dir = self.config.backend_dir / "htmlcov"
            self.print_colored(f"📊 Rapport de couverture généré: {coverage_dir}/index.html", Colors.OKGREEN)
        else:
            self.print_colored("❌ Erreur lors de la génération du rapport", Colors.FAIL)


def main():
    """Fonction principale"""
    parser = argparse.ArgumentParser(description="Test Runner pour le module temps réel")
    
    parser.add_argument(
        "--env",
        choices=["development", "testing", "ci"],
        default="development",
        help="Environnement de test"
    )
    
    parser.add_argument(
        "--category",
        choices=["unit", "integration", "performance", "security", "ml"],
        help="Catégorie de tests à exécuter"
    )
    
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Générer le rapport de couverture HTML"
    )
    
    parser.add_argument(
        "--check-deps",
        action="store_true",
        help="Vérifier les dépendances seulement"
    )
    
    parser.add_argument(
        "--test",
        help="Exécuter un test spécifique (format: fichier::classe::methode)"
    )
    
    parser.add_argument(
        "--parallel",
        type=int,
        default=4,
        help="Nombre de workers parallèles"
    )
    
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout en secondes"
    )
    
    args = parser.parse_args()
    
    # Configuration
    config = TestConfig()
    config.parallel_workers = args.parallel
    config.timeout = args.timeout
    
    # Initialisation du runner
    runner = TestRunner(config)
    
    try:
        # Configuration de l'environnement
        runner.setup_environment(args.env)
        
        # Vérification des dépendances
        if args.check_deps:
            if runner.check_dependencies():
                runner.print_colored("✅ Toutes les dépendances sont OK", Colors.OKGREEN)
                sys.exit(0)
            else:
                runner.print_colored("❌ Certaines dépendances manquent", Colors.FAIL)
                sys.exit(1)
        
        # Vérification automatique des dépendances
        if not runner.check_dependencies():
            runner.print_colored("⚠️  Certaines dépendances manquent, mais on continue...", Colors.WARNING)
        
        # Exécution d'un test spécifique
        if args.test:
            if "::" in args.test:
                test_file, test_name = args.test.split("::", 1)
            else:
                test_file, test_name = args.test, None
            runner.run_specific_test(test_file, test_name)
            return
        
        # Génération du rapport de couverture
        if args.coverage:
            runner.generate_coverage_report()
            return
        
        # Exécution d'une catégorie spécifique
        if args.category:
            result = runner.run_test_category(args.category)
            sys.exit(0 if result["success"] else 1)
        
        # Exécution complète par défaut
        results = runner.run_full_suite()
        sys.exit(0 if results["summary"]["overall_success"] else 1)
        
    except KeyboardInterrupt:
        runner.print_colored("\n⚠️  Interruption par l'utilisateur", Colors.WARNING)
        sys.exit(1)
    except Exception as e:
        runner.print_colored(f"\n❌ Erreur inattendue: {e}", Colors.FAIL)
        runner.logger.exception("Erreur lors de l'exécution des tests")
        sys.exit(1)


if __name__ == "__main__":
    main()
