"""
Analytics Tasks Package

Zentraler Einstiegspunkt für alle Analytics-Task-Module:
- Datenaggregation, Performance-Analyse, Report-Generierung, Trend-Berechnung
- Siehe README für Details
"""
from .data_aggregation import aggregate_data_task
from .performance_analysis import analyze_performance_task
from .report_generation import generate_report_task
from .trend_calculation import calculate_trends_task

__all__ = [
    "aggregate_data_task",
    "analyze_performance_task",
    "generate_report_task",
    "calculate_trends_task",
]
