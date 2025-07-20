"""
Spotify AI Agent - Business Intelligence Formatters
==================================================

Ultra-advanced business intelligence formatting system for executive dashboards,
KPI reports, financial analytics, and strategic insights.

This module handles complex BI formatting for:
- Executive dashboard visualizations
- Artist performance analytics and insights
- Revenue reporting and financial KPIs
- User engagement and behavioral analytics
- ML model performance business impact
- Market trend analysis and competitive intelligence
- Playlist analytics and recommendation effectiveness

Author: Fahed Mlaiel & Spotify AI Team
Version: 2.1.0
"""

import asyncio
import json
import math
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from decimal import Decimal, ROUND_HALF_UP
import structlog

logger = structlog.get_logger(__name__)


class BusinessMetricType(Enum):
    """Business metric types with descriptions."""
    KPI = ("kpi", "Key Performance Indicator")
    FINANCIAL = ("financial", "Financial Metric")
    ENGAGEMENT = ("engagement", "User Engagement Metric")
    PERFORMANCE = ("performance", "Performance Metric")
    GROWTH = ("growth", "Growth Metric")
    EFFICIENCY = ("efficiency", "Efficiency Metric")
    QUALITY = ("quality", "Quality Metric")


class ReportType(Enum):
    """Business report types."""
    EXECUTIVE_SUMMARY = "executive_summary"
    DETAILED_ANALYTICS = "detailed_analytics"
    COMPARATIVE_ANALYSIS = "comparative_analysis"
    TREND_ANALYSIS = "trend_analysis"
    PERFORMANCE_REVIEW = "performance_review"
    STRATEGIC_INSIGHTS = "strategic_insights"


class TimeGranularity(Enum):
    """Time granularity for business reports."""
    REAL_TIME = "real_time"
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"


@dataclass
class BusinessMetric:
    """Business metric data structure."""
    
    name: str
    value: Union[int, float, Decimal]
    metric_type: BusinessMetricType
    timestamp: datetime
    period: str
    target: Optional[Union[int, float]] = None
    previous_value: Optional[Union[int, float]] = None
    unit: Optional[str] = None
    currency: str = "USD"
    region: Optional[str] = None
    segment: Optional[str] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    
    @property
    def variance(self) -> Optional[float]:
        """Calculate variance from previous period."""
        if self.previous_value is not None and self.previous_value != 0:
            return ((float(self.value) - float(self.previous_value)) / float(self.previous_value)) * 100
        return None
    
    @property
    def target_achievement(self) -> Optional[float]:
        """Calculate target achievement percentage."""
        if self.target is not None and self.target != 0:
            return (float(self.value) / float(self.target)) * 100
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "value": float(self.value),
            "type": self.metric_type.value[0],
            "timestamp": self.timestamp.isoformat(),
            "period": self.period,
            "target": float(self.target) if self.target else None,
            "previous_value": float(self.previous_value) if self.previous_value else None,
            "variance": self.variance,
            "target_achievement": self.target_achievement,
            "unit": self.unit,
            "currency": self.currency,
            "region": self.region,
            "segment": self.segment,
            "confidence_interval": self.confidence_interval
        }


@dataclass
class FormattedBusinessReport:
    """Container for formatted business report."""
    
    title: str
    executive_summary: str
    content: str
    charts: List[Dict[str, Any]] = field(default_factory=list)
    tables: List[Dict[str, Any]] = field(default_factory=list)
    insights: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "title": self.title,
            "executive_summary": self.executive_summary,
            "content": self.content,
            "charts": self.charts,
            "tables": self.tables,
            "insights": self.insights,
            "recommendations": self.recommendations,
            "metadata": self.metadata
        }


class BaseBusinessFormatter:
    """Base class for business intelligence formatters."""
    
    def __init__(self, tenant_id: str, config: Optional[Dict[str, Any]] = None):
        self.tenant_id = tenant_id
        self.config = config or {}
        self.logger = logger.bind(tenant_id=tenant_id, formatter=self.__class__.__name__)
        self.currency = config.get('currency', 'USD')
        self.locale = config.get('locale', 'en_US')
        
    def format_currency(self, amount: Union[int, float, Decimal], currency: Optional[str] = None) -> str:
        """Format currency amount with proper locale."""
        currency = currency or self.currency
        
        if currency == 'USD':
            return f"${amount:,.2f}"
        elif currency == 'EUR':
            return f"€{amount:,.2f}"
        elif currency == 'GBP':
            return f"£{amount:,.2f}"
        else:
            return f"{amount:,.2f} {currency}"
    
    def format_percentage(self, value: float, precision: int = 1) -> str:
        """Format percentage with proper precision."""
        return f"{value:.{precision}f}%"
    
    def format_large_number(self, value: Union[int, float], precision: int = 1) -> str:
        """Format large numbers with appropriate suffixes."""
        if abs(value) >= 1_000_000_000:
            return f"{value/1_000_000_000:.{precision}f}B"
        elif abs(value) >= 1_000_000:
            return f"{value/1_000_000:.{precision}f}M"
        elif abs(value) >= 1_000:
            return f"{value/1_000:.{precision}f}K"
        else:
            return f"{value:.{precision}f}"
    
    def calculate_growth_rate(self, current: float, previous: float) -> Optional[float]:
        """Calculate growth rate between periods."""
        if previous == 0:
            return None
        return ((current - previous) / previous) * 100
    
    def get_variance_indicator(self, variance: Optional[float]) -> str:
        """Get visual indicator for variance."""
        if variance is None:
            return "➖"
        elif variance > 5:
            return "📈"
        elif variance < -5:
            return "📉"
        else:
            return "➖"


class BusinessIntelligenceFormatter(BaseBusinessFormatter):
    """Main business intelligence formatter for comprehensive reports."""
    
    async def format_executive_dashboard(self, data: Dict[str, Any]) -> FormattedBusinessReport:
        """Format executive dashboard with key metrics and insights."""
        
        title = f"Spotify AI Executive Dashboard - {self.tenant_id}"
        timestamp = datetime.now(timezone.utc)
        
        # Extract key metrics
        key_metrics = data.get('key_metrics', {})
        
        # Generate executive summary
        executive_summary = await self._generate_executive_summary(key_metrics, data)
        
        # Create visualizations
        charts = await self._create_executive_charts(key_metrics, data)
        
        # Generate insights
        insights = await self._analyze_business_insights(key_metrics, data)
        
        # Create recommendations
        recommendations = await self._generate_recommendations(insights, data)
        
        # Format main content
        content = await self._format_executive_content(key_metrics, insights, data)
        
        metadata = {
            "report_type": ReportType.EXECUTIVE_SUMMARY.value,
            "generated_at": timestamp.isoformat(),
            "period": data.get('period', 'current'),
            "tenant_id": self.tenant_id,
            "metrics_count": len(key_metrics),
            "currency": self.currency
        }
        
        return FormattedBusinessReport(
            title=title,
            executive_summary=executive_summary,
            content=content,
            charts=charts,
            insights=insights,
            recommendations=recommendations,
            metadata=metadata
        )
    
    async def _generate_executive_summary(self, metrics: Dict[str, Any], data: Dict[str, Any]) -> str:
        """Generate executive summary from key metrics."""
        
        summary_parts = []
        
        # Revenue summary
        if 'revenue' in metrics:
            revenue = metrics['revenue']
            revenue_str = self.format_currency(revenue['current'])
            
            if revenue.get('previous'):
                growth = self.calculate_growth_rate(revenue['current'], revenue['previous'])
                if growth:
                    growth_str = self.format_percentage(growth)
                    indicator = self.get_variance_indicator(growth)
                    summary_parts.append(f"Revenue: {revenue_str} ({indicator} {growth_str} vs previous period)")
                else:
                    summary_parts.append(f"Revenue: {revenue_str}")
            else:
                summary_parts.append(f"Revenue: {revenue_str}")
        
        # User metrics summary
        if 'users' in metrics:
            users = metrics['users']
            if 'active_users' in users:
                active_users = self.format_large_number(users['active_users'])
                summary_parts.append(f"Active Users: {active_users}")
            
            if 'engagement_rate' in users:
                engagement = self.format_percentage(users['engagement_rate'])
                summary_parts.append(f"Engagement Rate: {engagement}")
        
        # AI Performance summary
        if 'ai_performance' in metrics:
            ai_metrics = metrics['ai_performance']
            if 'model_accuracy' in ai_metrics:
                accuracy = self.format_percentage(ai_metrics['model_accuracy'])
                summary_parts.append(f"AI Model Accuracy: {accuracy}")
            
            if 'recommendation_hit_rate' in ai_metrics:
                hit_rate = self.format_percentage(ai_metrics['recommendation_hit_rate'])
                summary_parts.append(f"Recommendation Hit Rate: {hit_rate}")
        
        # Business impact summary
        business_impact = data.get('business_impact', {})
        if business_impact:
            impact_items = []
            
            if 'cost_savings' in business_impact:
                savings = self.format_currency(business_impact['cost_savings'])
                impact_items.append(f"Cost Savings: {savings}")
            
            if 'efficiency_improvement' in business_impact:
                efficiency = self.format_percentage(business_impact['efficiency_improvement'])
                impact_items.append(f"Efficiency Improvement: {efficiency}")
            
            if impact_items:
                summary_parts.append(f"Business Impact: {', '.join(impact_items)}")
        
        # Combine summary
        if summary_parts:
            summary = "🎵 **Spotify AI Performance Summary**\n\n"
            summary += "\n".join([f"• {part}" for part in summary_parts])
            
            # Add period context
            period = data.get('period', 'current period')
            summary += f"\n\n*Report covers: {period}*"
            
            return summary
        else:
            return "No key metrics available for executive summary."
    
    async def _create_executive_charts(self, metrics: Dict[str, Any], data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create chart configurations for executive dashboard."""
        
        charts = []
        
        # Revenue trend chart
        if 'revenue_trend' in data:
            revenue_chart = {
                "type": "line",
                "title": "Revenue Trend",
                "data": {
                    "labels": data['revenue_trend']['periods'],
                    "datasets": [{
                        "label": "Revenue",
                        "data": data['revenue_trend']['values'],
                        "borderColor": "#1DB954",
                        "backgroundColor": "rgba(29, 185, 84, 0.1)",
                        "tension": 0.4
                    }]
                },
                "options": {
                    "responsive": True,
                    "plugins": {
                        "title": {"display": True, "text": "Revenue Trend"},
                        "legend": {"display": False}
                    },
                    "scales": {
                        "y": {
                            "beginAtZero": True,
                            "ticks": {
                                "callback": "function(value) { return '$' + value.toLocaleString(); }"
                            }
                        }
                    }
                }
            }
            charts.append(revenue_chart)
        
        # User engagement metrics
        if 'user_metrics' in data:
            user_data = data['user_metrics']
            engagement_chart = {
                "type": "doughnut",
                "title": "User Engagement Breakdown",
                "data": {
                    "labels": ["Highly Engaged", "Moderately Engaged", "Low Engagement"],
                    "datasets": [{
                        "data": [
                            user_data.get('highly_engaged', 0),
                            user_data.get('moderately_engaged', 0),
                            user_data.get('low_engaged', 0)
                        ],
                        "backgroundColor": ["#1DB954", "#1ED760", "#A0D468"]
                    }]
                },
                "options": {
                    "responsive": True,
                    "plugins": {
                        "title": {"display": True, "text": "User Engagement Distribution"},
                        "legend": {"position": "bottom"}
                    }
                }
            }
            charts.append(engagement_chart)
        
        # AI Performance radar chart
        if 'ai_performance' in metrics:
            ai_data = metrics['ai_performance']
            ai_chart = {
                "type": "radar",
                "title": "AI Performance Metrics",
                "data": {
                    "labels": ["Accuracy", "Precision", "Recall", "F1-Score", "Latency"],
                    "datasets": [{
                        "label": "Current Performance",
                        "data": [
                            ai_data.get('accuracy', 0) * 100,
                            ai_data.get('precision', 0) * 100,
                            ai_data.get('recall', 0) * 100,
                            ai_data.get('f1_score', 0) * 100,
                            ai_data.get('latency_score', 0) * 100
                        ],
                        "backgroundColor": "rgba(29, 185, 84, 0.2)",
                        "borderColor": "#1DB954",
                        "pointBackgroundColor": "#1DB954"
                    }]
                },
                "options": {
                    "responsive": True,
                    "plugins": {
                        "title": {"display": True, "text": "AI Model Performance"},
                        "legend": {"display": False}
                    },
                    "scales": {
                        "r": {
                            "beginAtZero": True,
                            "max": 100,
                            "ticks": {"stepSize": 20}
                        }
                    }
                }
            }
            charts.append(ai_chart)
        
        # Top content performance
        if 'top_content' in data:
            content_data = data['top_content']
            content_chart = {
                "type": "bar",
                "title": "Top Performing Content",
                "data": {
                    "labels": [item['name'] for item in content_data[:10]],
                    "datasets": [{
                        "label": "Engagement Score",
                        "data": [item['score'] for item in content_data[:10]],
                        "backgroundColor": "#1DB954",
                        "borderColor": "#169C47",
                        "borderWidth": 1
                    }]
                },
                "options": {
                    "responsive": True,
                    "plugins": {
                        "title": {"display": True, "text": "Top 10 Content by Engagement"},
                        "legend": {"display": False}
                    },
                    "scales": {
                        "x": {"ticks": {"maxRotation": 45}},
                        "y": {"beginAtZero": True}
                    }
                }
            }
            charts.append(content_chart)
        
        return charts
    
    async def _analyze_business_insights(self, metrics: Dict[str, Any], data: Dict[str, Any]) -> List[str]:
        """Analyze metrics and generate business insights."""
        
        insights = []
        
        # Revenue insights
        if 'revenue' in metrics:
            revenue = metrics['revenue']
            if revenue.get('current') and revenue.get('previous'):
                growth = self.calculate_growth_rate(revenue['current'], revenue['previous'])
                if growth and growth > 10:
                    insights.append(f"🚀 Strong revenue growth of {self.format_percentage(growth)} indicates successful AI implementation and user adoption.")
                elif growth and growth < -5:
                    insights.append(f"⚠️ Revenue decline of {self.format_percentage(abs(growth))} requires immediate attention and strategy adjustment.")
                elif growth and abs(growth) < 2:
                    insights.append(f"📊 Revenue remains stable with minimal variance ({self.format_percentage(growth)}), indicating consistent performance.")
        
        # AI Performance insights
        if 'ai_performance' in metrics:
            ai_metrics = metrics['ai_performance']
            
            accuracy = ai_metrics.get('model_accuracy', 0)
            if accuracy > 0.9:
                insights.append(f"🎯 Exceptional AI model accuracy of {self.format_percentage(accuracy * 100)} demonstrates mature ML capabilities.")
            elif accuracy < 0.8:
                insights.append(f"🔧 AI model accuracy of {self.format_percentage(accuracy * 100)} is below optimal threshold and needs improvement.")
            
            hit_rate = ai_metrics.get('recommendation_hit_rate', 0)
            if hit_rate > 0.7:
                insights.append(f"✨ High recommendation hit rate of {self.format_percentage(hit_rate * 100)} shows excellent personalization effectiveness.")
            elif hit_rate < 0.5:
                insights.append(f"📈 Recommendation hit rate of {self.format_percentage(hit_rate * 100)} has room for improvement through better training data.")
        
        # User engagement insights
        if 'users' in metrics:
            user_metrics = metrics['users']
            
            engagement_rate = user_metrics.get('engagement_rate', 0)
            if engagement_rate > 0.6:
                insights.append(f"👥 Strong user engagement rate of {self.format_percentage(engagement_rate * 100)} indicates high content relevance and satisfaction.")
            elif engagement_rate < 0.3:
                insights.append(f"💡 Low user engagement rate of {self.format_percentage(engagement_rate * 100)} suggests need for improved content curation.")
            
            churn_rate = user_metrics.get('churn_rate', 0)
            if churn_rate < 0.05:
                insights.append(f"🔒 Excellent user retention with churn rate of only {self.format_percentage(churn_rate * 100)}.")
            elif churn_rate > 0.15:
                insights.append(f"🚨 High churn rate of {self.format_percentage(churn_rate * 100)} requires immediate retention strategy implementation.")
        
        # Cost efficiency insights
        if 'costs' in data:
            cost_data = data['costs']
            
            if 'ai_compute_cost' in cost_data and 'total_revenue' in metrics.get('revenue', {}):
                ai_cost = cost_data['ai_compute_cost']
                revenue = metrics['revenue']['current']
                cost_ratio = (ai_cost / revenue) * 100
                
                if cost_ratio < 5:
                    insights.append(f"💰 Excellent AI cost efficiency with compute costs at only {self.format_percentage(cost_ratio)} of revenue.")
                elif cost_ratio > 15:
                    insights.append(f"⚡ AI compute costs at {self.format_percentage(cost_ratio)} of revenue indicate optimization opportunities.")
        
        # Market position insights
        if 'market_data' in data:
            market_data = data['market_data']
            
            market_share = market_data.get('market_share', 0)
            if market_share > 0.3:
                insights.append(f"🏆 Dominant market position with {self.format_percentage(market_share * 100)} market share.")
            elif market_share < 0.1:
                insights.append(f"🎯 Opportunity for growth with current {self.format_percentage(market_share * 100)} market share.")
        
        return insights
    
    async def _generate_recommendations(self, insights: List[str], data: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on insights."""
        
        recommendations = []
        
        # Analyze insights to generate recommendations
        insight_text = " ".join(insights).lower()
        
        # Revenue-based recommendations
        if "revenue decline" in insight_text:
            recommendations.append("🔄 Implement aggressive user acquisition campaigns and review pricing strategy to reverse revenue decline.")
            recommendations.append("📊 Conduct detailed user feedback analysis to identify pain points affecting retention.")
        
        if "revenue growth" in insight_text and "strong" in insight_text:
            recommendations.append("🚀 Scale successful strategies and consider expanding to new markets or user segments.")
            recommendations.append("💡 Invest additional resources in AI model development to maintain competitive advantage.")
        
        # AI Performance recommendations
        if "accuracy" in insight_text and ("below" in insight_text or "improvement" in insight_text):
            recommendations.append("🤖 Implement advanced model training techniques and increase training data quality.")
            recommendations.append("🔬 Consider ensemble methods or newer AI architectures to improve model performance.")
        
        if "recommendation hit rate" in insight_text and "improvement" in insight_text:
            recommendations.append("🎯 Enhance user profiling algorithms and incorporate more behavioral signals.")
            recommendations.append("📈 A/B test different recommendation algorithms to optimize hit rates.")
        
        # User engagement recommendations
        if "low user engagement" in insight_text:
            recommendations.append("🎵 Diversify content recommendations and improve playlist generation algorithms.")
            recommendations.append("📱 Enhance user interface design and implement gamification features.")
        
        if "high churn rate" in insight_text:
            recommendations.append("🔒 Implement proactive retention campaigns targeting at-risk users.")
            recommendations.append("💝 Develop personalized re-engagement strategies based on user preferences.")
        
        # Cost optimization recommendations
        if "cost" in insight_text and ("optimization" in insight_text or "high" in insight_text):
            recommendations.append("⚡ Optimize AI model inference pipeline and implement efficient caching strategies.")
            recommendations.append("☁️ Evaluate cloud resource allocation and consider auto-scaling solutions.")
        
        # Market position recommendations
        if "market share" in insight_text and "opportunity" in insight_text:
            recommendations.append("🌍 Develop targeted marketing campaigns for underserved market segments.")
            recommendations.append("🤝 Consider strategic partnerships to accelerate market penetration.")
        
        # General AI enhancement recommendations
        if len(recommendations) < 3:  # Ensure we have enough recommendations
            recommendations.extend([
                "🔮 Invest in real-time personalization capabilities to improve user experience.",
                "📊 Implement advanced analytics to better understand user behavior patterns.",
                "🎯 Develop predictive models for content trend forecasting and demand planning."
            ])
        
        return recommendations[:6]  # Limit to top 6 recommendations
    
    async def _format_executive_content(self, metrics: Dict[str, Any], insights: List[str], data: Dict[str, Any]) -> str:
        """Format the main content section of the executive report."""
        
        content_sections = []
        
        # Key Performance Indicators section
        kpi_section = "## 📊 Key Performance Indicators\n\n"
        
        if 'revenue' in metrics:
            revenue = metrics['revenue']
            kpi_section += f"**Revenue**: {self.format_currency(revenue['current'])}"
            if revenue.get('target'):
                achievement = (revenue['current'] / revenue['target']) * 100
                kpi_section += f" ({self.format_percentage(achievement)} of target)"
            kpi_section += "\n"
        
        if 'users' in metrics:
            user_metrics = metrics['users']
            if 'active_users' in user_metrics:
                kpi_section += f"**Active Users**: {self.format_large_number(user_metrics['active_users'])}\n"
            if 'engagement_rate' in user_metrics:
                kpi_section += f"**Engagement Rate**: {self.format_percentage(user_metrics['engagement_rate'] * 100)}\n"
        
        content_sections.append(kpi_section)
        
        # AI Performance section
        if 'ai_performance' in metrics:
            ai_section = "## 🤖 AI Performance Metrics\n\n"
            ai_metrics = metrics['ai_performance']
            
            if 'model_accuracy' in ai_metrics:
                ai_section += f"**Model Accuracy**: {self.format_percentage(ai_metrics['model_accuracy'] * 100)}\n"
            if 'recommendation_hit_rate' in ai_metrics:
                ai_section += f"**Recommendation Hit Rate**: {self.format_percentage(ai_metrics['recommendation_hit_rate'] * 100)}\n"
            if 'inference_latency' in ai_metrics:
                ai_section += f"**Average Inference Latency**: {ai_metrics['inference_latency']:.2f}ms\n"
            
            content_sections.append(ai_section)
        
        # Business Impact section
        if 'business_impact' in data:
            impact_section = "## 💼 Business Impact\n\n"
            impact_data = data['business_impact']
            
            if 'cost_savings' in impact_data:
                impact_section += f"**Cost Savings**: {self.format_currency(impact_data['cost_savings'])}\n"
            if 'efficiency_improvement' in impact_data:
                impact_section += f"**Efficiency Improvement**: {self.format_percentage(impact_data['efficiency_improvement'])}\n"
            if 'new_revenue' in impact_data:
                impact_section += f"**New Revenue Generated**: {self.format_currency(impact_data['new_revenue'])}\n"
            
            content_sections.append(impact_section)
        
        # Insights section
        if insights:
            insights_section = "## 💡 Key Insights\n\n"
            for insight in insights:
                insights_section += f"• {insight}\n"
            content_sections.append(insights_section)
        
        return "\n".join(content_sections)


class SpotifyArtistFormatter(BaseBusinessFormatter):
    """Specialized formatter for artist performance analytics."""
    
    async def format_artist_analytics(self, artist_data: Dict[str, Any]) -> FormattedBusinessReport:
        """Format comprehensive artist analytics report."""
        
        artist_name = artist_data.get('name', 'Unknown Artist')
        title = f"🎤 Artist Analytics Report - {artist_name}"
        
        # Generate artist-specific insights
        insights = await self._analyze_artist_performance(artist_data)
        
        # Create artist performance charts
        charts = await self._create_artist_charts(artist_data)
        
        # Format artist content
        content = await self._format_artist_content(artist_data)
        
        # Generate recommendations for artist
        recommendations = await self._generate_artist_recommendations(artist_data, insights)
        
        # Create executive summary
        executive_summary = await self._generate_artist_summary(artist_data)
        
        metadata = {
            "report_type": "artist_analytics",
            "artist_id": artist_data.get('id'),
            "artist_name": artist_name,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "period": artist_data.get('period', 'last_30_days'),
            "tenant_id": self.tenant_id
        }
        
        return FormattedBusinessReport(
            title=title,
            executive_summary=executive_summary,
            content=content,
            charts=charts,
            insights=insights,
            recommendations=recommendations,
            metadata=metadata
        )
    
    async def _analyze_artist_performance(self, artist_data: Dict[str, Any]) -> List[str]:
        """Analyze artist performance and generate insights."""
        
        insights = []
        
        # Streaming performance analysis
        if 'streams' in artist_data:
            streams = artist_data['streams']
            total_streams = streams.get('total', 0)
            monthly_streams = streams.get('monthly', 0)
            
            if total_streams > 100_000_000:
                insights.append(f"🌟 Exceptional reach with {self.format_large_number(total_streams)} total streams, indicating massive global appeal.")
            elif total_streams > 10_000_000:
                insights.append(f"🎵 Strong performance with {self.format_large_number(total_streams)} total streams showing solid fanbase.")
            
            if monthly_streams > 5_000_000:
                insights.append(f"🔥 High current momentum with {self.format_large_number(monthly_streams)} monthly streams.")
        
        # Geographic performance
        if 'geographic_data' in artist_data:
            geo_data = artist_data['geographic_data']
            top_markets = geo_data.get('top_markets', [])
            
            if len(top_markets) > 10:
                insights.append(f"🌍 Strong international presence across {len(top_markets)} markets demonstrates global appeal.")
            
            if top_markets:
                primary_market = top_markets[0]
                market_share = primary_market.get('percentage', 0)
                if market_share > 50:
                    insights.append(f"🏠 Heavily concentrated in {primary_market['country']} ({self.format_percentage(market_share)} of streams).")
                elif market_share < 20:
                    insights.append(f"🌐 Well-diversified global audience with no single market dominance.")
        
        # Engagement analysis
        if 'engagement' in artist_data:
            engagement = artist_data['engagement']
            
            save_rate = engagement.get('save_rate', 0)
            if save_rate > 0.15:
                insights.append(f"💾 Excellent fan loyalty with {self.format_percentage(save_rate * 100)} save rate.")
            elif save_rate < 0.05:
                insights.append(f"📈 Opportunity to improve fan engagement (current save rate: {self.format_percentage(save_rate * 100)}).")
            
            skip_rate = engagement.get('skip_rate', 0)
            if skip_rate < 0.1:
                insights.append(f"🎯 High content quality with low skip rate of {self.format_percentage(skip_rate * 100)}.")
            elif skip_rate > 0.3:
                insights.append(f"⚠️ High skip rate of {self.format_percentage(skip_rate * 100)} suggests content relevance issues.")
        
        # AI recommendation performance
        if 'ai_metrics' in artist_data:
            ai_metrics = artist_data['ai_metrics']
            
            recommendation_score = ai_metrics.get('recommendation_score', 0)
            if recommendation_score > 0.8:
                insights.append(f"🤖 AI algorithms highly favor this artist (recommendation score: {self.format_percentage(recommendation_score * 100)}).")
            elif recommendation_score < 0.4:
                insights.append(f"🔧 Low AI recommendation score ({self.format_percentage(recommendation_score * 100)}) indicates optimization opportunities.")
        
        return insights
    
    async def _create_artist_charts(self, artist_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create artist-specific performance charts."""
        
        charts = []
        
        # Streaming trend over time
        if 'streaming_history' in artist_data:
            history = artist_data['streaming_history']
            streaming_chart = {
                "type": "line",
                "title": "Streaming Performance Trend",
                "data": {
                    "labels": [entry['date'] for entry in history],
                    "datasets": [{
                        "label": "Daily Streams",
                        "data": [entry['streams'] for entry in history],
                        "borderColor": "#1DB954",
                        "backgroundColor": "rgba(29, 185, 84, 0.1)",
                        "tension": 0.4
                    }]
                },
                "options": {
                    "responsive": True,
                    "plugins": {"title": {"display": True, "text": "Daily Streaming Trend"}},
                    "scales": {
                        "y": {
                            "beginAtZero": True,
                            "ticks": {"callback": "function(value) { return value.toLocaleString(); }"}
                        }
                    }
                }
            }
            charts.append(streaming_chart)
        
        # Geographic distribution
        if 'geographic_data' in artist_data:
            geo_data = artist_data['geographic_data']
            top_markets = geo_data.get('top_markets', [])[:10]
            
            geo_chart = {
                "type": "bar",
                "title": "Top Markets by Streams",
                "data": {
                    "labels": [market['country'] for market in top_markets],
                    "datasets": [{
                        "label": "Stream Percentage",
                        "data": [market['percentage'] for market in top_markets],
                        "backgroundColor": "#1DB954",
                        "borderColor": "#169C47",
                        "borderWidth": 1
                    }]
                },
                "options": {
                    "responsive": True,
                    "plugins": {"title": {"display": True, "text": "Geographic Distribution"}},
                    "scales": {
                        "y": {
                            "beginAtZero": True,
                            "ticks": {"callback": "function(value) { return value + '%'; }"}
                        }
                    }
                }
            }
            charts.append(geo_chart)
        
        # Top tracks performance
        if 'top_tracks' in artist_data:
            tracks = artist_data['top_tracks'][:10]
            
            tracks_chart = {
                "type": "horizontalBar",
                "title": "Top Tracks Performance",
                "data": {
                    "labels": [track['name'] for track in tracks],
                    "datasets": [{
                        "label": "Streams",
                        "data": [track['streams'] for track in tracks],
                        "backgroundColor": "#1DB954",
                        "borderColor": "#169C47",
                        "borderWidth": 1
                    }]
                },
                "options": {
                    "responsive": True,
                    "plugins": {"title": {"display": True, "text": "Top 10 Tracks by Streams"}},
                    "scales": {
                        "x": {
                            "beginAtZero": True,
                            "ticks": {"callback": "function(value) { return value.toLocaleString(); }"}
                        }
                    }
                }
            }
            charts.append(tracks_chart)
        
        return charts
    
    async def _format_artist_content(self, artist_data: Dict[str, Any]) -> str:
        """Format artist-specific content sections."""
        
        content_sections = []
        
        # Artist Overview
        overview_section = "## 🎤 Artist Overview\n\n"
        overview_section += f"**Artist**: {artist_data.get('name', 'Unknown')}\n"
        overview_section += f"**Genre**: {', '.join(artist_data.get('genres', ['Unknown']))}\n"
        overview_section += f"**Followers**: {self.format_large_number(artist_data.get('followers', 0))}\n"
        overview_section += f"**Total Tracks**: {artist_data.get('total_tracks', 0)}\n"
        content_sections.append(overview_section)
        
        # Performance Metrics
        if 'streams' in artist_data:
            performance_section = "## 📊 Performance Metrics\n\n"
            streams = artist_data['streams']
            
            performance_section += f"**Total Streams**: {self.format_large_number(streams.get('total', 0))}\n"
            performance_section += f"**Monthly Streams**: {self.format_large_number(streams.get('monthly', 0))}\n"
            performance_section += f"**Average Daily Streams**: {self.format_large_number(streams.get('daily_average', 0))}\n"
            
            if 'growth_rate' in streams:
                growth = streams['growth_rate']
                indicator = self.get_variance_indicator(growth)
                performance_section += f"**Growth Rate**: {indicator} {self.format_percentage(growth)}\n"
            
            content_sections.append(performance_section)
        
        # Engagement Analytics
        if 'engagement' in artist_data:
            engagement_section = "## 💝 Fan Engagement\n\n"
            engagement = artist_data['engagement']
            
            engagement_section += f"**Save Rate**: {self.format_percentage(engagement.get('save_rate', 0) * 100)}\n"
            engagement_section += f"**Skip Rate**: {self.format_percentage(engagement.get('skip_rate', 0) * 100)}\n"
            engagement_section += f"**Playlist Adds**: {self.format_large_number(engagement.get('playlist_adds', 0))}\n"
            engagement_section += f"**Share Rate**: {self.format_percentage(engagement.get('share_rate', 0) * 100)}\n"
            
            content_sections.append(engagement_section)
        
        # Revenue Impact
        if 'revenue' in artist_data:
            revenue_section = "## 💰 Revenue Impact\n\n"
            revenue = artist_data['revenue']
            
            revenue_section += f"**Total Revenue**: {self.format_currency(revenue.get('total', 0))}\n"
            revenue_section += f"**Monthly Revenue**: {self.format_currency(revenue.get('monthly', 0))}\n"
            revenue_section += f"**Revenue per Stream**: {self.format_currency(revenue.get('per_stream', 0), currency='USD')}\n"
            
            content_sections.append(revenue_section)
        
        return "\n".join(content_sections)
    
    async def _generate_artist_summary(self, artist_data: Dict[str, Any]) -> str:
        """Generate executive summary for artist."""
        
        artist_name = artist_data.get('name', 'Unknown Artist')
        summary = f"🎵 **{artist_name} Performance Summary**\n\n"
        
        # Key metrics summary
        key_points = []
        
        if 'streams' in artist_data:
            streams = artist_data['streams']
            total_streams = streams.get('total', 0)
            monthly_streams = streams.get('monthly', 0)
            
            key_points.append(f"Total Streams: {self.format_large_number(total_streams)}")
            key_points.append(f"Monthly Streams: {self.format_large_number(monthly_streams)}")
        
        if 'engagement' in artist_data:
            engagement = artist_data['engagement']
            save_rate = engagement.get('save_rate', 0)
            key_points.append(f"Fan Engagement: {self.format_percentage(save_rate * 100)} save rate")
        
        if 'revenue' in artist_data:
            revenue = artist_data['revenue']
            monthly_revenue = revenue.get('monthly', 0)
            key_points.append(f"Monthly Revenue: {self.format_currency(monthly_revenue)}")
        
        summary += "\n".join([f"• {point}" for point in key_points])
        
        # Add context
        period = artist_data.get('period', 'last 30 days')
        summary += f"\n\n*Report covers: {period}*"
        
        return summary
    
    async def _generate_artist_recommendations(self, artist_data: Dict[str, Any], insights: List[str]) -> List[str]:
        """Generate artist-specific recommendations."""
        
        recommendations = []
        
        # Analyze current performance for recommendations
        if 'engagement' in artist_data:
            engagement = artist_data['engagement']
            
            save_rate = engagement.get('save_rate', 0)
            if save_rate < 0.1:
                recommendations.append("🎯 Focus on creating more engaging content to improve fan loyalty and save rates.")
            
            skip_rate = engagement.get('skip_rate', 0)
            if skip_rate > 0.25:
                recommendations.append("🎵 Analyze track structure and hooks to reduce skip rates and improve retention.")
        
        # Geographic expansion opportunities
        if 'geographic_data' in artist_data:
            geo_data = artist_data['geographic_data']
            top_markets = geo_data.get('top_markets', [])
            
            if len(top_markets) < 5:
                recommendations.append("🌍 Develop marketing strategies for international expansion to diversify audience base.")
        
        # AI optimization
        if 'ai_metrics' in artist_data:
            ai_metrics = artist_data['ai_metrics']
            
            recommendation_score = ai_metrics.get('recommendation_score', 0)
            if recommendation_score < 0.6:
                recommendations.append("🤖 Optimize content metadata and collaboration patterns to improve AI recommendation algorithms.")
        
        # Revenue optimization
        if 'revenue' in artist_data:
            revenue = artist_data['revenue']
            per_stream = revenue.get('per_stream', 0)
            
            if per_stream < 0.003:  # Below average
                recommendations.append("💰 Explore premium content strategies and direct fan monetization to increase revenue per stream.")
        
        # General growth recommendations
        recommendations.extend([
            "📱 Leverage social media integration to amplify reach and engagement.",
            "🎤 Consider collaborative features and playlist placements to expand discovery.",
            "📊 Implement regular performance reviews to track progress against KPIs."
        ])
        
        return recommendations[:5]  # Limit to top 5 recommendations


class PlaylistAnalyticsFormatter(BaseBusinessFormatter):
    """Specialized formatter for playlist performance and recommendation analytics."""
    
    async def format_playlist_analytics(self, playlist_data: Dict[str, Any]) -> FormattedBusinessReport:
        """Format comprehensive playlist analytics report."""
        
        playlist_name = playlist_data.get('name', 'Unknown Playlist')
        title = f"📋 Playlist Analytics Report - {playlist_name}"
        
        # Generate playlist insights
        insights = await self._analyze_playlist_performance(playlist_data)
        
        # Create playlist charts
        charts = await self._create_playlist_charts(playlist_data)
        
        # Format playlist content
        content = await self._format_playlist_content(playlist_data)
        
        # Generate recommendations
        recommendations = await self._generate_playlist_recommendations(playlist_data, insights)
        
        # Create executive summary
        executive_summary = await self._generate_playlist_summary(playlist_data)
        
        metadata = {
            "report_type": "playlist_analytics",
            "playlist_id": playlist_data.get('id'),
            "playlist_name": playlist_name,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "period": playlist_data.get('period', 'last_30_days'),
            "tenant_id": self.tenant_id
        }
        
        return FormattedBusinessReport(
            title=title,
            executive_summary=executive_summary,
            content=content,
            charts=charts,
            insights=insights,
            recommendations=recommendations,
            metadata=metadata
        )
    
    async def _analyze_playlist_performance(self, playlist_data: Dict[str, Any]) -> List[str]:
        """Analyze playlist performance metrics."""
        
        insights = []
        
        # Follower analysis
        if 'followers' in playlist_data:
            followers = playlist_data['followers']
            total_followers = followers.get('total', 0)
            growth_rate = followers.get('growth_rate', 0)
            
            if total_followers > 1_000_000:
                insights.append(f"🌟 Massive audience reach with {self.format_large_number(total_followers)} followers.")
            elif total_followers > 100_000:
                insights.append(f"🎵 Strong community with {self.format_large_number(total_followers)} followers.")
            
            if growth_rate > 10:
                insights.append(f"🚀 Exceptional growth momentum with {self.format_percentage(growth_rate)} follower growth.")
            elif growth_rate < 0:
                insights.append(f"📉 Declining follower count ({self.format_percentage(abs(growth_rate))} decrease) needs attention.")
        
        # Engagement metrics
        if 'engagement' in playlist_data:
            engagement = playlist_data['engagement']
            
            avg_completion = engagement.get('avg_completion_rate', 0)
            if avg_completion > 0.7:
                insights.append(f"💯 Excellent playlist curation with {self.format_percentage(avg_completion * 100)} average completion rate.")
            elif avg_completion < 0.4:
                insights.append(f"⚠️ Low completion rate ({self.format_percentage(avg_completion * 100)}) suggests track ordering issues.")
            
            save_rate = engagement.get('save_rate', 0)
            if save_rate > 0.2:
                insights.append(f"❤️ High user satisfaction with {self.format_percentage(save_rate * 100)} of listeners saving the playlist.")
        
        # AI recommendation performance
        if 'ai_metrics' in playlist_data:
            ai_metrics = playlist_data['ai_metrics']
            
            discovery_score = ai_metrics.get('discovery_score', 0)
            if discovery_score > 0.8:
                insights.append(f"🔍 Excellent discoverability through AI with {self.format_percentage(discovery_score * 100)} discovery score.")
            
            personalization_score = ai_metrics.get('personalization_score', 0)
            if personalization_score > 0.75:
                insights.append(f"🎯 Highly personalized content with {self.format_percentage(personalization_score * 100)} personalization effectiveness.")
        
        return insights
    
    async def _create_playlist_charts(self, playlist_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create playlist-specific performance charts."""
        
        charts = []
        
        # Follower growth trend
        if 'follower_history' in playlist_data:
            history = playlist_data['follower_history']
            follower_chart = {
                "type": "line",
                "title": "Follower Growth Trend",
                "data": {
                    "labels": [entry['date'] for entry in history],
                    "datasets": [{
                        "label": "Followers",
                        "data": [entry['followers'] for entry in history],
                        "borderColor": "#1DB954",
                        "backgroundColor": "rgba(29, 185, 84, 0.1)",
                        "tension": 0.4
                    }]
                },
                "options": {
                    "responsive": True,
                    "plugins": {"title": {"display": True, "text": "Follower Growth Over Time"}},
                    "scales": {
                        "y": {
                            "beginAtZero": True,
                            "ticks": {"callback": "function(value) { return value.toLocaleString(); }"}
                        }
                    }
                }
            }
            charts.append(follower_chart)
        
        # Track performance within playlist
        if 'track_performance' in playlist_data:
            tracks = playlist_data['track_performance'][:15]  # Top 15 tracks
            
            track_chart = {
                "type": "bar",
                "title": "Track Performance in Playlist",
                "data": {
                    "labels": [f"{track['name']} - {track['artist']}" for track in tracks],
                    "datasets": [{
                        "label": "Completion Rate",
                        "data": [track['completion_rate'] * 100 for track in tracks],
                        "backgroundColor": "#1DB954",
                        "borderColor": "#169C47",
                        "borderWidth": 1
                    }]
                },
                "options": {
                    "responsive": True,
                    "plugins": {"title": {"display": True, "text": "Track Completion Rates"}},
                    "scales": {
                        "x": {"ticks": {"maxRotation": 45}},
                        "y": {
                            "beginAtZero": True,
                            "max": 100,
                            "ticks": {"callback": "function(value) { return value + '%'; }"}
                        }
                    }
                }
            }
            charts.append(track_chart)
        
        return charts
    
    async def _format_playlist_content(self, playlist_data: Dict[str, Any]) -> str:
        """Format playlist-specific content sections."""
        
        content_sections = []
        
        # Playlist Overview
        overview_section = "## 📋 Playlist Overview\n\n"
        overview_section += f"**Playlist**: {playlist_data.get('name', 'Unknown')}\n"
        overview_section += f"**Creator**: {playlist_data.get('creator', 'Unknown')}\n"
        overview_section += f"**Total Tracks**: {playlist_data.get('track_count', 0)}\n"
        overview_section += f"**Total Duration**: {playlist_data.get('duration_minutes', 0)} minutes\n"
        overview_section += f"**Followers**: {self.format_large_number(playlist_data.get('followers', {}).get('total', 0))}\n"
        content_sections.append(overview_section)
        
        # Performance Metrics
        if 'engagement' in playlist_data:
            performance_section = "## 📊 Performance Metrics\n\n"
            engagement = playlist_data['engagement']
            
            performance_section += f"**Average Completion Rate**: {self.format_percentage(engagement.get('avg_completion_rate', 0) * 100)}\n"
            performance_section += f"**Save Rate**: {self.format_percentage(engagement.get('save_rate', 0) * 100)}\n"
            performance_section += f"**Skip Rate**: {self.format_percentage(engagement.get('skip_rate', 0) * 100)}\n"
            performance_section += f"**Share Rate**: {self.format_percentage(engagement.get('share_rate', 0) * 100)}\n"
            
            content_sections.append(performance_section)
        
        # AI Performance
        if 'ai_metrics' in playlist_data:
            ai_section = "## 🤖 AI Performance\n\n"
            ai_metrics = playlist_data['ai_metrics']
            
            ai_section += f"**Discovery Score**: {self.format_percentage(ai_metrics.get('discovery_score', 0) * 100)}\n"
            ai_section += f"**Personalization Score**: {self.format_percentage(ai_metrics.get('personalization_score', 0) * 100)}\n"
            ai_section += f"**Recommendation Frequency**: {ai_metrics.get('recommendation_frequency', 0)} times/day\n"
            
            content_sections.append(ai_section)
        
        return "\n".join(content_sections)
    
    async def _generate_playlist_summary(self, playlist_data: Dict[str, Any]) -> str:
        """Generate executive summary for playlist."""
        
        playlist_name = playlist_data.get('name', 'Unknown Playlist')
        summary = f"📋 **{playlist_name} Performance Summary**\n\n"
        
        key_points = []
        
        if 'followers' in playlist_data:
            followers = playlist_data['followers']
            total_followers = followers.get('total', 0)
            key_points.append(f"Total Followers: {self.format_large_number(total_followers)}")
        
        if 'engagement' in playlist_data:
            engagement = playlist_data['engagement']
            completion_rate = engagement.get('avg_completion_rate', 0)
            key_points.append(f"Avg Completion: {self.format_percentage(completion_rate * 100)}")
        
        summary += "\n".join([f"• {point}" for point in key_points])
        
        period = playlist_data.get('period', 'last 30 days')
        summary += f"\n\n*Report covers: {period}*"
        
        return summary
    
    async def _generate_playlist_recommendations(self, playlist_data: Dict[str, Any], insights: List[str]) -> List[str]:
        """Generate playlist-specific recommendations."""
        
        recommendations = []
        
        # Engagement-based recommendations
        if 'engagement' in playlist_data:
            engagement = playlist_data['engagement']
            
            completion_rate = engagement.get('avg_completion_rate', 0)
            if completion_rate < 0.5:
                recommendations.append("🎵 Optimize track ordering and remove low-performing songs to improve completion rates.")
            
            save_rate = engagement.get('save_rate', 0)
            if save_rate < 0.15:
                recommendations.append("❤️ Enhance playlist metadata and descriptions to increase save rates.")
        
        # AI optimization recommendations
        if 'ai_metrics' in playlist_data:
            ai_metrics = playlist_data['ai_metrics']
            
            discovery_score = ai_metrics.get('discovery_score', 0)
            if discovery_score < 0.6:
                recommendations.append("🔍 Improve playlist tags and genre consistency to boost AI discovery.")
        
        # Growth recommendations
        recommendations.extend([
            "📱 Promote playlist across social channels to increase follower growth.",
            "🔄 Regularly update content to maintain engagement and freshness.",
            "🎯 Analyze user feedback to understand preference patterns."
        ])
        
        return recommendations[:4]


# Factory function for creating business formatters
def create_business_formatter(
    formatter_type: str,
    tenant_id: str,
    config: Optional[Dict[str, Any]] = None
) -> BaseBusinessFormatter:
    """
    Factory function to create business intelligence formatters.
    
    Args:
        formatter_type: Type of formatter ('executive', 'artist', 'playlist', 'revenue', 'engagement')
        tenant_id: Tenant identifier
        config: Configuration dictionary
        
    Returns:
        Configured formatter instance
    """
    formatters = {
        'executive': BusinessIntelligenceFormatter,
        'business_intelligence': BusinessIntelligenceFormatter,
        'artist': SpotifyArtistFormatter,
        'playlist': PlaylistAnalyticsFormatter,
        'revenue': BusinessIntelligenceFormatter,  # Can handle revenue reports
        'engagement': BusinessIntelligenceFormatter  # Can handle engagement reports
    }
    
    if formatter_type not in formatters:
        raise ValueError(f"Unsupported business formatter type: {formatter_type}")
    
    formatter_class = formatters[formatter_type]
    return formatter_class(tenant_id, config or {})
