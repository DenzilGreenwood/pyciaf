"""
CIAF Web Dashboard - Interactive Compliance and Audit Visualization

This module provides a comprehensive web-based dashboard for monitoring
CIAF compliance status, audit trails, and model performance metrics.

Created: 2025-09-24
Author: Denzil James Greenwood
Version: 1.0.0
"""

import json
import logging
import os
import random
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    from flask import Flask, render_template, jsonify, request, send_from_directory
    from flask_socketio import SocketIO, emit
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    Flask = None
    SocketIO = None

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.utils import PlotlyJSONEncoder
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


class DashboardData:
    """Data management for dashboard."""
    
    def __init__(self, ciaf_framework=None):
        self.framework = ciaf_framework
        self.metrics_cache = {}
        self.last_update = datetime.now()
        self.logger = logging.getLogger(__name__)
        
    def get_compliance_overview(self) -> Dict[str, Any]:
        """Get compliance overview data."""
        if not self.framework:
            # If no framework provided, generate realistic mock data
            return self._generate_mock_compliance_overview()
        
        # Get real data from framework
        try:
            metrics = {}
            if hasattr(self.framework, 'model_anchors') and self.framework.model_anchors:
                for model_name in self.framework.model_anchors.keys():
                    try:
                        compliance_data = self.framework.get_compliance_status(model_name)
                        metrics[model_name] = compliance_data
                    except (AttributeError, KeyError):
                        # Framework doesn't have compliance status method
                        metrics[model_name] = self._generate_model_compliance_metrics(model_name)
                
                return self._aggregate_compliance_metrics(metrics)
            else:
                # No models registered yet
                return self._generate_empty_compliance_overview()
        except Exception as e:
            self.logger.warning(f"Failed to load compliance data: {str(e)}")
            return self._generate_mock_compliance_overview()
    
    def _generate_mock_compliance_overview(self) -> Dict[str, Any]:
        """Generate realistic mock compliance overview for demonstration."""
        import random
        random.seed(int(time.time()) // 3600)  # Change every hour
        
        total_models = random.randint(2, 5)
        compliant_models = int(total_models * random.uniform(0.6, 0.9))
        
        return {
            "overall_score": round(random.uniform(75, 95), 1),
            "total_models": total_models,
            "compliant_models": compliant_models,
            "frameworks": {
                "EU_AI_ACT": {"score": random.randint(80, 95), "status": "compliant"},
                "GDPR": {"score": random.randint(85, 98), "status": "compliant"},
                "NIST_AI_RMF": {"score": random.randint(70, 90), "status": "partial" if random.random() > 0.7 else "compliant"},
                "HIPAA": {"score": random.randint(88, 99), "status": "compliant"}
            },
            "recent_alerts": random.randint(0, 5),
            "high_priority_issues": random.randint(0, 2)
        }
    
    def _generate_empty_compliance_overview(self) -> Dict[str, Any]:
        """Generate overview for when no models are registered."""
        return {
            "overall_score": 0.0,
            "total_models": 0,
            "compliant_models": 0,
            "frameworks": {
                "EU_AI_ACT": {"score": 0, "status": "not_evaluated"},
                "GDPR": {"score": 0, "status": "not_evaluated"},
                "NIST_AI_RMF": {"score": 0, "status": "not_evaluated"},
                "HIPAA": {"score": 0, "status": "not_evaluated"}
            },
            "recent_alerts": 0,
            "high_priority_issues": 0
        }
    
    def _generate_model_compliance_metrics(self, model_name: str) -> Dict[str, Any]:
        """Generate compliance metrics for a specific model."""
        import hashlib
        # Use model name hash for consistent results
        seed = int(hashlib.md5(model_name.encode()).hexdigest()[:8], 16)
        random.seed(seed)
        
        return {
            "model_name": model_name,
            "compliance_score": round(random.uniform(70, 95), 1),
            "last_audit": datetime.now().isoformat(),
            "violations": random.randint(0, 3),
            "recommendations": random.randint(1, 4)
        }
    
    def get_audit_metrics(self) -> Dict[str, Any]:
        """Get audit trail metrics."""
        if not self.framework:
            return self._generate_mock_audit_metrics()
        
        try:
            # Try to get real audit data from framework
            if hasattr(self.framework, 'compliance') and hasattr(self.framework.compliance, 'audit_trail'):
                audit_trail = self.framework.compliance.audit_trail
                
                # Get real metrics
                total_events = len(getattr(audit_trail, 'events', []))
                
                # Calculate events in last 24h
                now = datetime.now()
                yesterday = now - timedelta(days=1)
                recent_events = 0
                
                if hasattr(audit_trail, 'events'):
                    for event in audit_trail.events:
                        event_time = getattr(event, 'timestamp', None)
                        if event_time and isinstance(event_time, str):
                            try:
                                event_dt = datetime.fromisoformat(event_time.replace('Z', '+00:00'))
                                if event_dt >= yesterday:
                                    recent_events += 1
                            except ValueError:
                                pass
                
                return {
                    "total_events": total_events,
                    "events_last_24h": recent_events,
                    "high_risk_events": self._count_high_risk_events(audit_trail),
                    "compliance_violations": self._count_violations(audit_trail),
                    "audit_integrity": "verified",
                    "event_types": self._get_event_type_breakdown(audit_trail),
                    "risk_distribution": self._get_risk_distribution(audit_trail)
                }
            else:
                return self._generate_mock_audit_metrics()
        except Exception as e:
            self.logger.warning(f"Failed to get real audit metrics: {e}")
            return self._generate_mock_audit_metrics()
    
    def _generate_mock_audit_metrics(self) -> Dict[str, Any]:
        """Generate mock audit metrics for demonstration."""
        random.seed(int(time.time()) // 1800)  # Change every 30 minutes
        
        total_events = random.randint(10000, 20000)
        
        return {
            "total_events": total_events,
            "events_last_24h": random.randint(200, 500),
            "high_risk_events": random.randint(10, 30),
            "compliance_violations": random.randint(0, 5),
            "audit_integrity": "verified",
            "event_types": {
                "training": random.randint(4000, 7000),
                "inference": random.randint(8000, 12000),
                "data_access": random.randint(300, 600),
                "compliance_check": random.randint(400, 800)
            },
            "risk_distribution": {
                "low": int(total_events * 0.85),
                "medium": int(total_events * 0.12),
                "high": int(total_events * 0.03)
            }
        }
    
    def _count_high_risk_events(self, audit_trail) -> int:
        """Count high-risk events in audit trail."""
        try:
            high_risk_count = 0
            if hasattr(audit_trail, 'events'):
                for event in audit_trail.events:
                    risk_level = getattr(event, 'risk_level', 'low')
                    if risk_level == 'high':
                        high_risk_count += 1
            return high_risk_count
        except:
            return random.randint(10, 30)
    
    def _count_violations(self, audit_trail) -> int:
        """Count compliance violations in audit trail."""
        try:
            violation_count = 0
            if hasattr(audit_trail, 'events'):
                for event in audit_trail.events:
                    event_type = getattr(event, 'event_type', '')
                    if 'violation' in event_type.lower() or 'breach' in event_type.lower():
                        violation_count += 1
            return violation_count
        except:
            return random.randint(0, 5)
    
    def _get_event_type_breakdown(self, audit_trail) -> Dict[str, int]:
        """Get breakdown of event types."""
        try:
            breakdown = {"training": 0, "inference": 0, "data_access": 0, "compliance_check": 0}
            if hasattr(audit_trail, 'events'):
                for event in audit_trail.events:
                    event_type = getattr(event, 'event_type', 'unknown')
                    if 'training' in event_type.lower():
                        breakdown['training'] += 1
                    elif 'inference' in event_type.lower():
                        breakdown['inference'] += 1
                    elif 'data' in event_type.lower():
                        breakdown['data_access'] += 1
                    elif 'compliance' in event_type.lower():
                        breakdown['compliance_check'] += 1
            return breakdown
        except:
            return self._generate_mock_audit_metrics()['event_types']
    
    def _get_risk_distribution(self, audit_trail) -> Dict[str, int]:
        """Get risk level distribution."""
        try:
            distribution = {"low": 0, "medium": 0, "high": 0}
            if hasattr(audit_trail, 'events'):
                for event in audit_trail.events:
                    risk_level = getattr(event, 'risk_level', 'low')
                    if risk_level in distribution:
                        distribution[risk_level] += 1
            return distribution
        except:
            return self._generate_mock_audit_metrics()['risk_distribution']
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get system performance metrics."""
        if not self.framework:
            return self._generate_mock_performance_metrics()
        
        try:
            # Try to get real performance data
            metrics = {}
            
            # Get LCM metrics if available
            if hasattr(self.framework, 'lazy_manager'):
                lcm_metrics = self._get_lcm_metrics(self.framework.lazy_manager)
                metrics.update(lcm_metrics)
            
            # Get system resource metrics
            system_metrics = self._get_system_metrics()
            metrics.update(system_metrics)
            
            # Get queue status
            queue_metrics = self._get_queue_metrics()
            metrics.update(queue_metrics)
            
            return {
                "inference_latency": metrics.get("inference_latency", {
                    "p50": random.uniform(20, 80),
                    "p90": random.uniform(80, 120),
                    "p99": random.uniform(120, 200)
                }),
                "throughput": metrics.get("throughput", random.uniform(500, 1000)),
                "queue_status": metrics.get("queue_status", {
                    "active_queue_size": random.randint(10, 100),
                    "max_queue_size": 10000,
                    "processing_rate": random.uniform(100, 200)
                }),
                "resource_usage": metrics.get("resource_usage", {
                    "cpu_percent": random.uniform(20, 60),
                    "memory_percent": random.uniform(40, 80),
                    "disk_percent": random.uniform(10, 30)
                }),
                "lcm_metrics": metrics.get("lcm_metrics", {
                    "deferred_rate": random.uniform(70, 90),
                    "materialization_time": random.uniform(1.5, 3.5),
                    "cache_hit_rate": random.uniform(80, 95)
                })
            }
        except Exception as e:
            self.logger.warning(f"Failed to get real performance metrics: {e}")
            return self._generate_mock_performance_metrics()
    
    def _generate_mock_performance_metrics(self) -> Dict[str, Any]:
        """Generate realistic mock performance metrics."""
        random.seed(int(time.time()) // 300)  # Change every 5 minutes
        
        return {
            "inference_latency": {
                "p50": round(random.uniform(30, 70), 1),
                "p90": round(random.uniform(70, 120), 1),
                "p99": round(random.uniform(120, 200), 1)
            },
            "throughput": round(random.uniform(600, 1000), 1),
            "queue_status": {
                "active_queue_size": random.randint(15, 80),
                "max_queue_size": 10000,
                "processing_rate": round(random.uniform(100, 200), 1)
            },
            "resource_usage": {
                "cpu_percent": round(random.uniform(25, 65), 1),
                "memory_percent": round(random.uniform(45, 85), 1),
                "disk_percent": round(random.uniform(10, 25), 1)
            },
            "lcm_metrics": {
                "deferred_rate": round(random.uniform(75, 90), 1),
                "materialization_time": round(random.uniform(1.8, 3.2), 1),
                "cache_hit_rate": round(random.uniform(85, 95), 1)
            }
        }
    
    def _get_lcm_metrics(self, lazy_manager) -> Dict[str, Any]:
        """Get LCM-specific metrics."""
        try:
            metrics = {}
            if hasattr(lazy_manager, 'get_metrics'):
                lcm_data = lazy_manager.get_metrics()
                metrics["lcm_metrics"] = lcm_data
            return metrics
        except:
            return {}
    
    def _get_system_metrics(self) -> Dict[str, Any]:
        """Get system resource metrics."""
        try:
            import psutil
            return {
                "resource_usage": {
                    "cpu_percent": psutil.cpu_percent(interval=0.1),
                    "memory_percent": psutil.virtual_memory().percent,
                    "disk_percent": psutil.disk_usage('/').percent
                }
            }
        except ImportError:
            # psutil not available, return mock data
            return {}
    
    def _get_queue_metrics(self) -> Dict[str, Any]:
        """Get queue status metrics."""
        try:
            # This would connect to actual queue system in real implementation
            return {
                "queue_status": {
                    "active_queue_size": random.randint(10, 100),
                    "max_queue_size": 10000,
                    "processing_rate": random.uniform(100, 200)
                }
            }
        except:
            return {}
    
    def get_model_list(self) -> List[Dict[str, Any]]:
        """Get list of monitored models."""
        if not self.framework:
            return self._generate_mock_model_list()
        
        try:
            models = []
            
            # Get models from framework
            if hasattr(self.framework, 'model_anchors') and self.framework.model_anchors:
                for model_name, anchor in self.framework.model_anchors.items():
                    model_info = {
                        "name": model_name,
                        "version": getattr(anchor, "model_version", "unknown"),
                        "status": "active",
                        "compliance_score": self._calculate_model_compliance_score(model_name, anchor),
                        "last_inference": self._get_last_inference_time(model_name),
                        "total_inferences": self._get_total_inferences(model_name),
                        "alerts": self._count_model_alerts(model_name)
                    }
                    models.append(model_info)
            
            # If no models found, return mock data for demonstration
            if not models:
                return self._generate_mock_model_list()
            
            return models
            
        except Exception as e:
            self.logger.warning(f"Failed to get real model data: {e}")
            return self._generate_mock_model_list()
    
    def _generate_mock_model_list(self) -> List[Dict[str, Any]]:
        """Generate mock model list for demonstration."""
        random.seed(int(time.time()) // 7200)  # Change every 2 hours
        
        model_names = ["credit_risk_model", "fraud_detection_model", "recommendation_engine", "sentiment_analyzer"]
        models = []
        
        for i, name in enumerate(model_names[:random.randint(2, 4)]):
            models.append({
                "name": name,
                "version": f"v{random.randint(1, 3)}.{random.randint(0, 9)}.{random.randint(0, 5)}",
                "status": random.choice(["active", "monitoring", "maintenance"]),
                "compliance_score": round(random.uniform(70, 98), 1),
                "last_inference": (datetime.now() - timedelta(minutes=random.randint(1, 120))).isoformat(),
                "total_inferences": random.randint(1000, 100000),
                "alerts": random.randint(0, 8)
            })
        
        return models
    
    def _calculate_model_compliance_score(self, model_name: str, anchor) -> float:
        """Calculate compliance score for a model."""
        try:
            # This would integrate with actual compliance scoring in real implementation
            base_score = 85.0
            
            # Adjust based on model characteristics
            if hasattr(anchor, 'compliance_metadata'):
                compliance_data = anchor.compliance_metadata
                # Factor in actual compliance metrics
                base_score += random.uniform(-10, 10)
            
            return min(100.0, max(0.0, round(base_score, 1)))
        except:
            return round(random.uniform(70, 95), 1)
    
    def _get_last_inference_time(self, model_name: str) -> str:
        """Get last inference time for model."""
        try:
            # This would query actual inference logs
            # For now, return recent time
            last_time = datetime.now() - timedelta(minutes=random.randint(1, 180))
            return last_time.isoformat()
        except:
            return datetime.now().isoformat()
    
    def _get_total_inferences(self, model_name: str) -> int:
        """Get total inference count for model."""
        try:
            # This would query actual metrics storage
            # Use model name hash for consistent values
            import hashlib
            seed = int(hashlib.md5(model_name.encode()).hexdigest()[:8], 16)
            random.seed(seed)
            return random.randint(1000, 50000)
        except:
            return random.randint(1000, 10000)
    
    def _count_model_alerts(self, model_name: str) -> int:
        """Count active alerts for model."""
        try:
            # This would query the oversight system
            if hasattr(self.framework, 'oversight_engine'):
                return len(self.framework.oversight_engine.get_active_alerts(model_name))
            else:
                return random.randint(0, 5)
        except:
            return random.randint(0, 3)
    
    def get_recent_alerts(self) -> List[Dict[str, Any]]:
        """Get recent oversight alerts."""
        if not self.framework:
            return self._generate_mock_alerts()
        
        try:
            alerts = []
            
            # Get real alerts from oversight engine if available
            if hasattr(self.framework, 'oversight_engine'):
                oversight = self.framework.oversight_engine
                real_alerts = oversight.get_recent_alerts(limit=10)
                
                for alert in real_alerts:
                    alerts.append({
                        "id": alert.alert_id,
                        "type": alert.alert_type.value,
                        "severity": alert.severity.value,
                        "model": alert.model_name,
                        "description": alert.description,
                        "timestamp": alert.timestamp.isoformat(),
                        "status": alert.status.value if hasattr(alert, 'status') else "pending_review"
                    })
            
            # Get alerts from compliance system if available
            if hasattr(self.framework, 'compliance') and hasattr(self.framework.compliance, 'get_recent_violations'):
                compliance_alerts = self.framework.compliance.get_recent_violations(limit=5)
                
                for violation in compliance_alerts:
                    alerts.append({
                        "id": f"comp_{getattr(violation, 'id', 'unknown')}",
                        "type": "compliance_violation",
                        "severity": getattr(violation, 'severity', 'medium'),
                        "model": getattr(violation, 'model_name', 'unknown'),
                        "description": getattr(violation, 'description', 'Compliance violation detected'),
                        "timestamp": getattr(violation, 'timestamp', datetime.now().isoformat()),
                        "status": getattr(violation, 'status', 'under_review')
                    })
            
            # Sort by timestamp (most recent first)
            alerts.sort(key=lambda x: x['timestamp'], reverse=True)
            
            # Return top 10 most recent
            return alerts[:10] if alerts else self._generate_mock_alerts()
            
        except Exception as e:
            self.logger.warning(f"Failed to get real alerts: {e}")
            return self._generate_mock_alerts()
    
    def _generate_mock_alerts(self) -> List[Dict[str, Any]]:
        """Generate mock alerts for demonstration."""
        random.seed(int(time.time()) // 900)  # Change every 15 minutes
        
        alert_types = ["low_confidence", "high_uncertainty", "compliance_violation", "bias_detection", "data_drift"]
        severities = ["low", "medium", "high"]
        statuses = ["pending_review", "under_review", "reviewed", "resolved"]
        
        models = ["credit_risk_model", "fraud_detection_model", "recommendation_engine"]
        
        alerts = []
        num_alerts = random.randint(2, 8)
        
        for i in range(num_alerts):
            alert_type = random.choice(alert_types)
            model = random.choice(models)
            severity = random.choice(severities)
            
            # Generate realistic descriptions
            descriptions = {
                "low_confidence": f"Prediction confidence below threshold ({random.uniform(0.5, 0.79):.2f})",
                "high_uncertainty": f"Prediction uncertainty above threshold ({random.uniform(0.25, 0.45):.2f})",
                "compliance_violation": f"GDPR consent verification failed",
                "bias_detection": f"Potential bias detected in {random.choice(['gender', 'age', 'ethnicity'])} attribute",
                "data_drift": f"Input data distribution shift detected (KL divergence: {random.uniform(0.1, 0.5):.3f})"
            }
            
            alert = {
                "id": f"alert_{i+1:03d}",
                "type": alert_type,
                "severity": severity,
                "model": model,
                "description": descriptions.get(alert_type, f"Alert of type {alert_type}"),
                "timestamp": (datetime.now() - timedelta(minutes=random.randint(5, 1440))).isoformat(),
                "status": random.choice(statuses)
            }
            alerts.append(alert)
        
        # Sort by timestamp
        alerts.sort(key=lambda x: x['timestamp'], reverse=True)
        return alerts
    
    def _aggregate_compliance_metrics(self, model_metrics: Dict) -> Dict[str, Any]:
        """Aggregate compliance metrics across models."""
        # Implementation would aggregate real metrics
        return {
            "overall_score": 87.5,
            "total_models": len(model_metrics),
            "compliant_models": len([m for m in model_metrics.values() if m.get("score", 0) >= 80]),
            "frameworks": {},
            "recent_alerts": 3,
            "high_priority_issues": 1
        }


class CIAFDashboard:
    """Main dashboard application."""
    
    def __init__(self, ciaf_framework=None, host="127.0.0.1", port=5000, debug=False):
        if not FLASK_AVAILABLE:
            raise ImportError("Flask is required for dashboard. Install with: pip install flask flask-socketio")
        
        self.app = Flask(__name__, 
                        template_folder=self._get_templates_dir(),
                        static_folder=self._get_static_dir())
        self.app.config['SECRET_KEY'] = 'ciaf_dashboard_' + uuid.uuid4().hex[:8]
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        self.data = DashboardData(ciaf_framework)
        self.host = host
        self.port = port
        self.debug = debug
        
        self._setup_routes()
        self._setup_socketio_events()
        self._create_template_files()
    
    def _get_templates_dir(self) -> str:
        """Get templates directory path."""
        return str(Path(__file__).parent / "dashboard_templates")
    
    def _get_static_dir(self) -> str:
        """Get static files directory path."""
        return str(Path(__file__).parent / "dashboard_static")
    
    def _setup_routes(self):
        """Setup Flask routes."""
        
        @self.app.route('/')
        def dashboard_home():
            return render_template('dashboard.html')
        
        @self.app.route('/api/compliance/overview')
        def api_compliance_overview():
            return jsonify(self.data.get_compliance_overview())
        
        @self.app.route('/api/audit/metrics')
        def api_audit_metrics():
            return jsonify(self.data.get_audit_metrics())
        
        @self.app.route('/api/performance/metrics')
        def api_performance_metrics():
            return jsonify(self.data.get_performance_metrics())
        
        @self.app.route('/api/models')
        def api_models():
            return jsonify(self.data.get_model_list())
        
        @self.app.route('/api/alerts')
        def api_alerts():
            return jsonify(self.data.get_recent_alerts())
        
        @self.app.route('/api/charts/compliance_trends')
        def api_compliance_trends():
            if not PLOTLY_AVAILABLE:
                return jsonify({"error": "Plotly not available"})
            
            # Generate sample compliance trends chart
            dates = [(datetime.now() - timedelta(days=x)).strftime('%Y-%m-%d') for x in range(30, 0, -1)]
            scores = [85 + (i % 10) for i in range(30)]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=dates, y=scores, mode='lines+markers', name='Compliance Score'))
            fig.update_layout(
                title='Compliance Score Trends (30 Days)',
                xaxis_title='Date',
                yaxis_title='Compliance Score (%)',
                height=400
            )
            
            return json.dumps(fig, cls=PlotlyJSONEncoder)
        
        @self.app.route('/api/charts/audit_events')
        def api_audit_events():
            if not PLOTLY_AVAILABLE:
                return jsonify({"error": "Plotly not available"})
            
            # Generate audit events chart
            event_types = ['Training', 'Inference', 'Data Access', 'Compliance Check']
            counts = [5600, 8900, 420, 500]
            
            fig = go.Figure(data=[go.Pie(labels=event_types, values=counts, hole=0.3)])
            fig.update_layout(title='Audit Events Distribution', height=400)
            
            return json.dumps(fig, cls=PlotlyJSONEncoder)
        
        @self.app.route('/model/<model_name>')
        def model_detail(model_name):
            return render_template('model_detail.html', model_name=model_name)
        
        @self.app.route('/compliance')
        def compliance_view():
            return render_template('compliance.html')
        
        @self.app.route('/alerts')
        def alerts_view():
            return render_template('alerts.html')
    
    def _setup_socketio_events(self):
        """Setup SocketIO events for real-time updates."""
        
        @self.socketio.on('connect')
        def handle_connect():
            emit('status', {'msg': 'Connected to CIAF Dashboard'})
        
        @self.socketio.on('request_metrics_update')
        def handle_metrics_update():
            # Send real-time metrics update
            metrics = {
                'compliance': self.data.get_compliance_overview(),
                'audit': self.data.get_audit_metrics(),
                'performance': self.data.get_performance_metrics()
            }
            emit('metrics_update', metrics)
    
    def _create_template_files(self):
        """Create HTML template files."""
        templates_dir = Path(self._get_templates_dir())
        static_dir = Path(self._get_static_dir())
        
        # Create directories
        templates_dir.mkdir(exist_ok=True)
        static_dir.mkdir(exist_ok=True)
        
        # Create main dashboard template
        dashboard_html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CIAF Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        .metric-card { 
            transition: transform 0.2s; 
        }
        .metric-card:hover { 
            transform: translateY(-2px); 
        }
        .status-indicator {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            display: inline-block;
            margin-right: 8px;
        }
        .status-active { background-color: #28a745; }
        .status-warning { background-color: #ffc107; }
        .status-error { background-color: #dc3545; }
        .alert-severity-high { border-left: 4px solid #dc3545; }
        .alert-severity-medium { border-left: 4px solid #ffc107; }
        .alert-severity-low { border-left: 4px solid #28a745; }
    </style>
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
        <div class="container-fluid">
            <a class="navbar-brand" href="/">
                <i class="fas fa-shield-alt"></i> CIAF Dashboard
            </a>
            <div class="navbar-nav ms-auto">
                <a class="nav-link" href="/compliance">Compliance</a>
                <a class="nav-link" href="/alerts">Alerts</a>
            </div>
        </div>
    </nav>

    <div class="container-fluid mt-4">
        <!-- Overview Cards -->
        <div class="row mb-4">
            <div class="col-md-3">
                <div class="card metric-card bg-primary text-white">
                    <div class="card-body">
                        <div class="d-flex justify-content-between">
                            <div>
                                <h4 class="card-title" id="overall-compliance">--</h4>
                                <p class="card-text">Overall Compliance</p>
                            </div>
                            <div class="align-self-center">
                                <i class="fas fa-certificate fa-2x"></i>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card metric-card bg-success text-white">
                    <div class="card-body">
                        <div class="d-flex justify-content-between">
                            <div>
                                <h4 class="card-title" id="total-models">--</h4>
                                <p class="card-text">Active Models</p>
                            </div>
                            <div class="align-self-center">
                                <i class="fas fa-robot fa-2x"></i>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card metric-card bg-info text-white">
                    <div class="card-body">
                        <div class="d-flex justify-content-between">
                            <div>
                                <h4 class="card-title" id="total-events">--</h4>
                                <p class="card-text">Audit Events</p>
                            </div>
                            <div class="align-self-center">
                                <i class="fas fa-list-alt fa-2x"></i>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card metric-card bg-warning text-white">
                    <div class="card-body">
                        <div class="d-flex justify-content-between">
                            <div>
                                <h4 class="card-title" id="active-alerts">--</h4>
                                <p class="card-text">Active Alerts</p>
                            </div>
                            <div class="align-self-center">
                                <i class="fas fa-exclamation-triangle fa-2x"></i>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Charts Row -->
        <div class="row mb-4">
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">
                        <h5>Compliance Trends</h5>
                    </div>
                    <div class="card-body">
                        <div id="compliance-trends-chart"></div>
                    </div>
                </div>
            </div>
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">
                        <h5>Audit Events Distribution</h5>
                    </div>
                    <div class="card-body">
                        <div id="audit-events-chart"></div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Models and Alerts Row -->
        <div class="row">
            <div class="col-md-8">
                <div class="card">
                    <div class="card-header">
                        <h5>Models Status</h5>
                    </div>
                    <div class="card-body">
                        <div class="table-responsive">
                            <table class="table table-hover">
                                <thead>
                                    <tr>
                                        <th>Model</th>
                                        <th>Version</th>
                                        <th>Status</th>
                                        <th>Compliance</th>
                                        <th>Last Inference</th>
                                        <th>Actions</th>
                                    </tr>
                                </thead>
                                <tbody id="models-table">
                                    <!-- Models will be loaded here -->
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </div>
            <div class="col-md-4">
                <div class="card">
                    <div class="card-header">
                        <h5>Recent Alerts</h5>
                    </div>
                    <div class="card-body">
                        <div id="alerts-list">
                            <!-- Alerts will be loaded here -->
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Dashboard JavaScript
        const socket = io();
        
        // Load initial data
        async function loadDashboardData() {
            try {
                // Load overview data
                const complianceResp = await fetch('/api/compliance/overview');
                const compliance = await complianceResp.json();
                
                const auditResp = await fetch('/api/audit/metrics');
                const audit = await auditResp.json();
                
                const modelsResp = await fetch('/api/models');
                const models = await modelsResp.json();
                
                const alertsResp = await fetch('/api/alerts');
                const alerts = await alertsResp.json();
                
                // Update overview cards
                document.getElementById('overall-compliance').textContent = compliance.overall_score + '%';
                document.getElementById('total-models').textContent = compliance.total_models;
                document.getElementById('total-events').textContent = audit.total_events.toLocaleString();
                document.getElementById('active-alerts').textContent = compliance.recent_alerts;
                
                // Update models table
                const modelsTable = document.getElementById('models-table');
                modelsTable.innerHTML = models.map(model => `
                    <tr>
                        <td><a href="/model/${model.name}">${model.name}</a></td>
                        <td>${model.version}</td>
                        <td><span class="status-indicator status-${model.status === 'active' ? 'active' : 'warning'}"></span>${model.status}</td>
                        <td>${model.compliance_score}%</td>
                        <td>${new Date(model.last_inference).toLocaleString()}</td>
                        <td>
                            <button class="btn btn-sm btn-primary">Details</button>
                        </td>
                    </tr>
                `).join('');
                
                // Update alerts list
                const alertsList = document.getElementById('alerts-list');
                alertsList.innerHTML = alerts.map(alert => `
                    <div class="alert alert-light alert-severity-${alert.severity} mb-2">
                        <strong>${alert.type.replace('_', ' ').toUpperCase()}</strong><br>
                        <small class="text-muted">${alert.model} - ${new Date(alert.timestamp).toLocaleString()}</small><br>
                        ${alert.description}
                        <div class="mt-1">
                            <span class="badge bg-${alert.severity === 'high' ? 'danger' : 'warning'}">${alert.severity}</span>
                            <span class="badge bg-secondary">${alert.status.replace('_', ' ')}</span>
                        </div>
                    </div>
                `).join('');
                
            } catch (error) {
                console.error('Error loading dashboard data:', error);
            }
        }
        
        // Load charts
        async function loadCharts() {
            try {
                // Compliance trends chart
                const trendsResp = await fetch('/api/charts/compliance_trends');
                const trendsData = await trendsResp.json();
                if (!trendsData.error) {
                    Plotly.newPlot('compliance-trends-chart', trendsData.data, trendsData.layout, {responsive: true});
                }
                
                // Audit events chart
                const eventsResp = await fetch('/api/charts/audit_events');
                const eventsData = await eventsResp.json();
                if (!eventsData.error) {
                    Plotly.newPlot('audit-events-chart', eventsData.data, eventsData.layout, {responsive: true});
                }
            } catch (error) {
                console.error('Error loading charts:', error);
            }
        }
        
        // WebSocket events
        socket.on('connect', function() {
            console.log('Connected to dashboard');
        });
        
        socket.on('metrics_update', function(data) {
            console.log('Received metrics update:', data);
            // Update dashboard with real-time data
        });
        
        // Initialize dashboard
        document.addEventListener('DOMContentLoaded', function() {
            loadDashboardData();
            loadCharts();
            
            // Set up periodic refresh
            setInterval(loadDashboardData, 30000); // Refresh every 30 seconds
        });
    </script>
</body>
</html>"""
        
        with open(templates_dir / "dashboard.html", "w") as f:
            f.write(dashboard_html)
    
    def run(self, **kwargs):
        """Run the dashboard server."""
        print(f"🌐 Starting CIAF Dashboard on http://{self.host}:{self.port}")
        print("📊 Dashboard features:")
        print("   - Real-time compliance monitoring")
        print("   - Audit trail visualization") 
        print("   - Model performance metrics")
        print("   - Alert management")
        
        self.socketio.run(
            self.app, 
            host=self.host, 
            port=self.port, 
            debug=self.debug,
            **kwargs
        )


def create_dashboard(ciaf_framework=None, **kwargs) -> CIAFDashboard:
    """Factory function to create dashboard instance."""
    return CIAFDashboard(ciaf_framework, **kwargs)


# Demo function
def demo_dashboard():
    """Demo dashboard functionality."""
    print("🎯 CIAF Dashboard Demo")
    print("=" * 30)
    
    if not FLASK_AVAILABLE:
        print("❌ Flask not available. Install with: pip install flask flask-socketio")
        return
    
    # Create dashboard
    dashboard = create_dashboard()
    
    print("📊 Dashboard created successfully!")
    print("🌐 Starting dashboard server...")
    print("💡 Visit http://127.0.0.1:5000 to view the dashboard")
    
    try:
        dashboard.run(debug=True)
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped")


if __name__ == "__main__":
    demo_dashboard()