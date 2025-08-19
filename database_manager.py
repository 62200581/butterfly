#!/usr/bin/env python3
"""
Database Manager for Butterfly Classification Ecosystem
Provides high-level interface for database operations
"""

import os
import json
from datetime import datetime
from typing import List, Dict, Optional, Any
from database import ButterflyDatabase

class DatabaseManager:
    """High-level database management interface"""
    
    def __init__(self, db_path: str = "butterfly_ecosystem.db"):
        self.db = ButterflyDatabase(db_path)
    
    def get_system_overview(self) -> Dict[str, Any]:
        """Get comprehensive system overview"""
        stats = self.db.get_database_stats()
        
        # Get recent activity
        recent_classifications = self.db.get_classification_history(limit=5)
        recent_training = self.db.get_training_metrics(limit=5)
        all_species = self.db.get_species()
        
        overview = {
            "database_stats": stats,
            "recent_activity": {
                "classifications": len(recent_classifications),
                "training_updates": len(recent_training)
            },
            "species_diversity": {
                "total_species": len(all_species),
                "families": len(set(s.get('family') for s in all_species if s.get('family'))),
                "genera": len(set(s.get('genus') for s in all_species if s.get('genus')))
            },
            "system_health": {
                "database_size_mb": self._get_database_size(),
                "last_backup": self._get_last_backup_info(),
                "data_integrity": self._check_data_integrity()
            }
        }
        
        return overview
    
    def _get_database_size(self) -> float:
        """Get database file size in MB"""
        try:
            size_bytes = os.path.getsize(self.db.db_path)
            return round(size_bytes / (1024 * 1024), 2)
        except:
            return 0.0
    
    def _get_last_backup_info(self) -> str:
        """Get information about last backup"""
        backup_dir = "backups"
        if not os.path.exists(backup_dir):
            return "No backups found"
        
        backup_files = [f for f in os.listdir(backup_dir) if f.endswith('.db')]
        if not backup_files:
            return "No backups found"
        
        # Get most recent backup
        backup_files.sort(key=lambda x: os.path.getmtime(os.path.join(backup_dir, x)), reverse=True)
        latest_backup = backup_files[0]
        backup_time = datetime.fromtimestamp(os.path.getmtime(os.path.join(backup_dir, latest_backup)))
        return f"{latest_backup} ({backup_time.strftime('%Y-%m-%d %H:%M')})"
    
    def _check_data_integrity(self) -> Dict[str, bool]:
        """Check data integrity across tables"""
        integrity_checks = {}
        
        try:
            # Check foreign key relationships
            conn = self.db.get_connection()
            cursor = conn.cursor()
            
            # Check if classifications reference valid species
            cursor.execute("""
                SELECT COUNT(*) FROM classification_history ch
                LEFT JOIN species s ON ch.predicted_species_id = s.id
                WHERE ch.predicted_species_id IS NOT NULL AND s.id IS NULL
            """)
            orphaned_classifications = cursor.fetchone()[0]
            integrity_checks['foreign_keys_valid'] = orphaned_classifications == 0
            
            # Check for duplicate species names
            cursor.execute("""
                SELECT scientific_name, COUNT(*) FROM species 
                GROUP BY scientific_name HAVING COUNT(*) > 1
            """)
            duplicate_species = cursor.fetchall()
            integrity_checks['no_duplicate_species'] = len(duplicate_species) == 0
            
            # Check for valid training metrics
            cursor.execute("""
                SELECT COUNT(*) FROM training_metrics 
                WHERE epoch < 0 OR training_accuracy > 1 OR validation_accuracy > 1
            """)
            invalid_metrics = cursor.fetchone()[0]
            integrity_checks['training_metrics_valid'] = invalid_metrics == 0
            
            conn.close()
            
        except Exception as e:
            integrity_checks['database_accessible'] = False
            integrity_checks['error'] = str(e)
        
        return integrity_checks
    
    def manage_species(self, action: str, **kwargs) -> Dict[str, Any]:
        """Manage species data (add, update, delete, search)"""
        result = {"success": False, "message": "", "data": None}
        
        try:
            if action == "add":
                required_fields = ['scientific_name']
                if not all(field in kwargs for field in required_fields):
                    result["message"] = f"Missing required fields: {required_fields}"
                    return result
                
                species_id = self.db.add_species(**kwargs)
                if species_id:
                    result["success"] = True
                    result["message"] = f"Species added successfully with ID: {species_id}"
                    result["data"] = {"species_id": species_id}
                else:
                    result["message"] = "Failed to add species (possibly duplicate scientific name)"
            
            elif action == "update":
                if 'species_id' not in kwargs:
                    result["message"] = "Missing species_id for update"
                    return result
                
                species_id = kwargs.pop('species_id')
                if self.db.update_species(species_id, **kwargs):
                    result["success"] = True
                    result["message"] = f"Species {species_id} updated successfully"
                else:
                    result["message"] = f"Failed to update species {species_id}"
            
            elif action == "delete":
                if 'species_id' not in kwargs:
                    result["message"] = "Missing species_id for deletion"
                    return result
                
                species_id = kwargs['species_id']
                if self.db.delete_species(species_id):
                    result["success"] = True
                    result["message"] = f"Species {species_id} deleted successfully"
                else:
                    result["message"] = f"Failed to delete species {species_id}"
            
            elif action == "search":
                query = kwargs.get('query', '')
                if not query:
                    result["message"] = "Search query is required"
                    return result
                
                species = self.db.search_species(query)
                result["success"] = True
                result["data"] = species
                result["message"] = f"Found {len(species)} species matching '{query}'"
            
            elif action == "get":
                if 'species_id' in kwargs:
                    species = self.db.get_species(species_id=kwargs['species_id'])
                elif 'scientific_name' in kwargs:
                    species = self.db.get_species(scientific_name=kwargs['scientific_name'])
                else:
                    species = self.db.get_species()
                
                result["success"] = True
                result["data"] = species
                result["message"] = f"Retrieved {len(species) if isinstance(species, list) else 1} species"
            
            else:
                result["message"] = f"Unknown action: {action}"
        
        except Exception as e:
            result["message"] = f"Error: {str(e)}"
        
        return result
    
    def manage_training_data(self, action: str, **kwargs) -> Dict[str, Any]:
        """Manage training data and metrics"""
        result = {"success": False, "message": "", "data": None}
        
        try:
            if action == "add":
                required_fields = ['epoch']
                if not all(field in kwargs for field in required_fields):
                    result["message"] = f"Missing required fields: {required_fields}"
                    return result
                
                metrics_id = self.db.add_training_metrics(**kwargs)
                result["success"] = True
                result["message"] = f"Training metrics added successfully with ID: {metrics_id}"
                result["data"] = {"metrics_id": metrics_id}
            
            elif action == "get":
                limit = kwargs.get('limit')
                metrics = self.db.get_training_metrics(limit=limit)
                result["success"] = True
                result["data"] = metrics
                result["message"] = f"Retrieved {len(metrics)} training metrics"
            
            elif action == "analyze":
                metrics = self.db.get_training_metrics()
                if not metrics:
                    result["message"] = "No training metrics available for analysis"
                    return result
                
                # Analyze training trends
                analysis = self._analyze_training_trends(metrics)
                result["success"] = True
                result["data"] = analysis
                result["message"] = "Training analysis completed"
            
            else:
                result["message"] = f"Unknown action: {action}"
        
        except Exception as e:
            result["message"] = f"Error: {str(e)}"
        
        return result
    
    def _analyze_training_trends(self, metrics: List[Dict]) -> Dict[str, Any]:
        """Analyze training metrics for trends and insights"""
        if not metrics:
            return {}
        
        # Sort by epoch
        sorted_metrics = sorted(metrics, key=lambda x: x['epoch'])
        
        analysis = {
            "total_epochs": len(sorted_metrics),
            "epoch_range": {
                "start": sorted_metrics[0]['epoch'],
                "end": sorted_metrics[-1]['epoch']
            },
            "performance_trends": {},
            "overfitting_analysis": {},
            "recommendations": []
        }
        
        # Analyze performance trends
        epochs = [m['epoch'] for m in sorted_metrics]
        train_acc = [m.get('training_accuracy', 0) for m in sorted_metrics]
        val_acc = [m.get('validation_accuracy', 0) for m in sorted_metrics]
        train_loss = [m.get('training_loss', 0) for m in sorted_metrics]
        val_loss = [m.get('validation_loss', 0) for m in sorted_metrics]
        
        # Calculate trends
        if len(train_acc) > 1:
            train_acc_trend = "improving" if train_acc[-1] > train_acc[0] else "declining"
            val_acc_trend = "improving" if val_acc[-1] > val_acc[0] else "declining"
            
            analysis["performance_trends"] = {
                "training_accuracy": {
                    "trend": train_acc_trend,
                    "start": train_acc[0],
                    "end": train_acc[-1],
                    "improvement": round((train_acc[-1] - train_acc[0]) * 100, 2)
                },
                "validation_accuracy": {
                    "trend": val_acc_trend,
                    "start": val_acc[0],
                    "end": val_acc[-1],
                    "improvement": round((val_acc[-1] - val_acc[0]) * 100, 2)
                }
            }
        
        # Overfitting analysis
        if len(train_acc) > 1 and len(val_acc) > 1:
            train_val_gap = [abs(t - v) for t, v in zip(train_acc, val_acc)]
            avg_gap = sum(train_val_gap) / len(train_val_gap)
            max_gap = max(train_val_gap)
            
            analysis["overfitting_analysis"] = {
                "average_train_val_gap": round(avg_gap, 4),
                "max_train_val_gap": round(max_gap, 4),
                "overfitting_risk": "high" if avg_gap > 0.1 else "medium" if avg_gap > 0.05 else "low"
            }
        
        # Generate recommendations
        if analysis.get("overfitting_analysis", {}).get("overfitting_risk") == "high":
            analysis["recommendations"].append("Consider adding regularization or early stopping")
        
        if train_acc and train_acc[-1] < 0.8:
            analysis["recommendations"].append("Training accuracy is low - consider adjusting learning rate or model architecture")
        
        if val_acc and val_acc[-1] < 0.7:
            analysis["recommendations"].append("Validation accuracy is low - model may need more training data or different approach")
        
        return analysis
    
    def manage_training_metrics(self, operation: str, **kwargs) -> Dict[str, Any]:
        """Manage training metrics data"""
        try:
            if operation == "add":
                # Add new training metrics
                cursor = self.db.get_connection().cursor()
                cursor.execute("""
                    INSERT INTO training_metrics 
                    (epoch, training_loss, training_accuracy, training_f1_score, validation_loss, validation_accuracy, validation_f1_score, learning_rate)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    kwargs.get('epoch'),
                    kwargs.get('training_loss', 0),
                    kwargs.get('training_accuracy'),
                    kwargs.get('training_f1_score'),
                    kwargs.get('validation_loss', 0),
                    kwargs.get('validation_accuracy'),
                    kwargs.get('validation_f1_score'),
                    kwargs.get('learning_rate', 0.001)
                ))
                self.db.get_connection().commit()
                return {"status": "success", "message": "Training metrics added successfully"}
                
            elif operation == "get":
                # Get all training metrics
                cursor = self.db.get_connection().cursor()
                cursor.execute("""
                    SELECT epoch, training_loss, training_accuracy, training_f1_score, validation_loss, validation_accuracy, validation_f1_score, learning_rate, timestamp
                    FROM training_metrics 
                    ORDER BY epoch
                """)
                rows = cursor.fetchall()
                
                metrics = []
                for row in rows:
                    metrics.append({
                        'epoch': row[0],
                        'training_loss': row[1],
                        'training_accuracy': row[2],
                        'training_f1_score': row[3],
                        'validation_loss': row[4],
                        'validation_accuracy': row[5],
                        'validation_f1_score': row[6],
                        'learning_rate': row[7],
                        'timestamp': row[8]
                    })
                
                return {"status": "success", "data": metrics}
                
            elif operation == "delete":
                # Delete training metrics by epoch
                epoch = kwargs.get('epoch')
                if epoch:
                    cursor = self.db.get_connection().cursor()
                    cursor.execute("DELETE FROM training_metrics WHERE epoch = ?", (epoch,))
                    self.db.get_connection().commit()
                    return {"status": "success", "message": f"Training metrics for epoch {epoch} deleted successfully"}
                else:
                    return {"status": "error", "message": "Epoch is required for deletion"}
                    
            else:
                return {"status": "error", "message": f"Unknown operation: {operation}"}
                
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def manage_classifications(self, action: str, **kwargs) -> Dict[str, Any]:
        """Manage classification history"""
        result = {"success": False, "message": "", "data": None}
        
        try:
            if action == "add":
                required_fields = ['image_filename']
                if not all(field in kwargs for field in required_fields):
                    result["message"] = f"Missing required fields: {required_fields}"
                    return result
                
                classification_id = self.db.add_classification(**kwargs)
                result["success"] = True
                result["message"] = f"Classification recorded successfully with ID: {classification_id}"
                result["data"] = {"classification_id": classification_id}
            
            elif action == "get":
                limit = kwargs.get('limit', 100)
                classifications = self.db.get_classification_history(limit=limit)
                result["success"] = True
                result["data"] = classifications
                result["message"] = f"Retrieved {len(classifications)} classifications"
            
            elif action == "analyze":
                classifications = self.db.get_classification_history(limit=1000)
                if not classifications:
                    result["message"] = "No classification history available for analysis"
                    return result
                
                analysis = self._analyze_classification_history(classifications)
                result["success"] = True
                result["data"] = analysis
                result["message"] = "Classification analysis completed"
            
            else:
                result["message"] = f"Unknown action: {action}"
        
        except Exception as e:
            result["message"] = f"Error: {str(e)}"
        
        return result
    
    def _analyze_classification_history(self, classifications: List[Dict]) -> Dict[str, Any]:
        """Analyze classification history for insights"""
        if not classifications:
            return {}
        
        analysis = {
            "total_classifications": len(classifications),
            "confidence_analysis": {},
            "species_distribution": {},
            "performance_metrics": {},
            "recent_trends": {}
        }
        
        # Confidence analysis
        confidence_scores = [c.get('confidence_score', 0) for c in classifications if c.get('confidence_score')]
        if confidence_scores:
            analysis["confidence_analysis"] = {
                "average_confidence": round(sum(confidence_scores) / len(confidence_scores), 3),
                "min_confidence": round(min(confidence_scores), 3),
                "max_confidence": round(max(confidence_scores), 3),
                "high_confidence_count": len([c for c in confidence_scores if c > 0.8]),
                "low_confidence_count": len([c for c in confidence_scores if c < 0.5])
            }
        
        # Species distribution
        species_counts = {}
        for c in classifications:
            species_name = c.get('scientific_name', 'Unknown')
            species_counts[species_name] = species_counts.get(species_name, 0) + 1
        
        analysis["species_distribution"] = {
            "total_unique_species": len(species_counts),
            "most_common_species": max(species_counts.items(), key=lambda x: x[1]) if species_counts else None,
            "species_frequency": dict(sorted(species_counts.items(), key=lambda x: x[1], reverse=True)[:5])
        }
        
        # Performance metrics
        if confidence_scores:
            high_confidence_rate = len([c for c in confidence_scores if c > 0.8]) / len(confidence_scores)
            analysis["performance_metrics"] = {
                "high_confidence_rate": round(high_confidence_rate, 3),
                "average_processing_time": "N/A"  # Could be calculated if processing_time is available
            }
        
        return analysis
    
    def export_data(self, table_name: str, format: str = 'csv', backup: bool = False) -> Dict[str, Any]:
        """Export data from database tables"""
        result = {"success": False, "message": "", "data": None}
        
        try:
            # Create backup if requested
            if backup:
                backup_path = self.db.backup_database()
                result["backup_created"] = backup_path
            
            # Export data
            export_file = self.db.export_data(table_name, format)
            result["success"] = True
            result["message"] = f"Data exported successfully to {export_file}"
            result["data"] = {"export_file": export_file}
        
        except Exception as e:
            result["message"] = f"Export failed: {str(e)}"
        
        return result
    
    def get_database_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive database health report"""
        try:
            overview = self.get_system_overview()
            
            health_report = {
                "timestamp": datetime.now().isoformat(),
                "overview": overview,
                "recommendations": [],
                "status": "healthy"
            }
            
            # Check for potential issues
            if overview["system_health"]["data_integrity"].get("overfitting_risk") == "high":
                health_report["recommendations"].append("High overfitting risk detected - consider regularization")
                health_report["status"] = "warning"
            
            if overview["database_stats"]["classification_history_count"] > 1000:
                health_report["recommendations"].append("Large classification history - consider archiving old records")
            
            if overview["system_health"]["database_size_mb"] > 100:
                health_report["recommendations"].append("Database size is large - consider cleanup or archiving")
            
            if not overview["system_health"]["data_integrity"].get("database_accessible", True):
                health_report["status"] = "critical"
                health_report["recommendations"].append("Database accessibility issues detected")
            
            return health_report
        
        except Exception as e:
            return {
                "timestamp": datetime.now().isoformat(),
                "status": "error",
                "error": str(e),
                "recommendations": ["Check database connection and permissions"]
            }

def main():
    """Main function for testing the database manager"""
    print("🦋 Butterfly Database Manager")
    print("=" * 40)
    
    # Initialize database manager
    manager = DatabaseManager()
    
    # Get system overview
    print("\n📊 System Overview:")
    overview = manager.get_system_overview()
    print(f"Database Size: {overview['system_health']['database_size_mb']} MB")
    print(f"Total Species: {overview['species_diversity']['total_species']}")
    print(f"Total Families: {overview['species_diversity']['families']}")
    
    # Get database health report
    print("\n🏥 Database Health Report:")
    health = manager.get_database_health_report()
    print(f"Status: {health['status'].upper()}")
    if health['recommendations']:
        print("Recommendations:")
        for rec in health['recommendations']:
            print(f"  - {rec}")
    
    # Test species management
    print("\n🔍 Testing Species Management:")
    species_result = manager.manage_species("get")
    if species_result["success"]:
        print(f"Retrieved {len(species_result['data'])} species")
        print("Sample species:")
        for species in species_result['data'][:3]:
            print(f"  - {species['scientific_name']} ({species['common_name']})")
    
    # Test training data analysis
    print("\n📈 Testing Training Data Analysis:")
    training_result = manager.manage_training_data("analyze")
    if training_result["success"]:
        analysis = training_result['data']
        print(f"Total Epochs: {analysis.get('total_epochs', 0)}")
        if analysis.get('overfitting_analysis'):
            risk = analysis['overfitting_analysis'].get('overfitting_risk', 'unknown')
            print(f"Overfitting Risk: {risk.upper()}")
    
    print("\n✅ Database Manager test completed!")

if __name__ == "__main__":
    main()
