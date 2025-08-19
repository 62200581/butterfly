import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class ButterflyDataAnalyzer:
    """Comprehensive analysis of butterfly classification training data"""
    
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.data = None
        self.load_data()
        
    def load_data(self):
        """Load and preprocess training data"""
        try:
            self.data = pd.read_csv(self.csv_path)
            print(f"✅ Data loaded successfully: {len(self.data)} epochs")
            print(f"📊 Columns: {list(self.data.columns)}")
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            
    def get_training_summary(self):
        """Generate comprehensive training summary"""
        if self.data is None:
            return "No data loaded"
            
        summary = {
            'total_epochs': len(self.data),
            'final_training_accuracy': self.data['accuracy'].iloc[-1],
            'final_validation_accuracy': self.data['val_accuracy'].iloc[-1],
            'best_training_accuracy': self.data['accuracy'].max(),
            'best_validation_accuracy': self.data['val_accuracy'].max(),
            'overfitting_gap': self.data['accuracy'].iloc[-1] - self.data['val_accuracy'].iloc[-1],
            'training_convergence': self._check_convergence('accuracy'),
            'validation_convergence': self._check_convergence('val_accuracy')
        }
        
        return summary
    
    def _check_convergence(self, metric):
        """Check if a metric has converged"""
        last_5 = self.data[metric].tail(5)
        return (last_5.max() - last_5.min()) < 0.01
    
    def analyze_overfitting(self):
        """Analyze overfitting patterns"""
        if self.data is None:
            return "No data loaded"
            
        # Calculate overfitting metrics
        self.data['overfitting_gap'] = self.data['accuracy'] - self.data['val_accuracy']
        self.data['loss_gap'] = self.data['loss'] - self.data['val_loss']
        
        analysis = {
            'max_overfitting_gap': self.data['overfitting_gap'].max(),
            'current_overfitting_gap': self.data['overfitting_gap'].iloc[-1],
            'overfitting_trend': 'increasing' if self.data['overfitting_gap'].iloc[-1] > self.data['overfitting_gap'].iloc[0] else 'decreasing',
            'recommendation': self._get_overfitting_recommendation()
        }
        
        return analysis
    
    def _get_overfitting_recommendation(self):
        """Get recommendations based on overfitting analysis"""
        current_gap = self.data['accuracy'].iloc[-1] - self.data['val_accuracy'].iloc[-1]
        
        if current_gap > 0.05:
            return "High overfitting detected. Consider: early stopping, data augmentation, regularization, or reducing model complexity."
        elif current_gap > 0.02:
            return "Moderate overfitting. Monitor closely and consider regularization techniques."
        else:
            return "Good generalization. Model is performing well on both training and validation sets."
    
    def create_training_curves(self, save_path=None):
        """Create comprehensive training curves visualization"""
        if self.data is None:
            return "No data loaded"
            
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Training vs Validation Accuracy', 'Training vs Validation Loss', 
                          'F1 Score Progress', 'Overfitting Analysis'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Accuracy plot
        fig.add_trace(
            go.Scatter(x=self.data['Epoch'], y=self.data['accuracy'], 
                      name='Training Accuracy', line=dict(color='blue')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=self.data['Epoch'], y=self.data['val_accuracy'], 
                      name='Validation Accuracy', line=dict(color='red')),
            row=1, col=1
        )
        
        # Loss plot
        fig.add_trace(
            go.Scatter(x=self.data['Epoch'], y=self.data['loss'], 
                      name='Training Loss', line=dict(color='blue')),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=self.data['Epoch'], y=self.data['val_loss'], 
                      name='Validation Loss', line=dict(color='red')),
            row=1, col=2
        )
        
        # F1 Score plot
        fig.add_trace(
            go.Scatter(x=self.data['Epoch'], y=self.data['F1_score'], 
                      name='Training F1', line=dict(color='green')),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=self.data['Epoch'], y=self.data['val_F1_score'], 
                      name='Validation F1', line=dict(color='orange')),
            row=2, col=1
        )
        
        # Overfitting gap plot
        fig.add_trace(
            go.Scatter(x=self.data['Epoch'], y=self.data['overfitting_gap'], 
                      name='Overfitting Gap', line=dict(color='purple')),
            row=2, col=2
        )
        
        fig.update_layout(
            title="🦋 Butterfly Classification Training Progress",
            height=800,
            showlegend=True
        )
        
        if save_path:
            fig.write_html(save_path)
            
        return fig
    
    def generate_insights_report(self):
        """Generate comprehensive insights report"""
        if self.data is None:
            return "No data loaded"
            
        summary = self.get_training_summary()
        overfitting = self.analyze_overfitting()
        
        report = f"""
🦋 BUTTERFLY CLASSIFICATION TRAINING INSIGHTS REPORT
{'='*60}

📊 TRAINING SUMMARY:
• Total Epochs: {summary['total_epochs']}
• Final Training Accuracy: {summary['final_training_accuracy']:.3f} ({summary['final_training_accuracy']*100:.1f}%)
• Final Validation Accuracy: {summary['final_validation_accuracy']:.3f} ({summary['final_validation_accuracy']*100:.1f}%)
• Best Training Accuracy: {summary['best_training_accuracy']:.3f} ({summary['best_training_accuracy']*100:.1f}%)
• Best Validation Accuracy: {summary['best_validation_accuracy']:.3f} ({summary['best_validation_accuracy']*100:.1f}%)

🚨 OVERFITTING ANALYSIS:
• Current Overfitting Gap: {overfitting['current_overfitting_gap']:.3f}
• Maximum Overfitting Gap: {overfitting['max_overfitting_gap']:.3f}
• Trend: {overfitting['overfitting_trend'].title()}
• Recommendation: {overfitting['recommendation']}

🎯 PERFORMANCE INSIGHTS:
• Training Convergence: {'✅ Yes' if summary['training_convergence'] else '❌ No'}
• Validation Convergence: {'✅ Yes' if summary['validation_convergence'] else '❌ No'}
• Model Performance: {'🟢 Excellent' if summary['final_validation_accuracy'] > 0.95 else '🟡 Good' if summary['final_validation_accuracy'] > 0.90 else '🔴 Needs Improvement'}

💡 RECOMMENDATIONS:
1. {'Consider early stopping to prevent overfitting' if overfitting['current_overfitting_gap'] > 0.05 else 'Continue training as overfitting is minimal'}
2. {'Implement data augmentation to improve generalization' if overfitting['current_overfitting_gap'] > 0.03 else 'Data augmentation may not be necessary'}
3. {'Add regularization techniques (dropout, weight decay)' if overfitting['current_overfitting_gap'] > 0.04 else 'Current regularization appears adequate'}

🌟 NEXT STEPS:
• Monitor validation metrics closely
• Consider ensemble methods for better generalization
• Implement cross-validation for robust evaluation
• Explore transfer learning for improved performance
"""
        
        return report

# Example usage
if __name__ == "__main__":
    analyzer = ButterflyDataAnalyzer("training.csv.csv")
    
    # Generate insights
    print(analyzer.generate_insights_report())
    
    # Create visualizations
    fig = analyzer.create_training_curves("training_curves.html")
    print("📊 Training curves saved to 'training_curves.html'")
