import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, classification_report
import optuna
import warnings
warnings.filterwarnings('ignore')

class ButterflyModelOptimizer:
    """Advanced model optimization for butterfly classification"""
    
    def __init__(self, training_data_path):
        self.training_data_path = training_data_path
        self.training_data = None
        self.best_params = None
        self.optimization_history = []
        
    def load_training_data(self):
        """Load and preprocess training data"""
        try:
            self.training_data = pd.read_csv(self.training_data_path)
            print(f"✅ Training data loaded: {len(self.training_data)} epochs")
            return True
        except Exception as e:
            print(f"❌ Error loading training data: {e}")
            return False
    
    def analyze_training_patterns(self):
        """Analyze training patterns for optimization insights"""
        if self.training_data is None:
            return "No training data loaded"
        
        analysis = {
            'convergence_epoch': self._find_convergence_epoch(),
            'learning_rate_analysis': self._analyze_learning_rate(),
            'overfitting_trend': self._analyze_overfitting_trend(),
            'optimal_epochs': self._suggest_optimal_epochs(),
            'regularization_needs': self._assess_regularization_needs()
        }
        
        return analysis
    
    def _find_convergence_epoch(self):
        """Find the epoch where training converges"""
        val_acc = self.training_data['val_accuracy']
        threshold = 0.001
        
        for i in range(5, len(val_acc)):
            recent_values = val_acc.iloc[i-5:i]
            if recent_values.max() - recent_values.min() < threshold:
                return i
        
        return len(val_acc)
    
    def _analyze_learning_rate(self):
        """Analyze if learning rate is appropriate"""
        loss = self.training_data['loss']
        
        # Check for learning rate issues
        if loss.iloc[-1] > loss.iloc[0] * 0.5:
            return "Learning rate might be too high - consider reducing"
        elif loss.iloc[-1] < loss.iloc[0] * 0.01 and len(loss) < 10:
            return "Learning rate might be too low - consider increasing"
        else:
            return "Learning rate appears appropriate"
    
    def _analyze_overfitting_trend(self):
        """Analyze overfitting trends"""
        train_acc = self.training_data['accuracy']
        val_acc = self.training_data['val_accuracy']
        overfitting_gap = train_acc - val_acc
        
        # Check if overfitting is increasing
        if overfitting_gap.iloc[-1] > overfitting_gap.iloc[-5]:
            return "Overfitting is increasing - implement early stopping"
        elif overfitting_gap.iloc[-1] > 0.05:
            return "High overfitting detected - add regularization"
        else:
            return "Overfitting is under control"
    
    def _suggest_optimal_epochs(self):
        """Suggest optimal number of training epochs"""
        val_acc = self.training_data['val_accuracy']
        convergence_epoch = self._find_convergence_epoch()
        
        # Add buffer for stability
        optimal_epochs = min(convergence_epoch + 3, len(val_acc))
        
        return {
            'convergence_epoch': convergence_epoch,
            'suggested_epochs': optimal_epochs,
            'early_stopping_patience': 5
        }
    
    def _assess_regularization_needs(self):
        """Assess if regularization is needed"""
        overfitting_gap = self.training_data['accuracy'] - self.training_data['val_accuracy']
        
        if overfitting_gap.iloc[-1] > 0.05:
            return {
                'needed': True,
                'type': 'high',
                'recommendations': ['Dropout (0.3-0.5)', 'Weight decay (1e-4)', 'Data augmentation']
            }
        elif overfitting_gap.iloc[-1] > 0.02:
            return {
                'needed': True,
                'type': 'moderate',
                'recommendations': ['Dropout (0.2-0.3)', 'Weight decay (1e-5)']
            }
        else:
            return {
                'needed': False,
                'type': 'low',
                'recommendations': ['Current regularization adequate']
            }
    
    def optimize_hyperparameters(self, n_trials=100):
        """Optimize hyperparameters using Optuna"""
        if self.training_data is None:
            return "No training data loaded"
        
        def objective(trial):
            # Define hyperparameter search space
            lr = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
            batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
            dropout = trial.suggest_float('dropout', 0.1, 0.7)
            weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
            
            # Simulate training with these parameters
            # In a real implementation, this would train the actual model
            simulated_score = self._simulate_training(lr, batch_size, dropout, weight_decay)
            
            return simulated_score
        
        # Create study
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        self.best_params = study.best_params
        self.optimization_history = study.trials
        
        return {
            'best_params': study.best_params,
            'best_score': study.best_value,
            'n_trials': len(study.trials),
            'optimization_history': study.trials
        }
    
    def _simulate_training(self, lr, batch_size, dropout, weight_decay):
        """Simulate training performance based on hyperparameters"""
        # This is a simplified simulation
        # In reality, you would train the actual model
        
        # Base score from current training data
        base_score = self.training_data['val_accuracy'].iloc[-1]
        
        # Adjust based on hyperparameters
        lr_factor = 1.0 if 1e-4 <= lr <= 1e-3 else 0.8
        batch_factor = 1.0 if batch_size in [32, 64] else 0.9
        dropout_factor = 1.0 if 0.2 <= dropout <= 0.5 else 0.85
        weight_decay_factor = 1.0 if 1e-5 <= weight_decay <= 1e-4 else 0.9
        
        # Calculate adjusted score
        adjusted_score = base_score * lr_factor * batch_factor * dropout_factor * weight_decay_factor
        
        # Add some randomness to simulate real training
        noise = np.random.normal(0, 0.01)
        final_score = adjusted_score + noise
        
        return min(max(final_score, 0.0), 1.0)
    
    def generate_optimization_report(self):
        """Generate comprehensive optimization report"""
        if self.training_data is None:
            return "No training data loaded"
        
        patterns = self.analyze_training_patterns()
        
        report = f"""
🦋 BUTTERFLY MODEL OPTIMIZATION REPORT
{'='*60}

📊 TRAINING PATTERN ANALYSIS:
• Convergence Epoch: {patterns['convergence_epoch']}
• Learning Rate Status: {patterns['learning_rate_analysis']}
• Overfitting Trend: {patterns['overfitting_trend']}
• Optimal Epochs: {patterns['optimal_epochs']['suggested_epochs']}
• Early Stopping Patience: {patterns['optimal_epochs']['early_stopping_patience']}

🔧 REGULARIZATION ASSESSMENT:
• Regularization Needed: {'Yes' if patterns['regularization_needs']['needed'] else 'No'}
• Type: {patterns['regularization_needs']['type'].title()}
• Recommendations: {', '.join(patterns['regularization_needs']['recommendations'])}

🎯 OPTIMIZATION RECOMMENDATIONS:

1. EPOCH MANAGEMENT:
   • Implement early stopping with patience = {patterns['optimal_epochs']['early_stopping_patience']}
   • Target {patterns['optimal_epochs']['suggested_epochs']} epochs for optimal performance
   • Monitor validation metrics closely after epoch {patterns['convergence_epoch']}

2. LEARNING RATE:
   • Current status: {patterns['learning_rate_analysis']}
   • Consider learning rate scheduling if not already implemented
   • Reduce learning rate by factor of 0.1 when validation loss plateaus

3. REGULARIZATION:
   • {'Implement' if patterns['regularization_needs']['needed'] else 'Maintain'} regularization
   • Focus on: {', '.join(patterns['regularization_needs']['recommendations'])}
   • Monitor overfitting gap - keep below 0.05

4. DATA AUGMENTATION:
   • {'Implement' if patterns['overfitting_trend'].startswith('High') else 'Consider'} data augmentation
   • Techniques: rotation, scaling, brightness adjustment, noise injection
   • Target 2-3x data expansion for better generalization

5. MODEL ARCHITECTURE:
   • {'Consider' if patterns['regularization_needs']['type'] == 'high' else 'Maintain'} current architecture
   • {'Add' if patterns['regularization_needs']['needed'] else 'Keep'} dropout layers
   • {'Implement' if patterns['overfitting_trend'].startswith('High') else 'Monitor'} batch normalization

🌟 NEXT STEPS:
1. Run hyperparameter optimization (recommended: 100+ trials)
2. Implement early stopping with suggested patience
3. Add recommended regularization techniques
4. Test with data augmentation
5. Monitor validation metrics closely
6. Consider ensemble methods for final deployment

💡 PRO TIPS:
• Use learning rate finder to determine optimal learning rate range
• Implement model checkpointing to save best models
• Use cross-validation for robust hyperparameter selection
• Consider transfer learning from pre-trained models
• Monitor training curves in real-time for early intervention
"""
        
        return report
    
    def suggest_learning_rate_schedule(self):
        """Suggest learning rate scheduling strategy"""
        if self.training_data is None:
            return "No training data loaded"
        
        convergence_epoch = self._find_convergence_epoch()
        total_epochs = len(self.training_data)
        
        schedule = {
            'initial_lr': 1e-3,
            'schedule_type': 'step_decay',
            'decay_steps': [convergence_epoch // 2, convergence_epoch],
            'decay_factors': [0.5, 0.1],
            'min_lr': 1e-6,
            'warmup_epochs': min(5, total_epochs // 10)
        }
        
        return schedule
    
    def export_optimization_config(self, filename='optimization_config.json'):
        """Export optimization configuration to JSON"""
        if self.best_params is None:
            return "No optimization results to export"
        
        config = {
            'best_hyperparameters': self.best_params,
            'training_patterns': self.analyze_training_patterns(),
            'learning_rate_schedule': self.suggest_learning_rate_schedule(),
            'recommendations': {
                'early_stopping_patience': 5,
                'regularization': self.analyze_training_patterns()['regularization_needs']['recommendations'],
                'data_augmentation': True,
                'optimal_epochs': self.analyze_training_patterns()['optimal_epochs']['suggested_epochs']
            }
        }
        
        try:
            import json
            with open(filename, 'w') as f:
                json.dump(config, f, indent=2)
            print(f"✅ Optimization config exported to {filename}")
            return True
        except Exception as e:
            print(f"❌ Error exporting config: {e}")
            return False

# Example usage
if __name__ == "__main__":
    optimizer = ButterflyModelOptimizer("training.csv.csv")
    
    if optimizer.load_training_data():
        # Generate optimization report
        print(optimizer.generate_optimization_report())
        
        # Run hyperparameter optimization
        print("\n🔍 Running hyperparameter optimization...")
        results = optimizer.optimize_hyperparameters(n_trials=50)
        print(f"Best parameters: {results['best_params']}")
        print(f"Best score: {results['best_score']:.4f}")
        
        # Export configuration
        optimizer.export_optimization_config()
