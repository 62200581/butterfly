# 🦋 Butterfly Classification Ecosystem

A comprehensive full-stack web application for butterfly and moth image classification, featuring machine learning analytics, interactive dashboards, and a complete database management system.

![Butterfly Ecosystem](https://img.shields.io/badge/Butterfly-Classification-Ecosystem?style=for-the-badge&color=6a4c93)
![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)
![Flask](https://img.shields.io/badge/Flask-2.0+-green?style=for-the-badge&logo=flask)
![SQLite](https://img.shields.io/badge/SQLite-3.0+-yellow?style=for-the-badge&logo=sqlite)

## 🌟 **Features**

### **🎯 Core Functionality**
- **Image Classification**: Upload butterfly/moth images for instant species identification
- **Interactive Dashboard**: Real-time training progress visualization with Plotly charts
- **Database Management**: Complete CRUD operations for species and training data
- **Geographic Distribution**: Interactive world map showing species habitats

### **📊 Analytics & Insights**
- **Training Curves**: Visualize accuracy, F1-score, and loss progression
- **Performance Metrics**: Real-time model performance analysis
- **Overfitting Detection**: Advanced analytics for model optimization
- **Database Health**: Comprehensive system monitoring and statistics

### **🎨 User Experience**
- **Modern UI/UX**: Beautiful gradient designs with smooth animations
- **Responsive Design**: Works perfectly on all devices
- **Interactive Elements**: Hover effects, particle backgrounds, and smooth transitions
- **Professional Styling**: Bootstrap-based components with custom CSS enhancements

## 🚀 **Quick Start**

### **Prerequisites**
- Python 3.8 or higher
- pip package manager

### **Installation**

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/butterfly-classification-ecosystem.git
   cd butterfly-classification-ecosystem
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python app.py
   ```

4. **Access the application**
   - 🌐 **Main App**: http://127.0.0.1:5000
   - 📊 **Dashboard**: http://127.0.0.1:5000/dashboard
   - 🔍 **Classify**: http://127.0.0.1:5000/classify
   - 🗄️ **Database**: http://127.0.0.1:5000/database

## 🏗️ **Architecture**

### **Backend (Python/Flask)**
- **Flask Web Framework**: RESTful API endpoints and HTML serving
- **SQLite Database**: Lightweight, file-based database system
- **Data Analysis**: Pandas-based data processing and analytics
- **Model Optimization**: Simulated ML training and hyperparameter tuning

### **Frontend (HTML/CSS/JavaScript)**
- **Bootstrap 5**: Responsive UI components and grid system
- **Plotly.js**: Interactive data visualization and charts
- **Custom CSS**: Advanced animations, gradients, and modern styling
- **Font Awesome**: Scalable vector icons throughout the interface

### **Database Schema**
- **Species Table**: Butterfly/moth information and characteristics
- **Training Metrics**: ML model training progress and performance
- **Classification History**: User upload and prediction records
- **Model Versions**: Version control for ML model iterations

## 📁 **Project Structure**

```
butterfly-classification-ecosystem/
├── app.py                      # Main Flask application
├── database.py                 # Database initialization and schema
├── database_manager.py         # High-level database operations
├── data_analyzer.py           # Data analysis and insights
├── model_optimizer.py         # ML model optimization simulation
├── requirements.txt            # Python dependencies
├── templates/                  # HTML templates
│   ├── index.html             # Homepage with feature showcase
│   ├── dashboard.html         # Analytics dashboard
│   ├── classify.html          # Image classification interface
│   └── database_admin.html    # Database management interface
├── static/                     # Static assets (CSS, JS, images)
├── butterfly_ecosystem.db     # SQLite database (auto-generated)
└── README.md                  # This file
```

## 🔧 **Configuration**

### **Environment Variables**
The application uses default configurations, but you can customize:

```bash
# Database configuration
export DATABASE_URL="sqlite:///butterfly_ecosystem.db"

# Flask configuration
export FLASK_ENV="development"
export FLASK_DEBUG="True"
```

### **Database Initialization**
The database is automatically initialized on first run with:
- 15 butterfly species with detailed information
- Sample training metrics for dashboard visualization
- Model version tracking system

## 📊 **API Endpoints**

### **Core Endpoints**
- `GET /` - Homepage
- `GET /dashboard` - Analytics dashboard
- `GET /classify` - Image classification interface
- `GET /database` - Database management interface

### **API Endpoints**
- `GET /api/database-status` - Database health and statistics
- `GET /api/training-curves` - Training progress data
- `GET /api/insights-report` - ML model insights
- `POST /api/upload-image` - Image classification endpoint
- `POST /api/training-metrics/add` - Add new training data

## 🎨 **Customization**

### **Adding New Species**
```python
# Use the database manager to add new species
from database_manager import DatabaseManager

db_manager = DatabaseManager()
db_manager.manage_species('add', {
    'common_name': 'New Butterfly',
    'scientific_name': 'Lepidoptera novus',
    'family': 'Nymphalidae',
    'habitat': 'Tropical forests',
    'conservation_status': 'Least Concern'
})
```

### **Custom Styling**
The application uses CSS custom properties for easy theming:

```css
:root {
    --primary-color: #6a4c93;
    --secondary-color: #f8f9fa;
    --accent-color: #ff6b6b;
    --success-color: #28a745;
    --warning-color: #ffc107;
    --danger-color: #dc3545;
}
```

## 🚀 **Deployment**

### **Local Development**
```bash
python app.py
```

### **Production Deployment**
```bash
# Using Gunicorn
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app

# Using Waitress (Windows)
pip install waitress
waitress-serve --host=0.0.0.0 --port=5000 app:app
```

### **Docker Support**
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["python", "app.py"]
```

## 🤝 **Contributing**

We welcome contributions! Please feel free to submit issues and pull requests.

### **Development Setup**
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 **Acknowledgments**

- **Flask Community** for the excellent web framework
- **Bootstrap Team** for the responsive UI components
- **Plotly.js** for interactive data visualization
- **Font Awesome** for the beautiful icons

## 📞 **Support**

If you have any questions or need help:
- 📧 Create an issue on GitHub
- 🐛 Report bugs with detailed descriptions
- 💡 Suggest new features
- 📚 Check the documentation

---

**Made with ❤️ for the Butterfly Classification Community**

![GitHub stars](https://img.shields.io/github/stars/yourusername/butterfly-classification-ecosystem?style=social)
![GitHub forks](https://img.shields.io/github/forks/yourusername/butterfly-classification-ecosystem?style=social)
![GitHub issues](https://img.shields.io/github/issues/yourusername/butterfly-classification-ecosystem)
![GitHub pull requests](https://img.shields.io/github/issues-pr/yourusername/butterfly-classification-ecosystem)
