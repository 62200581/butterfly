import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import pandas as pd
import os

class ButterflyDatabase:
    """Database management system for the Butterfly Classification Ecosystem"""
    
    def __init__(self, db_path: str = "butterfly_ecosystem.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create species table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS species (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scientific_name TEXT UNIQUE NOT NULL,
                common_name TEXT,
                family TEXT,
                genus TEXT,
                habitat TEXT,
                distribution TEXT,
                conservation_status TEXT,
                description TEXT,
                image_url TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create training_metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS training_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                epoch INTEGER NOT NULL,
                training_loss REAL,
                training_accuracy REAL,
                training_f1_score REAL,
                validation_loss REAL,
                validation_accuracy REAL,
                validation_f1_score REAL,
                learning_rate REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create classification_history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS classification_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_filename TEXT NOT NULL,
                predicted_species_id INTEGER,
                confidence_score REAL,
                processing_time REAL,
                user_agent TEXT,
                ip_address TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (predicted_species_id) REFERENCES species (id)
            )
        ''')
        
        # Create model_versions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_versions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                version TEXT UNIQUE NOT NULL,
                model_path TEXT,
                accuracy REAL,
                f1_score REAL,
                training_date TIMESTAMP,
                hyperparameters TEXT,
                description TEXT,
                is_active BOOLEAN DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create optimization_runs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS optimization_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_name TEXT NOT NULL,
                optimization_type TEXT,
                parameters TEXT,
                results TEXT,
                duration REAL,
                status TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        
        # Populate with initial species data if table is empty
        self._populate_initial_species()
        self._populate_initial_training_data()
    
    def _populate_initial_species(self):
        """Populate the species table with real butterfly and moth species data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check if species table is empty
        cursor.execute("SELECT COUNT(*) FROM species")
        if cursor.fetchone()[0] > 0:
            conn.close()
            return
        
        # Real butterfly and moth species data with accurate information
        species_data = [
            ("Danaus plexippus", "Monarch Butterfly", "Nymphalidae", "Danaus", 
             "Milkweed meadows, gardens, fields, roadsides", "North America, Central America, South America, Australia, New Zealand", "Endangered",
             "The Monarch butterfly is famous for its incredible annual migration, traveling up to 3,000 miles from Canada to Mexico. Adults have bright orange wings with black veins and white spots along the edges. They lay eggs exclusively on milkweed plants, which provide food for caterpillars and make them toxic to predators."),
            
            ("Papilio machaon", "Old World Swallowtail", "Papilionidae", "Papilio",
             "Meadows, grasslands, gardens, coastal areas", "Europe, Asia, North America, North Africa", "Least Concern",
             "A large, striking butterfly with yellow wings marked with black stripes and blue spots. The hindwings have distinctive tail-like extensions. Adults feed on nectar from thistles, knapweeds, and other flowers. Caterpillars feed on plants in the carrot family."),
            
            ("Vanessa atalanta", "Red Admiral", "Nymphalidae", "Vanessa",
             "Woodlands, gardens, parks, urban areas", "Europe, Asia, North America, North Africa", "Least Concern",
             "A medium-sized butterfly with dark brown wings featuring bright red bands and white spots. Adults are strong fliers and can be seen year-round in some areas. They feed on tree sap, rotting fruit, and flower nectar. Caterpillars feed on nettles."),
            
            ("Pieris rapae", "Small White", "Pieridae", "Pieris",
             "Gardens, fields, meadows, urban areas", "Europe, Asia, North America, Australia, New Zealand", "Least Concern",
             "A small white butterfly with subtle gray markings on the wings. Males have one black spot on each forewing, while females have two. Adults feed on nectar from various flowers. Caterpillars feed on brassicas, including cabbage, broccoli, and mustard plants."),
            
            ("Aglais io", "European Peacock", "Nymphalidae", "Aglais",
             "Woodlands, gardens, parks, hedgerows", "Europe, Asia", "Least Concern",
             "A beautiful butterfly with reddish-brown wings featuring distinctive eye-like patterns on all four wings. These eye spots help deter predators. Adults hibernate over winter and emerge in spring. They feed on nectar and tree sap."),
            
            ("Limenitis archippus", "Viceroy", "Nymphalidae", "Limenitis",
             "Wetlands, marshes, riverbanks, meadows", "North America", "Least Concern",
             "The Viceroy is a master of mimicry, closely resembling the Monarch butterfly to avoid predation. It has orange wings with black veins and white spots, but is smaller than the Monarch. Adults feed on nectar, dung, and carrion. Caterpillars feed on willow and poplar leaves."),
            
            ("Colias eurytheme", "Orange Sulphur", "Pieridae", "Colias",
             "Fields, meadows, roadsides, gardens", "North America, Central America", "Least Concern",
             "A bright yellow-orange butterfly with black wing margins. Males are more brightly colored than females. Adults feed on nectar from clover, alfalfa, and other flowers. Caterpillars feed on legumes including clover, alfalfa, and vetch."),
            
            ("Speyeria cybele", "Great Spangled Fritillary", "Nymphalidae", "Speyeria",
             "Woodlands, meadows, prairies, gardens", "North America", "Least Concern",
             "A large, orange-brown butterfly with black spots and silver markings on the underside of wings. Adults feed on nectar from milkweeds, thistles, and other flowers. Caterpillars feed on violets. They are strong fliers and can be found in various habitats."),
            
            ("Battus philenor", "Pipevine Swallowtail", "Papilionidae", "Battus",
             "Woodlands, gardens, parks, riparian areas", "North America, Central America", "Least Concern",
             "A large black butterfly with iridescent blue-green hindwings. Adults feed on nectar from various flowers including milkweeds and thistles. Caterpillars feed on pipevines and Dutchman's pipe, making them toxic to predators."),
            
            ("Epargyreus clarus", "Silver-spotted Skipper", "Hesperiidae", "Epargyreus",
             "Fields, meadows, gardens, roadsides", "North America", "Least Concern",
             "A medium-sized skipper with brown wings featuring a distinctive silver spot on the underside of the hindwing. Adults feed on nectar from various flowers including milkweeds, thistles, and clovers. Caterpillars feed on legumes including locust trees and wisteria."),
            
            ("Polygonia interrogationis", "Question Mark", "Nymphalidae", "Polygonia",
             "Woodlands, gardens, parks, urban areas", "North America", "Least Concern",
             "A medium-sized butterfly with orange-brown wings featuring dark markings and irregular edges. The underside has a silver question mark pattern. Adults feed on tree sap, rotting fruit, and flower nectar. Caterpillars feed on elm, nettle, and hackberry."),
            
            ("Junonia coenia", "Common Buckeye", "Nymphalidae", "Junonia",
             "Fields, meadows, roadsides, gardens", "North America, Central America", "Least Concern",
             "A medium-sized butterfly with brown wings featuring distinctive eye spots and orange bands. Adults feed on nectar from various flowers including asters and milkweeds. Caterpillars feed on plantains, snapdragons, and other plants."),
            
            ("Phoebis sennae", "Cloudless Sulphur", "Pieridae", "Phoebis",
             "Fields, meadows, gardens, coastal areas", "North America, Central America, South America", "Least Concern",
             "A large, bright yellow butterfly with black wing margins. Adults are strong fliers and migrate seasonally. They feed on nectar from various flowers including lantana and hibiscus. Caterpillars feed on cassia and senna plants."),
            
            ("Libytheana carinenta", "American Snout", "Libytheidae", "Libytheana",
             "Woodlands, gardens, parks, urban areas", "North America, Central America", "Least Concern",
             "A unique butterfly with a long, beak-like projection on the head. It has brown wings with orange patches and white spots. Adults feed on nectar and tree sap. Caterpillars feed on hackberry leaves."),
            
            ("Morpho peleides", "Blue Morpho", "Nymphalidae", "Morpho",
             "Tropical rainforests, forest edges", "Central America, South America", "Least Concern",
             "A spectacular butterfly with brilliant blue wings that shimmer in flight. The underside is brown with eye spots for camouflage. Adults feed on rotting fruit and tree sap. Caterpillars feed on various plants including legumes and palms.")
        ]
        
        cursor.executemany('''
            INSERT INTO species (scientific_name, common_name, family, genus, habitat, 
                               distribution, conservation_status, description)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', species_data)
        
        conn.commit()
        conn.close()
        print(f"Successfully populated database with {len(species_data)} real butterfly species")
    
    def _populate_initial_training_data(self):
        """Populate training metrics from the CSV file if it exists"""
        if not os.path.exists('training.csv.csv'):
            return
        
        try:
            df = pd.read_csv('training.csv.csv')
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if training_metrics table is empty
            cursor.execute("SELECT COUNT(*) FROM training_metrics")
            if cursor.fetchone()[0] > 0:
                conn.close()
                return
            
            # Insert training data - map CSV columns to database columns
            for _, row in df.iterrows():
                cursor.execute('''
                    INSERT INTO training_metrics 
                    (epoch, training_loss, training_accuracy, training_f1_score,
                     validation_loss, validation_accuracy, validation_f1_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    row['Epoch'],  # CSV has 'Epoch' (capitalized)
                    row['loss'],  # CSV has 'loss'
                    row['accuracy'],  # CSV has 'accuracy'
                    row['F1_score'],  # CSV has 'F1_score'
                    row['val_loss'],  # CSV has 'val_loss'
                    row['val_accuracy'],  # CSV has 'val_accuracy'
                    row['val_F1_score']  # CSV has 'val_F1_score'
                ))
            
            conn.commit()
            conn.close()
            print(f"Successfully imported {len(df)} training epochs from CSV")
        except Exception as e:
            print(f"Error populating training data: {e}")
    
    def add_species(self, scientific_name: str, common_name: str = None, 
                    family: str = None, genus: str = None, habitat: str = None,
                    distribution: str = None, conservation_status: str = None,
                    description: str = None, image_url: str = None) -> int:
        """Add a new species to the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO species (scientific_name, common_name, family, genus, habitat,
                                   distribution, conservation_status, description, image_url)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (scientific_name, common_name, family, genus, habitat,
                  distribution, conservation_status, description, image_url))
            
            species_id = cursor.lastrowid
            conn.commit()
            return species_id
        except sqlite3.IntegrityError:
            return None
        finally:
            conn.close()
    
    def get_species(self, species_id: int = None, scientific_name: str = None) -> Optional[Dict]:
        """Get species information by ID or scientific name"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if species_id:
            cursor.execute("SELECT * FROM species WHERE id = ?", (species_id,))
        elif scientific_name:
            cursor.execute("SELECT * FROM species WHERE scientific_name = ?", (scientific_name,))
        else:
            cursor.execute("SELECT * FROM species ORDER BY scientific_name")
            rows = cursor.fetchall()
            conn.close()
            
            if not rows:
                return []
            
            columns = [description[0] for description in cursor.description]
            return [dict(zip(columns, row)) for row in rows]
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return None
        
        columns = [description[0] for description in cursor.description]
        return dict(zip(columns, row))
    
    def update_species(self, species_id: int, **kwargs) -> bool:
        """Update species information"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Build update query dynamically
        valid_fields = ['common_name', 'family', 'genus', 'habitat', 'distribution',
                       'conservation_status', 'description', 'image_url']
        
        update_fields = []
        values = []
        
        for field, value in kwargs.items():
            if field in valid_fields and value is not None:
                update_fields.append(f"{field} = ?")
                values.append(value)
        
        if not update_fields:
            conn.close()
            return False
        
        update_fields.append("updated_at = CURRENT_TIMESTAMP")
        values.append(species_id)
        
        query = f"UPDATE species SET {', '.join(update_fields)} WHERE id = ?"
        
        try:
            cursor.execute(query, values)
            conn.commit()
            success = cursor.rowcount > 0
        except Exception:
            success = False
        finally:
            conn.close()
        
        return success
    
    def delete_species(self, species_id: int) -> bool:
        """Delete a species from the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("DELETE FROM species WHERE id = ?", (species_id,))
            conn.commit()
            return cursor.rowcount > 0
        except Exception:
            return False
        finally:
            conn.close()
    
    def add_classification(self, image_filename: str, predicted_species_id: int = None,
                          confidence_score: float = None, processing_time: float = None,
                          user_agent: str = None, ip_address: str = None) -> int:
        """Add a new classification record"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO classification_history 
                (image_filename, predicted_species_id, confidence_score, processing_time,
                 user_agent, ip_address)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (image_filename, predicted_species_id, confidence_score,
                  processing_time, user_agent, ip_address))
            
            classification_id = cursor.lastrowid
            conn.commit()
            return classification_id
        finally:
            conn.close()
    
    def get_classification_history(self, limit: int = 100) -> List[Dict]:
        """Get classification history with species information"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT ch.*, s.scientific_name, s.common_name, s.family
            FROM classification_history ch
            LEFT JOIN species s ON ch.predicted_species_id = s.id
            ORDER BY ch.created_at DESC
            LIMIT ?
        ''', (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            return []
        
        columns = [description[0] for description in cursor.description]
        return [dict(zip(columns, row)) for row in rows]
    
    def get_training_metrics(self, limit: int = None) -> List[Dict]:
        """Get training metrics from the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if limit:
            cursor.execute("SELECT * FROM training_metrics ORDER BY epoch DESC LIMIT ?", (limit,))
        else:
            cursor.execute("SELECT * FROM training_metrics ORDER BY epoch")
        
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            return []
        
        columns = [description[0] for description in cursor.description]
        return [dict(zip(columns, row)) for row in rows]
    
    def add_training_metrics(self, epoch: int, training_loss: float = None,
                            training_accuracy: float = None, training_f1_score: float = None,
                            validation_loss: float = None, validation_accuracy: float = None,
                            validation_f1_score: float = None, learning_rate: float = None) -> int:
        """Add new training metrics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO training_metrics 
                (epoch, training_loss, training_accuracy, training_f1_score,
                 validation_loss, validation_accuracy, validation_f1_score, learning_rate)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (epoch, training_loss, training_accuracy, training_f1_score,
                  validation_loss, validation_accuracy, validation_f1_score, learning_rate))
            
            metrics_id = cursor.lastrowid
            conn.commit()
            return metrics_id
        finally:
            conn.close()
    
    def get_database_stats(self) -> Dict:
        """Get database statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        stats = {}
        
        # Count records in each table
        tables = ['species', 'training_metrics', 'classification_history', 'model_versions', 'optimization_runs']
        for table in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            stats[f"{table}_count"] = cursor.fetchone()[0]
        
        # Get latest training metrics
        cursor.execute("SELECT MAX(epoch) FROM training_metrics")
        latest_epoch = cursor.fetchone()[0]
        stats['latest_training_epoch'] = latest_epoch if latest_epoch else 0
        
        # Get classification accuracy trend
        cursor.execute("""
            SELECT AVG(confidence_score) as avg_confidence, 
                   COUNT(*) as total_classifications
            FROM classification_history 
            WHERE confidence_score IS NOT NULL
        """)
        confidence_data = cursor.fetchone()
        stats['avg_classification_confidence'] = confidence_data[0] if confidence_data[0] else 0
        stats['total_classifications'] = confidence_data[1] if confidence_data[1] else 0
        
        conn.close()
        return stats
    
    def search_species(self, query: str) -> List[Dict]:
        """Search species by name, family, or description"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        search_query = f"%{query}%"
        cursor.execute('''
            SELECT * FROM species 
            WHERE scientific_name LIKE ? OR common_name LIKE ? OR family LIKE ? OR description LIKE ?
            ORDER BY scientific_name
        ''', (search_query, search_query, search_query, search_query))
        
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            return []
        
        columns = [description[0] for description in cursor.description]
        return [dict(zip(columns, row)) for row in rows]
    
    def export_data(self, table_name: str, format: str = 'csv') -> str:
        """Export table data to CSV or JSON"""
        conn = sqlite3.connect(self.db_path)
        
        if format.lower() == 'csv':
            df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
            filename = f"{table_name}_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(filename, index=False)
        elif format.lower() == 'json':
            df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
            filename = f"{table_name}_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            df.to_json(filename, orient='records', indent=2)
        else:
            conn.close()
            raise ValueError("Unsupported format. Use 'csv' or 'json'")
        
        conn.close()
        return filename
    
    def backup_database(self, backup_path: str = None) -> str:
        """Create a backup of the database"""
        if not backup_path:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_path = f"butterfly_ecosystem_backup_{timestamp}.db"
        
        import shutil
        shutil.copy2(self.db_path, backup_path)
        return backup_path
    
    def get_connection(self):
        """Get a database connection for custom queries"""
        return sqlite3.connect(self.db_path)

# Database utility functions
def create_sample_data():
    """Create sample data for testing"""
    db = ButterflyDatabase()
    
    # Add more sample species
    sample_species = [
        ("Battus philenor", "Pipevine Swallowtail", "Papilionidae", "Battus",
         "Woodlands, gardens", "North America", "Least Concern",
         "Black butterfly with iridescent blue-green markings"),
        ("Limenitis arthemis", "Red-spotted Purple", "Nymphalidae", "Limenitis",
         "Forests, woodlands", "North America", "Least Concern",
         "Mimics the poisonous pipevine swallowtail"),
        ("Speyeria cybele", "Great Spangled Fritillary", "Nymphalidae", "Speyeria",
         "Meadows, fields", "North America", "Least Concern",
         "Large orange butterfly with silver spots on underside"),
        ("Colias philodice", "Clouded Sulphur", "Pieridae", "Colias",
         "Fields, meadows, gardens", "North America", "Least Concern",
         "Pale yellow butterfly common in open areas"),
        ("Lycaena phlaeas", "Small Copper", "Lycaenidae", "Lycaena",
         "Meadows, grasslands", "Europe, Asia, North America", "Least Concern",
         "Small orange butterfly with black markings")
    ]
    
    for species in sample_species:
        db.add_species(*species)
    
    print("Sample species data added to database")

if __name__ == "__main__":
    # Initialize database and create sample data
    db = ButterflyDatabase()
    create_sample_data()
    
    # Display database statistics
    stats = db.get_database_stats()
    print("\nDatabase Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Display sample species
    species = db.get_species()
    print(f"\nSpecies in database: {len(species)}")
    for s in species[:3]:  # Show first 3
        print(f"- {s['scientific_name']} ({s['common_name']})")
