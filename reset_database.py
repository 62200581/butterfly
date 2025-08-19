#!/usr/bin/env python3
"""
Reset and repopulate the butterfly database with real species data
"""

import os
import sqlite3

def reset_database():
    """Reset the database and repopulate with real species data"""
    
    # Remove existing database
    if os.path.exists('butterfly_ecosystem.db'):
        os.remove('butterfly_ecosystem.db')
        print("✅ Removed old database")
    
    # Import and initialize new database
    from database import ButterflyDatabase
    
    # Create new database instance
    db = ButterflyDatabase()
    print("✅ Created new database with real butterfly species")
    
    # Verify species were added
    conn = sqlite3.connect('butterfly_ecosystem.db')
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM species")
    species_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT scientific_name, common_name FROM species LIMIT 5")
    sample_species = cursor.fetchall()
    
    conn.close()
    
    print(f"✅ Database now contains {species_count} real butterfly species")
    print("\n📋 Sample species added:")
    for scientific, common in sample_species:
        print(f"   • {common} ({scientific})")
    
    print("\n🦋 Database reset complete! The classification system now uses real butterfly data.")

if __name__ == "__main__":
    reset_database()

