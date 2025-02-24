import sqlite3

def reset_database():
    """Delete all records from the images table to start fresh."""
    conn = sqlite3.connect('dataset.db')
    c = conn.cursor()

    # Clear existing images
    c.execute("DELETE FROM images")
    conn.commit()
    
    # Reset the auto-increment counter
    c.execute("DELETE FROM sqlite_sequence WHERE name='images'")
    conn.commit()

    conn.close()
    print("✅ Database has been reset.")

# Run the reset function
reset_database()
