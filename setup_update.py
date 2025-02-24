import sqlite3
from os import mkdir
from os.path import exists

# If folders don't exist on first run, make them
if not exists('classifications'): mkdir('classifications')
if not exists('inputs'): mkdir('inputs')
if not exists('objects'): mkdir('objects')
if not exists('dataset'): mkdir('dataset')

def initialize_database():
    """Create or update the database schema to store mask data properly."""
    conn = sqlite3.connect('dataset.db')
    c = conn.cursor()

    c.execute('''CREATE TABLE IF NOT EXISTS images(
                 object_id INTEGER PRIMARY KEY AUTOINCREMENT,
                 image_index INTEGER,
                 object_name TEXT,
                 x_coord INTEGER,
                 y_coord INTEGER,
                 width INTEGER,
                 height INTEGER,
                 mask BLOB,  -- Compressed mask of the visible object (binary format)
                 pixel_mask TEXT -- JSON array of pixel coordinates (to track segmentation data)
                 )''')

    conn.commit()
    conn.close()


initialize_database()
