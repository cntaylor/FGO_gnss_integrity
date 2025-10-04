import sqlite3

def setup_database(db_name="simulation_data.db"):
    """
    Connects to or creates the SQLite database and sets up the five tables
    based on the hierarchical data structure.
    """
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # Enable foreign key constraint enforcement
    cursor.execute("PRAGMA foreign_keys = ON;")

    # 1. Snapshots Table (Parent: Dataset)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS Snapshots (
        Snapshot_ID INTEGER PRIMARY KEY,
        Dataset_ID TEXT, -- Used for grouping sets of data, could be a run ID or experiment name
        Time REAL NOT NULL,
        True_Location_X REAL,
        True_Location_Y REAL,
        True_Location_Z REAL
    );
    """)

    # 2. Satellites Table (Child of Snapshots) - Stores the N satellite locations
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS Satellites (
        Satellite_ID INTEGER PRIMARY KEY,
        Snapshot_ID INTEGER,
        Satellite_Index INTEGER NOT NULL, -- 0-based index of the satellite
        Loc_X REAL,
        Loc_Y REAL,
        Loc_Z REAL,
        FOREIGN KEY (Snapshot_ID) REFERENCES Snapshots(Snapshot_ID) ON DELETE CASCADE
    );
    """)

    # 3. MC_Iterations Table (Child of Snapshots)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS MC_Iterations (
        MC_ID INTEGER PRIMARY KEY,
        Snapshot_ID INTEGER,
        MC_Iteration_Num INTEGER NOT NULL, -- 1, 2, 3... within this snapshot
        Real_or_Simulated INTEGER, -- Use 1 for True, 0 for False (BOOLEAN)
        FOREIGN KEY (Snapshot_ID) REFERENCES Snapshots(Snapshot_ID) ON DELETE CASCADE
    );
    """)

    # 4. Pseudoranges Table (Child of MC_Iterations) - Stores the N pseudoranges
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS Pseudoranges (
        PR_ID INTEGER PRIMARY KEY,
        MC_ID INTEGER,
        Satellite_Index INTEGER NOT NULL, -- Links to the satellite index used in the Satellites table
        Value REAL,
        FOREIGN KEY (MC_ID) REFERENCES MC_Iterations(MC_ID) ON DELETE CASCADE
    );
    """)

    # 5. Results Table (Child of MC_Iterations)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS Results (
        Result_ID INTEGER PRIMARY KEY,
        MC_ID INTEGER,
        Technique TEXT NOT NULL,
        Parameters_Used TEXT, -- Store as a JSON string or simple text
        Fault_Declared INTEGER, -- 1 for True, 0 for False (BOOLEAN)
        Protection_Level REAL,
        Est_Loc_X REAL,
        Est_Loc_Y REAL,
        Est_Loc_Z REAL,
        FOREIGN KEY (MC_ID) REFERENCES MC_Iterations(MC_ID) ON DELETE CASCADE
    );
    """)

    conn.commit()
    conn.close()
    print("Database and tables set up successfully.")

# Execute the setup function
setup_database()