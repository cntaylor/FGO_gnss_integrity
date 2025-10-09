import sqlite3
import json

def create_measurement_database(db_name="measurement_data.db"):
    """
    Creates an SQLite database file with the five required tables.
    """
    try:
        # 1. Connect to (or create) the SQLite database file
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()

        print(f"Database '{db_name}' connected successfully.")

        # --- 1. TIMESTEP TABLE ---
        # Stores true location information for the main object at each time step.
        # Primary Key is a composite of Dataset_ID and Snapshot_num.
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS Timesteps (
                Dataset_ID      TEXT    NOT NULL,
                Snapshot_num    INTEGER NOT NULL,
                Time            REAL    NOT NULL,
                True_Loc_X REAL    NOT NULL,
                True_Loc_Y REAL    NOT NULL,
                True_Loc_Z REAL    NOT NULL,
                PRIMARY KEY (Dataset_ID, Snapshot_num)
            );
        """)
        print("Table 'Timesteps' created.")

        # --- 2. SATELLITE LOCATION TABLE ---
        # Stores the location of each satellite for a given timestep.
        # Composite key links to Timesteps, plus Satellite_num for uniqueness.
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS Satellite_Locations (
                Dataset_ID      TEXT    NOT NULL,
                Snapshot_num    INTEGER NOT NULL,
                Satellite_num   INTEGER NOT NULL,
                Loc_X           REAL    NOT NULL,
                Loc_Y           REAL    NOT NULL,
                Loc_Z           REAL    NOT NULL,
                PRIMARY KEY (Dataset_ID, Snapshot_num, Satellite_num),
                FOREIGN KEY (Dataset_ID, Snapshot_num) 
                REFERENCES Timesteps (Dataset_ID, Snapshot_num)
            );
        """)
        print("Table 'Satellite_Locations' created.")
        
        # --- 3. MONTE CARLO PARAMETERS TABLE ---
        # Defines the parameters used to generate measurements (i.e., noise model).
        # We'll use a single unique ID column. The second row (MC_ID=1) will be 'real data'.
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS MC_Parameters (
                MC_Param_ID     INTEGER PRIMARY KEY,
                Description     TEXT NOT NULL,
                Parameters_JSON TEXT -- JSON string of the parameters
            );
        """)
        print("Table 'MC_Parameters' created.")
        
        # Add a placeholder for "real data" (no modification)
        real_data_params = json.dumps({"noise_type": "none", "std_dev": 0.0})
        cursor.execute("""
            INSERT INTO MC_Parameters (MC_Param_ID, Description, Parameters_JSON)
            VALUES (?, ?, ?)
        """, (1, 'Real Data (No Monte Carlo Simulation)', real_data_params))
        
        # --- 4. MONTE CARLO RUNS TABLE (The link table) ---
        # Links a specific timestep/snapshot to the set of parameters used to generate measurements.
        # This table represents one 'Monte Carlo Run'.
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS MC_Runs (
                MC_Run_ID       INTEGER PRIMARY KEY AUTOINCREMENT,
                Dataset_ID      TEXT    NOT NULL,
                Snapshot_num    INTEGER NOT NULL,
                MC_Param_ID     INTEGER NOT NULL,

                -- FK to ensure the time step exists
                FOREIGN KEY (Dataset_ID, Snapshot_num)
                    REFERENCES Timesteps (Dataset_ID, Snapshot_num),
                
                -- FK to ensure the parameters exist
                FOREIGN KEY (MC_Param_ID)
                    REFERENCES MC_Parameters (MC_Param_ID)
            );
        """)
        print("Table 'MC_Runs' created.")

        # --- 5. MEASUREMENTS TABLE ---
        # Stores the generated pseudorange measurements for each Monte Carlo run.
        # Composite key ensures a unique entry for each satellite within a run.
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS Measurements (
                MC_Run_ID       INTEGER NOT NULL,
                Satellite_num   INTEGER NOT NULL,
                Pseudorange     REAL    NOT NULL,
                PRIMARY KEY (MC_Run_ID, Satellite_num),
                
                -- FK to ensure the MC Run exists
                FOREIGN KEY (MC_Run_ID)
                    REFERENCES MC_Runs (MC_Run_ID)

            );
        """)
        print("Table 'Measurements' created.")

        # ----------------------------------------------------------------------
        # --- ENFORCEMENT & QUERYING VIA VIEW ---
        # ----------------------------------------------------------------------
        
        # A VIEW is created to join the three tables required to check integrity
        # and retrieve the full context of a measurement.
        cursor.execute("""
            CREATE VIEW IF NOT EXISTS Full_Measurement_View AS
            SELECT
                M.MC_Run_ID,
                R.Dataset_ID,
                R.Snapshot_num,
                M.Satellite_num,
                SL.Loc_X,
                SL.Loc_Y,
                SL.Loc_Z,
                M.Pseudorange
            FROM Measurements M
            JOIN MC_Runs R ON M.MC_Run_ID = R.MC_Run_ID
            JOIN Timesteps T ON R.Dataset_ID = T.Dataset_ID AND R.Snapshot_num = T.Snapshot_num
            JOIN Satellite_Locations SL ON 
                R.Dataset_ID = SL.Dataset_ID AND 
                R.Snapshot_num = SL.Snapshot_num AND 
                M.Satellite_num = SL.Satellite_num;
        """)
        print("VIEW 'Full_Measurement_View' created for easy querying and validation.")


        # Commit all changes and close the connection
        conn.commit()
        print("Database creation complete and changes saved.")

    except sqlite3.Error as e:
        print(f"An error occurred: {e}")
    finally:
        if conn:
            conn.close()

# --- EXECUTION ---
if __name__ == "__main__":
    create_measurement_database()


