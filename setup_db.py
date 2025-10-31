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

        # --- 1. SNAPSHOT TABLE ---
        # Stores true location information for the main object at each time step.
        # Primary Key is the snapshot.
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS Snapshots (
                Snapshot_ID         INTEGER PRIMARY KEY AUTOINCREMENT,
                Dataset             TEXT    NOT NULL,
                Time                REAL    NOT NULL,
                True_Loc_X          REAL    NOT NULL,
                True_Loc_Y          REAL    NOT NULL,
                True_Loc_Z          REAL    NOT NULL
            );
        """)
        print("Table 'Snapshots' created.")

        # --- 2. SATELLITE LOCATION TABLE ---
        # Stores the location of each satellite for a given Snapshot.
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS Satellite_Locations (
                Snapshot_ID     INTEGER NOT NULL,
                Satellite_num   INTEGER NOT NULL,
                Loc_X           REAL    NOT NULL,
                Loc_Y           REAL    NOT NULL,
                Loc_Z           REAL    NOT NULL,
                PRIMARY KEY (Snapshot_ID, Satellite_num),
                FOREIGN KEY (Snapshot_ID) 
                    REFERENCES Snapshots (Snapshot_ID)
            );
        """)
        print("Table 'Satellite_Locations' created.")
        
        # --- 3. MONTE CARLO RUN TABLE ---
        # Keeps track of all the "batches" created.  Each entry defines a "set" of samples
        # (all generated the same way.)  Records the parameters used to generate measurements 
        # (i.e., noise model).  
        # We'll use a single unique ID column. The first row (ID=0) will be 'real data'.
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS MC_Runs (
                MC_Run_ID     INTEGER PRIMARY KEY AUTOINCREMENT,
                Description     TEXT NOT NULL,
                Parameters_JSON TEXT -- JSON string of the parameters
            );
        """)
        print("Table 'MC_Runs' created.")
        
        # Add a placeholder for "real data" (no modification)
        real_data_params = json.dumps({"simulated": False})
        cursor.execute("""
            INSERT INTO MC_Runs (Description, Parameters_JSON)
            VALUES (?, ?)
        """, ('Real Data (No Monte Carlo Simulation)', real_data_params))
        
        # --- 4. MONTE CARLO SAMPLES TABLE (The link table) ---
        # Links a specific timestep/snapshot to the MC run used to generate measurements.
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS MC_Samples (
                MC_Sample_ID    INTEGER PRIMARY KEY AUTOINCREMENT,
                MC_Run_ID       INTEGER NOT NULL,
                Snapshot_ID     INTEGER NOT NULL,

                -- FK to ensure the time step exists
                FOREIGN KEY (Snapshot_ID)
                    REFERENCES Snapshots (Snapshot_ID),
                
                -- FK to ensure the parameters exist
                FOREIGN KEY (MC_Run_ID)
                    REFERENCES MC_Runs (MC_Run_ID)
            );
        """)
        print("Table 'MC_Samples' created.")

        # --- 5. MEASUREMENTS TABLE ---
        # Stores the generated pseudorange measurements for each Monte Carlo run.
        # Composite key ensures a unique entry for each satellite within a run.
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS Measurements (
                MC_Sample_ID    INTEGER NOT NULL,
                Satellite_num   INTEGER NOT NULL,
                Pseudorange     REAL    NOT NULL,
                Is_Outlier      INTEGER NULL,

                PRIMARY KEY (MC_Sample_ID, Satellite_num),
                
                -- FK to ensure the MC Run exists
                FOREIGN KEY (MC_Sample_ID)
                    REFERENCES MC_Samples (MC_Sample_ID)

            );
        """)
        print("Table 'Measurements' created.")

        # Enforce that the MC_sample_ID has the Satellite number being put into Measurements
        cursor.execute("""
            -- This syntax is specific to SQLite
            CREATE TRIGGER enforce_satellite_measurement_integrity
            BEFORE INSERT ON Measurements
            FOR EACH ROW
            BEGIN
                -- Look up the Snapshot_ID from the MC_Samples table
                SELECT CASE
                    WHEN NOT EXISTS (
                        SELECT 1 
                        FROM Satellite_Locations AS SL
                        JOIN MC_Samples AS MS ON MS.Snapshot_ID = SL.Snapshot_ID
                        WHERE 
                            MS.MC_Sample_ID = NEW.MC_Sample_ID AND 
                            SL.Satellite_num = NEW.Satellite_num
                    )
                    -- If the joint condition is NOT met, RAISE an abort error
                    THEN RAISE (ABORT, 'Integrity violation: Satellite does not exist for the linked snapshot.')
                END;
            END;
        """)

        store_method_results = True
        if store_method_results:
            # --- 6. METHODS TABLE ---
            # Stores the different estimation methods used.
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS Estimation_Methods (
                    Method_ID       INTEGER PRIMARY KEY AUTOINCREMENT,
                    Method_Name     TEXT    NOT NULL,
                    Parameters_JSON TEXT    -- JSON string of the parameters
                );
            """)
            print("Table 'Estimation_Methods' created.")

            # --- 7. ESTIMATION RESULTS TABLE ---
            # Stores the results of applying estimation methods to MC samples.

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS Estimation_Results (
                    -- Primary Key Columns
                    MC_Sample_ID  INTEGER NOT NULL,
                    Method_ID     INTEGER NOT NULL,

                    -- Required Data (3-element numpy array stored as a BLOB)
                    Error    BLOB    NOT NULL, -- X,y,z in NED about truth error = estimated - truth

                    -- Optional Data (BLOBs and TEXT)
                    Sat_Outliers  BLOB    NULL,          -- For numpy array of booleans, length of satellites for MC_sample
                    Covariance_Blob BLOB    NULL,          -- For 3x3 numpy array, NED covariance in meters
                    ARAIM_Results TEXT    NULL,          -- For JSON string

                    -- Define the Composite Primary Key
                    PRIMARY KEY (MC_Sample_ID, Method_ID)
                           
                    -- Foreign Keys
                    FOREIGN KEY (MC_Sample_ID)
                        REFERENCES MC_Samples (MC_Sample_ID),
                    FOREIGN KEY (Method_ID)
                        REFERENCES Estimation_Methods (Method_ID)
                );
            """)
            print("Table 'Estimation_Results' created.")

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