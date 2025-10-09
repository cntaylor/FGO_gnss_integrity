import sqlite3
import numpy as np



def snapshot_data(cursor, Dataset_ID, snapshot_num):
    '''
    Create a tuple for the corresponding Dataset_ID and snapshot_num
    The tuple has:
    1.  The truth location (3-element numpy array)
    2.  nx3 numpy array of satellite locations
    
    Returns nothing if ID's passed in are invalid
    '''

    #First, get the truth location
    cursor.execute("""
        SELECT T.True_Loc_X T.True_Loc_Y T.True_Loc_Z
        FROM Timesteps T
        WHERE Datatset_ID = ? AND Snapshot_num = ?
    """, (Dataset_ID, snapshot_num))
    truth_data = np.array(cursor.fetchone())
    if truth_data is not None:
        # Second, get all the satellite locations 
        # for the given location 
        cursor.execute("""
            SELECT S.Satellite_num S.Loc_X S.Loc_Y S.Loc_Z
            FROM Satellite_Locations S
            WHERE Dataset_ID = ? and Snapshot_num = ?
            ORDER BY Satellite_num ASC
        """, (Dataset_ID, snapshot_num))
        sat_data = cursor.fetchall()
        if sat_data:
            satellite_data = np.zeros(len(sat_data),3)
            for i,row in enumerate(sat_data):
                satellite_data[i] = np.array([row[1],row[2],row[3]])

            return (truth_data,sat_data)
        else:
            print('WARNING: Dataset_ID:',Dataset_ID,'and snapshot:',snapshot_num,'had no measurements')
    else:
        print('WARNING: Dataset_ID:',Dataset_ID,'and snapshot:',snapshot_num,'not valid')
    return None
    