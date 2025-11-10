import sqlite3
import numpy as np
import meas_db_utils as mdu

# This adds data from
# https://github.com/TUC-ProAut/libRSF/tree/master/datasets/
# Need the _GT.txt and _Input.txt files

if __name__ == "__main__":
    
    try:
        conn = sqlite3.connect("meas_data.db")
    except Exception as e:
        print("Database does not exist, creating new one.")
        mdu.create_measurement_database("meas_data.db")
        conn = sqlite3.connect("meas_data.db")

    # Chemnitz dataset
    if not mdu.add_measurement_data(
            conn = conn,
            dataset_name='Chemnitz',
            truth_filename='Chemnitz_GT.txt',
            input_filename='Chemnitz_Input.txt'
        ):
        print("Failed to add Chemnitz data!")
    # UrbanNav Data set.  It has "Deep", "Harsh", "Medium", "Obaida", and "Shinjuku" datasets
    #Deep
    if not mdu.add_measurement_data(
        conn = conn,
        dataset_name='UrbanNav_Deep',
        truth_filename='UrbanNav_HK_Deep_Urban_GT.txt',
        input_filename='UrbanNav_HK_Deep_Urban_Input.txt'
    ):
        print("Failed to add UrbanNav_Deep data!")
    #Harsh
    if not mdu.add_measurement_data(
        conn = conn,
        dataset_name='UrbanNav_Harsh',
        truth_filename='UrbanNav_HK_Harsh_Urban_GT.txt',
        input_filename='UrbanNav_HK_Harsh_Urban_Input.txt'
    ):
        print("Failed to add UrbanNav_Harsh data!")
    #Medium
    if not mdu.add_measurement_data(
        conn = conn,
        dataset_name='UrbanNav_Medium',
        truth_filename='UrbanNav_HK_Medium_Urban_GT.txt',
        input_filename='UrbanNav_HK_Medium_Urban_Input.txt'
    ):
        print("Failed to add UrbanNav_Medium data!")
    #Obaida
    if not mdu.add_measurement_data(
        conn = conn,
        dataset_name='UrbanNav_Obaida',
        truth_filename='UrbanNav_TK_Obaida_GT.txt',
        input_filename='UrbanNav_TK_Obaida_Input.txt'
    ):
        print("Failed to add UrbanNav_Obaida data!")
    #Shinjuku
    if not mdu.add_measurement_data(
        conn = conn,
        dataset_name='UrbanNav_Shinjuku',
        truth_filename='UrbanNav_TK_Shinjuku_GT.txt',
        input_filename='UrbanNav_TK_Shinjuku_Input.txt'
    ):
        print("Failed to add UrbanNav_Shinjuku data!")

    # And the SmartLoc dataset:  Frankfurt_Westend, Frankfurt_Main, Berlin_Gendarmenmarkt, and Berlin_Potsdamer
    #Frankfurt_Westend
    if not mdu.add_measurement_data(
        conn = conn,
        dataset_name='Frankfurt_Westend',
        truth_filename='Frankfurt_Westend_Tower_GT.txt',
        input_filename='Frankfurt_Westend_Tower_Input.txt'
    ):
        print("Failed to add Frankfurt_Westend data!")

    if not mdu.add_measurement_data(
        conn = conn,
        dataset_name='Frankfurt_Westend',
        truth_filename='Frankfurt_Westend_Tower_GT.txt',
        input_filename='Frankfurt_Westend_Tower_Input.txt'
    ):
        print("Failed to add Frankfurt_Westend data!")
    #Frankfurt_Main
    if not mdu.add_measurement_data(
        conn = conn,
        dataset_name='Frankfurt_Main',
        truth_filename='Frankfurt_Main_Tower_GT.txt',
        input_filename='Frankfurt_Main_Tower_Input.txt'
    ):
        print("Failed to add Frankfurt_Main data!")
    #Berlin_Gendarmenmarkt
    if not mdu.add_measurement_data(
        conn = conn,
        dataset_name='Berlin_Gendarmenmarkt',
        truth_filename='Berlin_Gendarmenmarkt_GT.txt',
        input_filename='Berlin_Gendarmenmarkt_Input.txt'
    ):
        print("Failed to add Berlin_Gendarmenmarkt data!")
    #Berlin_Potsdamer
    if not mdu.add_measurement_data(
        conn = conn,
        dataset_name='Berlin_Potsdamer',
        truth_filename='Berlin_Potsdamer_Platz_GT.txt',
        input_filename='Berlin_Potsdamer_Platz_Input.txt'
    ):
        print("Failed to add Berlin_Potsdamer data!")