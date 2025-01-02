import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm
import zipfile
import os

def create_gtfs_from_trips(trips_df, travel_time_threshold='travel_time_25'):
    """
    Convert trips dataframe to GTFS format using hours from data
    
    Args:
        trips_df (pd.DataFrame): DataFrame containing trip data
        travel_time_threshold (str): Column name for travel time threshold to use
    """
    gtfs = {}
    
    print("Creating GTFS files...")
    
    print("Creating stops.txt...")
    stops = create_stops(trips_df)
    gtfs['stops'] = stops
    
    print("Creating stop_times.txt...")
    stop_times = create_stop_times_multi(trips_df, travel_time_threshold)
    gtfs['stop_times'] = stop_times
    
    print("Creating trips.txt...")
    trips = create_trips(stop_times)
    gtfs['trips'] = trips
    
    # ... rest of the function remains the same ...
    
    return gtfs

def create_stop_times_multi(trips_df, travel_time_threshold='travel_time_25'):
    """
    Create stop_times.txt using departure hours and days from the data
    
    Args:
        trips_df (pd.DataFrame): DataFrame containing trip data
        travel_time_threshold (str): Column name for travel time threshold to use
    """
    # ... rest of the function remains the same until travel time calculation ...
    
            # Calculate arrival time using the travel time threshold
            base_time = datetime.strptime(dep_time, "%H:%M:%S")
            travel_time_delta = timedelta(seconds=int(row[travel_time_threshold]))
            arr_time = (base_time + travel_time_delta).strftime("%H:%M:%S")
            
            # ... rest of the function remains the same ...

def save_gtfs_files(gtfs, output_path, travel_time_threshold='travel_time_25'):
    """
    Save all GTFS files into a zip archive
    
    Args:
        gtfs (dict): Dictionary containing GTFS DataFrames
        output_path (str): Path to save the output files
        travel_time_threshold (str): Travel time threshold used (for filename)
    """
    temp_dir = os.path.join(output_path, 'temp_gtfs')
    os.makedirs(temp_dir, exist_ok=True)
    
    # Modified filename to include the threshold
    threshold_value = travel_time_threshold.split('_')[-1]  # Extract number from threshold name
    zip_path = os.path.join(output_path, f'gtfs_padam_territory_01_DRT_only_scenario_{threshold_value}.zip')
    
    try:
        print("Creating GTFS text files...")
        for filename, df in tqdm(gtfs.items(), desc="Saving temporary files"):
            file_path = os.path.join(temp_dir, f"{filename}.txt")
            df.to_csv(file_path, index=False)
        
        print("Creating zip archive...")
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for filename in tqdm(os.listdir(temp_dir), desc="Adding files to zip"):
                file_path = os.path.join(temp_dir, filename)
                zipf.write(file_path, filename)
        
        print(f"GTFS feed successfully saved to: {zip_path}")
        
    finally:
        print("Cleaning up temporary files...")
        for filename in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, filename))
        os.rmdir(temp_dir)



import pandas as pd
import os
import logging
from tqdm import tqdm
from zipfile import ZipFile
from io import StringIO

def run(PT_zip_path, DRT_zip_path, output_path):
    """
    Run the GTFS processing pipeline for zip files.
    
    Args:
        PT_zip_path: Path to the public transport GTFS zip file
        DRT_zip_path: Path to the DRT GTFS zip file
        output_path: Path where the output zip file should be saved
    """
    print("Processing Public Transport GTFS...")
    PT_GTFS = readGTFS_zip(PT_zip_path)
    print("\nProcessing DRT GTFS...")
    drt_gtfs = readGTFS_zip(DRT_zip_path)
    print("\nMerging GTFS datasets...")
    gtfs = joinGTFS(PT_GTFS, drt_gtfs)
    exportGTFS(output_path, gtfs)

def readGTFS_zip(zip_path):
    """
    Read GTFS files from a zip file into pandas dataframes.
    
    Args:
        zip_path: Path to the GTFS zip file
    
    Returns:
        Dictionary containing dataframes for each GTFS file
    """
    gtfs = {}
    required_files = [
        "agency.txt",
        "calendar_dates.txt",
        "calendar.txt",
        "routes.txt",
        "stop_times.txt",
        "stops.txt",
        "trips.txt"
    ]
    
    with ZipFile(zip_path, 'r') as zip_ref:
        # Get list of files in zip
        zip_files = zip_ref.namelist()
        
        # Create progress bar for file reading
        for filename in tqdm(required_files, desc=f"Reading {os.path.basename(zip_path)}", unit="file"):
            if filename in zip_files:
                # Read the file content from zip
                with zip_ref.open(filename) as file:
                    # Convert bytes to string and create DataFrame
                    content = StringIO(file.read().decode('utf-8'))
                    gtfs[filename.replace('.txt', '')] = pd.read_csv(content, sep=',', header=0)
                    tqdm.write(f"Successfully read {filename}")
            else:
                tqdm.write(f"Warning: Missing GTFS file: {filename} in {zip_path}")
                gtfs[filename.replace('.txt', '')] = pd.DataFrame()
            
    return gtfs

def joinGTFS(gtfs, drt_gtfs):
    """
    Merge two GTFS datasets.
    
    Args:
        gtfs: Dictionary of dataframes for the first GTFS dataset
        drt_gtfs: Dictionary of dataframes for the second GTFS dataset
    
    Returns:
        Dictionary containing merged dataframes
    """
    gtfsJOINED = {}
    files_to_merge = [
        "agency",
        "calendar_dates",
        "calendar",
        "routes",
        "stop_times",
        "stops",
        "trips"
    ]
    
    for f in tqdm(files_to_merge, desc="Merging datasets", unit="file"):
        gtfsJOINED[f] = pd.concat([gtfs[f], drt_gtfs[f]], ignore_index=True, sort=False)
        tqdm.write(f"Merged {f} datasets")
        
    return gtfsJOINED

def exportGTFS(output_path, gtfs):
    """
    Export the GTFS data to a zip file containing txt files.
    
    Args:
        output_path: Path where the output zip file should be saved
        gtfs: Dictionary containing the GTFS dataframes to export
    """
    print("\nPreparing to export GTFS files...")
    
    # Create a temporary directory for the txt files
    temp_dir = os.path.join(output_path, 'temp_gtfs')
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    
    filenames = ['stop_times', 'stops', 'trips', 'calendar', 'calendar_dates', 'routes', 'agency']
    
    try:
        # First, create all txt files
        print("Creating txt files...")
        for f in tqdm(filenames, desc="Creating txt files", unit="file"):
            output_file = os.path.join(temp_dir, f + '.txt')
            gtfs[f].to_csv(output_file, sep=',', index=False)
            tqdm.write(f"Created {f}.txt")
        
        # Then create the zip file
        zip_path = os.path.join(output_path, 'gtfs_PT_DRT_scenario_95.zip')
        print("\nCreating zip archive...")
        with ZipFile(zip_path, 'w') as zipf:
            for f in tqdm(filenames, desc="Adding to zip", unit="file"):
                txt_file = os.path.join(temp_dir, f + '.txt')
                zipf.write(txt_file, f + '.txt')
                tqdm.write(f"Added {f}.txt to zip")
        
        print(f"\nGTFS zip file successfully created at: {zip_path}")
    
    finally:
        # Clean up temporary files
        print("\nCleaning up temporary files...")
        for f in filenames:
            temp_file = os.path.join(temp_dir, f + '.txt')
            if os.path.exists(temp_file):
                os.remove(temp_file)
        if os.path.exists(temp_dir):
            os.rmdir(temp_dir)
        print("Cleanup completed")