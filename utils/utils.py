import h5py
import obspy
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from obspy import Trace, Stream, UTCDateTime
from obspy.signal.trigger import recursive_sta_lta, trigger_onset
import os

def get_file_names(self):
    """
    Get a list of unique file names inside the dir_path folder. For this
    we look into one specific file format (.asc) and remove the extension.

    Args:
        None

    Returns:
        file_list (list): List of file names in the folder without extensions
    """

    # Get the list of files in the folder with .asc extension
    ascii_file_names = [f for f in os.listdir(self.dir_path) if f.endswith('.asc')]
    # From the ascii_file_names list, remove the .asc extension
    self.file_names = [f.split('.')[0] for f in ascii_file_names]

    return self.file_names

