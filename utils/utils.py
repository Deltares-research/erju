import h5py
import obspy
from datetime import datetime

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




#def filterSignal():





def readOptaSenseProps(fpath):

    # Open the file for reading
    fp = h5py.File(fpath,'r')

    # Print the attributes for "Acquisition"
    print(fp['Acquisition'].attrs.keys())

    # Print the values associated with some important attributes
    print('\nThe gauge length is ',fp['Acquisition'].attrs['GaugeLength'],'m.')
    print('The pulse rate is ',fp['Acquisition'].attrs['PulseRate'],'Hz.')
    print('The spatial sampling interval is ',fp['Acquisition'].attrs['SpatialSamplingInterval'],'m.')


    # Create a new variable for the "RawDataTime" h5py dataset so we don't have to type so much
    rawDataTime = fp['Acquisition']['Raw[0]']['RawDataTime']

    # We can load the full contents of "RawDataTime" into a NumPy array
    rawDataTimeArr = rawDataTime[:]

    # Print the length of the array
    print('\nThere are',len(rawDataTimeArr),'values in \"RawDataTime\"')

    # Print the beginning and ending time of the dataset
    print('The first sample is at',datetime.utcfromtimestamp(rawDataTimeArr[0]*1e-6),\
        'and the last sample is at',datetime.utcfromtimestamp(rawDataTimeArr[-1]*1e-6))



