# This script is designed to find trains in the data and extract them to a separate file that is saved as binary.
# The data extraction is centered around a particular channel or set of channels

# Import the necessary libraries
from erju.process_FO_base import BaseFOData
from utils.file_utils import get_files_in_dir

def user_defined_params():
    """
    Enter the user-defined parameters here
    """
    dir_path = 'C:\Projects\erju\data\silixa'
    first_channel = 1
    last_channel = 1
    file_type = 'silixa'

    return dir_path, first_channel, last_channel, file_type


def main():
    # Get the user-defined parameters
    dir_path, first_channel, last_channel, file_type = user_defined_params()

    # Get the list of files in the directory
    file_list = get_files_in_dir(folder_path=dir_path, file_format='.tdms')

    # Create a FO class that will be used to extract the data
    fo_data = BaseFOData.create_instance(dir_path=dir_path,
                                         first_channel=first_channel,
                                         last_channel=last_channel,
                                         reader=file_type)

    # Extract the data
    fo_data.extract_data()



    # Print a message to the user
    print('The data has been successfully extracted and saved to a binary file.')


# Run the script
if __name__ == '__main__':
    main()