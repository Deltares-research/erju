import pickle


def observe_pickle_contents(file_path):
    """
    Opens a pickle file, loads its contents, and prints them.

    Args:
        file_path (str): The path to the pickle file.
    """
    try:
        # Open the pickle file in binary read mode
        with open(file_path, 'rb') as file:
            # Load the pickled object
            data = pickle.load(file)

        # Print the keys of the pickled object
        print("Keys of the pickled object:")
        print(data.keys())

        # Print key by key
        for key in data.keys():
            print(f"\nKey: {key}")
            print(data[key])


    except FileNotFoundError:
        print(f"Error: The file {file_path} does not exist.")
    except pickle.UnpicklingError:
        print(f"Error: The file {file_path} could not be unpickled.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


# Example usage
pickle_file_path = r'C:\Projects\erju\data\database\2_20201111_123200124000.pkl'  # Replace with your pickle file path
observe_pickle_contents(pickle_file_path)
