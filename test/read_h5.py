import h5py
import pandas as pd

def display_h5_as_dataframe(h5_file_name: str):
    with h5py.File(h5_file_name, 'r') as h5file:
        def load_dataset(name):
            try:
                data = h5file[name][()]
                # Check if data is 2D and convert to DataFrame
                if data.ndim == 2:
                    # Extract column names from attributes if available
                    columns = h5file[name].attrs.get('columns', None)
                    # Create DataFrame with or without column names
                    df = pd.DataFrame(data, columns=columns)
                else:
                    print(f"Dataset {name} is not 2D. Shape: {data.shape}")
                    df = pd.DataFrame(data)  # Attempt to convert any other shape to DataFrame
                return df
            except Exception as e:
                print(f"Could not load dataset {name}: {e}")
                return None

        def print_group(name, group):
            print(f"Group: {name}")
            # Iterate through items in the group
            for key in group.keys():
                item = group[key]
                if isinstance(item, h5py.Group):
                    print_group(f"{name}/{key}", item)
                elif isinstance(item, h5py.Dataset):
                    df = load_dataset(f"{name}/{key}")
                    if df is not None:
                        print(f"Dataset: {name}/{key}")
                        print(df.head())  # Display the first few rows
                        print("\n")

        # Start the inspection from the root group
        print_group("", h5file)

# Example usage:
file_path = 'C:\Projects\erju\erju\database.h5'
display_h5_as_dataframe(file_path)
