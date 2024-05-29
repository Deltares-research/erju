# Import the necessary libraries
import os
import pandas as pd
import matplotlib.pyplot as plt

# Define the path to the files
path = r'D:\csv_files_das\Culemborg_09112020'

# Get the list of files in the directory
file_names = [f for f in os.listdir(path) if f.endswith('.csv')]

# Get the first file name
file_name = file_names[0]
file_path = os.path.join(path, file_name)

# Open the file
data = pd.read_csv(file_path)

# Print the column names to understand the structure of the DataFrame
print("Column names in the CSV file:", data.columns)

# Assuming 'Unnamed: 0' contains the indices measured at 1000 Hz
if 'Unnamed: 0' in data.columns:
    data['Time'] = data['Unnamed: 0'] / 1000  # Convert to seconds
    data['Time'] = pd.to_datetime(data['Time'], unit='s').dt.strftime('%M:%S.%f').str[:-3]  # Convert to minutes and seconds format and truncate microseconds to milliseconds

# Print the first few rows to check the transformation
print(data.head())

# Plot the data with Time as the x-axis and Channel45 as the y-axis
ax = data.plot(x='Time', y='Channel45')

# Set the x-axis limit to 30 minutes
ax.set_xlim(0, 30 * 60 * 1000)  # 30 minutes in milliseconds

# Set major ticks to be every 5 minutes
ax.xaxis.set_major_locator(plt.MultipleLocator(5 * 60 * 1000))  # 5 minutes in milliseconds

# Set the labels and title
plt.xlabel('Time (MM:SS.SSS)')
plt.ylabel('Signal Intensity')
plt.title('Signal Intensity vs Time')
plt.grid(True)  # Show gridlines
plt.show()
