import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from protocol.datum.file_datum_reader import open_datum_file
from protocol.envelope import DataExtractorOutputEnvelope, ReceiveServerOutputEnvelope, Envelope
from protocol.envelope.indus_das_envelope import IndusDASEnvelope
from protocol.message import RSMessage
from protocol.message.de_message import DEMessage
import pathlib
from protocol.message import FileNotOpen, FilePackageReader
from protocol.envelope import IndusDASEnvelope
from protocol.message.rs_file_reader import open_rs_file


# THIS VERSION OF THE CODE IS SOMEWHAT WORKING>>>

focus_path = r'C:\Projects\erju\data\focus\sensor-1-full-00024170-10181607-10181806 1.bin'

#de_message = DEMessage()
#rs_message = RSMessage()


package_reader = FilePackageReader()
indus_das_envelope = IndusDASEnvelope()

package_read = 0
# read from file package by package
package_reader.open_file(focus_path)

# for package in package_reader.yield_raw_headers():
for offset in package_reader.yield_package_offsets_from_file():
    package_read += 1
    header = indus_das_envelope.decode(package_reader.mm[package_reader.indus_offset:package_reader.indus_offset + IndusDASEnvelope.SUMMARY_DATA_OFFSET])



all_data = []
nn = 0
message_dict = {}

signals_sum = np.zeros((20000,200))

with open_rs_file(focus_path) as file_reader:
    for package_number, message_bytes in file_reader.yield_messages():
        rs_envelope = ReceiveServerOutputEnvelope()
        message_dict[nn] = rs_envelope.decode(message_bytes)

        rs_message = RSMessage()
        full_data = rs_message.get_full_package(message_bytes)

        all_data.append(full_data['payload'])

        signals_sum[:,nn] = full_data['summary']

        nn+=1

all_data = np.array(all_data)

print(all_data.shape)

n = 100e6/header['sampling_freq']*header['spatial_decimation']
m = header['samplingFreq']/10*header['frameDecimation']


print(header)

signal = all_data[0,...].flatten()
fig,ax0 = plt.subplots(figsize=(10,5))
factor = header['conversion_factor']
ax0.plot(signal/factor)
ax0.set_ylabel('Phase (rads)')
ax0.set_xlabel('Samples')
plt.show()
plt.close()

fs_dec = header['sampling_freq']/header['frameDecimation']

print('the fs_dec is:', fs_dec)


tmax = 20000/fs_dec

fig,(ax0,ax1) = plt.subplots(1,2, figsize=(12,8))
ax0.imshow(signals_sum, aspect='auto',cmap='seismic_r')
ax0.set_ylim(0,4000)
ax0.set_title("sensor_1_full_00024170_10181607_10181806_1.bin")

ax1.imshow(np.log(np.abs(signals_sum)), aspect='auto',cmap='seismic_r')
ax1.set_ylim(0,4000)
ax1.set_title("Using log-scale")
plt.show()
plt.close()


fig,axs = plt.subplots(5, 1, figsize=(8,10))

for i in range(5):
    axs[i].plot(signals_sum[:,i]/np.max(np.abs(signals_sum[:,i]),axis=0))

    axs[i].set_xlim(500,4000)
    axs[i].set_ylim(-0.2,0.2)
    axs[i].set_title(f'Normalized amplitudes -- Channel {i+1}')

fig.tight_layout()
plt.show()
plt.close()