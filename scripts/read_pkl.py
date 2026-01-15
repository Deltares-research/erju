import pickle
import matplotlib.pyplot as plt

# Path to the pickle you created
pickle_path = r"P:\11210064-erju\holten\res-20240827_20240828-SPRA-ch_1194-dir_1\processed_data_event_20240828_062158.mat.pickle"

with open(pickle_path, "rb") as f:
    data = pickle.load(f)

# Extract time and trace_fo
time = data["time"]  # list of datetime objects
trace_fo = data["trace_fo"]

# In your pickle, trace_fo is a plain list of floats.
# (Other traces like trace_x are nested one level deeper.)
# Just in case, unwrap if nested:
if len(trace_fo) == 1 and isinstance(trace_fo[0], (list, tuple)):
    trace_fo = trace_fo[0]

plt.figure()
plt.plot(time, trace_fo)
plt.xlabel("Time")
plt.ylabel("FO raw trace")
plt.title("Fibre Optic trace vs Time")
plt.gcf().autofmt_xdate()  # nicer date formatting
plt.tight_layout()
plt.show()
