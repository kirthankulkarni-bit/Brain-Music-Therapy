import collections
import numpy as np
import time

# set up a rolling window that drops old values as new ones come in
window_size = 5
data_window = collections.deque(maxlen=window_size)

# fake stream of data points to test how it handles jumps
incoming_numbers = [0.4, 0.5, 1.8, 0.4, 1.9, 1.7, 1.6, 1.5, 0.3, 1.4]

print("--- starting data smoothing simulation ---")

for second, new_number in enumerate(incoming_numbers):
    # push the latest raw number into the rolling window
    data_window.append(new_number)
    
    # average the window contents to dampen sudden random spikes
    current_average = np.mean(data_window)
    
    print(f"Time: {second+1}s")
    print(f"  -> added number:   {new_number:.2f}")
    print(f"  -> current window: {list(data_window)}")
    print(f"  -> smooth average: {current_average:.2f}")
    
    # base decisions on the rolling trend instead of individual volatile points
    if current_average > 1.0:
        print("   DECISION: trigger high-value action")
    else:
        print("   DECISION: trigger low-value action")
    print("-" * 40)
    
    time.sleep(0.5)