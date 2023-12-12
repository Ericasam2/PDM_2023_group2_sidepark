import numpy as np
import csv
import matplotlib.pyplot as plt

# Parameters
amplitude = 10.0
frequency = 0.02  # Adjust the frequency as needed
duration = 1.0   # Adjust the duration of the sine wave
length = 400
sampling_rate = 200

# Initial position values
init_x = 0.0
init_y = 0.0

# Generate sine wave
x = np.linspace(0, length, int(sampling_rate * duration), endpoint=False)
y = amplitude * np.sin(2 * np.pi * frequency * x)

# Save to CSV file
csv_data = list(zip(x, y))
csv_file_path = "data/sine_wave_waypoints.csv"

with open(csv_file_path, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['X-axis', 'Y-axis'])
    csv_writer.writerows(csv_data)

print(f"Sine wave data saved to {csv_file_path}")

# Visualize the sine wave
plt.plot(x, y)
plt.title('Sine Wave')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.grid(True)
plt.show()

