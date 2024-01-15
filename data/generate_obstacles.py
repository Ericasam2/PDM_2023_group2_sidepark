import numpy as np
import csv
import matplotlib.pyplot as plt

'''
circle: description -- (center, radius)
rectangle: description -- (vertex, height, width, angle)
'''

""" shape = ['rectangle', 'circle']
description = [ [(50,0),3,5], [(20,5), 5] ]

# Save to CSV file
csv_data = list(zip(shape, description))
csv_file_path = "data/static_obstacles.csv"

with open(csv_file_path, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['shape', 'description'])
    csv_writer.writerows(csv_data)

print(f"Sine wave data saved to {csv_file_path}") """

shape = []
description = []
data_path = 'data/static_obstacles.csv'
with open(data_path, newline='') as f:    
    rows = list(csv.reader(f, delimiter=','))[1:-1]

number = len(rows)
for s,d in rows:
    shape.append(s)
    description.append(eval(d))
print("number: {}".format(number))
print('shape: {}'.format(shape))
print('description: {}, type: {}'.format(description, type(description[0])))

