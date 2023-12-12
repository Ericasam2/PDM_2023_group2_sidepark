# PDM_2023_group2_sidepark
## Simulator
The simulator is based on the [KinematicBicycleModel](https://github.com/winstxnhdw/KinematicBicycleModel/tree/main).

The way_points is in file `data/sine_wave_waypoints.csv`
* the current way_points is a sin wave
* the waypoint can be redefined using `data/generate_waypoints.py`

The obstacle is in file `data/static_obstacles.csv`
* now only the static obstacles are defined
* circle : (center, radius)
* rectangle : (vertex, height, width, angle)

### Requirement: 
Python >= 3.10\
numpy >= 1.22.2\
matplotlib >= 3.5.1\
scipy >= 1.8.0

### Demo:
```bash
git clone --recursive https://github.com/Ericasam2/PDM_2023_group2_sidepark.git
```
```bash
cd KinematicBicycleModel
python animate.py
```

### Generate way_points:
```bash
python data/generate_waypoints.py
```

### Demonstration:
![Demonstration ](KinematicBicycleModel/animation.gif)


