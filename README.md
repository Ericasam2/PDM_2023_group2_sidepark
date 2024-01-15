# Group 2:
## MPC and RRT* an approach for a single vehicle to parallel parking
### Introduction
This is the final project of the course "Planning and Decision Making", RO47005, ME, TUDelft. \
We implement the RRT* motion planner and the Model Predictive Control (MPC) to on a 4-wheel mobile to perform the side parking (parallel parking) in with obstacles in narrow street. \
The simulation is based on the [KinematicBicycleModel](https://github.com/winstxnhdw/KinematicBicycleModel/tree/main).

### Requirements
python >= 3.10\
casadi==3.6.4\
cvxpy==1.4.1\
do-mpc==4.6.4\
matplotlib==3.8.2\
numpy==1.26.2\
Rtree==1.1.0\
scikit-learn==1.3.2\
scipy==1.11.4


Install (only for conda user)
```bash
conda create -n py310 python=3.10
conda activate py310
pip3 install -r requirements.txt
```
### Simulation


The obstacles the simulation uses are stored in the `PDM_2023_group2_sidepark\KinematicBicycleModel\data\static_obstacles.csv` file.

#### Requirement:
All the requirements can be found in the `requirements.txt` file.


### How to run it:
1. Clone the repository:
```bash
git clone --recursive https://github.com/Ericasam2/PDM_2023_group2_sidepark.git
```
2. Check the simulation result using MPC and RRT*
```bash
cd <path_of_the_package>
python3 ./animate.py
```
3. Check the tracking performance of MPC on sin wave
```bash
cd <path_of_the_package>
python3 ./mpc_animate.py 
```

### Demonstration:
![Demonstration ](result/rrt_animation2.gif)


