# AEM667

### Overview
This is a repository to contain all my projects for AEM667, navigation and target tracking,
at the University of Alabama. 

### Results
The project subfolders all contain readmes with descriptions and select results from each project. More extensive results are in the /plots folder for each project. 

### Project List 
Project 1: Implement a linear kalman filter to estimate the error between two simulated clocks.  

Project 2: Process raw GPS base and rover receiver data into position estimates using double differenced psuedorange multilateration.

Project 3: Implement an error-state EKF to perform loose (position-level) GNSS/INS integration

Project 4: Implement a probability density association filter (PDAF) to track a single target in the presence of clutter and missed detections

Project 5: Implement an RFS-based gaussian-mixture probability hypothesis density (GM-PHD) filter to track multiple targets with clutter and missed detections

### Setup
- Install python
- Run the setup.sh script. This will create a virtual python environment, install the python requirements in that environment, and source the environment for you. 

### Running the projects
Navigate to the project directory you want to run, then run the main P#.py script. 
