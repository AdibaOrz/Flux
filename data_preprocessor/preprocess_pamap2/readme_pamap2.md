## PAMAP2 ##
- Official website: [link](https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring)
- Cite the dataset: [1]

### General Description ###
The PAMAP2 Physical Activity Monitoring dataset contains data of 18 different physical activities, 
performed by 9 subjects wearing 3 inertial measurement units and a heart rate monitor.


### Sensors ###
- 3 Colibri wireless inertial measurement units (IMU):
  - sampling frequency: 100Hz
  - position of the sensors:
       - 1 IMU over the wrist on the dominant arm 
       - 1 IMU on the chest 
       - 1 IMU on the dominant side's ankle 
- HR-monitor:
  - sampling frequency: ~9Hz


### Usage in our project ###
- We will use the IMU data only.
- IMU (acc, gyro) x 3 body locations = 6 modalities, if acc and gyro are treated separately.
- IMU x 3 body locations = 3 modalities, if acc and gyro are combined.
- Data from one participant is excluded due to the presence of only a single activity data [1]

### How to pre-process the dataset ###
1. Download the PAMAP2 dataset from the [official website](https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring).
   - The size of the dataset (compressed version) is `657MB`.
2. Unzip the dataset.
3. Run the `01_preprocess_pamap2.py` script to generate processed files. By default, it will use the standard scaling. This will generate all modality-aligned data and then split the dataset into train and test sets.

---
### References ###
[1] Attila Reiss and Didier Stricker. 2012. Introducing a new benchmarked dataset for activity monitoring. 
In 2012 16th international symposium on wearable computers. IEEE, 108–109.
