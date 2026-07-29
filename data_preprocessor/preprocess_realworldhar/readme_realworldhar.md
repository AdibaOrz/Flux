## RealWorld-HAR ##
- Official website: [link](https://www.uni-mannheim.de/dws/research/projects/activity-recognition/dataset/dataset-realworld/)
- Cite the dataset: [1]


### General Description ###
The data set covers acceleration, GPS, gyroscope, light, magnetic field, and sound level data of 
the activities climbing stairs down and up, jumping, lying, standing, sitting, running/jogging, and walking of 
fifteen subjects (age 31.9±12.4, height 173.1±6.9, weight 74.1±13.8, eight males and seven females). 
For each activity, the authors recorded simultaneously the acceleration of the body positions chest, forearm, head, 
shin, thigh, upper arm, and waist. Each subject performed each activity roughly 10 minutes except for jumping 
due to the physical exertion (~1.7 minutes). Concerning male and female, the amount of data is equally distributed.
Each movement was recorded by a video camera to facilitate the usage.

- 15 Subjects
- 6 Sensors
- Video recordings
- 7 Body positions (Walking, Running, Sitting, Standing, lying, Stairs Up, Stairs Down, Jumping)


### Sensors ###
- Acc, GYRO (IMU sensors) at seven body positions
- Light, Sound, Magnetic field, GPS


### Usage in our project ###
- Each participant was equipped with seven IMU devices positioned on seven different body parts, 
but we `omitted two of them due to incomplete activity coverage`. Thus, the dataset includes 
`ten modalities from five body locations` and two types of IMU sensors.
- We use only IMU sensors from 5 body locations.
- In total 10 modalities if acc and gyro are considered separately.
- In total 5 modalities if acc and gyro are combined as one modality.


### How to pre-process the dataset ###
1. Download the dataset from the [official website](https://www.uni-mannheim.de/dws/research/projects/activity-recognition/dataset/dataset-realworld/). Place the unzipped data into the `datasets/realworldhar/raw_unzipped` directory.
2. Run the `00_unzip_realworldhar.py` to unzip the dataset. The dataset is organized in hierarchical levels, and just unzipping manually will take a lot of time. The result of this script will be located in `datasets/realworldhar/raw` with directories for each user.
3. Run the `01_preprocess_realworldhar.py` to pre-process the dataset. The result of this script will be located in `datasets/realworldhar/processed` with train and test splits.

---
### References ###
[1] Timo Sztyler and Heiner Stuckenschmidt. 2016. On-body localization 1446 of wearable devices: 
An investigation of position-aware activity recognition. In 2016 IEEE International Conference on 
Pervasive Computing 1448 and Communications (PerCom). IEEE, 1–9.
