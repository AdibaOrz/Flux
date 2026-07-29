## Sleep-EDF-20 ##
- Official website: [link](https://www.physionet.org/content/sleep-edfx/1.0.0/)
- Cite the dataset: [1, 2]

### General Description ###
The sleep-edf database contains 197 whole-night PolySomnoGraphic sleep recordings, 
containing EEG, EOG, chin EMG, and event markers. Some records also contain respiration and body temperature. 
Corresponding hypnograms (sleep patterns) were manually scored by well-trained technicians according to 
the Rechtschaffen and Kales manual, and are also available. The data comes from two studies.

- The `labels` correspond to `five` types of sleep patterns (hypnograms).

### Sensors ###
- Electroencephalography (EEG) (two types), 
- Electrooculography (EOG), 
- chin EMG, 
- Respiration (Resp)

### Usage in our project ###
- Similar to previous works [3, 4], we use data from the `Sleep Cassette study`, which investigated the effects of age on sleep of healthy individuals.
- In total `5` modalities (`EEG Fpz-Cz, EEG Pz-Oz, EOG horizontal, chin EMG, Resp`) are used in our project.


### How to preprocess the dataset ###
1. The starting code is dependent on the full `Sleep-EDFX` dataset. 
   - Follow the instructions in `preprocess_sleepedfx/readme_sleepedfx.md` to download, and create merged (`sleepedfx/processed/merged.csv`) dataset.
2. Run the `01_preprocess_sleepedf20.py` to create train and test splits with the user ids of 20 participants (based on previous works). 

---
### References ###
[1] Ary L Goldberger, Luis AN Amaral, Leon Glass, Jeffrey M Hausdorff, Plamen Ch Ivanov, Roger G Mark, 
Joseph E Mietus, George B Moody, Chung-Kang Peng, and H Eugene Stanley. 2000. PhysioBank, PhysioToolkit, 
and PhysioNet: components of a new research resource for complex physiologic signals. circulation 101, 23 (2000), 
e215–e220.

[2] Bob Kemp, Aeilko H Zwinderman, Bert Tuk, Hilbert AC Kamphuisen, and Josefien JL Oberye. 2000.
Analysis of a sleep-dependent neuronal feedback loop: the slow-wave microcontinuity of the EEG. 
IEEE Transactions on Biomedical Engineering 47, 9 (2000), 1185–1194.

[3] Huy Phan, Fernando Andreotti, Navin Cooray, Oliver Y Chén, and Maarten De Vos. 2018. 
Joint classification and prediction CNN frame- work for automatic sleep stage classification. 
IEEE Transactions on Biomedical Engineering 66, 5 (2018), 1285–1296.

[4] Orestis Tsinalis, Paul M Matthews, Yike Guo, and Stefanos Zafeiriou. 2016. 
Automatic sleep stage scoring with single-channel EEG using convolutional neural networks. 
arXiv preprint arXiv:1610.01683 (2016).

