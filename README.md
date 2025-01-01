# CSCI1430 Final Project
Final Project for Fall 2024 CSCI1430 Computer Vision @ Brown University

Seyoung Jang, Camilo Becerra, Javier Nino-Sears

## Overview
Our project focuses on developing a real-time soccer match analysis program using the You Only Look Once (YOLO) AI object detection model. We fine-tune the model to improve its accuracy in tracking players, referees, and the ball. Then, we assign the players to teams based on colors of their uniforms using K-means clustering at pixel level. With this information, we measure each team's ball acquisition percentage in a match. 

A detailed report about the project's methodology and results can be found in Final project report pdf file.

### Dataset
We acquired data to train our YOLO model from the [Football Players Detection dataset](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc/dataset/2) hosted by Roboflow. This dataset consists of 600+ annotated images captured during football games, with bounding boxes labeling individual players on the field, enabling precise localization of players within the frame. 


## Modules Used
The following modules are used in this project:

- YOLO: AI object detection model
- Kmeans: Pixel segmentation and clustering to detect t-shirt color
- Speed and distance calculation per player

## Getting Started
1. Clone repository. 
2. Install following libraries: Python 3.x, ultralytics, OpenCV, NumPy, Matplotlib, Pandas. 
3. Import the video you want to test the model on into the "videos" folder and modify the path to the input dataset in `annotations.py` line 216: `path = videos/<name of imported video>`. 
4. Run file `annotations.py`. 
5. The annotated video is outputted to the annotated_videos folder.

## Known Bugs
For the player speed calculation, we were not able to get an accurate measure of each player's real-time speed in meters per second. The concept behind this was to track players across frames, which required refactoring much of our initial code to track player positions across frames via player ids for each team. Using the movement in position for players between each frames, the frame would display m/s values for each player (updating every 5 frames). Unfortunately, we were not able to get these values entirely accurate, due to time constraints and changing camera angles between shots. Increased utilization of camera geometry would have yielded more accurate results.