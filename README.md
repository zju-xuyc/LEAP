# The official implementation of LEAP
To establish the required environment
```
pip -r install requirement.txt
```

To test on a new dataset
1. Collect video dataset
2. Cut the video dataset into train data and test data.
3. Generate video mask and save into ./mask
4. Generate Pesudo Label for each dataset
```
python video_tracker.py --video_path YOUR_VIDEO_PATH --video_name YOUR_VIDEO_NAME --mask_path YOUR_MASK_PATH --save_path PATH_TO_SAVE_LABEL --fps 30
```
5. Download Detector weight from 
https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5x.pt 
https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5l.pt
https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5s.pt

Save weight files to ./weights

6. Train cross-frame associate model

    a. Prepare dataset
    ```
    python ./vehicle_reid/generate_reid.py
    ```
    b. Model Training
    ```
    bash ./vehicle_reid/run.sh
    ```
7. Train distilled model

    ```
    python ./distill_model/train_distill.py --weights STUDENT_WEIGHT_PATH --teacher TEACHER_WEIGHT_PATH --teacher-cfg TEACHER_MODEL_CONFIGS --data DATA_CONFIGS
    ```

8. Generate corresponding config file in ./configs


For a simple start
```
python main.py --config CONFIG_PATH --weight DETECTOR_WEIGHT --active_log --visualize
```






