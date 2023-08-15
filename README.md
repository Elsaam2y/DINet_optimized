# DINet: Deformation Inpainting Network for Realistic Face Visually Dubbing on High Resolution Video (AAAI2023)
![在这里插入图片描述](https://img-blog.csdnimg.cn/178c6b3ec0074af7a2dcc9ef26450e75.png)
[Paper](https://fuxivirtualhuman.github.io/pdf/AAAI2023_FaceDubbing.pdf) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;     [demo video](https://www.youtube.com/watch?v=UU344T-9h7M&t=6s)  &nbsp;&nbsp;&nbsp;&nbsp; Supplementary materials


# 🤔 How to achive this boost in inference latency?

To achieve this, several changes were implemented:
- Removed DeepSpeech and utilized wav2vec for instant feature extraction, leveraging the speed and power of torch.
- Trained a lightweight model to map the wav2vec features to DeepSpeech, maintaining the existing process.
- Enhanced frames extraction for improved speed.
- These adjustments contribute to a reduction of up to 60% in inference latency compared to the original implementation, all while maintaining quality.

Additionally, Docker has been introduced to facilitate faster, simpler, and more automated facial landmarks extraction.

Tested on:
- Ubuntu (18 and 20)
- Python version >= 3.9

# 📖 Prerequisites
To get started, follow these steps:

- Download the resources (asserts.zip) n [Google drive](https://drive.google.com/file/d/1FHpYJqGrIKsItG313aokXes03qJxFyxg/view?usp=share_link). Unzip the file and place the directory in the current directory (./). This zip file includes the model for mapping wav2vec to deepspeech, beside all other models.

- For running inference or training, you have two options:

## Option 1: Docker (most preferable)

Build the Docker image using the following command.
  
  ```
  docker build --target inference -t dinet . 
```


## Option 2: Conda Environment

Set up a Conda environment by executing the following commands.

```
  #Create the virtual environment 
  conda create -n dinet python=3.9

  #Activate the environment
  conda activate dinet

  #Install teh requirements 
  pip install -r requirements.txt
```


# 🚀 Inference

## Run inference with example videos: 

  ```
docker run --rm --gpus 'device=0' -v $PWD:/app dinet python3 inference.py --mouth_region_size=256 --source_video_path=./asserts/examples/testxxx.mp4 --source_openface_landmark_path=./asserts/examples/testxxx.csv --driving_audio_path=./asserts/examples/driving_audio_xxx.wav --pretrained_clip_DINet_path=./asserts/clip_training_DINet_256mouth.pth  
```

The results are saved in ./asserts/inference_result.

## Run inference with custom videos. 
First, use the following Docker command to extract facial landmarks for your video. Replace `input_video` with the correct name of your video. The output file will be saved in the root directory.

  ```
  docker run --rm -v "$PWD:/mnt" -it algebr/openface -c "cp /mnt/input_video.mp4 /tmp/video.mp4 && build/bin/FeatureExtraction -f /tmp/video.mp4 -2Dfp -out_dir /tmp && cp /tmp/video.csv /mnt/input_video.csv" 
```

Run inference using the following command.

```
docker run --rm --gpus 'device=0' -v $PWD:/app -it dinet python3 inference.py --mouth_region_size=256 --source_video_path=<path_to_your_video>.mp4 --source_openface_landmark_path=<path_to_openface_output>.csv --driving_audio_path=<path_to_your_audio_file>.wav --pretrained_clip_DINet_path=./asserts/clip_training_DINet_256mouth.pth  
```

The results are saved in ./asserts/inference_result.

Or using Conda.

```
  #Activate the environment
  conda activate dinet

  # Run the command
  python3 inference.py --mouth_region_size=256 --source_video_path=<path_to_your_video>.mp4 --source_openface_landmark_path=<path_to_openface_output>.csv --driving_audio_path=<path_to_your_audio_file>.wav --pretrained_clip_DINet_path=./asserts/clip_training_DINet_256mouth.pth  
```

# 🧠 Training
First, you need to build the Docker image with the training dependencies as follows.

  ```
  docker build --target training -t dinet-training . 
```
This command will create the necessary Docker image for training with the specified dependencies.

## Data Processing
We release the code of video processing on [HDTF dataset](https://github.com/MRzzm/HDTF). You can also use this code to process custom videos.

 1. Downloading videos from [HDTF dataset](https://github.com/MRzzm/HDTF). Splitting videos according to xx_annotion_time.txt and **do not** crop&resize videos.
 2. Resampling all split videos into **25fps** and put videos into "./asserts/split_video_25fps". You can see the two example videos in "./asserts/split_video_25fps". We use [software](http://www.pcfreetime.com/formatfactory/cn/index.html) to resample videos. We provide the name list of training videos in  our experiment. (pls see "./asserts/training_video_name.txt")
 3. Using [openface](https://github.com/TadasBaltrusaitis/OpenFace) to detect smooth facial landmarks of all videos. Putting all ".csv" results into "./asserts/split_video_25fps_landmark_openface". You can see the two example csv files in "./asserts/split_video_25fps_landmark_openface".

 4. Extracting frames from all videos and saving frames in "./asserts/split_video_25fps_frame". Run 
```python 
docker run --rm --gpus 'device=0' -it -v $PWD:/app dinet-training python3 data_processing.py --extract_video_frame --source_video_dir <PATH_TO_DATASET>
```
 5. Extracting audios from all videos and saving audios in "./asserts/split_video_25fps_audio". Run 
 ```python 
docker run --rm --gpus 'device=0' -it -v $PWD:/app dinet-training python3 data_processing.py --extract_audio --source_video_dir <PATH_TO_DATASET>
```
 6. Extracting deepspeech features from all audios and saving features in "./asserts/split_video_25fps_deepspeech". Run 
  ```python 
docker run --rm --gpus 'device=0' -it -v $PWD:/app dinet-training python3 data_processing.py --extract_deep_speech
```
 7.  Cropping faces from all videos and saving images in "./asserts/split_video_25fps_crop_face". Run
   ```python 
docker run --rm --gpus 'device=0' -it -v $PWD:/app dinet-training python3 data_processing.py --crop_face
```
 8. Generating training json file "./asserts/training_json.json". Run
   ```python 
docker run --rm --gpus 'device=0' -it -v $PWD:/app dinet-training python3 data_processing.py --generate_training_json
```

### Training models
We split the training process into **frame training stage** and **clip training stage**. In frame training stage, we use coarse-to-fine strategy, **so you can train the model in arbitrary resolution**.

#### Frame training stage.
In frame training stage, we only use perception loss and GAN loss.

 1. Firstly, train the DINet in 104x80 (mouth region is 64x64) resolution. Run 
   ```python 
docker run --rm --gpus 'device=0' -it -v $PWD:/app dinet-training python3 train_DINet_frame.py --augment_num=32 --mouth_region_size=64 --batch_size=24 --result_path=./asserts/training_model_weight/frame_training_64
```
You can stop the training when the loss converges (we stop in about 270 epoch).

 2. Loading the pretrained model (face:104x80 & mouth:64x64) and train the DINet in higher resolution (face:208x160 & mouth:128x128). Run
   ```python 
docker run --rm --gpus 'device=0' -it -v $PWD:/app dinet-training python3 train_DINet_frame.py --augment_num=100 --mouth_region_size=128 --batch_size=80 --coarse2fine --coarse_model_path=./asserts/training_model_weight/frame_training_64/xxxxxx.pth --result_path=./asserts/training_model_weight/frame_training_128
```
You can stop the training when the loss converges (we stop in about 200 epoch).

 3. Loading the pretrained model (face:208x160 & mouth:128x128) and train the DINet in higher resolution (face:416x320 & mouth:256x256). Run
   ```python 
docker run --rm --gpus 'device=0' -it -v $PWD:/app dinet-training python3 train_DINet_frame.py --augment_num=20 --mouth_region_size=256 --batch_size=12 --coarse2fine --coarse_model_path=./asserts/training_model_weight/frame_training_128/xxxxxx.pth --result_path=./asserts/training_model_weight/frame_training_256
```
You can stop the training when the loss converges (we stop in about 200 epoch). Keep in mind that you may need to adjust the batch size to start the training. 

#### Clip training stage.
In clip training stage, we use perception loss, frame/clip GAN loss and sync loss. Loading the pretrained frame model (face:416x320 & mouth:256x256), pretrained syncnet model (mouth:256x256) and train the DINet in clip setting. Run
   ```python 
docker run --rm --gpus 'device=0' -it -v $PWD:/app dinet-training python3 train_DINet_clip.py --augment_num=3 --mouth_region_size=256 --batch_size=3 --pretrained_syncnet_path=./asserts/syncnet_256mouth.pth --pretrained_frame_DINet_path=./asserts/training_model_weight/frame_training_256/xxxxxx.pth --result_path=./asserts/training_model_weight/clip_training_256
```
You can stop the training when the loss converges and select the best model (our best model is at 160 epoch).

## Acknowledge
The AdaAT is borrowed from [AdaAT](https://github.com/MRzzm/AdaAT). The deepspeech feature is borrowed from [AD-NeRF](https://github.com/YudongGuo/AD-NeRF). The basic module is borrowed from [first-order](https://github.com/AliaksandrSiarohin/first-order-model). Thanks for their released code.