# 3MDAD PyTorch Dataset
This is an unofficial implementation of the [3MDAD dataset](https://sites.google.com/site/benkhalifaanouar1/6-datasets).
Reference:
```sh
Imen Jegham, Anouar Ben Khalifa, Ihsen Alouani, Mohamed Ali Mahjoub, A novel public dataset for multimodal multiview and multispectral driver distraction analysis: 3MDAD, Signal Processing: Image Communication, Volume 88, October 2020, 115966, DOI: https://doi.org/10.1016/j.image.2020.115960.
```
The dataset is a multimodal multiview driver distraction detection dataset. that contains the following activities:
```sh
A1: Safe driving, 
A2: Doing hair and makeup, 
A3: Adjusting radio,  
A4: GPS operating, 
A5: Writing message using right hand, 
A6: Writing message using left hand, 
A7: Talking phone using right hand, 
A8: Talking phone using left hand, 
A9: Having picture, 
A10: Talking to passenger, 
A11: Singing or dancing, 
A12: Fatigue and somnolence, 
A13: Drinking using right hand, 
A14: Drinking using left hand, 
A15: Reaching behind, 
A16: Smoking.
```

## Download
Download the dataset [here](https://sites.google.com/site/benkhalifaanouar1/6-datasets)
For each modality and view, download the zip files in a folder in the following format `"{}_view_{}_data".format(view, time)`, where view is either side or front, time is either day or night according to the dataset. Create a zip folder under each category folder that contain the zip files. The dataset should look like the following:
```sh
3MDAD
    front_view_day_data
        zips
            Depth2.rar
            RGB2.rar
    front_view_night_data
    side_view_day_data
    side_view_night_data
    Annotations 3MDAD.rar
```

## UnZip
Run the script to extract 3MDAD dataset (please read the next before doing so):
```sh
python dataset/unzip_3MDAD.py --datadir /path_to_3MDAD
```
We have noticed some corrupted data from this data. The provided unzip_3MDAD.py will fix everything below. However, it's recommended you check out the code before running it to decide which conflict you want to solve. 
```sh
3MDAD/front_view_night_data/Depth2/S6/AC2/DN2_SUB6ACT2F234.tiff
3MDAD/front_view_day_data/Depth2/S2/AC7/Vid_dep_S2_AC7.mj2
```
3MDAD/side_view_day_data/Depth1/S31/AC10 and AC9 should be swapped. 
```sh
mv 3MDAD/side_view_day_data/Depth1/S31/AC10 3MDAD/side_view_day_data/Depth1/S31/AC9_
mv 3MDAD/side_view_day_data/Depth1/S31/AC9 3MDAD/side_view_day_data/Depth1/S31/AC10
mv 3MDAD/side_view_day_data/Depth1/S31/AC9_ 3MDAD/side_view_day_data/Depth1/S31/AC9
```

3MDAD also includes a list of .mat files in the annotations that encode the hand & face locations. To convert all .mat files to .csv files, upload the .mat files to Matlab and run convert_csv_3MDAD.m in matlab. 


## Dataset parameter
```sh
sub_id_lst: a list of subject ids. e.g. [1,2,3,4,5]. 
time: string. Choose from day, night
modality: a list of modalities to include in the dataset. e.g. [RGB, Depth]
bbox_gt: Boolean. Include bbox in the ground truth
view: a list of views from side, front. Can be [side, front] for multiview fusion
dataset_folder: parent dataset folder
video_fps: Int. if given, will sample 3D video tensors based on this fps. Otherwise, sample images.
video_sample_per_action: how many sample videos are extracted for a video
exhaustive_video_sampling: sample all continuous video clips based on video_fps and number of frames in video
```

## Dataloader example 
```sh
# video training 
threeMDAD = ThreeMDADDataset([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], time="day", view=["side"], modality=["RGB", "Depth"],
                                 bbox_gt=False, dataset_folder="/home/lang/Data/project/3MDAD/", transform=None,
                                 video_fps=16, exhaustive_video_sampling=True)
``` 