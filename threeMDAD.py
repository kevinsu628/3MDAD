# !/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2020 XXX
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import glob
import csv
import random

from torch.utils.data import Dataset
import torch
import numpy as np
from PIL import Image


class DriverSampler:
    def __init__(self, class_image_map, class_per_batch, batch_size):
        """
        A batch sampler of 3MDAD dataset.
        This dataset sampler limit the number of "mode" randomly sampled to class_per_batch
        e.g. if the mode is driver, batch size is 32, class_per_batch is 4:
        The sampler will randomly choose 4 drivers, sample 32/4 images for each driver
        e.g. if the mode is action, batch size is 32, class_per_batch is 4:
        The sampler will randomly choose 4 actions, sample 32/4 images for each action

        driver_image_map: a dictionary where the key is driver ID, value is the list
        of the images performed by this driver
        class_per_batch: int. Number of unique drivers/actions per batch
        batch_size: batch size
        """
        self.class_image_map = class_image_map
        # choose samples from a random two classes out of three for each batch
        self.class_per_batch = class_per_batch
        self.batch_size = batch_size
        self.images_per_class = int(self.batch_size / self.class_per_batch)
        self.n_batches = sum([len(d) for d in self.class_image_map.values()]) // batch_size

    def get_sample_array(self):
        batches = []
        for _ in range(self.n_batches):
            batch = []
            # for each batch, random choose class_per_batch drivers out of say all drivers
            classes_choice = random.sample(list(self.class_image_map.keys()), self.class_per_batch)
            for each_class in classes_choice:
                # gives me a images for this class
                sample_choice = random.sample(self.class_image_map[each_class], self.images_per_class)
                batch += sample_choice
            batches.append(batch)
        return batches

    def __iter__(self):
        return iter(self.get_sample_array())

    def __len__(self):
        return self.n_batches


# take a single image file, extract the ground truth  
def rgbBboxParser(data_path, frame_id):
    with open(data_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for i, row in enumerate(csv_reader):
            if int(i) == int(frame_id):
                hand_l = [row[1], row[2], row[3], row[4]]
                hand_r = [row[5], row[6], row[7], row[8]]
                head = [row[9], row[10], row[11], row[12]]
                bbox = {
                    "hand_l": hand_l,
                    "hand_r": hand_r,
                    "head": head
                }
                return bbox
    raise Exception("Frame not found")


def get_view_id(view, view_choice):
    return str(view_choice.index(view) + 1)


class ThreeMDADDataset(Dataset):
    """3MDAD dataset.
    collect data from the dataset folder by subject ids
    sub_id_lst: a list of subject ids
    time: string. Choose from day, night
    modality: a list of modalities to include in the dataset. e.g. [RGB1, Depth1]
    bbox_gt: Boolean. Include bbox in the ground truth
    view: a list of views from side, front
    dataset_folder: parent dataset folder
    video_fps: Int. if given, will sample 3D video tensors based on this fps
    video_sample_per_action: how many sample videos are extracted for a video
    exhaustive_video_sampling: sample all continuous video clips based on video_fps and number of frames in video
    """

    def __init__(self, sub_id_lst, time, view, modality, bbox_gt, dataset_folder, transform=None,
                 video_fps=None, exhaustive_video_sampling=False):
        self.image_lst = []
        self.video_lst = []
        self.video_frame_counter = {}
        self.view_choices = ["side", "front"]
        self.modality_choices = ["RGB", "Depth", "IR"]
        self.time_choices = ["day", "night"]
        self.transform = transform
        self.sub_id_lst = sub_id_lst
        self.driver_image_map = {}  # key: driver ID, value: image idx with this driver
        self.action_image_map = {i+1: [] for i in range(16)}  # key: action ID, value: image idx with this action

        self.video_fps = video_fps
        self.exhaustive_video_sampling = exhaustive_video_sampling

        assert time in self.time_choices
        assert len(view) > 0
        for v in view:
            assert v in self.view_choices
        assert len(modality) > 0
        for m in modality:
            assert m in self.modality_choices
        self.time = time
        self.view = view
        self.modality = modality
        self.bbox_gt = bbox_gt
        self.dataset_folder = dataset_folder
        self.ext_map = {"RGB": ".jpg", "Depth": ".tiff", "IR": ".tiff"}
        self.pre_map = {"RGB": "RGB", "Depth": "D", "IR": "IR"}

        self.target_folder = []
        if self.time == "day":
            self.target_folder = ["side_view_day_data", "front_view_day_data"]
        else:
            self.target_folder = ["side_view_night_data", "front_view_night_data"]
        self.target_folder = [x for v in self.view for x in self.target_folder if v in x]

        # we initialize image_lst by the following format:
        # e.g. SUB1ACT1F1
        tmp_view_id = get_view_id(self.view[0], self.view_choices)
        tmp_folder = os.path.join(os.path.join(self.dataset_folder, self.target_folder[0]),
                                  self.modality[0] + tmp_view_id)
        assert os.path.exists(tmp_folder), tmp_folder + " does not exist"
        counter = 0
        for each_sub in self.sub_id_lst:
            sub_id = each_sub
            self.driver_image_map[sub_id] = []
            each_sub = os.path.join(tmp_folder, "S" + str(each_sub))
            for each_act in glob.glob(os.path.join(each_sub, "AC*")):
                act_id = int(each_act[each_act.index("AC")+2:])
                frame_counter = 0
                for each_img in glob.glob(os.path.join(each_act, "*.*")):
                    data_name = os.path.basename(each_img).split('_')[1]
                    frame_counter += 1
                    self.video_frame_counter.update({data_name[:data_name.index("F")]: frame_counter})
                    self.image_lst.append(data_name.split(".")[0])
                    self.driver_image_map[sub_id].append(counter)
                    self.action_image_map[act_id].append(counter)
                    counter += 1
        if self.video_fps:
            self.video_lst = list(self.video_frame_counter.keys())
            if self.exhaustive_video_sampling:
                new_video_lst = []
                for each_video in self.video_lst:
                    num_clips = self.video_frame_counter[each_video] // self.video_fps
                    new_video_lst += [each_video+"_"+str(i) for i in range(num_clips)]
                self.video_lst = new_video_lst
            print("{} video samples in total. ".format(len(self.video_lst)))
        else:
            print("{} images in total. ".format(len(self.image_lst)))

    def __len__(self):
        if self.video_fps:
            return len(self.video_lst)
        else:
            return len(self.image_lst)

    def __getitem__(self, idx):
        if self.video_fps:
            chosen = self.video_lst[idx]
        else:
            chosen = self.image_lst[idx]
        gt = self.gtParser(chosen)
        sub_id = gt["subject"]
        act_id = gt["action"]

        sample = {"act": int(act_id) - 1,
                  "sub": int(sub_id) - 1}

        for view_folder in self.target_folder:
            view = view_folder.split("_")[0]
            view_folder = os.path.join(self.dataset_folder, view_folder)
            sample[view] = {}
            for mod in self.modality:
                prefix = self.pre_map[mod]
                if mod == "Depth" and self.time == "night":
                    prefix = self.pre_map[mod] + "N"  # the naming is insane in this dataset
                view_id = get_view_id(view, self.view_choices)
                if self.video_fps:
                    data_path = os.path.join(view_folder, mod + view_id + "/S{}/AC{}"
                                             .format(str(sub_id), str(act_id)))
                else:
                    data_path = os.path.join(view_folder, mod + view_id + "/S{}/AC{}/{}"
                                             .format(str(sub_id), str(act_id),
                                                     prefix + view_id + "_" + self.image_lst[idx] + self.ext_map[mod]))
                sample[view][mod] = self.readData(mod, data_path, chosen)

        return sample

    def getFrames(self, datadir, frame_start, seq_len):
        frames = []
        dummy_file = os.path.basename(glob.glob(os.path.join(datadir, "*"))[0])
        [f_name, ext] = dummy_file.split(".")
        for i in range(frame_start, frame_start+seq_len):
            f_name = f_name[:f_name.index("F")+1] + str(i+1) + "." + ext
            frames.append(os.path.join(datadir, f_name))
        return frames

    def readData(self, mod, data_path, chosen):
        if self.video_fps:
            if self.exhaustive_video_sampling:
                frame_start = int(chosen.split("_")[-1]) * self.video_fps
            else:
                if self.video_frame_counter[chosen] - self.video_fps < 0:
                    print("{} has a total of {} frames, less than {} fps defined.".format(
                        chosen, self.video_frame_counter[chosen], self.video_fps))
                    frame_start = 0
                else:
                    frame_start = random.randint(0, self.video_frame_counter[chosen]-self.video_fps)
            frames_path = self.getFrames(data_path, frame_start, self.video_fps)
            frames_tensor = [self.readImageData(mod, fp) for fp in frames_path]
            frames_tensor = torch.stack(frames_tensor, dim=0)
            #print(frames_tensor.shape)
            return frames_tensor
        else:
            return self.readImageData(mod, data_path)

    def readImageData(self, mod, data_path):
        data = None
        if (not os.path.exists(data_path)) and self.video_fps:
            print("Padding video...")
            data = Image.new("RGB", size=(640, 480))
            if isinstance(self.transform, dict) and "RGB" in self.transform:
                data = self.transform["Depth"](data)
                data = data.squeeze_(0)  # remove fake batch dimension
            else:
                data = torch.from_numpy(np.array(data).astype("uint8"))

        # IR Depth RGB all convert to (0-255) uint8 images
        elif mod.startswith('RGB'):
            data = Image.open(data_path)
            if isinstance(self.transform, dict) and "RGB" in self.transform:
                data = self.transform["RGB"](data)
                data = data.squeeze_(0)  # remove fake batch dimension
            else:
                data = torch.tensor(np.array(data))

        elif mod.startswith('IR'):
            ir_img = Image.open(data_path)
            data_1d = self.min_max_norm(np.array(ir_img), s=255)
            data = np.stack([data_1d, data_1d, data_1d], axis=2)
            if isinstance(self.transform, dict) and "IR" in self.transform:
                data = self.transform["IR"](Image.fromarray(data.astype(np.uint8)))
                data = data.squeeze_(0)  # remove fake batch dimension
            else:
                data = torch.from_numpy(np.array(data).astype("uint8"))

        elif mod.startswith('Depth'):
            depth_img = Image.open(data_path)
            data_1d = self.min_max_norm(np.array(depth_img), s=255)
            data = np.stack([data_1d, data_1d, data_1d], axis=2)
            if isinstance(self.transform, dict) and "Depth" in self.transform:
                data = self.transform["Depth"](Image.fromarray(data.astype(np.uint8)))
                data = data.squeeze_(0)  # remove fake batch dimension
            else:
                data = torch.from_numpy(np.array(data).astype("uint8"))

        else:
            raise Exception("Modality " + mod + " not from allowed list")

        return data

    # e.g.
    # data_str: e.g. 'SUB1ACT9F85'
    # csv_folder: dataset/3MDAD/csv
    def gtParser(self, data_str, bbox_gt=False, csv_folder=None):
        if self.exhaustive_video_sampling:
            data_str = data_str.split("_")[0]
        sub_idx = data_str.index("SUB")
        act_idx = data_str.index("ACT")
        sub_id = data_str[sub_idx + 3:act_idx]

        if not self.video_fps:
            frame_idx = data_str.index("F")
            act_id = data_str[act_idx + 3:frame_idx]
            frame_id = data_str[frame_idx + 1:]
        else:
            act_id = data_str[act_idx + 3:]

        gt = {
            "subject": sub_id,
            "action": act_id,
        }

        # if also include bbox ground truth
        if bbox_gt and not self.video_fps:
            csv_file = "RGB1S{}AC{}.csv".format(sub_id, act_id)
            bbox_location = os.path.join(csv_folder, csv_file)
            bbox_gt = rgbBboxParser(bbox_location, frame_id=frame_id)
            gt["bbox"] = bbox_gt
        return gt

    def get_driver_image_map(self):
        return self.driver_image_map

    def get_action_image_map(self):
        return self.action_image_map

    def min_max_norm(self, mtx, s=1):
        mtx = s * ((mtx - mtx.min()) / (mtx.max() - mtx.min() + 0.0001))
        return mtx


if __name__ == "__main__":
    # test video
    threeMDAD = ThreeMDADDataset([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], time="day", view=["side"], modality=["RGB", "Depth"],
                                 bbox_gt=False, dataset_folder="/home/lang/Data/project/3MDAD/", transform=None,
                                 video_fps=16, exhaustive_video_sampling=True)
    print(threeMDAD.video_lst)
    iterator = torch.utils.data.DataLoader(dataset=threeMDAD, shuffle=False)

    final = Image.new('RGB', (640*5, 480))
    counter = 0
    for batch in iterator:
        data = batch["side"]["Depth"][0]
        for i, each_img in enumerate(np.array(data).astype(np.uint8)):
            print(each_img.shape)
            final.paste(Image.fromarray(each_img), (i*640, 0))
        final.save("test{}.jpg".format(counter))
        counter += 1
        if counter == 3:
            break

    """
    # test images using driver sampler
    # this example shows how to get images from only 2 drivers out of 16 images in a batch
    threeMDAD = ThreeMDADDataset([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], time="day", view=["side"], modality=["RGB", "Depth"],
                                 bbox_gt=False, dataset_folder="/home/lang/Data/project/3MDAD/", transform=None)
    class_image_map = threeMDAD.get_action_image_map()  # threeMDAD.get_driver_image_map()
    iterator = torch.utils.data.DataLoader(dataset=threeMDAD,
                                           batch_sampler=DriverSampler(class_image_map, 4, batch_size=16),
                                           shuffle=False)
    for batch in iterator:
        # print(batch)
        for i, sample in enumerate(list(range(len(batch["side"]["RGB"])))):
            data = batch["side"]["RGB"][i]
            img1 = Image.fromarray(np.array(data).astype(np.uint8))

            data = batch["side"]["Depth"][i]
            img2 = Image.fromarray(np.array(data).astype(np.uint8))

            final = Image.new('RGB', (img1.width + img2.width, img1.height))
            final.paste(img1, (0, 0))
            final.paste(img2, (img2.width, 0))

            final.save("test{}.jpg".format(i))
        break

    # test images using normal sampler
    threeMDAD = ThreeMDADDataset([1], time="day", view=["side"], modality=["RGB", "Depth"],
                                bbox_gt=False, dataset_folder="/home/lang/Data/project/3MDAD/", transform=None)
    iterator = torch.utils.data.DataLoader(dataset=threeMDAD, shuffle=True)
    for batch in iterator:
        data = batch["side"]["RGB"][0]
        img1 = Image.fromarray(np.array(data).astype(np.uint8))

        data = batch["side"]["Depth"][0]
        img2 = Image.fromarray(np.array(data).astype(np.uint8))
        # np.savetxt("test.txt", np.array(data).astype(np.uint8)[:, :, 0])

        final = Image.new('RGB', (img1.width + img2.width, img1.height))
        final.paste(img1, (0, 0))
        final.paste(img2, (img2.width, 0))

        final.show()
        break"""