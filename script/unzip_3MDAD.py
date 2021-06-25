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


# This script extract the downloaded 3MDAD dataset

import patoolib
import glob
import os
import argparse


# %% Parse inputs
def parse_args():
    parser = argparse.ArgumentParser(description='3MDAD dataset preparation')
    parser.add_argument('--datadir', type=str, help='dataset directory', default='dataset', required=True)
    return parser.parse_args()


# swap two folder. e.g. AC9 -> AC10 ; AC10 -> AC9
# e.g. act1 = "side_view_day_data/RGB1/S31/AC9"
#      act2 = "side_view_day_data/RGB1/S31/AC10"
#      name1 = ACT9
#      name2 = ACT10
#      ext = "*.tiff"
def swap_folder(act1, act2, name1, name2, ext):
    # swap two mistakenly placed folder
    act1_full = os.path.join(datadir, act1)
    tmp = os.path.join(datadir, act2+"_")
    act2_full = os.path.join(datadir, act2)
    for each_file in glob.glob(os.path.join(act1_full, ext)):
        os.rename(each_file, each_file.replace(name1, name2))
    for each_file in glob.glob(os.path.join(act2_full, ext)):
        os.rename(each_file, each_file.replace(name2, name1))

    os.rename(act1_full, tmp)
    os.rename(act2_full, act1_full)
    os.rename(tmp, act2_full)
    print("Swapped two folders. {} <-> {}".format(act1, act2))


def get_num_files(datadir, ext):
    return len([name for name in os.listdir(datadir) if name.endswith(ext)])


def fix_dataset(datadir):
    """
    front_view_day_data
    ['RGB2', 'Depth2']
    Sub S2 act AC7: 131 != 132
    """
    rm_file = os.path.join(datadir, "front_view_day_data/Depth2/S2/AC7/Vid_dep_S2_AC7.mj2")
    if os.path.exists(rm_file):
        os.remove(rm_file)
        print("Removed " + rm_file)
    """
    front_view_day_data
    ['RGB2', 'Depth2']
    Sub S21 act AC4: 111 != 112
    """
    rm_file = os.path.join(datadir, "front_view_day_data/Depth2/S21/AC4/D2_SUB21ACT4F112.tiff")
    if os.path.exists(rm_file):
        os.remove(rm_file)
        print("Removed " + rm_file)
    """
    side_view_day_data
    ['Depth1', 'RGB1']
    Sub S31 act AC9: 138 != 117 (depth ac9 <-> ac10)
    Sub S31 act AC10: 117 != 138
    """
    side_view_day_depth_ac9 = "side_view_day_data/Depth1/S31/AC9"
    side_view_day_depth_ac10 = "side_view_day_data/Depth1/S31/AC10"
    swap_folder(side_view_day_depth_ac9, side_view_day_depth_ac10,
                "ACT9", "ACT10", "*tiff")
    print("Swaped " + side_view_day_depth_ac9 + " " + side_view_day_depth_ac9)

    """
    front_view_day_data
    ['Depth1', 'RGB1']
    Sub S31 act AC9: (depth ac9 <-> ac10)
    Sub S31 act AC10: (rgb ac9 <-> ac10)
    """
    front_view_day_rgb_ac9 = "front_view_day_data/RGB2/S31/AC9"
    front_view_day_rgb_ac10 = "front_view_day_data/RGB2/S31/AC10"
    swap_folder(front_view_day_rgb_ac9, front_view_day_rgb_ac10,
                "ACT9", "ACT10", "*jpg")
    print("Swaped " + front_view_day_rgb_ac9 + " " + front_view_day_rgb_ac10)

    front_view_day_depth_ac9 = "front_view_day_data/Depth2/S31/AC9"
    front_view_day_depth_ac10 = "front_view_day_data/Depth2/S31/AC10"
    swap_folder(front_view_day_depth_ac9, front_view_day_depth_ac10,
                "ACT9", "ACT10", "*tiff")
    print("Swaped " + front_view_day_depth_ac9 + " " + front_view_day_depth_ac10)


def unzip_dataset(datadir):
    target_folder = ["front_view_day_data", "front_view_night_data", "side_view_day_data", "side_view_night_data"]
    for tf in target_folder:
        tf_full = os.path.join(datadir, tf)
        assert os.path.exists(tf_full), tf + " not exists. Please make sure the folder name is correct"
        zip_folder = os.path.join(tf_full, "zips")
        assert os.path.exists(zip_folder), "zips folder not found in " + tf
        for each_rar in glob.glob(os.path.join(zip_folder, "*.rar")):
            print("Unzipping " + each_rar)
            patoolib.extract_archive(each_rar, outdir=tf_full)
    patoolib.extract_archive(os.path.join(datadir, "Annotations 3MDAD.rar"), outdir=datadir)
    os.rename(os.path.join(datadir, "Annotations 3MDAD"), os.path.join(datadir, "annotations"))


def count_dataset(datadir):
    target_folder = {
        "front_view_day_data": {},
        "front_view_night_data": {},
        "side_view_day_data": {},
        "side_view_night_data": {}
    }

    for tf in target_folder.keys():
        tf_full = os.path.join(datadir, tf)
        modalities_folder = os.listdir(tf_full)
        modalities_folder.remove("zips")
        target_folder[tf] = {}

        for mod in modalities_folder:
            target_folder[tf][mod] = {}
            full_mod_path = os.path.join(tf_full, mod)
            subs = [each_sub for each_sub in os.listdir(full_mod_path)]
            for sub in subs:
                target_folder[tf][mod][sub] = {}
                for act in os.listdir(os.path.join(full_mod_path, sub)):
                    target_folder[tf][mod][sub][act] = \
                        len([name for name in os.listdir(os.path.join(full_mod_path, sub + "/" + act))])

    return target_folder


# check if two modality folders' sub have the files
def mod_sanity_check(stat1, stat2):
    for sub in stat1.keys():
        for act in stat1[sub].keys():
            if stat1[sub][act] != stat2[sub][act]:
                print("Sub {} act {}: {} != {}".format(sub, act, stat1[sub][act], stat2[sub][act]))


if __name__ == "__main__":
    # dataset_folder = "/home/lang/Data/project/3MDAD/"
    args = parse_args()
    datadir = args.datadir
    # if unzip dataset
    unzip_dataset(datadir)

    stat = count_dataset(datadir)

    target_folder = ["front_view_day_data", "front_view_night_data", "side_view_day_data", "side_view_night_data"]
    print("Checking modality...")
    for tf in target_folder:
        print(tf)
        keys = list(stat[tf].keys())
        print(keys)
        mod_sanity_check(stat[tf][keys[0]], stat[tf][keys[1]])

    print("checking views...")
    mod_sanity_check(stat["front_view_day_data"]["RGB2"],
                     stat["side_view_day_data"]["RGB1"])

    mod_sanity_check(stat["front_view_night_data"]["IR2"],
                     stat["side_view_night_data"]["IR1"])

    print("*******************************************************")
    print("Fixing dataset")
    # fix_dataset(datadir)
    print("*******************************************************")
