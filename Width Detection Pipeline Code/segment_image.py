import argparse
import os

def prompt_user(prompt):
    """Makes sure the user wants to continue."""
    response = input(f"{prompt} (y/n): ")
    if response != "y":
        print("Exiting.")
        exit()

def make_folder_and_warn(folder):
    if not folder:
        return
    # # Create output directory
    # if os.path.isdir(folder):
    #     print(f"Warning: Output directory {folder} already exists!")
    #     prompt_user("Overwrite?")
    #     print("Deleting old directory...")
    #     os.system(f"rm -rf {folder}")
    # else:
    #     print(f"Creating output directory for {folder}")
    os.makedirs(folder, exist_ok=True)

# Parse all arguments
parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, required=True,
                    help="The input image")
parser.add_argument('--output', type=str, default="out",
                    help="The output base filepath (will append .csv and .png")
args = parser.parse_args()
make_folder_and_warn(os.path.dirname(args.output))

import cv2
import time
import tensorflow as tf
from model.pspunet import pspunet
from data_loader.display import create_mask
import numpy as np
gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
       gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=500)])
    except RuntimeError as e:
        print(e)

frame = cv2.imread(args.input)

ORIG_WIDTH = frame.shape[1]
ORIG_HEIGHT = frame.shape[0]
# IMG_WIDTH = ORIG_WIDTH // 64 * 8
# IMG_HEIGHT = ORIG_HEIGHT // 64 * 8
SCALE_FACTOR = max(480 / ORIG_WIDTH, 272 / ORIG_HEIGHT)
# Making it exactly 480 x 272 does not improve the model at all
IMG_WIDTH = round(ORIG_WIDTH * SCALE_FACTOR // 8 * 8)
IMG_HEIGHT = round(ORIG_HEIGHT * SCALE_FACTOR // 8 * 8)
print("Original shape", ORIG_WIDTH, ORIG_HEIGHT)
print("-> Quantized", IMG_WIDTH, IMG_HEIGHT)
n_classes = 7

model = pspunet((IMG_HEIGHT, IMG_WIDTH, 3), n_classes)
model.load_weights("pspunet_weight.h5");

import numpy as np

def predict_image(frame):
    frame2 = frame.copy()[tf.newaxis, ...] / 255.0
    
    frame = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame[tf.newaxis, ...]
    frame = frame/255

    pre = model.predict(frame)
    pre = create_mask(pre).numpy()
    print(pre.shape)

    # Remove the channel dimension if it's 1 (i.e., grayscale)
    if pre.shape[2] == 1:
        pre = pre[:, :, 0]
    
    # Resize the mask (temp change)
    # ORIG_WIDTH = 1792
    # ORIG_HEIGHT = 828
    resized_mask = cv2.resize(pre, (ORIG_WIDTH, ORIG_HEIGHT), interpolation=cv2.INTER_NEAREST)
    # frame2 = cv2.resize(frame2[0], (ORIG_WIDTH, ORIG_HEIGHT))
    # frame2 = frame2.copy()[tf.newaxis, ...]

    # Output mask
    np.savetxt(f"{args.output}.csv", resized_mask, fmt='%s', delimiter=",")
    
    # If you need to add the channel dimension back
    pre = resized_mask[:, :, np.newaxis]
    print("Output shape:", frame2[0].shape)

    frame2[0][(pre==1).all(axis=2)] += [0, 0, 0] #""bike_lane_normal", "sidewalk_asphalt", "sidewalk_urethane""
    frame2[0][(pre==2).all(axis=2)] += [0.5, 0.5,0] # "caution_zone_stairs", "caution_zone_manhole", "caution_zone_tree_zone", "caution_zone_grating", "caution_zone_repair_zone"]
    frame2[0][(pre==3).all(axis=2)] += [0.2, 0.7, 0.5] #"alley_crosswalk","roadway_crosswalk"
    frame2[0][(pre==4).all(axis=2)] += [0, 0.5, 0.5] #"braille_guide_blocks_normal", "braille_guide_blocks_damaged"
    frame2[0][(pre==5).all(axis=2)] += [0, 0, 0.5] #"roadway_normal","alley_normal","alley_speed_bump", "alley_damaged""
    frame2[0][(pre==6).all(axis=2)] += [0.5, 0, 0] #"sidewalk_blocks","sidewalk_cement" , "sidewalk_soil_stone", "sidewalk_damaged","sidewalk_other"
    frame2[0] = frame2[0]*255
    
    return frame2[0]

pre = predict_image(frame)
cv2.imwrite(f"{args.output}.png", pre)
print("Done!")