import json, os
import numpy as np

def hwf_to_K(hwf):
    K = np.eye(4)
    K[0][0] = hwf[2]
    K[0][2] = hwf[1] / 2.
    K[1][1] = hwf[2]
    K[1][2] = hwf[0] / 2.

    return K

def save_cameras(path, poses):
    cameras_dict = {}
    bottom = np.reshape([0, 0, 0, 1.], (1, 4))
    for i, pose in enumerate(poses):
        c2w = np.concatenate([pose[:, :4], bottom], axis=0)
        hwf = pose[:, 4]
        K = hwf_to_K(hwf)
        w2c = np.linalg.inv(c2w)
        img_size = [hwf[1], hwf[0]]
        cameras_dict[f"{i}"] = {}
        cameras_dict[f"{i}"]["K"] = K.flatten().tolist()
        cameras_dict[f"{i}"]["W2C"] = w2c.flatten().tolist()
        cameras_dict[f"{i}"]["img_size"] = img_size
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(cameras_dict, f, indent=4)