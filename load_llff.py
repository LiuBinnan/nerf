import numpy as np
import imageio.v3 as iio
import os
from utils import save_cameras

def normalize(x):
    return x / np.linalg.norm(x)

def avg_poses(poses):
    hwf = poses[..., -1]
    c = poses[..., 3].mean(0)
    vec2 = normalize(poses[..., 2].sum(0))
    vec1 = normalize(poses[..., 1].sum(0))
    vec0 = normalize(np.cross(vec1, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    avg_pose = np.stack([vec0, vec1, vec2, c, hwf[0]], axis=-1)

    return avg_pose

def recenter_poses(poses):
    bottom = np.array([0, 0, 0, 1.]).reshape(1, 4)
    avg_pose = avg_poses(poses)
    avg_pose = np.concatenate([avg_pose[..., :-1], bottom], axis=0)

    bottom = np.tile(np.reshape(bottom, (1, 1, 4)), (poses.shape[0], 1, 1))
    hwf = poses[..., -1:]
    poses = np.concatenate([poses[..., :4], bottom], axis=1)
    poses = np.linalg.inv(avg_pose) @ poses
    poses = np.concatenate([poses[:, :3], hwf], axis=-1)

    return poses

def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, rots, N):
    render_poses = []
    rads = np.append(rads, 1.)
    hwf = c2w[:, -1]

    for theta in np.linspace(0, 2. * np.pi * rots, N+1)[:-1]:
        xyz = np.array([np.cos(theta), -np.sin(theta), -np.sin(theta*zrate), 1.]) * rads
        c = np.dot(c2w[:, :-1], xyz)
        z = np.array([0, 0, -focal, 1.])
        vec2 = normalize(c - np.dot(c2w[:, :-1], z))
        vec0 = normalize(np.cross(up, vec2))
        vec1 = normalize(np.cross(vec2, vec0))
        render_poses.append(np.stack([vec0, vec1, vec2, c, hwf], axis=-1))

    return render_poses

def _load_data(basedir, factor=None):
    # load poses and bounds
    poses_bounds = np.load(os.path.join(basedir, 'poses_bounds.npy'))
    poses = poses_bounds[:, :15].reshape(-1, 3, 5) # [B, 3, 5]
    bds = poses_bounds[:, -2:]

    suffix = f'_{factor}' if factor is not None else ''

    # load images
    images_path = os.path.join(basedir, f'images{suffix}')
    images =[iio.imread(os.path.join(images_path, file))[..., :3]/255. for file in sorted(os.listdir(images_path))
            if file.endswith('jpg') or file.endswith('png') or file.endswith('JPG')]
    images = np.stack(images, axis=0) # [B, H, W, C]
    assert(images.shape[0] == poses.shape[0])

    # scale camera intrinsics
    sh = images.shape[1:3]
    poses[:, :2, 4] = np.array(sh)
    poses[:, 2, 4] = poses[:, 2, 4] * 1. / factor

    return poses, bds, images
   
def load_llff_data(basedir, factor=8, recenter=True, bd_factor=.75):
    poses, bds, images = _load_data(basedir, factor=8)
    poses = np.concatenate([poses[..., 1:2], -poses[..., 0:1], poses[..., 2:]], axis=-1)

    # Rescale if bd_factor is provided
    sc = 1. if bd_factor is None else 1./(bds.min() * bd_factor)
    poses[:,:3,3] *= sc
    bds *= sc # near -> 1.33

    if recenter:
        poses = recenter_poses(poses)

    avg_pose = avg_poses(poses)
    ## Get spiral
    # Get average pose
    up = normalize(poses[:, :3, 1].sum(0))

    # Find a reasonable "focus depth" for this dataset
    close_depth, inf_depth = bds.min()*.9, bds.max()*5.
    dt = .75
    mean_dz = 1./(((1.-dt)/close_depth + dt/inf_depth))
    focal = mean_dz
    # Get radii for spiral path
    shrink_factor = .8
    zdelta = close_depth * .2
    tt = poses[:,:3,3] # ptstocam(poses[:3,3,:].T, c2w).T
    rads = np.percentile(np.abs(tt), 90, 0)
    c2w_path = avg_pose
    N_views = 120
    N_rots = 2

    render_poses = render_path_spiral(c2w_path, up, rads, focal, zdelta, zrate=.5, rots=N_rots, N=N_views)
    render_poses = np.array(render_poses)

    dists = np.sum(np.square(avg_pose[:3,3] - poses[:,:3,3]), -1)
    i_test = np.argmin(dists)

    return images, poses, bds, render_poses, i_test

if __name__ == '__main__':
    basedir = 'C:\\Users\\LiuBinnan\\Desktop\\code\\nerf\\data\\nerf_example_data\\nerf_llff_data\\fern'
    images, poses, bds, render_poses, i_test = load_llff_data(basedir, factor=8, recenter=True)

    path = os.path.join(basedir, 'avg_cam.json')
    save_cameras(path, avg_poses(poses)[None, ...])
    path = os.path.join(basedir, 'cam.json')
    save_cameras(path, poses)
    path = os.path.join(basedir, 'render_cam.json')
    save_cameras(path, render_poses)