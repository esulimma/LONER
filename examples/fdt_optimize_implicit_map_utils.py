LIDAR_MIN_RANGE = 0.3 #http://www.oxts.com/wp-content/uploads/2021/01/Ouster-datasheet-revc-v2p0-os0.pdf
WARN_MOCOMP_ONCE = True
WARN_LIDAR_TIMES_ONCE = True
CHUNK_SIZE=2**12
CHUNK_SIZE=2**8

import os, sys

import rospy
import ros_numpy

import numpy as np
import torch

from attrdict import AttrDict
from sensor_msgs.msg import Image, PointCloud2
import os, sys

import cv2
from cv_bridge import CvBridge
import kornia
from scipy import ndimage

import open3d as o3d

PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    os.pardir))

sys.path.append(PROJECT_ROOT)

from src.common.sensors import Image, LidarScan
from src.common.ray_utils import LidarRayDirections
from common.frame import Frame
from analysis.fdt_common_utils.pcd_visualizer import *


def compute_sky_rays(frame: Frame):
    TOP_ROWS = 3
    HORIZON_OFFSET=10

    dirs = frame.lidar_points.ray_directions
    x,y,z = dirs[0], dirs[1], dirs[2]
    theta = torch.atan2(y, x).rad2deg().round().long()
    phi = torch.atan2(torch.sqrt(x**2 + y**2), z).rad2deg().round().long()

    phi_img = phi - phi.min()
    theta_img = theta-theta.min()
    theta_img[theta_img == 360] = 0

    depth_img = torch.zeros((phi_img.max()+1, 360), device=phi.device)
        
    depth_img[phi_img, theta_img] = 1

    depth_img = kornia.morphology.dilation(depth_img.unsqueeze(0).unsqueeze(0), torch.ones((3,3), device=depth_img.device))
    depth_img = kornia.morphology.erosion(depth_img, torch.ones((3,3), device=depth_img.device)).squeeze(0).squeeze(0)

    depth_img[:TOP_ROWS] = 1

    zero_locs = torch.where(depth_img == 0)
    zero_phi = (zero_locs[0] + phi.min()).deg2rad()
    zero_theta = (zero_locs[1] + theta.min()).deg2rad()

    z_out = torch.cos(zero_phi)
    y_out = torch.sin(zero_phi) * torch.sin(zero_theta)
    x_out = torch.sin(zero_phi) * torch.cos(zero_theta)

    zero_dirs = torch.vstack((x_out, y_out, z_out))

    r = frame.get_lidar_pose().get_rotation()

    zero_dirs_world = r @ zero_dirs
    x_w,y_w,z_w = zero_dirs_world[0], zero_dirs_world[1], zero_dirs_world[2]
    phi_w = 90 - torch.atan2(torch.sqrt(x_w**2 + y_w**2), z_w).rad2deg()
    sky_rays = zero_dirs_world[:, phi_w > HORIZON_OFFSET]

    frame.lidar_points.sky_rays = sky_rays

def get_msg_from_time_stamp(bag, lidar_topic, gt_timestamp, initiliase_from_zero):
    if initiliase_from_zero:
        initial_time = None
    else:
        initial_time = 0

    for _, msg, _ in bag.read_messages(topics=[lidar_topic]):
        if initial_time == None:
            initial_time = msg.header.stamp.to_sec()
        if np.round(msg.header.stamp.to_sec() - initial_time,5) == np.round(gt_timestamp,5):
            return msg, msg.header.stamp

def build_scan_from_msg(lidar_msg: PointCloud2, timestamp: rospy.Time, fov: dict = None, recomute_timestamps = False) -> LidarScan:

    lidar_data = ros_numpy.point_cloud2.pointcloud2_to_array(lidar_msg)

    fields = [f.name for f in lidar_msg.fields]
    
    time_key = None
    for f in fields:
        if "time" in f or f == "t":
            time_key = f
            break
    
    num_points = lidar_msg.width * lidar_msg.height

    xyz = torch.zeros((num_points, 3,), dtype=torch.float32)
    xyz[:,0] = torch.from_numpy(lidar_data['x'].copy().reshape(-1,))
    xyz[:,1] = torch.from_numpy(lidar_data['y'].copy().reshape(-1,))
    xyz[:,2] = torch.from_numpy(lidar_data['z'].copy().reshape(-1,))

    if fov is not None and fov.enabled:
        theta = torch.atan2(xyz[:,1], xyz[:,0]).rad2deg()
        theta[theta < 0] += 360
        point_mask = torch.zeros_like(xyz[:, 1])
        for segment in fov.range:
            local_mask = torch.logical_and(theta >= segment[0], theta <= segment[1])
            point_mask = torch.logical_or(point_mask, local_mask) 

        xyz = xyz[point_mask]
    dists = xyz.norm(dim=1)

    valid_ranges = dists > LIDAR_MIN_RANGE

    xyz = xyz[valid_ranges].T

    azimuth = torch.atan2(xyz[:, 1], xyz[:, 0]).rad2deg()
    azimuth[azimuth < 0] += 360
    
    global WARN_MOCOMP_ONCE

    if time_key is None:
        if WARN_MOCOMP_ONCE:
            print("Warning: LiDAR Data has No Associated Timestamps. Motion compensation is useless.")
            WARN_MOCOMP_ONCE = False
        timestamps = torch.full_like(xyz[0], timestamp.to_sec()).float()
    else:

        global WARN_LIDAR_TIMES_ONCE
        if recomute_timestamps:
            # This fix provided to me by the authors of Fusion Portable.
            lidar_indices = torch.arange(len(lidar_data[time_key].flatten()))
            h_resolution = 2048
            scan_period = 0.2
            timestamps = (lidar_indices % h_resolution) * 1.0/h_resolution * scan_period

        else:
            timestamps = torch.from_numpy((lidar_data[time_key]-lidar_data[time_key][0]).astype(np.float32)).reshape(-1,)
            # timestamps = torch.from_numpy((lidar_data[time_key]).astype(np.float32)).reshape(-1,)

        if fov is not None and fov.enabled:
            timestamps = timestamps[point_mask]
        timestamps = timestamps[valid_ranges]

        # This logic deals with the fact that some lidars report time globally, and others 
        # use the ROS timestamp for the overall time then the timestamps in the message are just
        # offsets. This heuristic has looked legit so far on the tested lidars (ouster and hesai).
        if timestamps.abs().max() > 1e7:
            if WARN_LIDAR_TIMES_ONCE:
                print("Timestamps look to be in nanoseconds. Scaling")
            timestamps *= 1e-9

        if timestamps[0] < -0.001:
            if WARN_LIDAR_TIMES_ONCE:
                print("Timestamps negative (velodyne?). Correcting")
                timestamps -= timestamps[0].clone()

        if timestamps[0] < 1e-2:
            if WARN_LIDAR_TIMES_ONCE:
                print("Assuming LiDAR timestamps within a scan are local.")
        else:
            if WARN_LIDAR_TIMES_ONCE:
                print("Assuming lidar timestamps within a scan are global.")
            timestamps = timestamps - timestamps[0] + timestamp.to_sec()
        WARN_LIDAR_TIMES_ONCE = False


        if timestamps[-1] - timestamps[0] < 1e-3:
            if WARN_MOCOMP_ONCE:
                print("Warning: Timestamps in LiDAR data aren't unique. Motion compensation is useless")
                WARN_MOCOMP_ONCE = False

            timestamps = torch.full_like(xyz[0], timestamp.to_sec()).float()

    timestamps = timestamps.float()

    dists = dists[valid_ranges].float()
    directions = (xyz / dists).float()

    timestamps, indices = torch.sort(timestamps)
    
    dists = dists[indices]
    directions = directions[:, indices]

    return LidarScan(directions.float().cpu(), dists.float().cpu(), timestamps.float().cpu())

def tf_to_settings(tf_msg):
    trans = tf_msg.transform.translation
    rot = tf_msg.transform.rotation

    xyz = [trans.x, trans.y, trans.z]
    quat = [rot.w, rot.x, rot.y, rot.z]

    return AttrDict({"xyz": xyz, "orientation": quat})

def build_image_from_msg(image_msg, timestamp, scale_factor, upsampling = None) -> Image:
    br = CvBridge()
    cv_img = br.imgmsg_to_cv2(image_msg, desired_encoding='rgb8')
    cv_img = cv2.resize(cv_img, (0,0), fx=scale_factor, fy=scale_factor)
    if (upsampling is not None) & (upsampling > 1):
        cv_img = upsample_image(cv_img, upsampling)
    pytorch_img = torch.from_numpy(cv_img / 255).float()
    return Image(pytorch_img, timestamp.to_sec())

def build_image_from_compressed_msg(image_msg, timestamp, upsampling = None) -> Image:
    br = CvBridge()
    cv_image = br.compressed_imgmsg_to_cv2(image_msg, "bgr8")
    if (upsampling is not None):
        if(upsampling > 1):
            cv_image = upsample_image(cv_image, upsampling)
    pytorch_img = torch.from_numpy(cv_image / 255).float()
    return Image(pytorch_img, timestamp.to_sec())

def build_image_from_compressed_mask(image_msg, timestamp, upsampling = None, enlarge = False) -> Image:
    br = CvBridge()
    cv_image = br.compressed_imgmsg_to_cv2(image_msg, "bgr8")
    if (upsampling is not None):
        if (upsampling > 1):
            cv_image = upsample_mask(cv_image, upsampling)
    if enlarge:
        cv_image = enlarge_mask(cv_image.astype(bool), 2.0)
        pytorch_img = torch.from_numpy(cv_image)
    else:
        pytorch_img = torch.from_numpy(cv_image).bool()
    return Image(pytorch_img, timestamp.to_sec())

def enlarge_mask(mask, factor):
    factor = int(factor)
    enlarged_mask = mask.copy()
    rows, cols = mask.shape
    for i in range(rows):
        for j in range(cols):
            if not mask[i, j]:
                for k in range(-factor, factor+1):
                    for l in range(-factor, factor+1):
                        if 0 <= i+k < rows and 0 <= j+l < cols:
                            enlarged_mask[i+k, j+l] = False
    return enlarged_mask

def upsample_mask(mask_array, factor = 2.0):
    mask_array_upsampled = ndimage.zoom(mask_array, factor)
    return mask_array_upsampled

def upsample_image(image_array, factor = 2.0):
    f = int(factor)
    image_array_upsampled = np.zeros((image_array.shape[0]*f, image_array.shape[1]*f, image_array.shape[2]))
    image_array_upsampled[:,:,0] = ndimage.zoom(image_array[:,:,0], factor)
    image_array_upsampled[:,:,1] = ndimage.zoom(image_array[:,:,1], factor)
    image_array_upsampled[:,:,2] = ndimage.zoom(image_array[:,:,2], factor)
    return image_array_upsampled

def compute_l1_depth(lidar_pose, ray_directions: LidarRayDirections, model_data, render_color: bool = False):
    with torch.no_grad():
        model, ray_sampler, world_cube, ray_range, device = model_data

        scale_factor = world_cube.scale_factor.to(device)
        size = ray_directions.lidar_scan.ray_directions.shape[1]
        depth_fine = torch.zeros((size,1), dtype=torch.float32).view(-1, 1)

        for chunk_idx in range(ray_directions.num_chunks):
            eval_rays = ray_directions.fetch_chunk_rays(chunk_idx, lidar_pose, world_cube, ray_range)
            eval_rays = eval_rays.to(device)
            results = model(eval_rays, ray_sampler, scale_factor, testing=True, return_variance=True, camera=render_color, render_strategy = 'threshold')
            depth_fine[chunk_idx * CHUNK_SIZE: (chunk_idx+1) * CHUNK_SIZE, :] = results['depth_fine'].unsqueeze(1)  * scale_factor

        gt_depth = ray_directions.lidar_scan.distances
        good_idx = torch.logical_and(gt_depth.flatten() > ray_range[0], gt_depth.flatten() < ray_range[1] - 0.25) 
        good_depth = depth_fine[good_idx]
        good_gt_depth = gt_depth[good_idx.flatten()]

        l1_loss = torch.nn.functional.l1_loss(good_depth.cpu().flatten(), good_gt_depth.cpu().flatten())


        return l1_loss

