## Utils
import os
import shutil
import numpy as np
from tqdm import tqdm
import open3d
import torch
import re


### Reorganise filepath
def reorganise_image_path(path2d: os.PathLike, path3d: os.PathLike, filetype: str):
    '''
    Reorganises dataset to folder structure required by the load utility function of OM3D
    
    Args:
        - path2d: absolute path to 2d data: images, intrinics and depth maps (originally downloaded from the openscene ETH repository)
        - path3d: absolute path to 3d data: pointclouds in `.pth` file format (originally downloaded from the openscene ETH repository)
        - filetype: the file extension in form of a string
                
    .. Returns:: None #FIXME
    '''
    
    with os.scandir(path2d) as it:
        subdirectories2d = np.array([entry.name for entry in it if not entry.name.startswith('.') and entry.is_dir()]).flatten()
        same_length = (np.array([len(x) for x in subdirectories2d]) == 12).all()
        print(same_length)
                
    with os.scandir(path3d ) as it:
        pth_filenames = np.array([entry.name for entry in it if not entry.name.startswith('.') and entry.is_file() and entry.name.endswith(filetype) ])
        
    # debug_lengths = (np.array([len(x) for x in pth_filenames]) == len('scene0000_00_vh_clean_2.pth')).all()
    prefixes = np.array([x.replace('_vh_clean_2.pth', '') for x in pth_filenames])
    print(prefixes.shape==subdirectories2d.shape)
    print(prefixes.dtype==subdirectories2d.dtype)
    
    dir_idxs = np.array([np.where(subdirectories2d==x)[0] for x in prefixes])
     
    for i,j in enumerate(dir_idxs):
        print(f"Copying scannet3d/[train/val]/{pth_filenames[i]} to scannet2d/{subdirectories2d[j]}")
        source = os.path.join(path3d + '/' + pth_filenames[i])
        out = os.path.join(path2d, prefixes[i] + '/' + pth_filenames[i])
        
        if not os.path.isfile(out):
            shutil.copy2(source, out)
            print(f"source: {source}, out: {out}")
        else:
            print("No need to copy the pointcloud, the .pth file exists alr")
    
        data_dir        = os.path.join(path2d, prefixes[i] + '/' + 'data')
        pose_dir        = os.path.join(path2d, prefixes[i] + '/' + 'data' + '/' + 'pose')
        intrinsics_dir  = os.path.join(path2d, prefixes[i] + '/' + 'data' + '/' + 'intrinsics')
        compressed_dir  = os.path.join(path2d, prefixes[i] + '/' + 'data_compressed')
        color_dir       = os.path.join(path2d, prefixes[i] + '/' + 'data_compressed' + '/' + 'color')
        depth_dir       = os.path.join(path2d, prefixes[i] + '/' + 'data_compressed' + '/' + 'depth')
        
        intrinsics_file  = os.path.join(path2d, 'intrinsics.txt') ## kinda const, but imma group it with the other paths
        
        # handle directory creation
        if not (os.path.isdir(data_dir) and os.path.isdir(compressed_dir)):
            os.makedirs(data_dir)
            os.makedirs(compressed_dir)
        
        if not (os.path.isdir(intrinsics_dir) and os.path.isdir(pose_dir)):
            os.makedirs(intrinsics_dir)
            os.makedirs(pose_dir)
                
        if not (os.path.isdir(color_dir) and os.path.isdir(depth_dir)):
            os.makedirs(color_dir)
            os.makedirs(depth_dir)
        
        color = os.path.join(path2d, prefixes[i] + '/' + 'color')
        depth = os.path.join(path2d, prefixes[i] + '/' + 'depth')
        pose = os.path.join(path2d, prefixes[i] + '/' + 'pose')
        label = os.path.join(path2d, prefixes[i] + '/' + 'label')
        
        print(f"{color} > {color_dir}")
        print(f"{depth} > {depth_dir}")
        print(f"{pose} > {pose_dir}")
        print(f"{intrinsics_file} > {intrinsics_dir}")
        
        if os.path.isdir(color):
            color_files = os.listdir(color)
            for file in tqdm(color_files):
                file =  os.path.join(color, file)
                shutil.move(file, color_dir)
            os.rmdir(color)
        else:
            print("Color dir doesnt exist (anymore)")
            
        if os.path.isdir(depth):
            depth_files = os.listdir(depth)
            for file in tqdm(depth_files):
                file =  os.path.join(depth, file)
                shutil.move(file, depth_dir)
            os.rmdir(depth)
        else:
            print("Depth dir doesnt exist (anymore)")
            
        if os.path.isdir(pose):
            pose_files = os.listdir(pose)
            for file in tqdm(pose_files):
                file =  os.path.join(pose, file)
                shutil.move(file, pose_dir)
            os.rmdir(pose)
        else:
            print("Pose dir doesnt exist (anymore)")
            
        if os.path.isdir(intrinsics_dir):
            print("Copying intrinsics")
            shutil.copy2(intrinsics_file, intrinsics_dir)
   
    return


def pth_to_ply(path3d: os.PathLike): # try to undo data-preprocessing done by openscene on the 3d files from scannet
    with os.scandir(path3d ) as it:
        pth_filenames = np.array([entry.name for entry in it if not entry.name.startswith('.') and entry.is_file()])
    ply_filenames = np.array([x.replace('_vh_clean_2.pth', '_vh_clean_2.ply') for x in pth_filenames])
    print(ply_filenames)
    
    out_dir = path3d + '/' + "ply"
    os.makedirs(out_dir)
    
    for i, file in enumerate(pth_filenames):
        source = os.path.join(path3d + '/' + file)
        # print(f"loading pth {source}")
        data = torch.load(source)
        coords, colors, labels = data
        features = (colors + 1) * 127.5
        assert coords.size == features.size
        # print(f"labels: {labels}")
        
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(coords)
        pcd.colors = open3d.utility.Vector3dVector(features)
        # pcd.orient_normals_towards_camera_location(pcd.get_center()) # to make normals face same direction
        
        out = os.path.join(out_dir + '/' + ply_filenames[i])
        print(f"saving to ply {out}")
        open3d.io.write_point_cloud(out , pcd)
        
    return


def add_metadata(metadata_path: os.PathLike, dataset_path: os.PathLike):
    '''
    Adds metadata files to the dataset used for preprocessing
    (e.g. intrinsics, poses, etc.) to the correct folder structure
    
    Args:
        - metadata_path: absolute path to metadata files
        - dataset_path: absolute path to dataset folder structure
                
    .. Returns:: None #FIXME
    '''
    
    with os.scandir(metadata_path) as it:
        metadata_dirs = np.array([entry.name for entry in it if not entry.name.startswith('.') and entry.is_dir()]).flatten()
        # print(metadata_dirs)
        # same_length = (np.array([len(x) for x in metadata_dirs]) == 12).all()
        # print(same_length)
        
    with os.scandir(dataset_path) as it:
        data_dirs = np.array([entry.name for entry in it if not entry.name.startswith('.') and entry.is_dir()]).flatten()
        # same_length = (np.array([len(x) for x in data_dirs]) == 12).all()
        # print(same_length)
    
    # dir_idxs = np.array([np.where(metadata_dirs==x)[0] for x in data_dirs])
    # print(dir_idxs)
    # print((metadata_dirs==data_dirs).all()) 
    assert metadata_dirs.shape==data_dirs.shape, "Metadata and data directories do not match in size"
    
      
    for scan_dir in metadata_dirs:
        source_dir = os.path.join(metadata_path, scan_dir)
        assert os.path.isdir(source_dir), f"Source directory {source_dir} does not exist"
        print(f"Source directory: {source_dir}")
        
        print(f"Copying metadata from {source_dir} to {dataset_path}")
        
        for file in tqdm(os.listdir(source_dir)):
            source_file = os.path.join(source_dir, file)
            if os.path.isfile(source_file):
                dst_dir = os.path.join(dataset_path, scan_dir)
                print(f"Copying {source_file} to {dst_dir}")
                shutil.copy(source_file, dst_dir)
            else:
                print(f"Source file {source_file} does not exist")
                continue
    
    return


def verify_number_of_files(path: os.PathLike, filenumber: int, breakpt: int = 0, verbose: bool = True):
    '''
    Verifies the number of files in a directory is equal to the expected number of files
    (e.g. 9 files for the scannet dataset)
    
    Args:
        - path: absolute path to dataset folder structure
        - filenumber: expected number of files in the directory
    '''
    
    with os.scandir(path) as it:
        subdirectories = np.array([entry for entry in it if not entry.name.startswith('.') and entry.is_dir()]).flatten()
        # print(subdirectories)
        same_length = (np.array([len(x.name) for x in subdirectories]) == 12).all()
        print(same_length)
        
    differences = np.array([]) # empty array to store differences
        
    for i,directory in enumerate(subdirectories):
        if verbose:
            print(f"Directory: {directory.name}")
        with os.scandir(directory) as it:
            files = np.array([entry for entry in it if not entry.name.startswith('.') and entry.is_file()]).flatten()
            if verbose:
                print(files)
                print(f"Number of files in {directory.name}: {len(files)}")
            # assert len(files) == filenumber, f"Number of files in {directory.name} ({len(files)}) is not equal to {filenumber}"
            if len(files) != filenumber:
                differences = np.append(differences, (directory.name, len(files)))
                if verbose:
                    print(f"Number of files in {directory.name} ({len(files)}) is not equal to {filenumber}")
                    
        if  breakpt != 0 and i == breakpt and verbose:
            print("Breaking at directory: ", directory.name)
            break
    
    print(f"Number of directories: {len(subdirectories)}")
    print(f"Number of directories with different number of files: {len(differences)//2}")
    print(f"Differences: {differences}")
    
    return
    

'''
The function calls below bring the data from the openscene dataset into
a) the correct file format and 
b) the correct folder structure
c) add metadata files to the dataset used for preprocessing
'''
# verify_number_of_files("ABSOLUTE PATH TO DATASET (.../datasets/data)", 9, breakpt=0, verbose=True)

# add_metadata("ABSOLUTE PATH TO METADATA FILES (eg., .../datasets/metadata)", "ABSOLUTE PATH TO DATASET (.../datasets/data/scans)")

# pth_to_ply("ABSOLUTE PATH TO DATASET 3D (.../datasets/data/scannet_3d)")

# reorganise_image_path("ABSOLUTE PATH TO DATASET 2D (.../datasets/data/scannet_2d)", "ABSOLUTE PATH TO DATASET 3D (.../datasets/data/scannet_3d)")
