## Utils
import os
import shutil
import numpy as np
from tqdm import tqdm

### Reorganise filepath
def reorganise_image_path(path2d: os.PathLike, path3d: os.PathLike):
    '''
    Reorganises dataset to folder structure required by the load utility function of OM3D
    
    Args:
        - path2d: absolute path to 2d data: images, intrinics and depth maps (originally downloaded from the openscene ETH repository)
        - path3d: absolute path to 3d data: pointclouds in `.pth` file format (originally downloaded from the openscene ETH repository)
                
    .. Returns:: None #FIXME
    '''
    
    with os.scandir(path2d) as it:
        subdirectories2d = np.array([entry.name for entry in it if not entry.name.startswith('.') and entry.is_dir()]).flatten()
        same_length = (np.array([len(x) for x in subdirectories2d]) == 12).all()
        print(same_length)
                
    with os.scandir(path3d ) as it:
        pth_filenames = np.array([entry.name for entry in it if not entry.name.startswith('.') and entry.is_file()])
        
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

reorganise_image_path("/mnt/c/users/vikra/desktop/ethz/msc_mech_eng/3dv/openmask3d/datasets/data/scannet_2d", "/mnt/c/users/vikra/desktop/ethz/msc_mech_eng/3dv/openmask3d/datasets/data/scannet_3d")
