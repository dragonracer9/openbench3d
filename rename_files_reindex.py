import os
from glob import glob
from multiprocessing import Pool
from tqdm import tqdm

def safe_rename_files(scene_dir):
    subfolders = ["data_compressed/color", "data_compressed/depth", "label", "data/pose"]
    for sub in subfolders:
        folder = os.path.join(scene_dir, sub)
        if not os.path.exists(folder):
            continue

        # 1. Rename all to .tmp
        for f in os.listdir(folder):
            base, ext = os.path.splitext(f)
            if not base.isdigit():
                continue
            src = os.path.join(folder, f)
            tmp = os.path.join(folder, f"{base}{ext}.tmp")
            os.rename(src, tmp)

        # 2. Rename .tmp to final target
        for f in os.listdir(folder):
            if not f.endswith(".tmp"):
                continue
            base, ext = os.path.splitext(f[:-4])
            new_idx = str(int(base) // 20)
            new_path = os.path.join(folder, f"{new_idx}{ext}")
            src = os.path.join(folder, f)
            if os.path.exists(new_path):
                print(f"[SKIP] Exists: {new_path}")
                continue
            os.rename(src, new_path)

def run_parallel_renaming(scenes, processes=8):
    with Pool(processes=processes) as pool:
        list(tqdm(pool.imap_unordered(safe_rename_files, scenes), total=len(scenes), desc="Renaming scenes"))

def main():
    scans_root = "some path to your files"
    scenes = [os.path.join(scans_root, d) for d in os.listdir(scans_root)
              if os.path.isdir(os.path.join(scans_root, d))]

    run_parallel_renaming(scenes)

if __name__ == "__main__":
    main()