import os
import shutil
import argparse
import itertools
import open3d as o3d
import numpy as np
import time, csv
from utilities import save_time_breakdown

def compute_iou_sets(set1, set2):
    """
    Compute IoU between two point sets (each is a set of tuple points).
    """
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    intersection = set1 & set2
    union = set1 | set2
    return len(intersection) / len(union)

def compute_iou(pcd1, pcd2):
    """
    Backward-compatible: Compute IoU between two point clouds by direct point overlap.
    """
    pts1 = np.asarray(pcd1.points)
    pts2 = np.asarray(pcd2.points)
    set1 = set(map(tuple, pts1))
    set2 = set(map(tuple, pts2))
    return compute_iou_sets(set1, set2)

def main():
    parser = argparse.ArgumentParser(description="Two-stage filtering of mesh directories by IoU of part.ply files")
    parser.add_argument("--input_dir", required=True, help="Path to the input directory containing subdirectories with part.ply")
    parser.add_argument("--output_dir", required=True, help="Path to the output directory to copy filtered subdirectories")
    parser.add_argument("--iou_threshold", type=float, default=0.8, help="IoU threshold above which to consider meshes duplicates (first stage)")
    args = parser.parse_args()
    
    folder = "/home/troye/ssd/Zhao_SP/opdm_code/scene_output"
    
    for subfolder in os.listdir(folder):
        print(f"Processing subfolder: {subfolder}")
        subfolder_path = os.path.join(folder, subfolder)
        if not os.path.isdir(subfolder_path):
            continue

        # input_dir = args.input_dir
        input_dir = subfolder_path
        # output_dir = args.output_dir
        output_dir = os.path.join(args.output_dir, subfolder)

        # start timer
        start_time = time.time()

        first_stage_thresh = args.iou_threshold
        candidate_thresh = 0.2
        second_stage_cover_thresh = 0.8

        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)

        mesh_list = []  # list of tuples: (subdir_path, point_cloud)

        # Read all part.ply files
        for subdir in os.listdir(input_dir):
            subdir_path = os.path.join(input_dir, subdir)
            if os.path.isdir(subdir_path):
                ply_file = os.path.join(subdir_path, "part.ply")
                if os.path.exists(ply_file):
                    mesh = o3d.io.read_point_cloud(ply_file)
                    if not mesh.is_empty():
                        mesh_list.append((subdir_path, mesh))

        num_meshes = len(mesh_list)
        if num_meshes == 0:
            print("No valid part.ply point clouds found in input directory.")
            continue

        # Precompute point sets for efficiency
        point_sets = []
        for _, mesh in mesh_list:
            pts = np.asarray(mesh.points)
            point_sets.append(set(map(tuple, pts)))

        # Pairwise IoU cache: keys are (i, j) with i < j
        pairwise_iou = {}

        # First stage: pairwise comparison to remove near-duplicates
        keep = [True] * num_meshes

        for (i, (path_i, mesh_i)), (j, (path_j, mesh_j)) in itertools.combinations(enumerate(mesh_list), 2):
            set_i = point_sets[i]
            set_j = point_sets[j]
            iou = compute_iou_sets(set_i, set_j)
            pairwise_iou[(i, j)] = iou

            if keep[i] and keep[j]:
                name_i = os.path.basename(path_i)
                name_j = os.path.basename(path_j)
                if name_i == "mesh_088":
                    print(f"[First Stage] Comparing {name_i} and {name_j}: IoU = {iou:.4f}")
                if iou > first_stage_thresh:
                    # name_i = os.path.basename(path_i)
                    # name_j = os.path.basename(path_j)
                    print(f"[First Stage] Comparing {name_i} and {name_j}: IoU = {iou:.4f} > {first_stage_thresh}, deduplication triggered.")
                    # Keep the mesh with more points
                    if len(mesh_i.points) >= len(mesh_j.points):
                        keep[j] = False
                    else:
                        keep[i] = False

        # Second stage: From largest point count to smallest, use combinations of smaller preserved meshes to cover the current mesh
        kept_indices = [idx for idx, flag in enumerate(keep) if flag]
        # sort by number of points descending
        kept_indices.sort(key=lambda i: len(mesh_list[i][1].points), reverse=True)

        for curr_idx in kept_indices:
            if not keep[curr_idx]:
                continue  # Might have already been filtered out earlier in the second stage

            curr_set = point_sets[curr_idx]
            curr_name = os.path.basename(mesh_list[curr_idx][0])
            curr_point_count = len(mesh_list[curr_idx][1].points)

            # Candidates: still preserved, fewer points, and IoU with curr > candidate_thresh
            candidate_indices = []
            for other_idx in range(num_meshes):
                if other_idx == curr_idx:
                    continue
                if not keep[other_idx]:
                    continue  # Only use surviving sub-meshes
                other_point_count = len(mesh_list[other_idx][1].points)
                if other_point_count >= curr_point_count:
                    continue  # Only use smaller meshes to cover the larger mesh

                key = (other_idx, curr_idx) if other_idx < curr_idx else (curr_idx, other_idx)
                iou = pairwise_iou.get(key)
                if iou is None:
                    # Fallback calculation (usually all pairs i<j have been computed in the first stage)
                    iou = compute_iou_sets(curr_set, point_sets[other_idx])
                    pairwise_iou[key] = iou
                if iou > candidate_thresh:
                    candidate_indices.append(other_idx)

            if not candidate_indices:
                continue  # No qualified sub-meshes

            # Enumerate all non-empty combinations of candidates to see if they can cover curr
            covered = False
            for r in range(1, len(candidate_indices) + 1):
                for combo in itertools.combinations(candidate_indices, r):
                    merged_set = set()
                    for idx in combo:
                        merged_set |= point_sets[idx]
                    iou_combo = compute_iou_sets(curr_set, merged_set)
                    if iou_combo >= second_stage_cover_thresh:
                        combo_names = [os.path.basename(mesh_list[idx][0]) for idx in combo]
                        print(f"[Second Stage] Mesh {curr_name} is covered by smaller combination {combo_names} with IoU = {iou_combo:.4f} ≥ {second_stage_cover_thresh}. Filtering out {curr_name}.")
                        keep[curr_idx] = False
                        covered = True
                        break
                if covered:
                    break

        # Final copy of kept subdirectories
        kept_count = sum(1 for flag in keep if flag)
        os.makedirs(output_dir, exist_ok=True)
        for flag, (subdir_path, _) in zip(keep, mesh_list):
            if flag:
                dest = os.path.join(output_dir, os.path.basename(subdir_path))
                shutil.copytree(subdir_path, dest)

        print(f"Filtering complete. Kept {kept_count} out of {num_meshes} subdirectories.")

        # end timer
        end_time = time.time()
        duration = end_time - start_time

        # --- TIME BREAKDOWN LOGGING ---
        save_time_breakdown(
            input_dir=input_dir,
            output_dir=output_dir,
            module_name="Module 5.1: Mesh Deduplication and Filtering",
            start_time=start_time,
            end_time=end_time,
            duration=duration
        )

if __name__ == "__main__":
    main()
