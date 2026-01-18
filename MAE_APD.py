import cv2
import os
import re
import numpy as np

def extract_segment(video_path, segment_index, segment_width=256, target_size=(256, 256)):
    cap = cv2.VideoCapture(video_path)
    frames = []
    start_col = segment_index * segment_width
    end_col = start_col + segment_width

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        segment = frame[:, start_col:end_col, :]
        segment = cv2.resize(segment, target_size)
        frames.append(segment)
    
    cap.release()
    return np.array(frames)

def calculate_mae(ground_truth_frames, sample_segment):
    return np.mean(np.abs(ground_truth_frames - sample_segment))

def load_video_frames(video_path, segment_col_start=768, segment_width=256, target_size=(256, 256)):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        segment = frame[:, segment_col_start:segment_col_start + segment_width, :]
        frame_resized = cv2.resize(segment, target_size)
        frames.append(frame_resized)
    cap.release()
    return np.array(frames) / 255.0

def compute_apd_for_videos_efficient(video_paths):
    M = len(video_paths)
    pairwise_total = 0.0
    total_frames = None

    for i in range(M):
        for j in range(M):
            frames_i = load_video_frames(video_paths[i])
            frames_j = load_video_frames(video_paths[j])

            num_frames = min(len(frames_i), len(frames_j))
            if total_frames is None:
                total_frames = num_frames

            f_i = frames_i[:num_frames].reshape(num_frames, -1)
            f_j = frames_j[:num_frames].reshape(num_frames, -1)

            diff = f_i - f_j
            pairwise_total += np.sum(np.abs(diff))

    APD = pairwise_total / (M**2 * total_frames * 256 * 256 * 3)
    return APD

# === Main Execution Starts ===

sample_videos_folder = "./log/test-reconstruction-vox/vox_8-16_GRU-SNF_recon_diversity_100"
ground_truth_folder = "./FOMM/datasets/voxceleb/test"

sample_videos_by_identity = {}
for video in os.listdir(sample_videos_folder):
    suffix_match = re.search(r"-\d+\.mp4$", video)
    if not suffix_match:
        print(f"Skipping video without valid suffix: {video}")
        continue
    identity = video[:video.find('.mp4')]
    sample_videos_by_identity.setdefault(identity, []).append(os.path.join(sample_videos_folder, video))

lowest_20_maes_by_group = {}

for gt_video in os.listdir(ground_truth_folder):
    if not gt_video.endswith(('.mp4', '.png')):
        continue

    gt_identity = gt_video[:gt_video.find('.mp4')]
    ground_truth_path = os.path.join(ground_truth_folder, gt_video)

    if gt_identity not in sample_videos_by_identity:
        print(f"Ground truth identity {gt_identity} not found in sample videos.")
        continue

    ground_truth_frames = []

    if gt_video.endswith('.mp4'):
        cap = cv2.VideoCapture(ground_truth_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_resized = cv2.resize(frame, (256, 256))
            ground_truth_frames.append(frame_resized)
        cap.release()

    elif gt_video.endswith('.png'):
        img = cv2.imread(ground_truth_path)
        frame_width = img.shape[0]
        num_frames = img.shape[1] // frame_width
        for i in range(num_frames):
            frame = img[:, i * frame_width:(i + 1) * frame_width]
            frame_resized = cv2.resize(frame, (256, 256))
            ground_truth_frames.append(frame_resized)
    # if gt_video.endswith('.mp4'):
    #     cap = cv2.VideoCapture(ground_truth_path)
    #     frame_idx = 0  # initialize frame counter
    #     while True:
    #         ret, frame = cap.read()
    #         if not ret:
    #             break
    #         if frame_idx % 2 == 0:  # keep only every 2nd frame
    #             frame_resized = cv2.resize(frame, (256, 256))
    #             ground_truth_frames.append(frame_resized)
    #         frame_idx += 1
    #     cap.release()
    # elif gt_video.endswith('.png'):
    #     img = cv2.imread(ground_truth_path)
    #     frame_width = img.shape[0]
    #     num_frames = img.shape[1] // frame_width
    #     for i in range(num_frames):
    #         if i % 2 == 0:  # keep only every 2nd frame
    #             frame = img[:, i * frame_width:(i + 1) * frame_width]
    #             frame_resized = cv2.resize(frame, (256, 256))
    #             ground_truth_frames.append(frame_resized)

    ground_truth_frames = np.array(ground_truth_frames) / 255.0
    frames_to_keep = (ground_truth_frames.shape[0] // 24) * 24
    ground_truth_frames = ground_truth_frames[:frames_to_keep]

    group_mae = []
    for sample_video_path in sample_videos_by_identity[gt_identity]:
        sample_segment = extract_segment(sample_video_path, segment_index=3)
        sample_segment = sample_segment.astype(np.float32) / 255.0
        mae = calculate_mae(ground_truth_frames, sample_segment)
        group_mae.append((mae, sample_video_path))

    sorted_group_mae = sorted(group_mae, key=lambda x: x[0])
    lowest_20_maes = sorted_group_mae[:20]
    lowest_20_maes_by_group[gt_identity] = lowest_20_maes

    print(f"\nIdentity: {gt_identity}")
    for mae, path in lowest_20_maes:
        print(f"  Video: {os.path.basename(path)}, MAE: {mae}")

# === Calculate APD per identity ===

apd_values_by_identity = {}
average_group_maes = []

for identity, maes_and_paths in lowest_20_maes_by_group.items():
    group_avg_mae = np.mean([mae for mae, _ in maes_and_paths])
    average_group_maes.append(group_avg_mae)

    video_paths = [path for _, path in maes_and_paths]
    apd = compute_apd_for_videos_efficient(video_paths)
    apd_values_by_identity[identity] = apd

    print(f"\nIdentity: {identity}, Avg MAE: {group_avg_mae}")
    print(f"\nIdentity: {identity}, APD: {apd}")

# Final overall stats
overall_avg_mae = np.mean(average_group_maes)
overall_apd = np.mean(list(apd_values_by_identity.values()))
print(f"\n=== Final Summary ===")
print(f"Overall Average MAE: {overall_avg_mae}")
print(f"Overall Average APD: {overall_apd}")