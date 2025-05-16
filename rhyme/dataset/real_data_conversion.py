from typing import Sequence, Tuple, Dict, Optional, Union
import os
import pathlib
import numpy as np
import av
import zarr
import numcodecs
import multiprocessing
import concurrent.futures
from tqdm import tqdm
from rhyme.common.replay_buffer import ReplayBuffer, get_optimal_chunks
from rhyme.common.cv2_util import get_image_transform
from rhyme.dataset.video_recorder import read_video
from rhyme.codecs.imagecodecs_numcodecs import (register_codecs, Jpeg2k)

register_codecs()

def check_and_process_npz(directory, npz_files, is_human=False):
    # Path to the stats.npz file
    stats_path = os.path.join(directory, "stats.npz")

    # If the file exists, load episode_starts and episode_lengths from it
    if os.path.isfile(stats_path):
        print(f"'stats.npz' exists, loading data...")
        data = np.load(stats_path, allow_pickle=True)
        episode_starts = data['episode_starts'].tolist()
        episode_lengths = data['episode_lengths'].tolist()

    else:
        print(f"'stats.npz' does not exist, creating and saving data...")
        episode_starts = []
        episode_lengths = []
        start = 0
        
        # Iterate over the npz files and compute starts and lengths
        for episode_idx, f in enumerate(npz_files):
            raw_episode = np.load(f, allow_pickle=True)["episode" if not is_human else "human_video"]
            episode_starts.append(start)
            episode_length = len(raw_episode)
            episode_lengths.append(episode_length)
            start += episode_length
        
        # Save episode_starts and episode_lengths to stats.npz
        np.savez(stats_path, episode_starts=episode_starts, episode_lengths=episode_lengths)

    return episode_starts, episode_lengths

def get_all_files(root, file_extension, contain=None) -> list[str]:
    files = []
    for folder, _, fs in os.walk(root):
        for f in fs:
            if file_extension is not None:
                if f.endswith(file_extension):
                    if contain is None or contain in os.path.join(folder, f):
                        files.append(os.path.join(folder, f))
            else:
                if contain in f:
                    files.append(os.path.join(folder, f))
    return files



def real_data_to_replay_buffer(
        dataset_path: str,
        out_store: Optional[zarr.ABSStore] = None,
        out_resolutions: Union[None, tuple, Dict[str, tuple]] = None,  # (width, height)
        lowdim_keys: Optional[Sequence[str]] = None,
        image_keys: Optional[Sequence[str]] = None,
        lowdim_compressor: Optional[numcodecs.abc.Codec] = None,
        image_compressor: Optional[numcodecs.abc.Codec] = None,
        n_decoding_threads: int = multiprocessing.cpu_count(),
        n_encoding_threads: int = multiprocessing.cpu_count(),
        max_inflight_tasks: int = multiprocessing.cpu_count() * 5,
        read_top_n = None,
        verify_read: bool = True) -> ReplayBuffer:
    """
    It is recommended to use before calling this function
    to avoid CPU oversubscription
    cv2.setNumThreads(1)
    threadpoolctl.threadpool_limits(1)

    out_resolution:
        if None:
            use video resolution
        if (width, height) e.g. (1280, 720)
        if dict:
            camera_0: (1280, 720)
    image_keys: ['camera_0', 'camera_1']
    """
    if out_store is None:
        out_store = zarr.MemoryStore()
    if n_decoding_threads <= 0:
        n_decoding_threads = multiprocessing.cpu_count()
    if n_encoding_threads <= 0:
        n_encoding_threads = multiprocessing.cpu_count()
    if image_compressor is None:
        image_compressor = Jpeg2k(level=50)

    # verify input
    input = pathlib.Path(os.path.expanduser(dataset_path))
    in_zarr_path = input.joinpath('replay_buffer.zarr')
    in_video_dir = input.joinpath('videos')
    assert in_zarr_path.is_dir()
    assert in_video_dir.is_dir()

    in_replay_buffer = ReplayBuffer.create_from_path(str(in_zarr_path.absolute()), mode='r')

    # save lowdim data to single chunk
    chunks_map = dict()
    compressor_map = dict()
    for key, value in in_replay_buffer.data.items():
        chunks_map[key] = value.shape
        compressor_map[key] = lowdim_compressor

    print('Loading lowdim data')
    out_replay_buffer = ReplayBuffer.copy_from_store(src_store=in_replay_buffer.root.store,
                                                     store=out_store,
                                                     keys=lowdim_keys,
                                                     chunks=chunks_map,
                                                     compressors=compressor_map)

    # worker function
    def put_img(zarr_arr, zarr_idx, img):
        try:
            zarr_arr[zarr_idx] = img
            # make sure we can successfully decode
            if verify_read:
                _ = zarr_arr[zarr_idx]
            return True
        except Exception as e:
            return False

    n_cameras = 0
    camera_idxs = set()
    if image_keys is not None:
        n_cameras = len(image_keys)
        camera_idxs = set(int(x.split('_')[-1]) for x in image_keys)
    else:
        # estimate number of cameras
        episode_video_dir = in_video_dir.joinpath(str(0))
        episode_video_paths = sorted(episode_video_dir.glob('*.mp4'), key=lambda x: int(x.stem))
        camera_idxs = set(int(x.stem) for x in episode_video_paths)
        n_cameras = len(episode_video_paths)

    n_steps = in_replay_buffer.n_steps
    episode_starts = in_replay_buffer.episode_ends[:] - in_replay_buffer.episode_lengths[:]
    episode_lengths = in_replay_buffer.episode_lengths
    
    if read_top_n is not None:
        episode_starts = episode_starts[:read_top_n]
        episode_lengths = episode_lengths[:read_top_n]
        
    timestamps = in_replay_buffer['timestamp'][:]
    dt = timestamps[1] - timestamps[0]
    
    with tqdm(total=n_steps * n_cameras, desc="Loading image data", mininterval=1.0) as pbar:
        # one chunk per thread, therefore no synchronization needed
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_encoding_threads) as executor:
            futures = set()
            for episode_idx, episode_length in enumerate(episode_lengths):
                episode_video_dir = in_video_dir.joinpath(str(episode_idx))
                episode_start = episode_starts[episode_idx]

                episode_video_paths = sorted(episode_video_dir.iterdir(), key=lambda x: int(x.stem))

                print(episode_video_paths)
                for video_path_folder in episode_video_paths:
                    print(video_path_folder)
                    camera_idx = int(video_path_folder.stem)
                    if image_keys is not None:
                        # if image_keys provided, skip not used cameras
                        if camera_idx not in camera_idxs:
                            continue
                    video_path = video_path_folder.joinpath('color.mp4')
                    # read resolution
                    with av.open(str(video_path.absolute())) as container:
                        video = container.streams.video[0]
                        vcc = video.codec_context
                        this_res = (vcc.width, vcc.height)
                    in_img_res = this_res

                    arr_name = f'camera_{camera_idx}'
                    print(arr_name)
                    # figure out save resolution
                    out_img_res = in_img_res
                    if isinstance(out_resolutions, dict):
                        if arr_name in out_resolutions:
                            out_img_res = tuple(out_resolutions[arr_name])
                    elif out_resolutions is not None:
                        out_img_res = tuple(out_resolutions)

                    # allocate array
                    if arr_name not in out_replay_buffer:
                        ow, oh = out_img_res
                        _ = out_replay_buffer.data.require_dataset(name=arr_name,
                                                                   shape=(n_steps, oh, ow, 3),
                                                                   chunks=(1, oh, ow, 3),
                                                                   compressor=image_compressor,
                                                                   dtype=np.uint8)
                    arr = out_replay_buffer[arr_name]

                    image_tf = get_image_transform(input_res=in_img_res, output_res=out_img_res, bgr_to_rgb=False)
                    for step_idx, frame in enumerate(
                            read_video(video_path=str(video_path),
                                       dt=dt,
                                       img_transform=image_tf,
                                       thread_type='FRAME',
                                       thread_count=n_decoding_threads)):
                        if len(futures) >= max_inflight_tasks:
                            # limit number of inflight tasks
                            completed, futures = concurrent.futures.wait(futures,
                                                                         return_when=concurrent.futures.FIRST_COMPLETED)
                            for f in completed:
                                if not f.result():
                                    raise RuntimeError('Failed to encode image!')
                            pbar.update(len(completed))

                        global_idx = episode_start + step_idx
                        futures.add(executor.submit(put_img, arr, global_idx, frame))

                        if step_idx == (episode_length - 1):
                            break
            completed, futures = concurrent.futures.wait(futures)
            for f in completed:
                if not f.result():
                    raise RuntimeError('Failed to encode image!')
            pbar.update(len(completed))
    return out_replay_buffer

def portal_real_data_to_replay_buffer(
        dataset_path: str,
        out_store: Optional[zarr.ABSStore] = None,
        out_resolutions: Union[None, tuple, Dict[str, tuple]] = None,  # (width, height)
        lowdim_keys: Optional[Sequence[str]] = None,
        image_keys: Optional[Sequence[str]] = None,
        lowdim_compressor: Optional[numcodecs.abc.Codec] = None,
        image_compressor: Optional[numcodecs.abc.Codec] = None,
        n_decoding_threads: int = multiprocessing.cpu_count(),
        n_encoding_threads: int = multiprocessing.cpu_count(),
        max_inflight_tasks: int = multiprocessing.cpu_count() * 5,
        read_top_n = None,
        verify_read: bool = True,
        rewrite=False) -> ReplayBuffer:
    """
    It is recommended to use before calling this function
    to avoid CPU oversubscription
    cv2.setNumThreads(1)
    threadpoolctl.threadpool_limits(1)

    out_resolution:
        if None:
            use video resolution
        if (width, height) e.g. (1280, 720)
        if dict:
            camera_0: (1280, 720)
    image_keys: ['camera_0', 'camera_1']
    """
    # TODO: camera name needs to change when doing it from the reconstructed human video
    if out_store is None:
        out_store = zarr.MemoryStore()
    if n_decoding_threads <= 0:
        n_decoding_threads = multiprocessing.cpu_count()
    if n_encoding_threads <= 0:
        n_encoding_threads = multiprocessing.cpu_count()
    if image_compressor is None:
        image_compressor = Jpeg2k(level=50)

    # verify input
    input = pathlib.Path(os.path.expanduser(dataset_path))
    in_video_dir = input.joinpath('videos')
    in_demo_dir = input.joinpath('demos')
    assert in_video_dir.is_dir()
    assert in_demo_dir.is_dir()

    # worker function
    def put_img(zarr_arr, zarr_idx, img):
        try:
            zarr_arr[zarr_idx] = img
            # make sure we can successfully decode
            if verify_read:
                _ = zarr_arr[zarr_idx]
            return True
        except Exception as e:
            return False


    
    # compute episode starts and lengths
    npz_files = list(sorted(get_all_files(in_demo_dir, "npz")))

    episode_starts, episode_lengths = check_and_process_npz(in_demo_dir, npz_files, is_human=False)
    
    n_steps = episode_starts[-1] + episode_lengths[-1]
    n_cameras = 1

    if read_top_n is not None:
        episode_starts = episode_starts[:read_top_n]
        episode_lengths = episode_lengths[:read_top_n]
        
    dt = 0.1 # b/c 10 FPS

    # zarr_path = input.joinpath('replay_buffer.zarr')
    zarr_path = input.joinpath('replay_buffer.zarr')

    
    if zarr_path.exists() and not rewrite:
        # If the Zarr file exists, open it in read-only mode
        print("Zarr file already exists. Opening in read-only mode.")
        zarr_root = zarr.open(zarr_path, mode='r')
        # Return or raise an error as needed, since we don't want to overwrite or modify anything
        return ReplayBuffer(zarr_root)
    else:
        # If the Zarr file does not exist, create it
        zarr_root = zarr.open(zarr_path, mode='w')
        data_group = zarr_root.create_group('data')
        meta_group = zarr_root.create_group('meta')
        episode_ends_group = meta_group.create_group('episode_ends')
        meta_group['episode_ends'] = np.array(episode_starts[:]) + np.array(episode_lengths[:])

    out_replay_buffer = ReplayBuffer(zarr_root)

    # zarr_root = zarr.open(input.joinpath('replay_buffer.zarr'), mode='r+')
    
    # out_replay_buffer = ReplayBuffer(zarr_root)  # 'w' means create a new file or overwrite

    
    with tqdm(total=n_steps * n_cameras, desc="Loading image data", mininterval=1.0) as pbar:
        # one chunk per thread, therefore no synchronization needed
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_encoding_threads) as executor:
            futures = set()
            for episode_idx, episode_length in enumerate(episode_lengths):
                episode_video_dir = in_video_dir.joinpath(str(episode_idx))
                episode_start = episode_starts[episode_idx]

                video_path_folder = episode_video_dir.joinpath('color.mp4')
                episode_video_paths = [video_path_folder]
                print(episode_video_paths)
                for video_path in episode_video_paths:
                    # read resolution
                    with av.open(str(video_path.absolute())) as container:
                        video = container.streams.video[0]
                        vcc = video.codec_context
                        this_res = (vcc.width, vcc.height)
                    in_img_res = this_res

                    arr_name = f'third_person_cam'
                    print(arr_name)
                    # figure out save resolution
                    out_img_res = in_img_res
                    if isinstance(out_resolutions, dict):
                        if arr_name in out_resolutions:
                            out_img_res = tuple(out_resolutions[arr_name])
                    elif out_resolutions is not None:
                        out_img_res = tuple(out_resolutions)

                    # allocate array
                    if arr_name not in out_replay_buffer:
                        ow, oh = out_img_res
                        _ = out_replay_buffer.data.require_dataset(name=arr_name,
                                                                   shape=(n_steps, oh, ow, 3),
                                                                   chunks=(1, oh, ow, 3),
                                                                   compressor=image_compressor,
                                                                   dtype=np.uint8)
                    arr = out_replay_buffer[arr_name]

                    image_tf = get_image_transform(input_res=in_img_res, output_res=out_img_res, bgr_to_rgb=False)
                    for step_idx, frame in enumerate(
                            read_video(video_path=str(video_path),
                                       dt=dt,
                                       img_transform=image_tf,
                                       thread_type='FRAME',
                                       thread_count=n_decoding_threads)):
                        if len(futures) >= max_inflight_tasks:
                            # limit number of inflight tasks
                            completed, futures = concurrent.futures.wait(futures,
                                                                         return_when=concurrent.futures.FIRST_COMPLETED)
                            for f in completed:
                                if not f.result():
                                    raise RuntimeError('Failed to encode image!')
                            pbar.update(len(completed))

                        global_idx = episode_start + step_idx
                        futures.add(executor.submit(put_img, arr, global_idx, frame))

                        if step_idx == (episode_length - 1):
                            break
            completed, futures = concurrent.futures.wait(futures)
            for f in completed:
                if not f.result():
                    raise RuntimeError('Failed to encode image!')
            pbar.update(len(completed))
    return out_replay_buffer