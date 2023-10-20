import re, os, json
import cv2

# import youtube_dl
import numpy as np
from pytube import Playlist, YouTube
from tqdm import tqdm
from queue import Queue
from functools import partial
from concurrent.futures.process import ProcessPoolExecutor
from concurrent.futures import as_completed
from matplotlib import pyplot as plt


# Function to download and process a video from a YouTube playlist
def process_youtube_playlist(playlist_url, output_folder, skip_first_n=1):
    playlist = Playlist(playlist_url)
    playlist._video_regex = re.compile(r"\"url\":\"(/watch\?v=[\w-]*)")

    for idx, video_url in enumerate(playlist.video_urls):
        if idx < skip_first_n:
            continue
        video = YouTube(video_url)
        # print(dir(video))
        # print(json.dumps(video.vid_info, indent=4))
        vid_duration = float(video.vid_info["videoDetails"]["lengthSeconds"])
        print(vid_duration)
        video_stream = video.streams.filter(file_extension="mp4").get_by_itag("137")

        if video_stream:
            print(f"Downloading: {video.title}")
            video_stream.download(output_path=output_folder)

            # Process the downloaded video
            video_filename = os.path.join(output_folder, video.title + ".mp4")
            process_video(video_filename, vid_duration)
            break


dx = [-1, 0, 1, 0]
dy = [0, -1, 0, 1]


def clear_teacher_with_bfs(binary_image):
    q = Queue()
    # vis = np.zeros(binary_image.shape, dtype=bool)
    # print(np.where(binary_image[-1, :] == 0)[0])
    for pos in np.where(binary_image[-1, :] == 0)[0]:
        q.put((binary_image.shape[0] - 1, pos))
        binary_image[binary_image.shape[0] - 1, pos] = 255

    while not q.empty():
        sx, sy = q.get()
        # print(sx, sy)
        for i in range(4):
            nx, ny = sx + dx[i], sy + dy[i]
            if (
                nx < 0
                or ny < 0
                or nx >= binary_image.shape[0]
                or ny >= binary_image.shape[1]
                or binary_image[nx, ny] == 255
                # or vis[nx, ny]
            ):
                continue
            # vis[nx, ny] = 1
            binary_image[nx, ny] = 255
            q.put((nx, ny))

    return binary_image


# def frame_score(frame: np.ndarray) -> int:
def frame_score(frame_array, stride, h, w, t, frame_filename):
    threshold_fn = partial(
        cv2.threshold, thresh=t, maxval=255, type=cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    catch_threshold = lambda image: threshold_fn(image)[1]

    def write_image(image):
        cv2.imwrite(frame_filename, image)
        return image

    sm = 6
    transforms = [
        partial(np.min, axis=2),
        partial(
            cv2.morphologyEx,
            op=cv2.MORPH_ERODE,
            kernel=np.ones((stride * sm + 1, stride * sm + 1), dtype=np.uint8),
            iterations=1,
        ),
        catch_threshold,
        clear_teacher_with_bfs,
        partial(
            cv2.morphologyEx,
            op=cv2.MORPH_DILATE,
            kernel=np.ones((stride * sm + 1, stride * sm + 1), dtype=np.uint8),
            iterations=1,
        ),
        # write_image,
        np.sum,
    ]
    val = frame_array
    for f in transforms:
        val = f(val)

    return val


# You can call this function with your frame array, stride, rectangle dimensions (h and w), and threshold (t) to calculate the sum of pixel values as described in your request. Here's how you can use it:

# python
# Copy code
# # Example usage
# frame_array = np.random.randint(0, 256, size=(200, 200, 3), dtype=np.uint8)  # Replace with your frame array
# stride = 5
# h = 20
# w = 30
# t = 100

# result = frame_score(frame_array, stride, h, w, t)
# print("Sum of pixel values:", result)
# Replace the frame_array variable with your actual frame array, and adjust the values of stride, h, w, and t as needed. The function will return the sum of pixel values after performing the specified operations on the frame.


# Is this conversation helpful so far?


def iter_or_cnt_frames(video_filename, iter_fn=None, skip_frames=0):
    cap = cv2.VideoCapture(video_filename)
    frame_count = 0
    frame_scores = []
    if iter_fn is None:
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        return frame_count
    with tqdm(
        total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), desc="Submit frames"
    ) as pbar, ProcessPoolExecutor(max_workers=1, max_tasks_per_child=1) as executor:
        futures = []
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            # Process the frame (e.g., apply additional operations as needed)
            if frame_count >= skip_frames:
                frame_array = np.array(frame)
                frame_filename = os.path.join(
                    output_folder, f"frame_{frame_count:06d}.jpg"
                )
                # cv2.imwrite(frame_filename, frame_array)
                futures.append((frame_count, executor.submit(iter_fn, frame_array, 1, 300, 400, 127, frame_filename)))
                # futures.append((frame_count, iter_fn(frame_array, 1, 300, 400, 127, frame_filename)))
                # print(
                #     iter_fn(frame_array, 1, 300, 400, 127, frame_filename)
                #     / (frame_array.shape[0] * frame_array.shape[1])
                # )
            # Save the frame array or do further processing here

            frame_count += 1
            pbar.update(1)

        cap.release()
        for fram_cnt, fut in tqdm(as_completed(futures), desc="Receiving frames"):
            frame_scores.append((fram_cnt, fut.result() / (frame_array.shape[0] * frame_array.shape[1])))
    
    x, y = zip(*frame_scores)
    plt.plot(x, y)
    plt.show()
    print(f"Processed {frame_count} frames from {video_filename}")
    return frame_scores


# Function to parse a video into frames and represent each frame as a NumPy array
def process_video(video_filename, vid_duration):
    frame_cnt = iter_or_cnt_frames(video_filename)
    frame_rate = frame_cnt // vid_duration
    # print(frame_rate)
    iter_or_cnt_frames(video_filename, frame_score, 15 * frame_rate)
    # print(frame_score(frame_array, 3, 300, 400, 200, frame_filename))


if __name__ == "__main__":
    playlist_url = "https://www.youtube.com/playlist?list=PLyqSpQzTE6M_ax2pAhetbzpZC5YNjOnMX"  # Replace with your playlist URL
    output_folder = "output"  # Output folder to store downloaded videos

    process_youtube_playlist(playlist_url, output_folder)
