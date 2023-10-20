import re, os, json
import cv2
# import youtube_dl
import numpy as np
from pytube import Playlist, YouTube
from tqdm import tqdm

# Function to download and process a video from a YouTube playlist
def process_youtube_playlist(playlist_url, output_folder, skip_first_n=1):
    playlist = Playlist(playlist_url)
    playlist._video_regex = re.compile(r"\"url\":\"(/watch\?v=[\w-]*)")

    for idx, video_url in enumerate(playlist.video_urls):
        if idx < skip_first_n: continue
        video = YouTube(video_url)
        # print(dir(video))
        # print(json.dumps(video.vid_info, indent=4))
        vid_duration = float(video.vid_info["videoDetails"]["lengthSeconds"])
        print(vid_duration)
        video_stream = video.streams.filter(file_extension="mp4").get_by_itag("137")

        if video_stream:
            print(f'Downloading: {video.title}')
            video_stream.download(output_path=output_folder)

            # Process the downloaded video
            video_filename = os.path.join(output_folder, video.title + ".mp4")
            process_video(video_filename, vid_duration)
            break

# def frame_score(frame: np.ndarray) -> int:
def frame_score(frame_array, stride, h, w, t, frame_filename):
    # Make the lower right rectangle of width (h, w) as 0
    frame_array[-h:, -w:] = np.zeros(3, dtype=np.uint8)
    # cv2.imshow("", frame_array)
    # Pad the image with zeros in all directions up to stride
    pad_width = stride
    # frame_array = np.pad(frame_array, pad_width, mode='constant', constant_values=np.zeros(3, dtype=np.uint8))

    grayscale_image = cv2.cvtColor(frame_array, cv2.COLOR_BGR2GRAY)
    # Apply max filter to each (stride*2 + 1) x (stride*2 + 1) box
    max_filtered = cv2.dilate(grayscale_image, np.ones((stride * 2 + 1, stride * 2 + 1), dtype=np.uint8))

    cv2.imwrite(frame_filename, grayscale_image)
    # Make grayscale

    # Binarize the image with threshold t
    _, binary_image = cv2.threshold(grayscale_image, t, 255, cv2.THRESH_BINARY)

    # print(frame_array.shape)
    # input()
    # Sum the values of each pixel
    pixel_sum = np.sum(binary_image)

    return pixel_sum
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


def iter_or_cnt_frames(video_filename, iter_fn=None, skip_frames = 0):
    cap = cv2.VideoCapture(video_filename)
    frame_count = 0
    if iter_fn is None: 
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        return frame_count
    with tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))) as pbar:
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break
            
            # Process the frame (e.g., apply additional operations as needed)
            if frame_count >= skip_frames:
                frame_array = np.array(frame)
                frame_filename = os.path.join(output_folder, f'frame_{frame_count:06d}.jpg')
                # cv2.imwrite(frame_filename, frame_array)

                iter_fn(frame_array, 1, 300, 400, 150, frame_filename)
            # Save the frame array or do further processing here

            frame_count += 1
            pbar.update(1)

        cap.release()

    print(f'Processed {frame_count} frames from {video_filename}')
    return frame_count

# Function to parse a video into frames and represent each frame as a NumPy array
def process_video(video_filename, vid_duration):
    frame_cnt = iter_or_cnt_frames(video_filename)
    frame_rate  = frame_cnt // vid_duration
    print(frame_rate)
    iter_or_cnt_frames(video_filename, frame_score, 600*frame_rate)
    # print(frame_score(frame_array, 3, 300, 400, 200, frame_filename))
    

if __name__ == '__main__':
    playlist_url = 'https://www.youtube.com/playlist?list=PLyqSpQzTE6M_ax2pAhetbzpZC5YNjOnMX'  # Replace with your playlist URL
    output_folder = 'output'  # Output folder to store downloaded videos

    process_youtube_playlist(playlist_url, output_folder)
