import os
import cv2
import sys
import time
import queue
import logging
import schedule
import threading
import http.server
import socketserver
import numpy as np

from pathlib import Path
from datetime import datetime, timedelta


# -----------------------CONFIG

# INSERT Local IPv4 of your camera
CONFIG_ATOM_CAM_IP = "192.168.0.45"

# INSERT path to video file if proccessing files (+ ".mp4")
CONFIG_VIDEOFILE_PATH = None

# EDIT Camera's exposure time in seconds: eat CPU Usage
CONFIG_EXPOSURE_TIME = 8

# INSERT output file's directory, None == current dir
CONFIG_OUTPUT_DIRECTORY = "/output_files"

# EDIT start of observation (2000 = 20:00 )
# USE "xxxx" to don't use scheduler and start immidiatelly
CONFIG_PLANED_START = "2000"

# EDIT end of observation (0500 = 05:00 )
CONFIG_PLANED_END = "0500"

# EDIT HOGH_LINES algorithm lines size in mm
CONFIG_MIN_HOGH_LINES = 40

# USE GPU
CONFIG_USE_OPENCL = False

# IGNORE warnings in Stderr from FFmpeg
CONFIG_NO_FFMPEG_WARNING = True

# SPIN LAN http.server for file access (access with http://IP:PORT)
CONFIG_USE_HTTP_SERVER = True

# PORT for htttp.server
CONFIG_HTTP_SERVER_PORT= 9999

# USE Complex difference function (usually more sensitive)
CONFIG_USE_COMPLEX_DIFFERENCE = False

# LOGGING config
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [ %(levelname)s ] - %(message)s')

# -----------------------/CONFIG

# GLOBAL VARIABLE FOR HTTP-SERVER (DON'T TOUCH :P)
GLOBAL_HTTP_SERVER = None

def get_substract_difference(img_list, mask):
    """
    Computes simple differences between consecutive frames, with optimization.
    Applies the mask to the input images via bitwise_or before subtraction.

    Args:
      img_list: List of sequential frames
      mask: Binary mask (same size as image, 1-channel or 3-channel)

    Returns:
      A list of difference images.
    """
    diff_results = []
    num_frames = len(img_list)

    if num_frames < 2:
        return []

    processed_mask = None
    if mask is not None:
        if mask.ndim == 2 and img_list[0].ndim == 3:
            processed_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        else:
            processed_mask = mask

    for i in range(num_frames - 1):
        img1 = img_list[i]
        img2 = img_list[i+1]

        if processed_mask is not None:
            img1 = cv2.bitwise_or(img1, processed_mask) # 
            img2 = cv2.bitwise_or(img2, processed_mask)

        frame_diff = cv2.subtract(img1, img2) # 
        diff_results.append(frame_diff)

    return diff_results


def get_absdiff_difference(img_list, mask):
    """
    Computes absolute differences between consecutive frames,
    with optimized mask pre-processing and efficient operations.
    """
    num_frames = len(img_list)
    if num_frames <= 1:
        return []

    diff_results = []
    
    inverted_mask = None
    if mask is not None:
        if mask.ndim == 2:
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        inverted_mask = cv2.bitwise_not(mask) # 

    for i in range(num_frames - 1):
        img1 = img_list[i]
        img2 = img_list[i+1]

        frame_diff = cv2.absdiff(img1, img2) # 

        if inverted_mask is not None:
            frame_diff = cv2.bitwise_and(frame_diff, inverted_mask)

        diff_results.append(frame_diff)

    return diff_results



def get_brightest(img_list):
    """
    Compare for brightest picture

    Args:
      img_list: lists of photos

    Returns:
      Brightest picture
    """
    if not img_list:
        return None
    return np.maximum.reduce(img_list)


def detect_hough_lines(img, length):
    """
    Detect meteors based on line-shaped diffrences in photos.

    Args:
      img: target of processins
      length: minimal pixel size for detections using HoughLinesP
    Returns:
      Detection result
    """
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    canny = cv2.Canny(blur, 100, 200, 3)

    return cv2.HoughLinesP(canny, 1, np.pi/180, 25, minLineLength=length, maxLineGap=5)



class AtomCamera:
    def __init__(self, video_url, 
                 output=None, end_time="0500",
                 minLineLength=30, opencl=False):
        self._running = False
        self.capture = None
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 10 # Max attempts before giving up
        self._base_reconnect_delay = 5   # Initial delay in seconds (2^0 * 5s)
        self._max_reconnect_delay = 60   # Maximum delay between retries

        self.opencl = opencl
        self.url = video_url

        self.connect()
        if not self.capture:
            logging.error("Initial camera connection failed. Exiting.")
            raise RuntimeError("Initial camera connection failed after multiple attempts.")
        
        self.FPS = min(int(self.capture.get(cv2.CAP_PROP_FPS)), 60)
        self.HEIGHT = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.WIDTH = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))

        if output:
            output_dir = Path(output)
            output_dir.mkdir(exist_ok=True)
        else:
            output_dir = Path('.')
        self.output_dir = output_dir
        self.mp4 = Path(video_url).suffix == '.mp4'

        now = datetime.now()
        t = datetime.strptime(end_time, "%H%M")
        self.end_time = datetime(
            now.year, now.month, now.day, t.hour, t.minute)
        if now > self.end_time:
            self.end_time = self.end_time + timedelta(hours=24)

        logging.info(f"Planed end at: {self.end_time}")
        self.now = now

 
        # Create a mask for Timestamp
        if self.opencl:
            zero = cv2.UMat((self.HEIGHT, self.WIDTH), cv2.CV_8UC3)
        else:
            zero = np.zeros((self.HEIGHT, self.WIDTH, 3), np.uint8) 

        # mask ATOM Cam timestamp
        self.mask = cv2.rectangle(
            zero, (1390, 1010), (self.HEIGHT, self.WIDTH), (255, 255, 255), -1)

        self.min_length = minLineLength
        self.image_queue = queue.Queue(maxsize=200)


    def close(self):
        """Explicitly releases all resources held by the AtomCam instance."""
        logging.info(f"Closing AtomCam resources at: {datetime.now().strftime('%Y/%m/%d %H:%M:%S')}")
        if self.capture:
            self.capture.release()
            self.capture = None
        self._running = False


    def connect(self):
        """
        Attempts to connect to the video source with exponential backoff.
        Resets _reconnect_attempts on success.
        """
        if self.capture:
            self.capture.release()
            self.capture = None

        while True:
            try:
                logging.info(f"Attempting to connect to camera: {self.url}")
                self.capture = cv2.VideoCapture(self.url)

                if self.capture.isOpened():
                    logging.info("Camera connection successful.")
                    self._reconnect_attempts = 0
                    return True
                else:
                    logging.warning("Failed to open camera. Retrying...")

            except Exception as e:
                logging.error(f"Error during camera connection attempt: {e}")

            self._reconnect_attempts += 1
            if self._reconnect_attempts > self._max_reconnect_attempts:
                logging.critical(f"Max reconnect attempts ({self._max_reconnect_attempts}) reached. Giving up.")
                self._running = False
                return False

            delay = min(self._base_reconnect_delay * (2 ** (self._reconnect_attempts -1)), self._max_reconnect_delay)
            logging.info(f"Retrying connection in {delay:.1f} seconds (attempt {self._reconnect_attempts}/{self._max_reconnect_attempts})...")
            time.sleep(delay)


    def stop(self):
        self._running = False


    def queue_streaming(self):
        """
        Connect RTSP in thread and add data to queue
        """
        logging.info("Threaded streaming started.")

        frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT)) if self.mp4 else -1
        self._running = True

        while self._running:
            try:
                ret, frame = self.capture.read()

                if ret:
                    # GPU
                    if self.opencl:
                        frame = cv2.UMat(frame)

                    try:
                        self.image_queue.put((datetime.now(), frame), timeout=1) # Add timeout to prevent indefinite block
                        self._reconnect_attempts = 0 # Reset attempts on successful frame read
                    except queue.Full:
                        logging.warning("Image queue is full, dropping frame.")
                        # Optionally, try to get and discard the oldest frame:
                        # self.image_queue.get_nowait()
                        # self.image_queue.put((datetime.now(), frame))

                    # if MP4 file provided
                    if self.mp4 and int(self.capture.get(cv2.CAP_PROP_POS_FRAMES)) >= frame_count:
                        logging.info("End of MP4 file reached.")
                        self._running = False
                        break # Break the loop if MP4 ends
                else:
                    logging.warning("Failed to read frame from camera. Attempting to reconnect...")
                    # If read fails, try to reconnect using the backoff logic
                    if not self.connect():
                        logging.critical("Failed to reconnect after multiple attempts. Stopping streaming.")
                        self._running = False
                        break # Exit the loop if reconnection fails completely

            except cv2.error as e:
                logging.error(f"OpenCV error during streaming: {e}. Attempting to reconnect...")
                if not self.connect():
                    logging.critical("Failed to reconnect after OpenCV error. Stopping streaming.")
                    self._running = False
                    break
            except Exception as e:
                logging.error(f"An unexpected error occurred in queue_streaming: {e}. Attempting to reconnect...")
                if not self.connect():
                    logging.critical("Failed to reconnect after unexpected error. Stopping streaming.")
                    self._running = False
                    break
        logging.info("Queue streaming thread stopped.")


    def dequeue_streaming(self, exposure=1):
        """
        Take data out fom queue, detect meteors and draw it
        """
        num_frames = int(self.FPS * exposure)
        logging.info(f"Dequeuing streaming with {num_frames} frames per exposure.")

        while self._running:
            img_list = []
            t0 = None

            for n in range(num_frames):
                try:
                    (t, frame) = self.image_queue.get(timeout=5)
                except queue.Empty:
                    logging.warning("Queue is empty, waiting for more frames...")
                    if not self._running and self.image_queue.empty():
                        logging.info("Producer stopped and queue is empty. Exiting dequeue_streaming.")
                        break
                    continue # Keep waiting if producer is still running

                if len(img_list) == 0:
                    t0 = t
                    img_list.append(frame)
                else:
                    dt = t - t0
                    if dt.total_seconds() < exposure: # Use total_seconds for timedelta comparison
                        img_list.append(frame)
                    else:
                        # Stop if exposure time exceeded, process current batch
                        # Put the current frame back for the next batch as it exceeded the exposure
                        self.image_queue.put_nowait((t, frame)) # Put it back without blocking
                        break

            if len(img_list) > 2:
                self.composite_img = get_brightest(img_list)
                self.detect_meteor(img_list)

            # End of session for streaming mode
            now = datetime.now()
            if not self.mp4 and now > self.end_time:
                logging.info(f"End of observation at: {now}")
                self._running = False
                break

        logging.info("Dequeue streaming thread stopped.")


    def detect_meteor(self, img_list):
        """
        Detect meteors from a list of images.
        """
        now = datetime.now()

        if len(img_list) <= 2:
            logging.debug("Not enough frames for differencing.")
            return

        if CONFIG_USE_COMPLEX_DIFFERENCE:
            diff_img = get_brightest(get_absdiff_difference(img_list, self.mask))

        else:
            diff_img = get_brightest(get_substract_difference(img_list, self.mask))

        try:
            detected = detect_hough_lines(diff_img, self.min_length)
            if detected is not None:
                logging.info("A possible meteor was detected.")

                filename = now.strftime("%Y%m%d%H%M%S")
                image_path = self.output_dir / f"{filename}.jpg"
                if not cv2.imwrite(str(image_path), self.composite_img):
                    logging.error(f"Failed to save image: {image_path}")
                else:
                    logging.info(f"Image saved: {image_path}")

                video_path = self.output_dir / f"movie-{filename}.mp4"
                self.save_movie(img_list, str(video_path))

        except Exception as e:
            logging.error(f"Exception occurred during meteor detection: {e}")


    def save_movie(self, img_list, pathname):
        """
        Create a video file from list of photos

        Args:
          imt_list: list of photos
          pathname: output path name
        """
        size = (self.WIDTH, self.HEIGHT)
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

        video = cv2.VideoWriter(pathname, fourcc, self.FPS, size)

        if not video.isOpened():
            logging.error(f"Failed to open video writer for {pathname}. Check codec/permissions.")
            return
        
        for img in img_list:
            if img is not None:
                if self.opencl and isinstance(img, cv2.UMat):
                    try:
                        video.write(img.get())
                    except cv2.error as e:
                        logging.warning(f"Failed to write UMat to video, trying as numpy array. Error: {e}")
                        video.write(img.get())
                else:
                    video.write(img)
            else:
                logging.warning("Skipping None frame during video saving.")
        video.release()
        logging.info(f"Video saved: {pathname}")


# ----------------------------------------------------------------


def start_http_server(output_dir, port=8000):
    """
    Starts an HTTP server to serve files from the specified directory.
    Stores the server instance in GLOBAL_HTTP_SERVER.
    """
    global GLOBAL_HTTP_SERVER

    serve_directory = Path(output_dir).resolve()
    serve_directory.mkdir(parents=True, exist_ok=True)
    original_working_directory = Path.cwd()

    try:
        os.chdir(serve_directory)
        logging.info(f"HTTP SERVER: Serving files from: {serve_directory.resolve()} on port {port}")

        Handler = http.server.SimpleHTTPRequestHandler
        # Create the server instance and store it
        GLOBAL_HTTP_SERVER = socketserver.TCPServer(("", port), Handler)
        logging.info("HTTP SERVER: started. Press Ctrl+C to stop main program.")
        GLOBAL_HTTP_SERVER.serve_forever() # This will block the thread
    except KeyboardInterrupt:
        logging.info("HTTP SERVER: thread received KeyboardInterrupt (this should be handled by main thread).")
    except Exception as e:
        logging.error(f"HTTP SERVER: error: {e}")
    finally:
        # Check if the server was successfully created before attempting to shutdown
        if GLOBAL_HTTP_SERVER:
            pass

        os.chdir(original_working_directory)
        logging.info(f"HTTP SERVER: thread: Changed back to original directory: {original_working_directory}")


def initialize():
    output_directory_absolute_path: Path

    if CONFIG_OUTPUT_DIRECTORY:
        output_directory_absolute_path = Path(CONFIG_OUTPUT_DIRECTORY).resolve()
    else:
        # If CONFIG_OUTPUT_DIRECTORY is None, use the current working directory's absolute path
        output_directory_absolute_path = Path.cwd().resolve()

    output_directory_absolute_path.mkdir(parents=True, exist_ok=True)

    http_server_thread = None
    if CONFIG_USE_HTTP_SERVER:
        if not isinstance(CONFIG_HTTP_SERVER_PORT, int):
            logging.error(f"Port number '{CONFIG_HTTP_SERVER_PORT}' is not an integer.")
            raise RuntimeError
        
        if CONFIG_HTTP_SERVER_PORT < 1 or CONFIG_HTTP_SERVER_PORT > 65535:
            logging.error(f"Port number {CONFIG_HTTP_SERVER_PORT} is out of valid range (1-65535).")
            raise RuntimeError
        
        http_server_thread = threading.Thread(
            target=start_http_server,
            args=(str(output_directory_absolute_path), CONFIG_HTTP_SERVER_PORT),
            name="HTTPServerThread"
        )

        http_server_thread.daemon = True
        http_server_thread.start()


    if CONFIG_VIDEOFILE_PATH:
        url = CONFIG_VIDEOFILE_PATH
    else:
        url = f"rtsp://6199:4003@{CONFIG_ATOM_CAM_IP}/live"

    atom = None
    t_in = None

    try:
        atom = AtomCamera(url, str(output_directory_absolute_path), CONFIG_PLANED_END, CONFIG_MIN_HOGH_LINES, CONFIG_USE_OPENCL)

        logging.info(f"Starting observation at: {datetime.now().strftime('%Y/%m/%d %H:%M:%S')}")

        t_in = threading.Thread(target=atom.queue_streaming, name="CaptureThread")
        t_in.start()

        atom.dequeue_streaming(CONFIG_EXPOSURE_TIME)
    except RuntimeError as e: # Catch the specific exception raised in __init__
        logging.critical(f"Application failed to start: {e}")

    except KeyboardInterrupt:
        logging.info("KeyboardInterrupt detected. Signaling threads to stop.")
        if atom:
            atom.stop()

    except Exception as e:
        logging.critical(f"An unhandled error occurred in start_record: {e}")
        if atom:
            atom.stop()

    finally:
        if atom:
            logging.info("Waiting for capture thread to finish...")
            if t_in and t_in.is_alive():
                t_in.join(timeout=10)
                if t_in.is_alive():
                    logging.warning("Capture thread did not terminate cleanly within timeout.")
            atom.close()

        global GLOBAL_HTTP_SERVER # Access the global variable
        if CONFIG_USE_HTTP_SERVER and GLOBAL_HTTP_SERVER:
            logging.info("Shutting down HTTP server gracefully...")
            GLOBAL_HTTP_SERVER.shutdown()
            http_server_thread.join(timeout=5)
            if http_server_thread.is_alive():
                logging.warning("HTTP server thread did not terminate cleanly within timeout.")
            else:
                logging.info("HTTP server successfully shut down.")

        logging.info("Application shutdown complete.")



if __name__ == '__main__':
    if CONFIG_NO_FFMPEG_WARNING:
        os.environ['OPENCV_FFMPEG_LOGLEVEL'] = 'quiet'
    
    if CONFIG_PLANED_START == "xxxx":
        initialize()

    else:
        try:
            dt_object = datetime.strptime(CONFIG_PLANED_START, "%H%M")
            time_str_hh_mm = dt_object.strftime("%H:%M")

        except ValueError as e:
            logging.error(f"Error parsing time '{CONFIG_PLANED_START}': {e}")
            raise RuntimeError

        schedule.every().day.at(time_str_hh_mm).do(initialize)
        while True:
            schedule.run_pending()
            time.sleep(1)
