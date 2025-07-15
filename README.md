# **Headless Meteor Detection System**

This Python script provides a real-time meteor detection system using OpenCV. It connects to an RTSP camera feed (or processes a local MP4 file), analyzes frames for meteor-like patterns, and saves detected events as images and video clips. Additionally, it can spin up a local HTTP server to easily view the captured meteor events. Script is focused for headless servers, as it doesn’t show any preview.

## **Features**

* **Scheduled start and end.
* **Real-time Video Processing**: Connects to RTSP streams for continuous monitoring.  
* **MP4 File Processing**: Can also process pre-recorded MP4 video files.  
* **Meteor Detection**: Utilizes multiple subtraction algorithms and Hough Line Transform to identify linear patterns indicative of meteors.  
  Configurable with: \[CONFIG\_USE\_COMPLEX\_DIFFERENCE\] and \[CONFIG\_USE\_BACKGROUND\_SUB\_METHOD\] flags.  
* **Event Saving**: Automatically saves composite images and video clips of detected meteor events.  
* **Robust Camera Connection**: Implements exponential backoff for connection retries to handle network instability.  
* **Configurable Parameters**: Easily adjust detection sensitivity, exposure time, output directory, and more via configuration variables.  
* **Local HTTP Server**: Serves the output directory via a simple HTTP server for convenient access to detected events.  
* **Timestamp Masking**: Automatically masks out the camera's timestamp overlay to prevent false positives.  
* **Graceful Shutdown**: Ensures all threads and resources (camera, HTTP server) are properly closed upon exit.

## **Requirements**

* Python 3.x  
* OpenCV (opencv-python)  
* NumPy
* Schedule

## **Installation**

1. Clone or Download the Script:  
   Save the provided Python script (e.g., atomcam.py) to your local machine.  
2. Install Dependencies:  
   It's highly recommended to use a Python virtual environment.  
   python3 \-m venv venv  
   source venv/bin/activate  \# On Windows: venv\\Scripts\\activate  
   pip install opencv-python numpy

## **Configuration**

All configurable parameters are located at the top of the camera\_atom.py script. Edit these variables directly to suit your setup:

\# Add IP of your camera  
CONFIG\_ATOM\_CAM\_IP \= "192.168.0.45" \# Your camera's RTSP IP address

\# Video source: Set to None for RTSP stream, or provide a path to an MP4 file  
CONFIG\_VIDEOFILE\_PATH \= None \# Example: "/path/to/your/video.mp4"

CONFIG\_EXPOSURE\_TIME \= 5 \# Exposure time in seconds for frame accumulation

\# Output directory for saved images and videos  
CONFIG\_OUTPUT\_DIRECTORY \= "/home/suomi/Desktop/meteor-detect/output\_files" \# Recommended: Use an absolute path

\# Start time for observation (JST) "hhmm" \=\> (ex. 2000 for 8:00 PM)  
CONFIG_PLANED_START = "2000"

\# End time for observation (JST) "hhmm" \=\> (ex. 0500 for 5:00 AM)  
CONFIG\_PLANED\_END \= "0500"

\# Minimum line length in pixels for HoughLinesP detection  
CONFIG\_MIN\_HOGH\_LINES \= 30

\# Use OpenCL for GPU acceleration (requires OpenCV built with OpenCL support)  
CONFIG\_USE\_OPENCL \= False

\# Suppress FFmpeg warnings from OpenCV (these are usually harmless but verbose)  
CONFIG\_NO\_FFMPEG\_WARNING \= True

\# Enable or disable the built-in HTTP server  
CONFIG\_USE\_HTTP\_SERVER \= True  
CONFIG\_HTTP\_SERVER\_PORT \= 8000 \# Port for the HTTP server

\# Enable Complex difference function  
CONFIG\_USE\_COMPLEX\_DIFFERENCE \= False (\*makes detection more sensitive)

\# Enable Advanced Background extraction (\*experimental; do not use during daylight)  
CONFIG\_USE\_BACKGROUND\_SUB\_METHOD \= False

**Important Notes:**

* **CONFIG\_OUTPUT\_DIRECTORY**: It's highly recommended to use an **absolute path** for this variable (e.g., /home/youruser/meteor\_detections). This ensures consistent file saving regardless of where you run the script from or if the HTTP server changes the working directory.  
* **CONFIG\_HTTP\_SERVER\_PORT**: Choose an available port. Common choices are 8000, 8080, 5000, etc. Avoid ports below 1024 as they often require root privileges.

## **Usage**

1. **Configure the script** as described above.  
2. **Run the script** from your terminal:  
   python camera\_atom.py

3. **Monitor Logs**: The script will print log messages to your terminal, indicating connection status, detection events, and file saving.  
4. Access HTTP Server: If CONFIG\_USE\_HTTP\_SERVER is True, you can access the saved images and videos by opening your web browser and navigating to:  
   http://localhost:8000 (replace 8000 with your configured port)  
5. **Stop the Script**: Press Ctrl+C in your terminal to gracefully shut down the application and the HTTP server.

## **How It Works (Briefly)**

1. **Initialization**: The AtomCam class connects to the video source and sets up parameters.  
2. **Frame Capture (queue\_streaming thread)**: A dedicated thread continuously reads frames from the camera/video file and puts them into a queue. It handles connection retries with exponential backoff.  
3. **Frame Processing (dequeue\_streaming loop)**: The main thread retrieves frames from the queue in batches, corresponding to the CONFIG\_EXPOSURE\_TIME.  
4. **Difference and Background Subtraction**: Depending on used flags, script will run simple difference/complex difference and background subtraction for each frame, a background subtractor (MOG2) is applied to generate a foreground mask, highlighting moving objects.  
5. **Masking**: A pre-defined mask is applied to ignore the camera's timestamp overlay.  
6. **Temporal Accumulation**: Foreground masks over the exposure time are combined to create a composite image that emphasizes trails.  
7. **Meteor Detection**: The detect function applies Gaussian blur, Canny edge detection, and cv2.HoughLinesP to find linear patterns (potential meteors) in the processed difference image.  
8. **Saving Events**: If a meteor is detected, the composite image and a short video clip of the event are saved to the CONFIG\_OUTPUT\_DIRECTORY.  
9. **HTTP Server**: If enabled, a separate thread serves the CONFIG\_OUTPUT\_DIRECTORY via a simple HTTP server, allowing you to view the saved files through a web browser.

## **Troubleshooting**

* **HTTP server error: \[Errno 98\] Address already in use**:  
  * The port specified in CONFIG\_HTTP\_SERVER\_PORT is already in use by another application or a previous instance of your script that didn't shut down cleanly.  
  * **Solution 1**: Wait 10-30 seconds after stopping the script before restarting.  
  * **Solution 2**: Change CONFIG\_HTTP\_SERVER\_PORT to a different number (e.g., 8001, 8080).  
  * **Solution 3 (Linux/macOS)**: Find and kill the process using the port:  
    lsof \-i :\<PORT\_NUMBER\>  
    \# (Look for PID in output, then)  
    kill \-9 \<PID\>

* **Initial camera connection failed. Exiting.**:  
  * The script could not connect to your camera's RTSP stream.  
  * **Check CONFIG\_ATOM\_CAM\_IP**: Ensure the IP address is correct and the camera is online and accessible on your network.  
  * **Check RTSP URL**: Verify the full RTSP URL (e.g., rtsp://username:password@IP/live). The default rtsp://6199:4003@IP/live is specific to Atom Cam.  
  * **Firewall**: Ensure no firewall (on your machine or network) is blocking the connection.  
  * **Camera Settings**: Confirm the camera's RTSP server is enabled.  
* **Too Many False Positives (Daylight)**:  
  * Background subtraction (CONFIG\_USE\_BACKGROUND\_SUB\_METHOD flag) struggles with dynamic backgrounds (trees, water, shadows, clouds).  
  * **Adjust MOG2 parameters**: In AtomCam.\_\_init\_\_, experiment with higher history (e.g., 1000 to 2000\) and varThreshold (e.g., 32 to 64\) in cv2.createBackgroundSubtractorMOG2.  
  * **Stronger Blurring**: Increase the kernel size for cv2.GaussianBlur applied before background subtraction (e.g., (7, 7\) or (9, 9)).  
  * **More Aggressive Morphological Operations**: Use larger kernels for cv2.MORPH\_OPEN and cv2.MORPH\_CLOSE after background subtraction to clean up noise (e.g., (5, 5\) for open, (10, 10\) for close).  
  * **Contour Filtering**: Implement additional filtering based on cv2.findContours results (e.g., cv2.contourArea, aspect ratio, solidity) to discard non-meteor shapes.

## **License**

This project is open-source and available under the MIT License.
