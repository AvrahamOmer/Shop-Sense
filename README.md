# SHOP SENSE - SERVER SIDE

## ABOUT

Welcome to the Server Side README for the Shop Sense Client-Server System! In this document, we'll provide an overview of the server-side application, its functionality, and how to set it up.

## GETTING STARTED

### Prerequisites

Before you begin, ensure you have met the following requirements:

- **Python**: Make sure you have Python installed. You can download it from [python.org](https://www.python.org/downloads/).

### Installation

1. **Clone the Repository**:

   - Clone the Shop Sense Server Side repository to your local machine.

   ```bash
   git clone https://github.com/YourUsername/shop-sense-server.git
   ```

2. **Navigate to the Project Directory**:

   - Change your current directory to the project folder.

   ```bash
   cd shop-sense-server
   ```

3. **Create a Virtual Environment** (optional but recommended):

   - Create a virtual environment to isolate project dependencies.

   ```bash
   python -m venv venv
   ```

4. **Activate the Virtual Environment** (Linux/macOS):

   ```bash
   source venv/bin/activate
   ```

   On Windows:

   ```bash
   venv\Scripts\activate
   ```

5. **Install Dependencies**:

   - Install the required Python dependencies.

   ```bash
   pip install -r requirements.txt
   ```

6. **Install FFmpeg**:

   - To work with video processing, you'll need FFmpeg. Here's how to install it:

     - On macOS, you can use Homebrew:

       ```bash
       brew install ffmpeg
       ```

     - On Windows, you can follow the instructions on the [FFmpeg website](https://ffmpeg.org/download.html) to download and install it.
     - After installation, add the path to the FFmpeg executable in the lib.py file. You can do this by adding the following lines to your Python code before importing moviepy:

       ```python
       import os
       os.environ["IMAGEIO_FFMPEG_EXE"] = "/path/to/ffmpeg"
       ```

### Usage

1. **Start the Server**:

   - Start the Python server application.

   ```bash
   python api.py
   ```

   The server should now be running and listening for incoming requests.

2. Go to the client side to start the client app: [SHOP-SENSE-WEB](https://github.com/AvrahamOmer/Shop-Sense-web)

3. **Object Tracking Configuration**:

   - The server includes an object tracking module that can be configured based on your needs. You can modify the following settings in the object tracking file:

   ```python
   duration = 1  # Time in seconds
   skip_detect = 5  # Do object detection every n frames (to not skip any frame, set skip_detect = 1)
   desired_interval = 2  # Take every n frames (to not skip any frame, set desired_interval = 1)
   ```

   - Adjust the values of `duration`, `skip_detect`, and `desired_interval` according to your specific requirements. These settings control the behavior of the object tracking process.

4. **Sort Configuration**:

   - The server includes a tracking model - SORT that can be configured based on your needs. You can modify the following settings in the object tracking file:

   ```python
    max_age = 2
    min_hits = 3
    iou_threshold = 0.3

   ```
