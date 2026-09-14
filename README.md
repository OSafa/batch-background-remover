### NOTE: This tool's creation was aided with Generative AI usage, specifically using Antigravity 2.0 and Gemini Flash 3.8

## Batch Background Remover
### Description
A batch background removal tool targeting portrait pictures and a pipeline for cropping out an area with a user-defined height, and outputting the files to a .DDS format.
The web UI uses a python FastAPI based python server with server side rendering for the actual interface.


### Requirements
- A GPU which supports CUDA 13.
- Python 3.11 installed on your machine.
- Windows operating system (included .bat files, and the outputting of .DDS files requires windows executables that may not run on other operating systems)

### Installation Procedure
1. Clone this repository, or download the code base as a ZIP File.
2. Install the requirements via setup.bat
3. Download the texconv utility required for DDS format conversion from the [Official Microsoft DirectXTex GitHub repository](https://github.com/microsoft/DirectXTex/releases), and place it in a directory called tools
(i.e.: The final path for the texconv.exe file should be tools/texconv.exe)
4. Download the YOLO 11 pose tensors from the [official Ultralytics YOLO11 HuggingFace repository](https://huggingface.co/Ultralytics/YOLO11/blob/a01aaa06caeff788b052e193acb76b3f21571b3a/yolo11n-pose.pt) and place them in the top level directory
(i.e.: setup.bat and yolo11n-pose.pt should be in the same folder)

## How to run
For the web UI, run `run_gui.bat`.
For the batch cli, use `python cli.py` with the appropriate command line arguments.

### Web UI Samples
<img width="2508" height="1325" alt="image" src="https://github.com/user-attachments/assets/953364f6-9aa7-4e3f-bc1c-b1b4e34aa705" />

<img width="2508" height="1323" alt="image" src="https://github.com/user-attachments/assets/9cbd4dc4-2d82-43fd-8c73-196fd0b8f65f" />

<img width="2508" height="1324" alt="image" src="https://github.com/user-attachments/assets/2198257a-2e73-498a-b8cf-be50757792ae" />

Image of former US President taken from [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:President_Barack_Obama.jpg) and is used as per allowed by images in the Public Domain.
