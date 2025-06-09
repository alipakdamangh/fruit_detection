1.Install python 3.12.10, cudatoolkit 12.2.2, cuDNN 9.9.0
* note that installing cuda libreries is not neccessary
as the pytorch installation automatically adds and uses it*
*the listed requirements and pytorch libreries that are used
here are compatible with any version of the python 3.12 installtion,
I've started the project with 3.12.0 and later updated to 3.12.10
without any chagnes or issues but for the sake of less bugs and stability
use the latest realese (at the time of writing this 3.12.10) of python 3.12*

2.Verify installation with
"python --version"  
"pip --version"

3.Update pip:
"python.exe -m pip install --upgrade pip"

4. Create a virtual environment (while in the project directory):
"py -3.12 -m venv .venv"

5.Activate the virtual environment:
".venv\Scripts\activate"

6.Select the python interpreter with .venv and the correct python version

7.Install the pytorch version with gpu support first so that ultralytics won't install a conflicting package:
pip3 install --default-timeout=1000 torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

"If the download fails try clearing the pip cache and the already installed torch libs and try downloading again:
"pip uninstall torch torchvision torchaudio" => just in case they are installed
"pip cache purge"

8.Install the package: 
"pip install -r requirements.txt"
*install the cpu version of torch
if the gpu version is not compatible 
with your hardware with the commadn below*
"pip install torch"

9.download the yolov8 model version you want (n,s,m,l) => nano, small, medium, large
*the smaller the model the faster it is but for me
the best balance between accuracy and speed is the s model*

10.dowload ip cam for phone camera ip generation
*or any other camera app that gives you
an video output through a static ip address*

11.download arduino ide, selet the board and the port in the ide and upload the arduino board script 
*change the port in the code if its different*

12.The empty folders dataset, figures, yolov8_models are placeholders for their 
own repective data as mentioned bellow: 
- dataset folder should contain these folders: train, test, val *you can ignore test and only use val*
- figures folder will contain the drawn matric and metrics after the model_training.py sceript is executed
- yolov8_models folder should contain the files for the downloded yolov8 models (n,s,m,l, ...)