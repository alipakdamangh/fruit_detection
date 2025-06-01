1.Install python 3.12.0, cudatoolkit 12.2.2, cuDNN 9.9.0:

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
* install the cpu version of torch if the gpu version is not compatible with your hardware (it's not listed in the requirements):
"pip install torch"

9.download the yolov8 model version you want (n,s,m,l) => nano, small, medium, large

10.dowload ip cam for phone camera ip generation

11.download arduino ide, selet the board and the port in the ide and upload the arduino board script 
(change the port in the code if its different)