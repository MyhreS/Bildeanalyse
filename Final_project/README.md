# .py version
This is the version that runs using .py files on your local computer. It is proved working on a Windows 10 machine. 
Each file is labeled 1, 2, 3, etc. This is the order of how they have been executed.
The 4_training and 5_testing uses weights I have trained from Google Colab when testing. 

NOTE: The '4_training' will require much memory. If you don't have enough then it can't run all at once. This is because of the size of the
dataset used.


# Install
These should be installed in this folder.
```
pip install opencv-python
pip install pandas
pip install matplotlib
pip install imgaug
pip install numpy
pip install torch
git clone https://github.com/ultralytics/yolov5  
pip install -U -r yolov5/requirements.txt
pip install scikit-learn
```