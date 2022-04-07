# Assigment 2 Guide
All the files that is runnable is created in jupiter notebooks meant to run on Google Colab. This is because it was a huge amount of dependencies that could make testing the code difficult. So by having everything run on Google Colab that problem was solved.

This is simple to setup. Just follow this little guide.

## Setup guide
### Step 1:
Add "Bildeanalyse-assignment2-resources.zip" file to your Google drive. Make sure it is in the root directory else you will have to change the path in the nootebooks. Do not unzip it manually. This is done in the notebooks. This .zip file contains the data (images) and my trained models.
### Step 2:
Upload all the jupiter notebooks to Google Colab
### Step 3:
Open the notebooks that is now on Google Colab. For how to run each of them follow the guide under the "Setup" headline in the notebooks. They are pretty much plug and play. They will all promt you for  access to your Google drive to be able to fetch the .zip file that they all use.

## Other information
This folder also contains the directory "Tools_used". These are just code I used to annotate and resize the images. Its there only for proof of work.

I followed guides on how to do YOLO and RCNN object detection. Much of the code comes from following them. The links to the guides is adden in each of the notebooks.

NOTE:
If some code does not work it is the reason of canvas download / upload. I have had problems with Github making my pre-trained models not work when uploading to a Github repository to then download from on Google Colab. Thats why the .zip fil is added to Google drive instead of a simple !git clone. The notebooks is tested multiple times using different accounts. So they work.
