SKIN COLOR AND HAIR COLOR DETECTION USING DEEP LEARNING, IMAGE PROCESSING AND MACHINE LEARNING.

Environment:
Ubuntu 16.04 LTS
Ram 8GB
Intel® Core™ i5-7200U CPU @ 2.50GHz × 4 

Framework : Pytorch
Face detection : dlib

No of classes for skin color 
1- White
2- Brown
3- Black

No of classes for Hair color
1- Black
2- Grey
3- Blonde

#Note - Hair model is confused between grey and blonde color [have some ideas for better classification between them]
 
<--------------------------------------"Steps to run the code"----------------------------->

1- Create a Virtual environment in ubuntu.
[sudo apt-get install virtualenv]
#check the steps for creating and activating the virtual env

2- Install the requirement.txt in that virtual env. [pip3 install -r requirement.txt]
3- Put your test images into #test folder
4- download the weight file and put that file in ckpt folder. 

Link is below:

https://drive.google.com/open?id=1xmNGIBP6ORc5M75sbyNBP2J8oeeryjo8

   
5- Run the code [python3 submit_code.py]

6- Check the prediction in output folder.

<------------------------------------------------------------------------------------------------->

Special Thanks to YBIGTA
https://github.com/YBIGTA/pytorch-hair-segmentation

#Note - Due to less time and less data model's accuracy is not that much good but there are lots of areas for improvement.
#Gaurav kothyari
#gkothari25@gmail.com 
![Image description](Wet-n-Wild-Reserve-Your-Cabana-Pale-Skin.png)
