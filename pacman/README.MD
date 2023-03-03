Berkeley CS188 Pacman AI
====================================
Hi! This folder includes my work on Berkeley's CS188 Pacman AI class. Note that the project is incomplete; I look forward to finishing it when my courseload affords me the time.

Labs 1-9 mark ascending levels of complexity in the AI. It would be a waste of time and device storage for you to run every single folder. In order to judge my mastery of Artificial Intelligence, I suggest you look to Lab 7, 8, and 9.

If you'd like to download an individual folder from a Github repository, you should start by navigating to https://download-directory.github.io/. In a separate tab, open either Lab 7, 8, or 9 on Github and copy the URL. Type the URL into the prompted box in the download website. Once the folder is downloaded, place it into your desired location on your device.

In order to run any of this code, please ensure that you have python3.6 downloaded. Afterwards, please type the following into your terminal, substituting "<env-name>" with your choice of environment name:
>conda create --name <env-name> python=3.6
  
>source activate <env-name>

To discontinue the python environment, type:
>conda deactivate

To run Lab 7, navigate to the appropriate folder in your terminal and run the following command:   
>python crawler.py   

<img width="1006" alt="crawler" src="https://user-images.githubusercontent.com/124002750/222548890-669d71a2-5230-4a83-a991-ce7a4adc471e.png">
Click the following link to see a video of the crawler agent learning to walk in real time:<br>

<br>

>https://drive.google.com/file/d/15RCYfuyj1qs9z03j0-RdSRij3EHuJO9y/view?usp=sharing

Look! He's just a little guy who's trying his best! You can poke around with the learning policy adjustments to make him learn faster or slower, less or more stable.  

To run Lab 8, go to the appropriate directory in your terminal and run the following:
>python pacman.py -p PacmanQAgent -n 10 -l smallGrid -a numTraining=10
>
>python pacman.py -p ApproximateQAgent -a extractor=SimpleExtractor -x 50 -n 60 -l mediumGrid

The first line trains the Pacman AI to avoid ghosts and find food pellets, while the second line shows a fully trained Pacman! If you'd like to see a video of these things happening instead of downloading the repository and running manually, click the following links which are pasted as the before video and after video respectively:

>https://drive.google.com/file/d/1UWdIaSr4_IHi4WqXfaEUs51KszkygR9T/view?usp=sharing
>https://drive.google.com/file/d/1g4l_AM0PadHufqn90dmgnakSsl06m1Gt/view?usp=sharing




To run Lab 9, similarly navigate to the appropriate folder in your terminal and run the following commands:  
>python autograder.py -q q1   
python autograder.py -q q2   
python autograder.py -q q3    
python autograder.py -q q4   

<img width="636" alt="pacman performance" src="https://user-images.githubusercontent.com/124002750/222544017-b2e3659e-2eba-4114-946c-fb81a33f725c.png">
You will notice that this pacman is very different from others! He is the one that hunts down the ghosts! The course suggests that this is a mark of pacman's distant past, being hunter before he became hunted. As you run the code, you'll notice that he becomes increasingly good at hunting down ghosts. Originally, he's bumbling around in the dark, but slowly he becomes faster and faster at finding the ghosts and eliminating them. Yeesh!

