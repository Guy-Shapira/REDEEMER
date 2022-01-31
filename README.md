# REDEEMER: Reinforcement Learning Based CEP Pattern Miner for Knowledge Extraction

## Background
REDEEMER is a CEP pattern miner that can generate new unobserved and meaninful patterns while minimzing the amount of labels needed for training.

## Datasets
The following datasets were used during our evaluations:
- *StarPilot*: This dataset was generated using OpanAI's [Procgen](https://github.com/openai/procgen) source code.
  All game objects were modified they would track information about themselves (e.g., location, speed, ect.) and send it to a log file every 5 game frames.
  We also trained using OpenAI's [procgen-train](https://github.com/openai/train-procgen) repository. Agents are trained with an aim of longevity.
  This resulted in longer games and provided us with massive data-streams.

  In our modified game, different objects track different attributes. For example, the player's spaceship tracks information about its speed, location and health points, while the finish only tracks information about its location.
  Each record represents a primitive event, with the object identifier used as the event type.
    The records in the dataset consisting of a
     frame number (timestamp), event (object id), 
     x coordinates, y coordinates, velocity in x axis, velocity in y axis, and remaining health points.
     If any of this attribuates was not relevant for the object, "-1" value was used instade.
    The following table demonstrates a sample of this dataset
    
    | Event type (object identifier)| Frame number| X cord | Y cord | X speed | Y speed | Health Points |
    | :-----------------------------:|:-----------:|:------:|:------:|:-------:|:-------:|:-------------:|
    |          539                  |  player     | 1.2562 | 9.9813 | 0       | 0       |  1            |
    |          567                  |  bullet1    | 1.3884 | 8.2798 | 0.8     | 0       |  -1           |
    |          1056                 |  flyer1     | 16.5   | 5.136  | -0.299  | -0.015  |  2            |
    
    **A large sample of games created from the same agent are provided [here](https://drive.google.com/file/d/1Cw3QpB45fdLIaavFkx-HqS_TMRV3w6wJ/view). 
    However, we added in the README.md file in GPU contains a link to a sample of that data we used which was based over a few minutes of the football game**.


- *Football dataset*:  This dataset was taken from the DEBS 2013 Grand
 Challenge: https://debs.org/grand-challenges/2013/.
 The link for downloading the dataset is: http://www2.iis.fraunhofer.de/sports-analytics/full-game.gz (it weighs 4G).
The data originates from wireless sensors embedded in the soccer players' shoes and a ball used during a single match; the data spans the duration of the entire game. 
The sensors for the players produced data at a frequency of 200Hz, while the sensor in the ball produced data at a frequency of 2000Hz. 
Each row in the dataset contains a sensor identifier, which is used for the event type, along with the timestamp in pico-seconds, the location's x, y, z coordinates in mm, the velocity, and acceleration. 
This dataset contains 49,576,080 records (i.e., primitive events).
As mentioned in our paper, we sampled down this in order to create a data-stream that can be comprehend by a human expert, contact us if you have question regarding the downsample method.

    **Due to its large size, we did not attach the complete dataset. 
    However, we added in the README.md file in GPU contains a link to a sample of that data we used which was based over a few minutes of the football game**.
    In the sample file every record contains: an event type (sensor id), a unique timestamp, object's coordinates, velocity and acceleration along the x,y,z axes.


    The following table demonstrates a sample of this dataset:
    
    | Event type (sensor ID)| Timestamp | x | y | z | v | a | vx | vy | vz | ax | ay | az |
    | :----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|
    | 67 | 10634731737624712 | 26679 | -580 | 194 | 140437 | 607509 | 2074 | -855 | -9744 | 9661 |    2063 | -1549 |
    | 66 | 10634732514566388 | 27813 | -558 | 67 | 142069 | 1233916 | -5400 | -636 | 8392 | -158 | -7878 | 6156 |
    | 75 | 10634735140223609 | 26254 | -1097 | -154 | 82907 | 837089 | -539 | 9671 | 2485 | 7827 | 6169 | -816 |



- *GPU Cluster*: This dataset contains information about servers that are used as part of a GPU computational cluster.

  Data was gatherd for over a month from all GPUs periodically.
  A portion of the data can be downloaded from: https://bit.ly/3obUBbd. **contact us if you are interested in the full data-set**.
  This dataset contains 43,447,118 records and each record contains the server id + status (explained in paper), timestamp, Memory usage (per GPU) , Power usage (per GPU), Minmum and Mamimum values of those between samples (per GPU).
  
  In our work we used a subgroup of all the possiable attributes, we removed colmuns that had very high corrleation in order to align with our hardware limitations.

## Extensions 
  If you intend to use other data-sets, follow instructions in <em>Model/open_OpenCEP.py</em> script (using --help flag).
  You should also create a new subfolder in the <em>OpenCEP/plugin</em> folder, and are welcomed to use the ToyExample for help.
  Feel free to contact us regarding any issues!

## Dependencies
The preferred way for running the experiments (as we performed) is by using our supplied [conda environment](https://github.com/Guy-Shapira/REDEEMER/blob/master/REDEEMER_ENV.yaml).


 
 Feel free to contact us if more explanation is needed :)
