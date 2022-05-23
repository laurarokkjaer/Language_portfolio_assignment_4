
# Language Analytics - Spring 2022
# Portfolio Assignment 4

This repository contains the code and descriptions from the fourth assigned project of the Spring 2022 module Language Analytics as part of the bachelor's tilvalg in Cultural Data Science at Aarhus University - whereas the overall Language Analytics portfolio (zip-file) consist of 5 projects, 4 class assignments + 1 self-assigned.

## Repo structure
### This repository has the following directory structure:

| **Folder** | **Description** |
| ----------- | ----------- |
| ```input``` | Contains the input data (will be empty) |
| ```output``` | Contains the results (outputs like plots or reports)  |
| ```src``` | Contains code for assignment 4 |
| ```utils``` | Contains utility functions written by [Ross](https://pure.au.dk/portal/en/persons/ross-deans-kristensenmclachlan(29ad140e-0785-4e07-bdc1-8af12f15856c).html), and which have been used in the assignments |

Also containing a ```MITLICENSE``` for guidelines of how to reproduce and use the data in this repository, as well as a ```.txt``` reqirements-file, where the required installments will be listed.

## Assignment description
The official description of the assignment from github/brightspace: [assignment description](https://github.com/CDS-AU-DK/cds-language/blob/main/assignments/assignment4.md).

The assignment for this week builds on these concepts and techniques. We're going to be working with the data in the folder CDS-LANG/toxic and trying to see if we can predict whether or not a comment is a certain kind of toxic speech. You should write two scripts which do the following:

- The first script should perform benchmark classification using standard machine learning approaches
  - This means CountVectorizer() or TfidfVectorizer(), LogisticRegression classifier
  - Save the results from the classification report to a text file
- The second script should perform classification using the kind of deep learning methods we saw in class
  - Keras Embedding layer, Convolutional Neural Network
  - Save the classification report to a text file


### The goal of the assignment 
The goal of this assignment was to show that I can perform text classification using both classical machine learning approaches, as well as using more sophisticated deep learning approaches. The scripts that I create for this assignment can also be reused and modified for use on other text data in a tabular format.

### Data source
The data used in this assignment is the in class flowers-folder from UCloud (shared-drive/CDS-VIS/flowers). 

Link to flowers dataset: [flowers dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/17/index.html).


## Methods
To solve this assignment i have worked with ```opencv``` in order to both calculate the histograms as well as for the general image processing (using the ```calcHist```, ```imread```, ```normalize``` and ```compareHist```). Futhermore i used the ```jimshow``` and ```jimshow_channel``` from the ```utils```-folder, along with the ```matplotlib``` for plotting and visualisation.

## Usage (reproducing results)
These are the steps you will need to follow in order to get the script running and working:
- load the given data into ```input```
- make sure to install and import all necessities from ```requirements.txt``` 
- change your current working directory to the folder above src in order to get access to the input, output and utils folder as well 
- the following should be written in the command line:

      - cd src (changing the directory to the src folder in order to run the script)
      
      - python image_search.py (calling the function within the script)
      
- when processed, there will be a messagge saying that the script has succeeded and that the outputs can be seen in the output folder 



## Discussion of results
The result of this script is an image which contains one target flower image and the calculated three similar images, as well as the calculated distance scores of the images. Furthermore, a csv file is made with the results (similar images). 

For further development, it could have been interesting to look at how to make the script run with a user defined input. Since this code have already been through a transision from jupiter notebook to .py script, it would not have been much change to do. For the user to parse an argument via the command line when running the code, the script would have been more reproduceble/reuseble, because of the fact that the user wpuld be able to define the target image themselves. 


