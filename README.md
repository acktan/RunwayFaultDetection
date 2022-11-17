[![pipeline status](https://gitlab.code.hfactory.io/adrian.tan/colasproject/badges/main/pipeline.svg)](https://gitlab.code.hfactory.io/adrian.tan/colasproject/-/commits/main)
[![coverage report](https://gitlab.code.hfactory.io/adrian.tan/colasproject/badges/main/coverage.svg)](https://gitlab.code.hfactory.io/adrian.tan/colasproject/-/commits/main)

# Multi-label classification of runway images to determine the different defects they may have 
**Colas x MSc X-HEC Data Science for Business**

## Presentation of the project

### Quick overview of the project 
This computer vision challenge has been proposed to sereval teams from the MSc X-HEC Data Science for Business by the AI / Data Science team of the building group COLAS, subsidiary of the Bouygues Group. COLAS works on several businesses (Road, rail, manufacturing, quarries, pipeline, ...)

### Objective 
Identify asphalt mixed defeact on airport track, and rank them with a clear data visualization. Identify airport with the best business opportunity for Colas, regarding its activities. Extend the consideration to all kind of Colas Business opportunities

### Step for the chalenge
1. Collect french airport open data
The objective of this step is to identify & collect open data of french airports (name, geolocalization...).
We used the BD Carto & BD Orthophoto from the French public organization IGN : https://geoservices.ign.fr/bdcarto

2. Collect french orthophotos
The objective was to collect IGN's orthophotos (each picture 25k x 25k pixels represents a square of 5km per 5km). Again, we found them on the IGN opendata platform : https://geoservices.ign.fr/bdortho#telechargement

3. Extract airport images
Here the objective was to extract picture for each airport. To achieve this objective, we linked the airports geolocalization from step 1 & orthophotos from step 2 to extract airports pictures.

4. Extract runway images
Similarly, the objective was to extract picture for each runaway from each airport. We used the tool [Label Studio](https://labelstud.io/guide/) to label the runaways manually before extracting them with Python

5. Modeling
Here, the objective was to Train a computer vision model to predict damages (multi label image classification). Metrics is weighted F1 score. Shout out to [Ashref Maiza](https://github.com/ashrefm/) who wrote this  really cool [article](https://towardsdatascience.com/multi-label-image-classification-in-tensorflow-2-0-7d4cf8a4bc72) on which we based a large part of our model.

6. Business opportunities
This step was particularly important to the Colas team. We had to identify similar uses cases or new business opportunities for COLAS Group based on the computer vision technology.

7. Pitch
Last but not least, we presented the team's results and especially the potential of the technology for the Colas Group during a presentation in front of several senior COlAS collaborators (Chief Digital Officer, Chief Data Officer, Digital and Core business applications Director, Chief Data Officer, Lead Data/AI, ...) and HEC professors (head of Corporate Partnership Dpt, Prof. Chair Holder of Bouygues Chair “Smart city and the common good”, Associate Dean for Research)

## Requirements
The requirements.txt file should list all Python libraries that your notebooks depend on, and they will be installed using
    pip install -r requirements.txt

## Structure of the repository
1. Inputs
The inputs folder contains all the inputs needed to run the pipeline. It is composed of three subfolders:
- AirportExtractionSample_Step3:
This folder contains three subfolders:
BD_CARTA which contains information regarding the geometry of airports in France
Orthophotos_Geom which contains the geometry of orthophotos downloaded from [ING website](https://geoservices.ign.fr/bdortho). Note that only the geometry of orthophotos for department 93 are available.
Orthophotos which contains the actual orthophotos from the ING website. Note that the user should add the photos in the folder when running the pipeline locally and update the path "bdortho_input_path" in the config file.

- Model
This folder contains two files: the "labels_train.csv" which contains the name of images in the training set and their labels and the template_test.csv which contains the name of images for the test set. Two other folders are also available there: the "train" folder contains the images for the training set and the "test" folder contains the images for the test set.

- RunwayExtractionSample_Step4
This folder contains a csv file which is the output of a manual labeling tool called [Label Studio](https://labelstud.io/guide/index.html#Quick-start). The file will contain the image name and the the x, y, width, height and rotation coordinates corresponding to runways in each image. Note that currently only a sample of images have been labeled.

2. Outputs
The outputs folder contains all the outputs that are created throughout the pipeline. For each step, the output is stored in a separate folder. For example, the extraction of airports from orthophotos are stored in the "ExtractionAirports" folder, the extraction of runways is stored in the "ExtractionRunways" folder, the predictions are stored as a csv file in the Inference folder, the model is saved in the Model folder and the unit tests outputs are stored in the Outputs_Test folder.

3. Params
The Params folder contains a Config folder with a config.json file where all parameters used in the pipeline are set. It also contains a Logs with a logs.log file where the loggings are stored when the pipeline is run.

4. src
The src folder is the folder that contains all the pipeline. The main.py file is used to run the whole pipeline. The user would need to run "python main.py" from the src folder to run the pipeline.
The first step is to extract the airports from orthophotos, this is done by the Extractairports class in the extract_airports.py file. The second step is to extract the runways from cropped airport images, and this is done by the Extractrunways class in the extract_runways.py file. Next, we move on to the modelling. The user can choose to train a model or load a model that has been previously saved, ensuring that the parameters in the config are set up correctly: set the model -> train parameter to True or False and set the model -> model_name parameter to a model that is available in your Outputs->Model folder. The data is prepared with a DataLoader class in the dataloader.py file, the model is created with a class in the model.py file and is trained in the train.py file. Finally, the model is evaluated using the Evaluate class in the evaluation_model.py file and the predictions are inferred from the Infererence class in the inference.py file. Note that if the parameter "save_preds" is set to True in the config file, then the predictions output will be saved in the Outputs->Inference folder.

5. Unit tests
The unit tests folder is used to test the functions that are in the src file. In order to run the tests, run "pytest" in the terminal while being at the root. This folder has a separate config file where different parameters are set for the unit tests. It also follows the same structure as the src file in order to ensure that each step of the pipeline is tested.