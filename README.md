# COPML Example 2 - Detecting Pneumonia in Chest X-Rays
This repoistory is used as the second example in the [Continuous Operations for Production Machine Learning](https://linktothis.com) (COPML) document that provides a framework for maintaning machine learning projects in production. The goal is to build an image classifier model predict the likelihood that a patient has pneumonia based on an image of their chest x-ray. The primary goal of this repository is to build an image classifier that predict whether a patient has pneumonia and if its a viral or bacterial pnemonia from a chest xray image. The project uses tensorflow and transfer learning with MobileNetV2 model, trained on a labled chest x-ray image dataset available on [Kaggle](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia).

![app](images/app.png)

The aim is to show and end-to-end process of how to build an application in CML that can take a new image and 
make a prediction on that image in real time use two separate models. The project also uploads the image data 
to an object store and then pulls the data from that object store during model training. There is also a front-end application to allow users to test and display the model predictions on test images in real time. 

## Project Structure

The project is organized with the following folder structure:

```
.
├── app/            # Assets needed to support the front end application
├── code/           # Scripts and files needed to create the various project artifacts
├── data/           # The full image dataset used for model training and other useful data
├── images/         # Images used for the README and documentation
├── notebooks/      # Notebooks used during the model building process 
├── models/         # Directory to hold trained models
├── cdsw-build.sh   # Shell script used to build environment for experiments and models
├── README.md
├── LICENSE.txt
└── requirements.txt
```

By following the notebooks, scripts, and documentation in the `code` directory, you will understand how to perform similar tasks on CML, as well as how to use the platform's major features to your advantage. 


<!-- These features include:

- Data ingestion and uploading to 
- Hive table creation and querying
- Streamlined model development
- Point-and-click model deployment to a RESTful API endpoint
- Application hosting for deploying frontend ML applications

We will focus our attention on working within CML, using all it has to offer, while glossing over the details that are simply standard data science, and in particular, pay special attention to data ingestion and processing at scale with Spark. -->

## Deploying on CML

There are three ways to launch the this prototype on CML:

1. **From Prototype Catalog** - Navigate to the Prototype Catalog on a CML workspace, select the "Airline Delay Prediction" tile, click "Launch as Project", click "Configure Project"
2. **As ML Prototype** - In a CML workspace, click "New Project", add a Project Name, select "ML Prototype" as the Initial Setup option, copy in the [repo URL](https://github.com/fletchjeff/cml_xray_classifier), click "Create Project", click "Configure Project"
3. **Manual Setup** - In a CML workspace, click "New Project", add a Project Name, select "Git" as the Initial Setup option, copy in the [repo URL](https://github.com/fletchjeff/cml_xray_classifier), click "Create Project". Then, follow the steps listed [in this document](code/README.md) in order

<!-- If you deploy this project as an Applied ML Prototype (AMP) (options 1 or 2 above), you will need to specify whether to run the project with `STORAGE_MODE` set to `local` or `external`. Running in external mode requires having external storage configured on your CML workspace and triggers the project to ingest, process, and store ~20GB of raw data using Spark. Running in local mode will bypass the data ingestion and manipulation steps by using the `data/preprocessed_flight_data.tgz` file to train a model and deploy the application. While running the project as an AMP will install, setup, and build all project artifacts for you, it may still be instructive to review the documentation and files in the [code](code/) directory. -->