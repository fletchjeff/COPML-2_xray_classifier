# COPML Example 2 - Detecting Pneumonia in Chest X-Rays
This repoistory is used as the second example in the [Continuous Operations for Production Machine Learning](https://linktothis.com) (COPML) document that provides a framework for maintaning machine learning projects in production. The goal is to build an image classifier model using Deep Learning to predict the likelihood that a patient has pneumonia from a chest x-ray. 

This section follows the standard workflow for implementing machine learning projects. As this project should be deployed as an AMP, most of the artifacts should already be deployed. 

_Note: If you did not deploy the project as an AMP, you need to initialise the project first._

**Initialize the Project**

There are a couple of steps needed at the start to configure the Project and Workspace settings so each step will run sucessfully. If you did **not** deploy this project as an AMP, you **must** run the project bootstrap before running other steps. If you just want to launch the project with all the artefacts already built without going through each step manually, then you can also redploy deploy the complete project and an AMP.

*Project bootstrap*

Open the file `_bootstrap.py` in a normal workbench python3 session. You only need a 2 vCPU / 8 GiB instance. You don't need to us GPU during the installation however. Once the session is loaded, click **Run > Run All Lines**. This will script will create an Environment Variable for the project called **STORAGE**, which is the root of default file storage location for the Hive Metastore in the DataLake (e.g. `s3a://my-default-bucket`). It will also fetch additional xray images from the web and and upload the images used in the project to `$STORAGE/user/$HADOOP_USER_NAME/data/xray/`. A smaller selection of images are provided as part this git repo in the `data` folder.

## Step 1: Clarify Business Requirements
*Note:* This example is for the sole purpose of illustrating a technical implementation and should not be considered a source of actual medical advice or opinion. 

Pneumonia is a serious life-threatening condition. This has become clear over the course of the Covid-19 pandemic. It has a number of causes including bacteria, viruses and fungi. The treatment for pneumonia varies depending on the cause. Consequently, it’s really important to accurately identify if the patient has pneumonia and if so, what type, so that the right course of drugs can be administered. In a busy, pressurised hospital setting, unexpected events can lead to resource constraints which in turn can lead to delays in diagnosing pneumonia. The use of a machine learning model to help with a timely diagnosis could help to save lives and reduce the burden on healthcare practitioners. A machine learning model that is capable of making high confidence predictions (that is embedded in a system with appropriate checks) has the potential to reduce costs and delays associated with blood tests and extensive consultations with specialists. Obviously, lower confidence predictions will still follow the conventional diagnostic workflow. The business requirement in this scenario is the smooth and timely prediction of likely pneumonia cases (with minimal false negatives) while reducing the number of non-pneumonia cases that are directed for blood tests and examination by consultants.  

## Step 2: Assess Available Data
The dataset for the model training comprises digitized x-ray images from a range of patients – some with various types of pneumonia and others with uninfected lungs. Digitized x-ray images are fairly widely used and the data format is well understood. This dataset needs to be made accessible to the data science team for model training. The sample project uploads the training image dataset to the connected data store (for CML this could be an S3 bucket). This could also be implemented using the Cloudera Operational Database (COD), as its Apache HBase Medium Object Storage (MOB) feature means it is well-suited to serving images.

### Data Ingest
The data ingest process runs as part of the `_bootstrap.py` process that was run earlier. Specifically additional xray images are pulled from a public S3 bucket and extracted to the CML project's file system. The images are then uploaded to the configured object storage bucket using. This currently only works for S3 using [smart_open](https://github.com/RaRe-Technologies/smart_open) and the `boto3_client()` function from [CMLBootstrap](https://github.com/fletchjeff/cmlbootstrap).

``` python
for i,image in enumerate(glob.glob("data/train/{}/*".format(directory))):
    pil_image = Image.open(image)
    write_url  = image_storage.format('train',directory,image.split("/")[-1])
    smart_open_writer = open(write_url, 'wb', transport_params={'client': client})
    pil_image.save(smart_open_writer)
    smart_open_writer.close()
```
The images are already divided into test and train sets and the upload function uploads to the object store accordingly.

## Step 3: Develop a Data Science Plan
The next step is for the data science team to explore the dataset and come up with the plan for model development and mode of deployment. The business requirement is to reduce the time to get a diagnosis, and minimise the use of specialists and/or additional blood tests. This needs to be done with the view that the model should also make the fewest possible number of false predictions that a patient does not have pneumonia when they actually do.  The model supports this requirement by minimizing false negatives as much as possible and optimizing the accuracy of the classification. A reasonable plan for achieving this would be to create an image classifier using transfer learning on one of the new generation, pretrained image classifier models. So our machine learning solution will comprise two specific models: one model capable of predicting if the patient has pneumonia and a second for predicting the type of pneumonia. The first model needs to be optimized to reduce the number of false negatives (high sensitivity / recall). Adjusting the threshold for classification into either group should minimise false negatives. The second model needs to be optimized for accuracy as the requirement is for more certainty as to the type of pneumonia. This is likely a complex computation task that will require many nodes of GPU during the initial model training process and would be best implemented using a public cloud-based CML with GPU nodes for the duration of the training and optimization processes.

### Explore Data
The project includes a Jupyter Notebook that does some basic data exploration and visualistaion. It is to show how this would be part of the data science workflow.

![data](../images/data.png)

Open a Jupyter Notebook session (rather than a work bench): python3, 1 CPU, 2 GB and open the `notebooks/data_exploration.ipynb` file. 

At the top of the page click **Cells > Run All**.


## Step 4: Model Deployment 
A production version of this model would involve a pipeline that captures new images from an x-ray that is flagged by the radiologist as requiring pneumonia identification. This image (or in some cases multiple) of the patient’s lungs would be sent to an API endpoint in CML to provide a prediction from the classifier. The data from each call to the API needs to be stored to calculate overall model performance and any result that is below an acceptable confidence threshold needs to trigger an alert to the attending medical practitioner to then either request blood test or request human assistance from someone with expertise in this field. Of critical importance here are availability and history. The model must always be available and the lineage of the data on which it was trained must be tracked in order to support auditability and reproducibility requirements. It is also necessary to store all predictions made by the specific model deployed and the detail for the image used. This allows the ability to confirm that the specific deployment will make the same prediction given the same image and to cover the Auditability requirement listed in section 4. CML provides the ability to track the image details, the prediction and a unique identifier for the model used. CML also keeps copies of previously deployed models so they can be redeployed for testing if required. 

## Step 5: Model Operations
Once the machine learning models are deployed, the classifier’s performance should be validated against human assessments and blood tests. Further, a random sample of the high confidence predictions made by the model should also be validated in this way. 

In a scenario such as this, where the number of positive cases of pneumonia are low relative to the total number of people x-rayed, an imbalanced classification could be a complicating factor. It could result in a situation where a model appears to be making accurate predictions (and by extension, fulfilling business requirements) but in reality it’s simply reflecting statistical probabilities. 

In order to better understand this point, let’s assume 1 in every 100 patients has pneumonia. This implies that 99% of x-ray images will have no indication of pneumonia. If our model simply classified every x-ray as normal (i.e. it never detects pneumonia) then most of the time the model would accurately classify x-ray images, it would be wrong in only 1% of cases. This would superficially satisfy the business success metrics as the number of cases flagged for corroboration through blood tests or manual assessment, would fall. This would in turn reduce the time and costs associated with pneumonia detection. However the real life implications of failing to identify a case of pneumonia would be very serious. Once the problem was eventually detected the project would, rightly, be considered a failure. 
