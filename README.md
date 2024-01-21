# MMRotate Model Adapter

## Introduction

An [MMRotate](https://github.com/dataloop-ai-apps/mmrotate/tree/main) Model Adapter implementation for Dataloop

## Requirements

```commandline
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    pip install -r requirements.txt
    mim install mmengine
    mim install "mmcv>=2.0.0"
    mim install mmdet
```

### Installing in the Platform

To clone the model from our AI Library, review
our [documentation](https://developers.dataloop.ai/tutorials/model_management/create_new_model_ui/chapter/)

### Installing via the SDK

To install MMRotate via SDK, all that is necessary is to clone the model from the AI Library to your own project:

```python
import dtlpy as dl

project = dl.projects.get('My Project')
public_model = dl.models.get(model_name="<Public-Model-Name>")
model = project.models.clone(from_model=public_model,
                             model_name='<Public-Model-Name>',
                             project_id=project.id)
```

For more options when installing the model, check
this [page](https://developers.dataloop.ai/tutorials/model_management/ai_library/chapter/#finetune-on-a-custom-dataset).

## Deployment

After installing the pretrained model or fine-tuning it on your data, it is necessary to deploy it, so it can be used
for prediction.

### Deploying with the Platform

In the Model Management page of your project, find a pretrained or fine-tuned version of your <Model Name> model and
click the three dots in the right of the model's row and select the "Deploy" option:

<img src="assets/MM_page.png" alt="Model Management - Versions Tab Image">

Here you can choose the instance, minimum and maximum number of replicas and queue size of the service that will run the
deployed model (for more information on these parameters,
check [the documentation](https://developers.dataloop.ai/tutorials/faas/advance/chapter/#autoscaler)):

<img src="assets/deployment_1.png" alt="Model Management - Deployment Page 1">

Proceed to the next page and define the service fields (which are
explained [here](https://developers.dataloop.ai/tutorials/faas/custom_environment_using_docker/chapter/)).

<img src="assets/deployment_2.png" alt="Model Management - Deployment Page 2">

After this, your model is deployed and ready to run inference.

### Deploying with the SDK

To deploy with the default service configuration defined in the package:

```python
import dtlpy as dl

model_entity = dl.models.get(model_id='<model-id>')
model_entity.deploy()
```

For more information and how to set specific service settings for the deployed model, check
the [documentation](https://developers.dataloop.ai/tutorials/model_management/ai_library/chapter/#clone-and-deploy-a-model)
.

## Testing

Once the model is deployed, you can test it by going to the Model Management, selecting the <Model Name> model and then
going to the test tab. Drag and drop or select an image to the image area:

<img src="assets/cat_test_1.png" alt="Model Page - Model Test tab image uploaded">

click the test button and wait for the prediction to be done:

<img src="assets/cat_test_2.png" alt="Model Page - Model Test tab image prediction complete">

## Prediction

### Predicting in the Platform

The best way to perform predictions in the platform is to add a "Predict Node" to a pipeline:

<img src="assets/pipeline.png" alt="Predict Pipeline Node">

Click [here](https://developers.dataloop.ai/onboarding/08_pipelines/) for more information on Dataloop Pipelines.

### Predicting with the SDK

The deployed model can be used to run prediction on batches of images:

```python
import dtlpy as dl

model_entity = dl.models.get(model_id='<model-id>')
item_id_0 = '<item-id-0>'
results = model_entity.predict_items([item_id_0])
print(results)
```

For more information and
options, [check the documentation](https://developers.dataloop.ai/tutorials/model_management/ai_library/chapter/#predict-items).#
mmocr
