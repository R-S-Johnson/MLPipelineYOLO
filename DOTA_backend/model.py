from typing import List, Dict, Optional
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
from ultralytics import YOLO
import numpy as np
import requests
import base64
import boto3
import os


class NewModel(LabelStudioMLBase):
    """Custom ML Backend model
    """
    
    def setup(self):
        """Configure any parameters of your model here
        """
        self.set("model_version", "0.0.1")

    @staticmethod
    def _convert_rotated_rect_to_params(obb):
        x1, y1, x2, y2, x3, y3, x4, y4 = np.array(obb).flatten()
        # Calculate center of the rotated rectangle
        cx = (x1 + x2 + x3 + x4) / 4.0
        cy = (y1 + y2 + y3 + y4) / 4.0

        # Calculate width and height of the rotated rectangle
        width = max(x1, x2, x3, x4) - min(x1, x2, x3, x4)
        height = max(y1, y2, y3, y4) - min(y1, y2, y3, y4)

        # Calculate rotation angle of the rectangle
        dx = x2 - x1
        dy = y2 - y1
        rotation = np.arctan2(dy, dx)

        # Calculate top-left corner (x, y) of the unrotated rectangle
        x = cx - width / 2.0
        y = cy - height / 2.0

        # Return the parameters (x, y, width, height, rotation)
        return [x, y, width, height, rotation]

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
        """ Write your inference logic here
            :param tasks: [Label Studio tasks in JSON format](https://labelstud.io/guide/task_format.html)
            :param context: [Label Studio context in JSON format](https://labelstud.io/guide/ml_create#Implement-prediction-logic)
            :return model_response
                ModelResponse(predictions=predictions) with
                predictions: [Predictions array in JSON format](https://labelstud.io/guide/export.html#Label-Studio-JSON-format-of-annotated-tasks)
        """
        print(f'''\
        Run prediction on {tasks}
        Received context: {context}
        Project ID: {self.project_id}
        Label config: {self.label_config}
        Parsed JSON Label config: {self.parsed_label_config}
        Extra params: {self.extra_params}''')

        # example for resource downloading from Label Studio instance,
        # you need to set env vars LABEL_STUDIO_URL and LABEL_STUDIO_API_KEY
        # path = self.get_local_path(tasks[0]['data']['image_url'], task_id=tasks[0]['id'])

        # example for simple classification
        # return [{
        #     "model_version": self.get("model_version"),
        #     "score": 0.12,
        #     "result": [{
        #         "id": "vgzE336-a8",
        #         "from_name": "sentiment",
        #         "to_name": "text",
        #         "type": "choices",
        #         "value": {
        #             "choices": [ "Negative" ]
        #         }
        #     }]
        # }]

        predictions = []
        s3client = boto3.client(
            's3',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
        )
        model = YOLO('/models/YOLOv8n-obb.pt')

        for task in tasks:
            # Get image from s3
            s3url = task['data']['image'].split('/')
            bucket = s3url[2]
            object_key = '/'.join(s3url[3:])
            # Generate presigned URL for the image
            presigned_url = s3client.generate_presigned_url(
                'get_object',
                Params={'Bucket': bucket, 'Key': object_key},
                ExpiresIn=3600  # URL expires in 3600 seconds (1 hour)
            )
            response = requests.get(presigned_url)
            with open('/targetImage.png', 'wb') as f:
                f.write(response.content)

            # Run prediction on image and get obb predictions
            result = model('/targetImage.png')[0]
            obbs = [NewModel._convert_rotated_rect_to_params(obb) for obb in result.obb.xyxyxyxyn.tolist()]
            confs = result.obb.conf.tolist()
            classes = result.obb.cls.tolist()
            class_names = result.names

            # Format for Label Studio
            predict_results = [{
                "from_name": "label",
                "to_name": "image",
                "type": "rectanglelabels",
                "value": {
                    "rectanglelabels": [class_names[clas]],
                    "x": obb[0],
                    "y": obb[1],
                    "width": obb[2],
                    "height": obb[3],
                    "rotation": np.degrees(obb[4])
                },
                "score": conf,
            } for obb, conf, clas in zip(obbs, confs, classes)]

            prediction = {
                "model_version": "YOLOv8n-obb.pt",
                "result": predict_results
            }
            predictions.append(prediction)

        return ModelResponse(predictions=predictions)
    
#    def fit(self, event, data, **kwargs):
#        """
#        This method is called each time an annotation is created or updated
#        You can run your logic here to update the model and persist it to the cache
#        It is not recommended to perform long-running operations here, as it will block the main thread
#        Instead, consider running a separate process or a thread (like RQ worker) to perform the training
#        :param event: event type can be ('ANNOTATION_CREATED', 'ANNOTATION_UPDATED')
#        :param data: the payload received from the event (check [Webhook event reference](https://labelstud.io/guide/webhook_reference.html))
#        """
#
#        # use cache to retrieve the data from the previous fit() runs
#        old_data = self.get('my_data')
#        old_model_version = self.get('model_version')
#        print(f'Old data: {old_data}')
#        print(f'Old model version: {old_model_version}')
#
#        # store new data to the cache
#        self.set('my_data', 'my_new_data_value')
#        self.set('model_version', 'my_new_model_version')
#        print(f'New data: {self.get("my_data")}')
#        print(f'New model version: {self.get("model_version")}')
#
#        print('fit() completed successfully.')
