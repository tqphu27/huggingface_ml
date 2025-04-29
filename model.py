#model_marketplace.config
# {"token_length": "4018", "accuracy": "70", "precision": "fp16", "sampling_frequency:": "44100", "mono": true, "fps": "74", "resolution": "480", "image_width": "1080", "image_height": "1920", "framework": "transformers", "dataset_format": "llm", "dataset_sample": "[id on s3]", "weights": [
#     {
#       "name": "DeepSeek-V3",
#       "value": "deepseek-ai/DeepSeek-V3",
#       "size": 20,
#       "paramasters": "685B",
#       "tflops": 14, 
#       "vram": 20,
#       "nodes": 10
#     },
# {
#       "name": "DeepSeek-V3-bf16",
#       "value": "opensourcerelease/DeepSeek-V3-bf16",
#       "size": 1500,
#       "paramasters": "684B",
#       "tflops": 80, 
#       "vram": 48,
#       "nodes": 10
#     }
#   ], "cuda": "11.4", "task":["text-generation", "text-classification", "text-summarization", "text-ner", "question-answering"]}
from io import BytesIO
import io
import math
import os
from typing import List, Dict, Optional
# from label_studio_ml.model import LabelStudioMLBase
# from label_studio_ml.response import ModelResponse
from flask import send_file
from transformers import pipeline
# import torchaudio
import torch
import soundfile as sf
import requests
import base64
import scipy

# MODEL_NAME = os.getenv('MODEL_NAME', 'facebook/opt-125m')
# _model = pipeline('text-generation', model=MODEL_NAME)


# class HuggingFaceLLM(LabelStudioMLBase):
#     """Custom ML Backend model
#     """

#     MAX_LENGTH = int(os.getenv('MAX_LENGTH', 50))

#     def setup(self):
#         """Configure any paramaters of your model here
#         """
#         self.set("model_version", f'{self.__class__.__name__}-v0.0.1')

#     def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
#         """ Write your inference logic here
#             :param tasks: [AIxBlock tasks in JSON format](https://labelstud.io/guide/task_format.html)
#             :param context: [AIxBlock context in JSON format](https://labelstud.io/guide/ml_create#Implement-prediction-logic)
#             :return model_response
#                 ModelResponse(predictions=predictions) with
#                 predictions: [Predictions array in JSON format](https://labelstud.io/guide/export.html#Label-Studio-JSON-format-of-annotated-tasks)
#         """
#         from_name, to_name, value = self.label_interface.get_first_tag_occurence('TextArea', 'Text')
#         predictions = []
#         for task in tasks:
#             text = self.preload_task_data(task, task['data'][value])
#             result = _model(text, max_length=self.MAX_LENGTH)
#             generated_text = result[0]['generated_text']
#             # cut the `text` prefix
#             generated_text = generated_text[len(text):].strip()
#             predictions.append({
#                 'result': [{
#                     'from_name': from_name,
#                     'to_name': to_name,
#                     'type': 'textarea',
#                     'value': {
#                         'text': [generated_text]
#                     }
#                 }],
#                 'model_version': self.get('model_version')
#             })
        
#         return ModelResponse(predictions=predictions, model_version=self.get("model_version"))

from typing import List, Dict, Optional
from aixblock_ml.model import AIxBlockMLBase
import torch.distributed as dist
import os
import torch
import os
import subprocess
import random
import asyncio
import logging
import logging
import base64
import hmac
import json
import hashlib
import zipfile
import subprocess
import shutil
import threading
import requests
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("aixblock-mcp")

HOST_NAME = os.environ.get('HOST_NAME',"http://127.0.0.1:8080")

def download_checkpoint(weight_zip_path, project_id, checkpoint_id, token):
    url = f"{HOST_NAME}/api/checkpoint_model_marketplace/download/{checkpoint_id}?project_id={project_id}"
    payload = {}
    headers = {
        'accept': 'application/json',
        'Authorization': f'Token {token}'
    }
    response = requests.request("GET", url, headers=headers, data=payload) 
    checkpoint_name = response.headers.get('X-Checkpoint-Name')

    if response.status_code == 200:
        with open(weight_zip_path, 'wb') as f:
            f.write(response.content)
        return checkpoint_name
    
    else: 
        return None

def download_dataset(data_zip_dir, project_id, dataset_id, token):
    # data_zip_path = os.path.join(data_zip_dir, "data.zip")
    url = f"{HOST_NAME}/api/dataset_model_marketplace/download/{dataset_id}?project_id={project_id}"
    payload = {}
    headers = {
        'accept': 'application/json',
        'Authorization': f'Token {token}'
    }

    response = requests.request("GET", url, headers=headers, data=payload)
    dataset_name = response.headers.get('X-Dataset-Name')
    if response.status_code == 200:
        with open(data_zip_dir, 'wb') as f:
            f.write(response.content)
        return dataset_name
    else:
        return None

def upload_checkpoint(checkpoint_model_dir, project_id, token):
    url = f"{HOST_NAME}/api/checkpoint_model_marketplace/upload/"

    payload = {
        "type_checkpoint": "ml_checkpoint",
        "project_id": f'{project_id}',
        "is_training": True
    }
    headers = {
        'accept': 'application/json',
        'Authorization': f'Token {token}'
    }

    checkpoint_name = None

    # response = requests.request("POST", url, headers=headers, data=payload) 
    with open(checkpoint_model_dir, 'rb') as file:
        files = {'file': file}
        response = requests.post(url, headers=headers, files=files, data=payload)
        checkpoint_name = response.headers.get('X-Checkpoint-Name')

    print("upload done")
    return checkpoint_name

def read_dataset(file_path):
    # Kiểm tra xem thư mục /content/ có tồn tại không
    if os.path.isdir(file_path):
        files = os.listdir(file_path)
        # Kiểm tra xem có file json nào không
        for file in files:
            if file.endswith(".json"):
            # Đọc file json
                with open(os.path.join(file_path, file), "r") as f:
                    data = json.load(f)

                return data
    return None

def is_correct_format(data_json):
    try:
        for item in data_json:
            if not all(key in item for key in ['instruction', 'input', 'output']):
                return False
        return True
    except Exception as e:
        return False
    
def conver_to_hf_dataset(data_json):
    formatted_data = []
    for item in data_json:
        for annotation in item['annotations']:
            question = None
            answer = None
            for result in annotation['result']:
                if result['from_name'] == 'question':
                    question = result['value']['text'][0]
                elif result['from_name'] == 'answer':
                    answer = result['value']['text'][0]
            if question and answer:
                formatted_data.append({
                    'instruction': item['data']['text'],
                    'input': question,
                    'output': answer
                })
    return formatted_data

    # dataset = Dataset.from_list(formatted_data)

class MyModel(AIxBlockMLBase):

    @mcp.tool()
    def action(self, command, **kwargs):
        # huggingface-cli login --token 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 --add-to-git-credential
        from huggingface_hub import login 
       
        print(f"""
                command: {command}
              """)
        if command.lower() == "train":
            try:
                # checkpoint = kwargs.get("checkpoint")
                # aixblock 
                #
                args = ('dummy', )

                clone_dir = os.path.join(os.getcwd())
                epochs = kwargs.get("num_epochs", 10)
                imgsz = kwargs.get("imgsz", 224)
                project_id = kwargs.get("project_id")
                token = kwargs.get("token")
                checkpoint_version = kwargs.get("checkpoint_version")
                checkpoint_id = kwargs.get("checkpoint")
                dataset_version = kwargs.get("dataset_version")
                dataset_id = kwargs.get("dataset")
                channel_log = kwargs.get("channel_log", "training_logs")
                world_size = kwargs.get("world_size", "1")
                rank = kwargs.get("rank", "0")
                master_add = kwargs.get("master_add")
                master_port = kwargs.get("master_port", "12345")
                # entry_file = kwargs.get("entry_file")
                configs = kwargs.get("configs")
                hf_access_token = kwargs.get("hf_access_token", "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")
                login(token = hf_access_token)
                def func_train_model(clone_dir, project_id, imgsz, epochs, token, checkpoint_version, checkpoint_id, dataset_version, dataset_id):
                    print("func_train_model")
                    # os.makedirs(f'{clone_dir}/data_zip', exist_ok=True)

                    # weight_path = os.path.join(clone_dir, f"models")
                    # dataset_path = "data"
                    # datasets_train = "alpaca_dataset"
                    # models_train = "stas/tiny-random-llama-2"

                    # if checkpoint_version and checkpoint_id:
                    #     weight_path = os.path.join(clone_dir, f"models/{checkpoint_version}")
                    #     if not os.path.exists(weight_path):
                    #         weight_zip_path = os.path.join(clone_dir, "data_zip/weights.zip")
                    #         checkpoint_name = download_checkpoint(weight_zip_path, project_id, checkpoint_id, token)
                    #         if checkpoint_name:
                    #             print(weight_zip_path)
                    #             with zipfile.ZipFile(weight_zip_path, 'r') as zip_ref:
                    #                 zip_ref.extractall(weight_path)

                    # if dataset_version and dataset_id:
                    #     dataset_path = os.path.join(clone_dir, f"datasets/{dataset_version}")
                    #     if not os.path.exists(dataset_path):
                    #         data_zip_dir = os.path.join(clone_dir, "data_zip/data.zip")
                    #         dataset_name = download_dataset(data_zip_dir, project_id, dataset_id, token)
                    #         if dataset_name: 
                    #             # if not os.path.exists(dataset_path):
                    #             with zipfile.ZipFile(data_zip_dir, 'r') as zip_ref:
                    #                 zip_ref.extractall(dataset_path)

                    #     data_json = read_dataset(dataset_path)
                    #     if data_json:
                    #         if is_correct_format(data_json):
                    #             formatted_data = data_json
                    #         else:
                    #             formatted_data = conver_to_hf_dataset(data_json)

                    #         with open("./llama_recipes/dataset/data_platform.json", 'w') as f:
                    #             json.dump(formatted_data, f, ensure_ascii=False, indent=4)

                    #         datasets_train = "data_platform"
                    # # files = [os.path.join(weight_path, filename) for filename in os.listdir(weight_path) if os.path.isfile(os.path.join(weight_path, filename))]
                    # # train_dir = os.path.join(os.getcwd(),f"yolov9/runs/train")

                    # # script_path = os.path.join(os.getcwd(),f"yolov9/train.py")

                    # train_dir = os.path.join(os.getcwd(), "models")
                    # log_dir = os.path.join(os.getcwd(), "logs")
                    # os.makedirs(train_dir, exist_ok=True)
                    # os.makedirs(log_dir, exist_ok=True)

                    # if dataset_version:
                    #     log_profile =  os.path.join(log_dir, "dataset_version")
                    # else:
                    #     log_profile =  os.path.join(log_dir, "profiler")

                    # # train_dir = os.path.join(os.getcwd(), "models")

                    # os.environ["LOGLEVEL"] = "ERROR"

                    # if configs and configs["entry_file"] != "":
                    #     command = [
                    #         "torchrun",
                    #         "--nproc_per_node", "1", #< count gpu card in compute
                    #         "--rdzv-backend", "c10d",
                    #         "--node-rank", f'{rank}',
                    #         "--nnodes", f'{world_size}',
                    #         "--rdzv-endpoint", f'{master_add}:{master_port}',
                    #         "--master-addr", f'{master_add}',
                    #         "--master-port", f'{master_port}',
                    #         f'{configs["entry_file"]}'
                    #     ]

                    #     if configs["arguments"] and len(configs["arguments"])>0:
                    #         args = configs["arguments"]
                    #         for arg in args:
                    #             command.append(arg['name'])
                    #             command.append(arg['value'])

                    # else:
                    #     command = [
                    #         "torchrun",
                    #         "--nproc_per_node", "1", #< count gpu card in compute
                    #         "--rdzv-backend", "c10d",
                    #         "--node-rank", f'{rank}',
                    #         "--nnodes", f'{world_size}',
                    #         "--rdzv-endpoint", f'{master_add}:{master_port}',
                    #         "--master-addr", f'{master_add}',
                    #         "--master-port", f'{master_port}',
                    #         "llama_recipes/finetuning.py",
                    #         "--model_name", f'{models_train}',
                    #         "--use_peft", 
                    #         "--num_epochs", f'{epochs}',
                    #         "--batch_size_training", "2",
                    #         "--peft_method", "lora",
                    #         "--dataset", f'{datasets_train}',
                    #         "--save_model",
                    #         "--dist_checkpoint_root_folder", "model_checkpoints",
                    #         "--dist_checkpoint_folder", "fine-tuned",
                    #         "--pure_bf16",
                    #         "--save_metrics",
                    #         "--output_dir", "/app/models/",
                    #         "--use_profiler",
                    #         "--profiler_dir", f'{log_profile}'
                    #     ]

                    # # subprocess.run(command, shell=True)
                    # run_train(command, channel_log)

                    # checkpoint_model = f'{train_dir}'
                    # checkpoint_model_zip = f'{train_dir}.zip'
                    # shutil.make_archive(checkpoint_model_zip, 'zip', checkpoint_model)

                    # if os.path.exists(checkpoint_model_zip):
                    #     # print(checkpoint_model)
                    #     checkpoint_name = upload_checkpoint(checkpoint_model_zip, project_id, token)
                    #     if checkpoint_name:
                    #         models_train = "/app/models/"

                train_thread = threading.Thread(target=func_train_model, args=(clone_dir, project_id, imgsz, epochs, token, checkpoint_version, checkpoint_id, dataset_version, dataset_id, ))

                train_thread.start()

                return {"message": "train completed successfully"}
                # # use cache to retrieve the data from the previous fit() runs
              
                # os.environ['MASTER_ADDR'] = 'localhost'
                # port = 29500 + random.randint(0, 500)
                # os.environ['MASTER_PORT'] = f'{port}'
                # print(f"Using localhost:{port=}")
                
                # torch.multiprocessing.spawn(self.debug_sp(0), nprocs=1, args=args)

                # return {"message": "train completed successfully"}
            except Exception as e:
                return {"message": f"train failed: {e}"}

        elif command.lower() == "tensorboard":
            def run_tensorboard():
                # train_dir = os.path.join(os.getcwd(), "{project_id}")
                # log_dir = os.path.join(os.getcwd(), "logs")
                p = subprocess.Popen(f"tensorboard --logdir ./logs --host 0.0.0.0 --port=6006", stdout=subprocess.PIPE, stderr=None, shell=True)
                out = p.communicate()
                print(out)

            tensorboard_thread = threading.Thread(target=run_tensorboard)
            tensorboard_thread.start()
            return {"message": "tensorboardx started successfully"}
        
        elif command.lower() == "dashboard":
            link = promethus_grafana.generate_link_public("ml_00")
            return {"Share_url": link}
          
        elif command.lower() == "predict":
            from huggingface_hub import login 
            import torch
            try:
                hf_access_token = kwargs.get("hf_access_token", "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")
                login(token = hf_access_token)
            except Exception as e:
                return {"message": f"predict failed: {e}"}
                
            prompt = kwargs.get("prompt", "")
            model_id = kwargs.get("model_id", "")
            text = kwargs.get("text", "")
            token_length = kwargs.get("token_lenght", 50)
            task = kwargs.get("task", "")

            predictions = []
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

            if task == "text-generation":
                    # {"project":"296","params":{"task":"text-generation","model_id":"facebook/opt-125m"}}
                _model = pipeline("text-generation", model=model_id)
                result = _model(prompt, max_length=token_length)
                generated_text = result[0]['generated_text']

            elif task == "summarization":
                _model = pipeline("summarization", model=model_id)
                result = _model(prompt, max_length=token_length)
                generated_text = result[0]['summary_text']
            elif task == "question-answering":
                # {"project":"296","params":{"task":"question-answering","model_id":"deepset/roberta-base-squad2"}}
                _model = pipeline("question-answering", model=model_id)
                result = _model(question=prompt, context=text)
                generated_text = result['answer']

            elif task == "translation":
                    # {"project":"296","params":{"task":"translation","model_id":"google-t5/t5-small"}}
                source = kwargs.get("source", "")
                target = kwargs.get("target", "")
                if len(target) == 0 :
                    return {"message": "predict failed", "result": None}
                _model = pipeline("translation_"+source+"_to_"+target,model=model_id)
                result = _model(text)
                generated_text = result[0]['translation_text']

            elif task == "text-classification":
                # {"project":"296","params":{"task":"text-classification","model_id":"distilbert-base-uncased"}}
                _model = pipeline("text-classification", model=model_id)
                result = _model(prompt)
                generated_text = result[0]['label'] #, Score: {result[0]['score']}"

            elif task == "sentiment-analysis":
                    # {"project":"296","params":{"task":"sentiment-analysis","model_id":"tabularisai/robust-sentiment-analysis"}}
                _model = pipeline("sentiment-analysis", model=model_id)
                result = _model(prompt)
                generated_text = result[0]['label'] #f"Sentiment: {result[0]['label']}, Score: {result[0]['score']}"

            elif task == "ner":
                # {"project":"296","params":{"task":"ner","model_id":"kaiku03/bert-base-NER-finetuned_custom_complain_dataset_NER9"}}
                _model = pipeline("ner", model=model_id)
                result = _model(text)
                entities = [(entity['word'], entity['entity']) for entity in result]
                generated_text = entities #f"Named Entities: {entities}"

            elif task == "fill-mask":
                _model = pipeline("fill-mask", model=model_id)
                result = _model(prompt)
                generated_text = result[0]['sequence'] #f"Masked Fill: {result[0]['sequence']}"

            elif task == "text2text-generation":
                    # {"project":"296","params":{"task":"text-generation","model_id":"facebook/opt-125m"}}
                _model = pipeline("text2text-generation", model=model_id)
                result = _model(prompt)
                generated_text = result[0]['generated_text']

            elif task == "multiple-choice":
                # {"project":"296","params":{"task":"multiple-choice","model_id":"iarfmoose/t5-base-question-generator"}}
                _model = pipeline("multiple-choice", model=model_id)
                result = _model(context=prompt, choices=text)
                generated_text = result[0]['answer'] #f"Choice: {result[0]['answer']}"
                
            elif task == "object-detection":
                image_64 = kwargs.get("image")
                model_id = kwargs.get("model_id", "facebook/detr-resnet-50")
                object_detector = pipeline("object-detection", model=model_id)
                object_detection = object_detector(image_64, device=device)

                generated_text = object_detection[0]

            elif task == "image-classification":
                image_64 = kwargs.get("image")
                model_id = kwargs.get("model_id", "google/vit-base-patch16-224")
                image_classification = pipeline("image-classification", model=model_id)
                image_classification = image_classification(image_64, device=device)

                generated_text = image_classification[0]

            elif task == "image-segmentation":
                image_64 = kwargs.get("image")
                model_id = kwargs.get("model_id", "facebook/mask2former-swin-large-coco-panoptic")
                image_segmentation = pipeline("image-segmentation", model=model_id)
                image_segmentation = image_segmentation(image_64, device=device)
                generated_text = image_segmentation[0]
                from io import BytesIO
                import cv2
                import numpy as np
                from PIL import Image

                def mask_to_points(pil_mask: Image.Image):
                    mask_array = np.array(pil_mask)  # Convert PIL to NumPy array

                    # Threshold để chắc chắn là nhị phân (0 hoặc 255)
                    _, thresh = cv2.threshold(mask_array, 127, 255, cv2.THRESH_BINARY)

                    # Tìm contours (đường bao)
                    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    # Chuyển mỗi contour thành list point [(x1, y1), (x2, y2), ...]
                    points = [contour.squeeze().tolist() for contour in contours if contour.shape[0] >= 3]

                    return points
                    
                points = mask_to_points(generated_text['mask'])
                generated_text['mask'] = points

            elif task == "video-classification":
                #  {"project":"296","params":{"task":"video-classification","model_id":"sayakpaul/videomae-base-finetuned-kinetics-finetuned-ucf101-subset"}}
                video_url = kwargs.get("video_url")
                video_classificatio = pipeline("video-classification", model=model_id)
                result=video_classificatio(video_url, device=device)
                generated_text = result[0]
            
            elif task == "text-to-speech":
                model_id = kwargs.get("model_id", "microsoft/speecht5_tts")
                _model = pipeline("text-to-speech", model=model_id, device=device)
                from datasets import load_dataset
                import torch

                embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
                speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
                # You can replace this embedding with your own as well.

                speech = _model(prompt, forward_params={"speaker_embeddings": speaker_embedding})

                buffer = io.BytesIO()
                sf.write(buffer, speech["audio"], samplerate=speech["sampling_rate"], format='WAV')
                audio_bytes = buffer.getvalue()
                audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
                generated_text = audio_base64
            
            elif task == "text-to-audio":
                model_id = kwargs.get("model_id", "facebook/musicgen-small")
                synthesiser = pipeline("text-to-audio", model_id, device=device)
                music = synthesiser(prompt, forward_params={"do_sample": True})
                buffer = io.BytesIO()
                scipy.io.wavfile.write(buffer, rate=music["sampling_rate"], data=music["audio"])

                # Lấy bytes và mã hóa base64
                audio_bytes = buffer.getvalue()
                audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
                generated_text = audio_base64

            elif task == "automatic-speech-recognition":
                import librosa
                import requests
                import soundfile as sf
                import torch
                
                model_id = kwargs.get("model_id", "openai/whisper-small")
                audio_url = kwargs.get("audio_url")


                # Chọn device tự động
                device = "cuda:0" if torch.cuda.is_available() else "cpu"

                # Tạo pipeline cho ASR
                pipe = pipeline(
                    "automatic-speech-recognition",
                    model="openai/whisper-small",
                    chunk_length_s=30,
                    device=device,
                )

                # Load audio thật từ máy bạn
                def load_audio_from_url(url):
                    response = requests.get(url)
                    if response.status_code != 200:
                        raise Exception(f"Failed to download audio: {response.status_code}")
                    audio_bytes = io.BytesIO(response.content)
                    speech_array, sampling_rate = sf.read(audio_bytes)
                    return speech_array, sampling_rate

                # URL file audio của bạn

                # Load audio từ URL
                speech_array, sampling_rate = load_audio_from_url(audio_url)

                # Nếu sampling_rate không phải 16kHz -> resample
                if sampling_rate != 16000:
                    speech_array = librosa.resample(speech_array, orig_sr=sampling_rate, target_sr=16000)
                    sampling_rate = 16000

                output = pipe(
                    {"array": speech_array, "sampling_rate": sampling_rate},
                    batch_size=8,
                    return_timestamps=True,   # Lưu ý: phải dùng "chunk"
                )["chunks"]

                # Format lại cho đẹp: timestamp tuple
                formatted_output = [
                    {
                        "timestamp": (chunk["timestamp"][0], chunk["timestamp"][1]),
                        "text": chunk["text"]
                    }
                    for chunk in output
                ]

                generated_text = formatted_output
            
            elif task == "image-to-text":
                # {"project":"296","params":{"task":"image-to-text","model_id":"Salesforce/blip-image-captioning-base"}}
                from PIL import Image
                image = image_64.replace('data:image/png;base64,', '')
                model_id = kwargs.get("model_id", "Salesforce/blip-image-captioning-base")
                image_text = pipeline("image-to-text", model=model_id)
                result = image_text(Image.open(io.BytesIO(base64.b64decode(image))).convert('RGB'))
                generated_text = result[0]['generated_text']

            elif task == "text-to-image":
                # {"project":"296","params":{"task":"text-to-image","model_id":"akurei/waifu-diffusion"}}
                from diffusers import StableDiffusionPipeline

                pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
                pipe = pipe.to("cuda")
                image = pipe(prompt).images[0]  
                buffer = io.BytesIO()
                # Lưu ảnh vào buffer dưới dạng PNG hoặc một định dạng phù hợp
                image.save(buffer, format="PNG")
                # Đặt con trỏ của buffer về vị trí đầu tiên
                buffer.seek(0)
                # Mã hóa buffer thành chuỗi base64
                generated_text = base64.b64encode(buffer.getvalue()).decode('utf-8')

            elif task == "text-to-video":
                # {"project":"296","params":{"task":"text-to-video","model_id":"damo-vilab/text-to-video-ms-1.7b"}}
                import torch
                from diffusers import DiffusionPipeline
                from diffusers.utils import export_to_video

                pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
                pipe.enable_model_cpu_offload()

                # memory optimization
                pipe.enable_vae_slicing()
                video_frames = pipe(prompt, num_frames=64).frames[0]
                video_path = export_to_video(video_frames)
                with open(video_path, "rb") as videoFile:
                    base64_video = base64.b64encode(videoFile.read())
                    # Mã hóa buffer thành chuỗi base64
                    generated_text = base64_video

            elif task == "image-to-video":
                # {"project":"296","params":{"task":"image-to-video","model_id":"THUDM/CogVideoX-5b-I2V"}}

                import torch
                from diffusers import CogVideoXImageToVideoPipeline
                from diffusers.utils import export_to_video, load_image
                from PIL import Image, ImageDraw

                # prompt = "A vast, shimmering ocean flows gracefully under a twilight sky, its waves undulating in a mesmerizing dance of blues and greens. The surface glints with the last rays of the setting sun, casting golden highlights that ripple across the water. Seagulls soar above, their cries blending with the gentle roar of the waves. The horizon stretches infinitely, where the ocean meets the sky in a seamless blend of hues. Close-ups reveal the intricate patterns of the waves, capturing the fluidity and dynamic beauty of the sea in motion."
                # image = load_image(image="cogvideox_rocket.png")
                image_64 = kwargs.get("image")
                image = image_64.replace('data:image/png;base64,', '')
                pipe = CogVideoXImageToVideoPipeline.from_pretrained(
                    model_id,
                    torch_dtype=torch.bfloat16
                )
                
                pipe.vae.enable_tiling()
                pipe.vae.enable_slicing()

                video = pipe(
                    prompt=prompt,
                    image=Image.open(io.BytesIO(base64.b64decode(image))).convert('RGB'),
                    num_videos_per_prompt=1,
                    num_inference_steps=24,
                    num_frames=24,
                    guidance_scale=6,
                    generator=torch.Generator(device="cuda").manual_seed(42),
                ).frames[0]

                export_to_video(video, "output.mp4", fps=8)
                with open("output.mp4", "rb") as videoFile:
                    base64_video = base64.b64encode(videoFile.read())
                    # Mã hóa buffer thành chuỗi base64
                    generated_text = base64_video
            
            else:
                raise ValueError(f"Task type '{task}' not supported")
            
            predictions.append({
                'result': [{
                    'from_name': "generated_text",
                    'to_name': "text_output",
                    'type': 'textarea',
                    'value': {
                        'text': [generated_text]
                    }
                }],
                'model_version': ""
            })

            return {"message": "predict completed successfully", "result": predictions}
            # except Exception as e:
            #     print(e)
            #     return {"message": "predict failed", "result": None}
        
        elif command.lower() == "stop":
            subprocess.run(["pkill", "-9", "-f", "llama_recipes/finetuning.py"])
            return {"message": "command not supported", "result": "Done"}
        
        elif command.lower() == "action-example":
            return {"message": "Done", "result": "Done"}
        
        else:
            return {"message": "command not supported", "result": None}
            
            
            # return {"message": "train completed successfully"}
        
    @mcp.tool()
    def model(self, **kwargs):
        
        import gradio as gr
        from transformers import pipeline
        task = kwargs.get("task", "text-generation")
        model_id = kwargs.get("model_id","meta-llama/Llama-3.2-1B-Instruct")
        import sys
       
        class Logger:
            def __init__(self, filename):
                self.terminal = sys.stdout
                self.log = open(filename, "w")

            def write(self, message):
                self.terminal.write(message)
                self.log.write(message)
                
            def flush(self):
                self.terminal.flush()
                self.log.flush()
                
            def isatty(self):
                return False    

        sys.stdout = Logger("output.log")
        def read_logs():
            sys.stdout.flush()
            with open("output.log", "r") as f:
                return f.read()
        from huggingface_hub import login 
        hf_access_token = kwargs.get("hf_access_token", "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")
        login(token = hf_access_token)

        def generate_response(user_input):
            # Initialize the text-generation pipeline with your model
            pipe = pipeline(task, model=model_id,token = "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")
            # Generate the response using the pipeline
            result = pipe(user_input) #, max_length=400
            print(result)
            read_logs()
            return result

        def generate_text2text_response(input_text):
            from huggingface_hub import login 
            hf_access_token = kwargs.get("hf_access_token", "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")
            login(token = hf_access_token)
            if torch.cuda.is_available():
                if torch.cuda.is_bf16_supported():
                    dtype = torch.bfloat16
                else:
                    dtype = torch.float16

                print("CUDA is available.")
                
                _model = pipeline(
                    "text-generation",
                    model="meta-llama/Llama-3.2-1B-Instruct", #"meta-llama/Llama-3.2-1B-Instruct", #"meta-llama/Llama-3.2-3B", meta-llama/Llama-3.3-70B-Instruct
                    torch_dtype=dtype, 
                    device_map="auto",  # Hoặc có thể thử "cpu" nếu không ổn,
                    max_new_tokens=256,
                    token = "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"
                )
            else:
                print("No GPU available, using CPU.")
                _model = pipeline(
                    "text-generation",
                    model="meta-llama/Llama-3.2-1B-Instruct", #"meta-llama/Llama-3.2-1B-Instruct", #"meta-llama/Llama-3.2-3B", meta-llama/Llama-3.3-70B-Instruct
                    device_map="cpu",
                    max_new_tokens=256,
                    token = "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"
                )
            messages = [
                {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
                {"role": "user", "content": input_text},
            ]
            # outputs = _model(
            #     messages,
            #     max_new_tokens=256,
            # )
            result = _model(messages, max_length=100)
            generated_text = result[0]['generated_text']
             
            # if input_text and prompt_text:
            #     generated_text = qa_with_context(_model, input_text, prompt_text)
            # elif input_text and not prompt_text:
            #     generated_text = qa_without_context(_model, prompt_text)
            # else:
            #     generated_text = qa_with_context(_model, prompt_text)
            read_logs()
            return generated_text
        def summarization_response(user_input):
            from langchain_huggingface.llms import HuggingFacePipeline
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

            model_id = "meta-llama/Llama-3.2-1B-Instruct"
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(model_id)
            pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=1024,token = "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")
            hf = HuggingFacePipeline(pipeline=pipe)
            from langchain_core.prompts import PromptTemplate

            template = """Summarize the following text into a single, concise paragraph focusing on the key ideas and important points:

                                Text: 
                                {context}

                                Summary:
            """
            prompt = PromptTemplate.from_template(template)

            chain = prompt | hf

            # context = """Elon Reeve Musk FRS (/ˈiːlɒn/; born June 28, 1971) is a businessman known for his key roles in the space company SpaceX and the automotive company Tesla, Inc. His other involvements include ownership of X Corp., the company that operates the social media platform X (formerly Twitter), and his role in the founding of the Boring Company, xAI, Neuralink, and OpenAI. Musk is the wealthiest individual in the world; as of December 2024, Forbes estimates his net worth to be US$432 billion.[2]

            # A member of the wealthy South African Musk family, Musk was born in Pretoria and briefly attended the University of Pretoria before immigrating to Canada at the age of 18, acquiring citizenship through his Canadian-born mother. Two years later, he matriculated at Queen's University at Kingston in Canada. Musk later transferred to the University of Pennsylvania and received bachelor's degrees in economics and physics. He moved to California in 1995 to attend Stanford University but never enrolled in classes, and with his brother Kimbal co-founded the online city guide software company Zip2. The startup was acquired by Compaq for $307 million in 1999. That same year, Musk co-founded X.com, a direct bank. X.com merged with Confinity in 2000 to form PayPal. In 2002, Musk acquired US citizenship, and that October eBay acquired PayPal for $1.5 billion. Using $100 million of the money he made from the sale of PayPal, Musk founded SpaceX, a spaceflight services company, in 2002.

            # In 2004, Musk was an early investor in electric-vehicle manufacturer Tesla Motors, Inc. (later Tesla, Inc.), providing most of the initial financing and assuming the position of the company's chairman. He later became the product architect and, in 2008, the CEO. In 2006, Musk helped create SolarCity, a solar energy company that was acquired by Tesla in 2016 and became Tesla Energy. In 2013, he proposed a hyperloop high-speed vactrain transportation system. In 2015, he co-founded OpenAI, a nonprofit artificial intelligence research company. The following year Musk co-founded Neuralink, a neurotechnology company developing brain–computer interfaces, and The Boring Company, a tunnel construction company. In 2018 the U.S. Securities and Exchange Commission (SEC) sued Musk, alleging that he had falsely announced that he had secured funding for a private takeover of Tesla. To settle the case Musk stepped down as the chairman of Tesla and paid a $20 million fine. In 2022, he acquired Twitter for $44 billion, merged the company into the newly-created X Corp. and rebranded the service as X the following year. In March 2023, Musk founded xAI, an artificial-intelligence company.

            # Musk's actions and expressed views have made him a polarizing figure. He has been criticized for making unscientific and misleading statements, including COVID-19 misinformation, promoting right-wing conspiracy theories, and endorsing an antisemitic trope; he has since apologized for the latter, but continued endorsing such statements. His ownership of Twitter has been controversial because of the layoffs of large numbers of employees, an increase in hate speech, misinformation and disinformation posts on the website, and changes to website features, including verification.

            # By early 2024, Musk became active in American politics as a vocal and financial supporter of Donald Trump, becoming Trump's second-largest individual donor in October 2024. In November 2024, Trump announced that he had chosen Musk along with Vivek Ramaswamy to co-lead Trump's planned Department of Government Efficiency (DOGE) advisory board which will make recommendations on improving government efficiency through measures such as slashing "excess regulations" and cutting "wasteful expenditures".?"""

            final_summary =  chain.invoke({"context": user_input})
            print(final_summary)
            import re
             # Sử dụng regex để trích xuất phần Summary
            summary = re.search(r"Summary:\s*(.+)", final_summary, re.DOTALL)

            if summary:
                final_summary = re.sub(r"[^\w\s.,!?]", "", summary.group(1)).strip()
                print("Summary:", final_summary)
            else:
                print("No summary found.")
            read_logs()
            return final_summary
            # Initialize the text-generation pipeline with your model
            # pipe = pipeline(task, model=model_id)
            # result = pipe(user_input) #, max_length=400
            # print(result)
            # import json
            # return json.dumps(result)
        def question_answering_response(context_textbox,question_textbox):
            from langchain_huggingface.llms import HuggingFacePipeline
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

            model_id = "meta-llama/Llama-3.2-1B-Instruct"
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(model_id)
            pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=100,token = "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")
            hf = HuggingFacePipeline(pipeline=pipe)
            from langchain_core.prompts import PromptTemplate

            template = """
            Here is the context: 
            {context}

            Based on the above context, provide an answer to the following question: 
            {question}

            Answer:
            """
            prompt = PromptTemplate.from_template(template)

            chain = prompt | hf
            final_answer = chain.invoke({"question": question_textbox, "context": context_textbox})
            print(final_answer)
            import re
             # Sử dụng regex để trích xuất phần Summary
            answer = re.search(r"Answer:\s*(.+)", final_answer, re.DOTALL)

            if answer:
                final_answer = re.sub(r"[^\w\s.,!?]", "", answer.group(1)).strip()
                print("Answer:", final_answer)
            else:
                print("No Answer found.")
            read_logs()
            return final_answer
            # Initialize the text-generation pipeline with your model
        #     pipe = pipeline(task, model=model_id)
        #    # Generate the response using the pipeline
        #     QA_input = {
        #         'question': question_textbox,
        #         'context': context_textbox
        #     }
        #     result = pipe(QA_input, max_length=400)
        #     print(result)
        #     import json
        #     return json.dumps(result)
        def translation_response(user_input,_source_language, _target_language): #,audio_source,input_audio_mic,input_audio_file
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
            t = AutoTokenizer.from_pretrained(model_id)
            m = AutoModelForSeq2SeqLM.from_pretrained(model_id)
            _model = pipeline(task, model=m, tokenizer=t, src_lang=_source_language, tgt_lang=_target_language,token = "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")
            result = _model(user_input)
            print(result)
            # import json
            # return json.dumps(result)
            read_logs()
            return  result[0]['translation_text']
        def run_image_classification(user_input):
            # Initialize the text-generation pipeline with your model
            pipe = pipeline(task, model=model_id,token = "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")
            result = pipe(user_input)
            print(result)
            read_logs()
            import json
            return json.dumps(result)
        # def run_detect_objects(user_input):
        #     # Initialize the text-generation pipeline with your model
        #     pipe = pipeline(task, model=model_id)
        #    # Generate the response using the pipeline
        #     response = pipe(user_input)[0]['generated_text']
        #     print(response)
        #     return response
        with gr.Blocks() as demo_text_generation:
            with gr.Row():
                with gr.Column():
                    with gr.Group():
                        input_text = gr.Textbox(label="Input text")
                    btn = gr.Button("Translate")
                with gr.Column():
                    output_text = gr.Textbox(label="Output text")

            # gr.Examples(
            #     inputs=[input_text, source_language, target_language],
            #     outputs=output_text,
            #     fn=generate_response,
            #     api_name=False,
            # )

            gr.on(
                triggers=[input_text.submit, btn.click],
                fn=generate_response,
                inputs=[input_text],
                outputs=output_text,
                api_name=task,
            )
        def generate_audio(prompt):
             # Initialize the text-generation pipeline with your model
            pipe = pipeline(task, model=model_id,token = "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")
            # Generate the response using the pipeline facebook/musicgen-medium
            result = pipe(prompt)
            print(result)
            # Audio(output["audio"], rate=output["sampling_rate"])
            read_logs()
            return result["audio"]
        with gr.Blocks() as demo_summarization:
            with gr.Row():
                with gr.Column():
                    with gr.Group():
                        input_text = gr.Textbox(label="Input text")
                    btn = gr.Button("Submit")
                with gr.Column():
                   output_text = gr.Textbox(label="Response:") #  gr.Label(label="Response: ")
                    

            # gr.Examples(
            #     inputs=[input_text, source_language, target_language],
            #     outputs=output_text,
            #     fn=summarization_response,
            #     api_name=False,
            # )

            gr.on(
                triggers=[input_text.submit, btn.click],
                fn=summarization_response,
                inputs=[input_text],
                outputs=output_text,
                api_name=task,
            )
        with gr.Blocks() as demo_question_answering:
            with gr.Row():
                with gr.Column():
                    with gr.Group():
                        context_textbox = gr.Textbox(label="Context text")
                        question_textbox = gr.Textbox(label="Question text")
                       
                    btn = gr.Button("Submit")
                with gr.Column():
                    output_text =   gr.Textbox(label="Response:")
            

            # gr.Examples(
            #    inputs=[input_text],
            #     outputs=output_text,
            #     fn=question_answering_response,
            #     api_name=False,
            # )

            gr.on(
                triggers=[input_text.submit, btn.click],
                fn=question_answering_response,
                inputs=[context_textbox,question_textbox],
                outputs=output_text,
                api_name=task,
            )
        language_code_to_name = {
            "af": "Afrikaans",
            "am": "Amharic",
            "ar": "Modern Standard Arabic",
            "ar": "Moroccan Arabic",
            "ar": "Egyptian Arabic",
            "as": "Assamese",
            "as": "Asturian",
            "az": "North Azerbaijani",
            "be": "Belarusian",
            "be": "Bengali",
            "bo": "Bosnian",
            "bu": "Bulgarian",
            "ca": "Catalan",
            "ce": "Cebuano",
            "ce": "Czech",
            "ck": "Central Kurdish",
            "cm": "Mandarin Chinese",
            "cy": "Welsh",
            "da": "Danish",
            "de": "German",
            "el": "Greek",
            "en": "English",
            "es": "Estonian",
            "eu": "Basque",
            "fi": "Finnish",
            "fr": "French",
            "ga": "West Central Oromo",
            "gl": "Irish",
            "gl": "Galician",
            "gu": "Gujarati",
            "he": "Hebrew",
            "hi": "Hindi",
            "hr": "Croatian",
            "hu": "Hungarian",
            "hy": "Armenian",
            "ib": "Igbo",
            "in": "Indonesian",
            "is": "Icelandic",
            "it": "Italian",
            "ja": "Javanese",
            "jp": "Japanese",
            "ka": "Kamba",
            "ka": "Kannada",
            "ka": "Georgian",
            "ka": "Kazakh",
            "ke": "Kabuverdianu",
            "kh": "Halh Mongolian",
            "kh": "Khmer",
            "ki": "Kyrgyz",
            "ko": "Korean",
            "la": "Lao",
            "li": "Lithuanian",
            "lt": "Luxembourgish",
            "lu": "Ganda",
            "lu": "Luo",
            "lv": "Standard Latvian",
            "ma": "Maithili",
            "ma": "Malayalam",
            "ma": "Marathi",
            "mk": "Macedonian",
            "ml": "Maltese",
            "mn": "Meitei",
            "my": "Burmese",
            "nl": "Dutch",
            "nn": "Norwegian Nynorsk",
            "no": "Norwegian Bokm\u00e5l",
            "np": "Nepali",
            "ny": "Nyanja",
            "oc": "Occitan",
            "or": "Odia",
            "pa": "Punjabi",
            "pb": "Southern Pashto",
            "pe": "Western Persian",
            "po": "Polish",
            "po": "Portuguese",
            "ro": "Romanian",
            "ru": "Russian",
            "sl": "Slovak",
            "sl": "Slovenian",
            "sn": "Shona",
            "sn": "Sindhi",
            "so": "Somali",
            "sp": "Spanish",
            "sr": "Serbian",
            "sw": "Swedish",
            "sw": "Swahili",
            "ta": "Tamil",
            "te": "Telugu",
            "tg": "Tajik",
            "tg": "Tagalog",
            "th": "Thai",
            "tu": "Turkish",
            "uk": "Ukrainian",
            "ur": "Urdu",
            "uz": "Northern Uzbek",
            "vi": "Vietnamese",
            "xh": "Xhosa",
            "yo": "Yoruba",
            "yu": "Cantonese",
            "zl": "Colloquial Malay",
            "zs": "Standard Malay",
            "zu": "Zulu",
        }


        # Source langs: S2ST / S2TT / ASR don't need source lang
        # T2TT / T2ST use this
        text_source_language_codes = [
            "af",
            "am",
            "ar",
            "ar",
            "ar",
            "as",
            "az",
            "be",
            "be",
            "bo",
            "bu",
            "ca",
            "ce",
            "ce",
            "ck",
            "cm",
            "cy",
            "da",
            "de",
            "el",
            "en",
            "es",
            "eu",
            "fi",
            "fr",
            "ga",
            "gl",
            "gl",
            "gu",
            "he",
            "hi",
            "hr",
            "hu",
            "hy",
            "ib",
            "in",
            "is",
            "it",
            "ja",
            "jp",
            "ka",
            "ka",
            "ka",
            "kh",
            "kh",
            "ki",
            "ko",
            "la",
            "li",
            "lu",
            "lu",
            "lv",
            "ma",
            "ma",
            "ma",
            "mk",
            "ml",
            "mn",
            "my",
            "nl",
            "nn",
            "no",
            "np",
            "ny",
            "or",
            "pa",
            "pb",
            "pe",
            "po",
            "po",
            "ro",
            "ru",
            "sl",
            "sl",
            "sn",
            "sn",
            "so",
            "sp",
            "sr",
            "sw",
            "sw",
            "ta",
            "te",
            "tg",
            "tg",
            "th",
            "tu",
            "uk",
            "ur",
            "uz",
            "vi",
            "yo",
            "yu",
            "zs",
            "zu",
        ]

        TEXT_SOURCE_LANGUAGE_NAMES = sorted([language_code_to_name[code] for code in text_source_language_codes])
        ASR_TARGET_LANGUAGE_NAMES = TEXT_SOURCE_LANGUAGE_NAMES
        with gr.Blocks() as demo_translation:
            with gr.Row():
                with gr.Column():
                    # with gr.Row() as audio_box:
                    #     audio_source = gr.Radio(
                    #         label="Audio source",
                    #         choices=["file", "microphone"],
                    #         value="file",
                    #     )
                    #     input_audio_mic = gr.Audio(
                    #         label="Input speech",
                    #         type="filepath",
                    #         source="microphone",
                    #         visible=False,
                    #     )
                    #     input_audio_file = gr.Audio(
                    #         label="Input speech",
                    #         type="filepath",
                    #         source="upload",
                    #         visible=True,
                    #     )
                    with gr.Group():
                        input_text = gr.Textbox(label="Input text")
                        _source_language = gr.Dropdown(
                            label="Source language",
                            choices=ASR_TARGET_LANGUAGE_NAMES,
                            value="English",
                        )
                        _target_language = gr.Dropdown(
                            label="Target language",
                            choices=ASR_TARGET_LANGUAGE_NAMES,
                            value="English",
                        )
                    btn = gr.Button("Submit")
                with gr.Column():
                    output_text =  gr.Textbox(label="Response:")

            # gr.Examples(
            #     inputs=[input_text, source_language, target_language],
            #     outputs=output_text,
            #     fn=translation_response,
            #     api_name=False,
            # )

            gr.on(
                triggers=[input_text.submit, btn.click],
                fn=translation_response,
                inputs=[input_text, _source_language, _target_language], #,audio_source,input_audio_mic,input_audio_file
                outputs=output_text,
                api_name=task,
            )
        
        def generate_classification_response(input_text,categories_text):
            from langchain_huggingface.llms import HuggingFacePipeline
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

            model_id = "meta-llama/Llama-3.2-1B-Instruct"
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(model_id)
            pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=1024,token = "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")
            hf = HuggingFacePipeline(pipeline=pipe)
            from langchain_core.prompts import PromptTemplate

            template = """Classify the following text into one of the following categories:
                {categories}

                The text is: {context}

                Classification:
            """
            prompt = PromptTemplate.from_template(template)
            final_classification = ""
            # class Joke(BaseModel):
            #     Answer: str = Field(description="answer of user")
            #     Intent: str = Field(description="type of the intent")

            # https://github.com/langchain-ai/langchain/discussions/21661
            # from langchain_core.output_parsers import JsonOutputParser
            # parser = JsonOutputParser(pydantic_object=Joke)

            chain = prompt | hf

            # context = """Elon Reeve Musk FRS (/ˈiːlɒn/; born June 28, 1971) is a businessman known for his key roles in the space company SpaceX and the automotive company Tesla, Inc. His other involvements include ownership of X Corp., the company that operates the social media platform X (formerly Twitter), and his role in the founding of the Boring Company, xAI, Neuralink, and OpenAI. Musk is the wealthiest individual in the world; as of December 2024, Forbes estimates his net worth to be US$432 billion.[2]

            # A member of the wealthy South African Musk family, Musk was born in Pretoria and briefly attended the University of Pretoria before immigrating to Canada at the age of 18, acquiring citizenship through his Canadian-born mother. Two years later, he matriculated at Queen's University at Kingston in Canada. Musk later transferred to the University of Pennsylvania and received bachelor's degrees in economics and physics. He moved to California in 1995 to attend Stanford University but never enrolled in classes, and with his brother Kimbal co-founded the online city guide software company Zip2. The startup was acquired by Compaq for $307 million in 1999. That same year, Musk co-founded X.com, a direct bank. X.com merged with Confinity in 2000 to form PayPal. In 2002, Musk acquired US citizenship, and that October eBay acquired PayPal for $1.5 billion. Using $100 million of the money he made from the sale of PayPal, Musk founded SpaceX, a spaceflight services company, in 2002.

            # In 2004, Musk was an early investor in electric-vehicle manufacturer Tesla Motors, Inc. (later Tesla, Inc.), providing most of the initial financing and assuming the position of the company's chairman. He later became the product architect and, in 2008, the CEO. In 2006, Musk helped create SolarCity, a solar energy company that was acquired by Tesla in 2016 and became Tesla Energy. In 2013, he proposed a hyperloop high-speed vactrain transportation system. In 2015, he co-founded OpenAI, a nonprofit artificial intelligence research company. The following year Musk co-founded Neuralink, a neurotechnology company developing brain–computer interfaces, and The Boring Company, a tunnel construction company. In 2018 the U.S. Securities and Exchange Commission (SEC) sued Musk, alleging that he had falsely announced that he had secured funding for a private takeover of Tesla. To settle the case Musk stepped down as the chairman of Tesla and paid a $20 million fine. In 2022, he acquired Twitter for $44 billion, merged the company into the newly-created X Corp. and rebranded the service as X the following year. In March 2023, Musk founded xAI, an artificial-intelligence company.

            # Musk's actions and expressed views have made him a polarizing figure. He has been criticized for making unscientific and misleading statements, including COVID-19 misinformation, promoting right-wing conspiracy theories, and endorsing an antisemitic trope; he has since apologized for the latter, but continued endorsing such statements. His ownership of Twitter has been controversial because of the layoffs of large numbers of employees, an increase in hate speech, misinformation and disinformation posts on the website, and changes to website features, including verification.

            # By early 2024, Musk became active in American politics as a vocal and financial supporter of Donald Trump, becoming Trump's second-largest individual donor in October 2024. In November 2024, Trump announced that he had chosen Musk along with Vivek Ramaswamy to co-lead Trump's planned Department of Government Efficiency (DOGE) advisory board which will make recommendations on improving government efficiency through measures such as slashing "excess regulations" and cutting "wasteful expenditures".?"""
            # categories = "Tesla Motors,SpaceX,paypal"
            result = chain.invoke({"context": input_text,"categories":categories_text})
            print(result)
            import re
            # Sử dụng regex để trích xuất phần Classification
            classification = re.search(r"Classification:\s*(.*)", result)
            
            if classification:
                final_classification = re.sub(r"[^\w\s]", "", classification.group(1)).strip()
                print("Classification:", final_classification)
            else:
                print("No classification found.")
            read_logs()
            return final_classification
        
        with gr.Blocks() as demo_text_classification:
            with gr.Row():
                with gr.Column():
                    with gr.Group():
                        input_text = gr.Textbox(label="Input text")
                        categories_text = gr.Textbox(label="Categories text")
                    btn = gr.Button("Submit")
                with gr.Column():
                    output_text = gr.Textbox(label="Response:")

            
            gr.on(
                triggers=[input_text.submit,categories_text.submit, btn.click],
                fn=generate_classification_response,
                inputs=[input_text,categories_text],
                outputs=output_text,
                api_name=task,
            )
        def sentiment_classifier(text):
            try:
                sentiment_classifier = pipeline("sentiment-analysis",token = "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")
                sentiment_response = sentiment_classifier(text)
                label = sentiment_response[0]['label']
                score = sentiment_response[0]['score']
                print(sentiment_response)
                read_logs()
                import json
                return f"label:{label} score:{score}"
            except Exception as e:
                return str(e)
        with gr.Blocks() as demo_sentiment_analysis:
            with gr.Row():
                with gr.Column():
                    with gr.Group():
                        input_text = gr.Textbox(label="Input text")
                    btn = gr.Button("Submit")
                with gr.Column():
                    
                    label_text = gr.Label(label="Label: ")
                    score_text = gr.Label(label="Score: ")

            # gr.Examples(
            #     inputs=[input_text, source_language, target_language],
            #     outputs=output_text,
            #     fn=generate_response,
            #     api_name=False,
            # )

            gr.on(
                triggers=[input_text.submit, btn.click],
                fn=sentiment_classifier,
                inputs=[input_text],
                outputs=[label_text,score_text],
                api_name=task,
            )
        def predict_entities(text,categories_text):
            from langchain_huggingface.llms import HuggingFacePipeline
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

            model_id = "meta-llama/Llama-3.2-1B-Instruct"
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(model_id)
            pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=1024,token = "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")
            hf = HuggingFacePipeline(pipeline=pipe)
            from langchain_core.prompts import PromptTemplate

            template = """The text is: {context}

                Extract all named entities from the context and classify them into the categories:
                {categories}

                Named Entities-classification:
            """
            prompt = PromptTemplate.from_template(template)

            chain = prompt | hf

            # context = """Elon Reeve Musk FRS (/ˈiːlɒn/; born June 28, 1971) is a businessman known for his key roles in the space company SpaceX and the automotive company Tesla, Inc. His other involvements include ownership of X Corp., the company that operates the social media platform X (formerly Twitter), and his role in the founding of the Boring Company, xAI, Neuralink, and OpenAI. Musk is the wealthiest individual in the world; as of December 2024, Forbes estimates his net worth to be US$432 billion.[2]

            # A member of the wealthy South African Musk family, Musk was born in Pretoria and briefly attended the University of Pretoria before immigrating to Canada at the age of 18, acquiring citizenship through his Canadian-born mother. Two years later, he matriculated at Queen's University at Kingston in Canada. Musk later transferred to the University of Pennsylvania and received bachelor's degrees in economics and physics. He moved to California in 1995 to attend Stanford University but never enrolled in classes, and with his brother Kimbal co-founded the online city guide software company Zip2. The startup was acquired by Compaq for $307 million in 1999. That same year, Musk co-founded X.com, a direct bank. X.com merged with Confinity in 2000 to form PayPal. In 2002, Musk acquired US citizenship, and that October eBay acquired PayPal for $1.5 billion. Using $100 million of the money he made from the sale of PayPal, Musk founded SpaceX, a spaceflight services company, in 2002.

            # In 2004, Musk was an early investor in electric-vehicle manufacturer Tesla Motors, Inc. (later Tesla, Inc.), providing most of the initial financing and assuming the position of the company's chairman. He later became the product architect and, in 2008, the CEO. In 2006, Musk helped create SolarCity, a solar energy company that was acquired by Tesla in 2016 and became Tesla Energy. In 2013, he proposed a hyperloop high-speed vactrain transportation system. In 2015, he co-founded OpenAI, a nonprofit artificial intelligence research company. The following year Musk co-founded Neuralink, a neurotechnology company developing brain–computer interfaces, and The Boring Company, a tunnel construction company. In 2018 the U.S. Securities and Exchange Commission (SEC) sued Musk, alleging that he had falsely announced that he had secured funding for a private takeover of Tesla. To settle the case Musk stepped down as the chairman of Tesla and paid a $20 million fine. In 2022, he acquired Twitter for $44 billion, merged the company into the newly-created X Corp. and rebranded the service as X the following year. In March 2023, Musk founded xAI, an artificial-intelligence company.

            # Musk's actions and expressed views have made him a polarizing figure. He has been criticized for making unscientific and misleading statements, including COVID-19 misinformation, promoting right-wing conspiracy theories, and endorsing an antisemitic trope; he has since apologized for the latter, but continued endorsing such statements. His ownership of Twitter has been controversial because of the layoffs of large numbers of employees, an increase in hate speech, misinformation and disinformation posts on the website, and changes to website features, including verification.

            # By early 2024, Musk became active in American politics as a vocal and financial supporter of Donald Trump, becoming Trump's second-largest individual donor in October 2024. In November 2024, Trump announced that he had chosen Musk along with Vivek Ramaswamy to co-lead Trump's planned Department of Government Efficiency (DOGE) advisory board which will make recommendations on improving government efficiency through measures such as slashing "excess regulations" and cutting "wasteful expenditures".?"""
            # categories = "Tesla Motors,SpaceX,paypal"
            final_entities = chain.invoke({"context": text,"categories": categories_text})
            print(final_entities)
            import re
             # Sử dụng regex để trích xuất phần Summary
             # Trích xuất phần "Named Entities-classification:" và parse các NER
            ner_classification = re.search(r"Named Entities-classification:\s*(.*)", final_entities, re.DOTALL)

            if ner_classification:
                # Lấy danh sách các entity từ kết quả, chia theo dòng
                final_entities = ner_classification.group(1).strip()
                # entities = entities_text.split("\n")

                # # Duyệt qua các entity và chuyển đổi thành format mong muốn
                # for entity in entities:
                #     match = re.match(r"\d+\.\s*(\w+):\s*(.*)", entity.strip())
                #     if match:
                #         entity_type = match.group(1).upper()  # Loại entity (Person, Location, Organization)
                #         entity_value = match.group(2).strip()  # Giá trị entity
                        
                #         # Kiểm tra nếu value có nhiều địa điểm, tách ra
                #         if entity_type == 'LOCATION' and ',' in entity_value:
                #             # Tách value nếu chứa dấu phẩy
                #             location_values = [val.strip() for val in entity_value.split(',')]
                #             # Thêm từng phần vào final_entities dưới dạng các đối tượng riêng biệt
                #             for location in location_values:
                #                 final_entities.append({"type": "LOCATION", "value": location})
                #         else:
                #             # Thêm entity vào danh sách nếu không phải LOCATION hoặc không có dấu phẩy
                #             final_entities.append({"type": entity_type, "value": entity_value})
            read_logs()
            return final_entities
            #  # Initialize the text-generation pipeline with your model
            # pipe = pipeline(task, model=model_id)
            # # Use the loaded model to identify entities in the text
            # entities = pipe(text)
            # # Highlight identified entities in the input text
            # highlighted_text = text
            # for entity in entities:
            #     entity_text = text[entity['start']:entity['end']]
            #     replacement = f"<span style='border: 2px solid green;'>{entity_text}</span>"
            #     highlighted_text = highlighted_text.replace(entity_text, replacement)
            # return highlighted_text
        with gr.Blocks() as demo_ner:
            with gr.Row():
                with gr.Column():
                    with gr.Group():
                        input_text = gr.Textbox(label="Input text")
                        categories_text = gr.Textbox(label="Categories text")
                    btn = gr.Button("Submit")
                with gr.Column():
                    output_text = gr.HTML()

            # gr.Examples(
            #     inputs=[input_text],
            #     outputs=output_text,
            #     fn=generate_response,
            #     api_name=False,
            # )

            gr.on(
                triggers=[input_text.submit,categories_text.submit, btn.click],
                fn=predict_entities,
                inputs=[input_text,categories_text],
                outputs=output_text,
                api_name=task,
            )
        
        with gr.Blocks() as demo_text2text_generation:
            with gr.Row():
                with gr.Column():
                    with gr.Group():
                        input_text = gr.Textbox(label="Input text")
                    btn = gr.Button("Submit")
                with gr.Column():
                    output_text = gr.Textbox(label="Output text")

            # gr.Examples(
            #     inputs=[input_text],
            #     outputs=output_text,
            #     fn=generate_response,
            #     api_name=False,
            # )

            gr.on(
                triggers=[input_text.submit, btn.click],
                fn=generate_text2text_response,
                inputs=[input_text],
                outputs=output_text,
                api_name=task,
            )

        import questiongenerator as qs
        from questiongenerator import QuestionGenerator
        qg = QuestionGenerator()

        def Extract_QA(qlist):
                i=0
                question_i= qlist[i]['question']
                Choices_ans= []
                Choice_is_correct=[]
                for j in range(4):
                    Choices_ans= Choices_ans+ [qlist[i]['answer'][j]['answer']]
                    Choice_is_correct= Choice_is_correct+ [qlist[i]['answer'][j]['correct']]
                
                Q=f"""
                    Q: {question_i}
                    A. {Choices_ans[0]}
                    B. {Choices_ans[1]}
                    C. {Choices_ans[2]}
                    D. {Choices_ans[3]}   
                    """
                xs=['A','B','C','D']
                result = [x for x, y in zip(xs, Choice_is_correct) if y ]
                A= f"""
                    The rigth answer is: {result[0]}
                    """
                return (Q,A)
        def ReurnAnswer(input_text):
            qlist= qg.generate(input_text, num_questions=1, answer_style="multiple_choice")
            Q,A= Extract_QA(qlist)
            read_logs()
            return A

        def GetQuestion(input_text):
            qlist= qg.generate(input_text, num_questions=1, answer_style="multiple_choice")
            Q,A= Extract_QA(qlist)
            read_logs()
            return Q

        with gr.Blocks() as demo_multiple_choice:
            with gr.Row():
                input_text = gr.Textbox(label="Input text")
            with gr.Row():
                with gr.Column():
                    Gen_Question = gr.Button(value="Show the Question")
                    Gen_Answer = gr.Button(value="Show the Answer")
                    
                with gr.Column():
                    question = gr.Textbox(label="Question(s)")
                    Answer = gr.Textbox(label="Answer(s)")
            Gen_Question.click(GetQuestion, inputs=input_text, outputs=question, api_name="QuestionGenerator")
            Gen_Answer.click(ReurnAnswer, inputs=input_text, outputs=Answer, api_name="AnswerGenerator")
            gr.on(
               triggers=[Gen_Question.click,Gen_Answer.click],
                # fn=run_object_detectionn,
                inputs=[Gen_Question,Gen_Answer],
                outputs=[question,Answer],
                api_name=task,
            )
        def run_object_detectionn(image):
             # Initialize the text-generation pipeline with your model
           
            object_detector = pipeline("object-detection", model=model_id)
            from PIL import Image, ImageDraw
            # Draw bounding box definition
            def draw_bounding_box(im, score, label, xmin, ymin, xmax, ymax, index, num_boxes):
                """ Draw a bounding box. """

                print(f"Drawing bounding box {index} of {num_boxes}...")

                # Draw the actual bounding box
                im_with_rectangle = ImageDraw.Draw(im)  
                im_with_rectangle.rounded_rectangle((xmin, ymin, xmax, ymax), outline = "red", width = 5, radius = 10)

                # Draw the label
                im_with_rectangle.text((xmin+35, ymin-25), label, fill="white", stroke_fill = "red")
                # Draw the score
                im_with_rectangle.text((xmin+35, ymin-25), score, fill="white", stroke_fill = "green")

                # Return the intermediate result
                return im
            # Open the image
            # with Image.open(image).convert('RGB') as im:

            # Perform object detection
            bounding_boxes = object_detector(image)

            # Iteration elements
            num_boxes = len(bounding_boxes)
            index = 0

            # Draw bounding box for each result
            for bounding_box in bounding_boxes:

                # Get actual box
                box = bounding_box["box"]

                # Draw the bounding box
                im = draw_bounding_box(im, bounding_box["score"], bounding_box["label"],\
                    box["xmin"], box["ymin"], box["xmax"], box["ymax"], index, num_boxes)

                # Increase index by one
                index += 1

            # generated_text = {
            #     'x': box['x'],
            #     'y': box['y'],
            #     'width': box['xmin']-box["xmax"],
            #     'height':box['ymin']-box["ymax"],
            # }
            return im
            
        with gr.Blocks() as demo_object_detection:
            with gr.Row():
                with gr.Column():
                    with gr.Group():
                        input_text = gr.Image(label="Upload Image", type="pil")
                      
                    btn = gr.Button("Submit")
                with gr.Column():
                    output_text = gr.Image(label="Submit")
            gr.on(
               triggers=[btn.click],
                fn=run_object_detectionn,
                inputs=[input_text],
                outputs=output_text,
                api_name=task,
            )

       
        with gr.Blocks() as demo_image_classification:
            with gr.Row():
                with gr.Column():
                    with gr.Group():
                        input_text = gr.Image(label="Upload Image", type="pil" )
                      
                    btn = gr.Button("Submit")
                with gr.Column():
                    output_text = gr.Textbox(label="Output text")


            gr.on(
               triggers=[btn.click],
                fn=run_image_classification,
                inputs=[input_text],
                outputs=output_text,
                api_name=task,
            )
        def run_image_segmentation(user_input):
             # Initialize the text-generation pipeline with your model
            pipe = pipeline(task, model=model_id,token = "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")
            result = pipe(user_input)
            print(result)
            # result[0]["mask"]
            # result[0]["label"]
            # result[0]["score"]
            import numpy as np
            from PIL import Image, ImageDraw, ImageFont
            # def overlay_mask(img, mask, alpha=0.5):
            #     import cv2
            #     # Define color mapping
            #     colors = {
            #         0: [255, 0, 0],   # Class 0 - Red
            #         1: [0, 255, 0],   # Class 1 - Green
            #         2: [0, 0, 255]    # Class 2 - Blue
            #         # Add more colors for additional classes if needed
            #     }

            #     # Create a blank colored overlay image
            #     overlay = np.zeros_like(img)

            #     # Map each mask value to the corresponding color
            #     for class_id, color in colors.items():
            #         overlay[mask == class_id] = color

            #     # Blend the overlay with the original image
            #     output = cv2.addWeighted(img, 1 - alpha, overlay, alpha, 0)

            #     return output
            import cv2
            import numpy as np
            def drawOverlay(image,logo,alpha=1.0,x=0, y=0, scale=1.0):
                (h, w) = image.shape[:2]
                image = np.dstack([image, np.ones((h, w), dtype="uint8") * 255])
                overlay = cv2.resize(logo, None,fx=scale,fy=scale)
                (wH, wW) = overlay.shape[:2]
                output = image.copy()
                # blend the two images together using transparent overlays
                try:
                    if x<0 : x = w+x
                    if y<0 : y = h+y
                    if x+wW > w: wW = w-x  
                    if y+wH > h: wH = h-y
                    print(x,y,wW,wH)
                    overlay=cv2.addWeighted(output[y:y+wH, x:x+wW],alpha,overlay[:wH,:wW],1.0,0)
                    output[y:y+wH, x:x+wW ] = overlay
                except Exception as e:
                    print("Error: Logo position is overshooting image!")
                    print(e)
                output= output[:,:,:3]
                return output
            def draw_mask(mask, draw, random_color=False, label=''):
                if random_color:
                    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), 153)
                else:
                    color = (30, 144, 255, 153)

                nonzero_coords = np.transpose(np.nonzero(mask))

                for coord in nonzero_coords:
                    draw.point(coord[::-1], fill=color)
                # draw.text((box[0], box[1]), str(label), fill="white")
                # draw.text((box[0], box[1]), label)
           

            from PIL import Image, ImageDraw
            size = user_input.size # w, h
            mask_image = Image.new('RGBA', size, color=(0, 0, 0, 0))
            mask_draw = ImageDraw.Draw(mask_image)
            for item in result:
                draw_mask(item["mask"], mask_draw, random_color=True)
            image = np.ascontiguousarray(user_input)
            
            # Blend the overlay with the original image
            return drawOverlay(image,np.array(mask_image))
            # return cv2.addWeighted(image, 1 - alpha, np.zeros_like(mask_image), alpha, 0)
            # return result
        with gr.Blocks() as demo_image_segmentation:
            with gr.Row():
                with gr.Column():
                    with gr.Group():
                        input_text = gr.Image(label="Upload Image", type="pil" )
                      
                    btn = gr.Button("Submit")
                with gr.Column():
                    output_text = gr.Image(label="Image Segmentation")

            gr.on(
                triggers=[btn.click],
                fn=run_image_segmentation,
                inputs=[input_text],
                outputs=output_text,
                api_name=task,
            )
        with gr.Blocks() as demo_fill_mask:
            with gr.Row():
                with gr.Column():
                    with gr.Group():
                        input_text = gr.Textbox(label="Input text")
                    btn = gr.Button("Submit")
                with gr.Column():
                    output_text = gr.Textbox(label="Output text")

            # gr.Examples(
            #     inputs=[input_text, source_language, target_language],
            #     outputs=output_text,
            #     fn=generate_response,
            #     api_name=False,
            # )

            gr.on(
                triggers=[input_text.submit, btn.click],
                fn=generate_response,
                inputs=[input_text],
                outputs=output_text,
                api_name=task,
            )
            
        def run_video_classification(user_input):
             # Initialize the text-generation pipeline with your model
            pipe = pipeline(task, model=model_id,token = "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")
            result = pipe(user_input)
            print(result)
            read_logs()
            import json
            return json.dumps(result)
        with gr.Blocks() as demo_video_classification:
            with gr.Row():
                with gr.Column():
                    with gr.Group():
                        input_text = gr.Video(label="Input Video", format="mp4")
                      
                    btn = gr.Button("Submit")
                with gr.Column():
                     output_text = gr.Textbox(label="Output text")
            gr.on(
                triggers=[btn.click],
                fn=run_video_classification,
                inputs=[input_text],
                outputs=output_text,
                api_name=task,
            )
        # transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base.en")

        def transcribe(microphone,input_upload):
             # Initialize the text-generation pipeline with your model
            pipe = pipeline(task, model=model_id,token = "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")
            if microphone != None:
                audio = microphone
            else:
                audio = input_upload
            read_logs()
            return pipe(audio)["text"] 
            # import numpy as np
            # if microphone != None:
            #     sr, y = microphone
            # else:
            #     sr, y = input_upload

            # # Convert to mono if stereo
            # if y.ndim > 1:
            #     y = y.mean(axis=1)
                
            # y = y.astype(np.float32)
            # y /= np.max(np.abs(y))

            # return pipe({"sampling_rate": sr, "raw": y})["text"] 
        with gr.Blocks() as demo_automatic_peech_recognition:
            with gr.Row():
                with gr.Column():
                    with gr.Group():
                        input_microphone =gr.Audio(sources=["microphone"], type="filepath",label="microphone", streaming=True)
                        input_upload =gr.Audio(sources=["upload"], type="filepath",label="upload")
                      
                    btn = gr.Button("Submit")
                with gr.Column():
                    output_text = gr.Textbox(label="Output text")


            gr.on(
                triggers=[btn.click],
                fn=transcribe,
                inputs=[input_microphone,input_upload],
                outputs=output_text,
                api_name=task
            )
            # gr.Interface(
            #     fn=transcribe,
            #     inputs=[
            #         gr.Audio(source="microphone", type="filepath", streaming=True),
            #     ],
            #     outputs=[
            #         "textbox",
            #     ],
            #     live=True)
        with gr.Blocks() as demo_text_to_audio:
            with gr.Row():
                with gr.Column():
                    with gr.Group():
                        input_text = gr.Textbox(label="Input text")
                      
                    btn = gr.Button("Submit")
                with gr.Column():
                    output_text =  gr.Audio(
                            label="Translated speech",
                            autoplay=False,
                            streaming=False,
                            type="numpy",
                        )


            gr.on(
                triggers=[input_text.submit, btn.click],
                fn=generate_audio,
                inputs=[input_text],
                outputs=output_text,
                api_name=task,
            )

        with gr.Blocks() as demo_image_to_text:
            with gr.Row():
                with gr.Column():
                    with gr.Group():
                        input_text = gr.Image(label="Upload Image", type="pil" )
                      
                    btn = gr.Button("Submit")
                with gr.Column():
                    output_text = gr.Textbox(label="Output text")


            gr.on(
               triggers=[btn.click],
                fn=run_image_classification,
                inputs=[input_text],
                outputs=output_text,
                api_name=task,
            )
        def text_to_image(input_text):
            from diffusers import StableDiffusionPipeline

            pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
            pipe = pipe.to("cuda")
            result=pipe(input_text)
            print(result)
            image = result.images[0]  

            return image
        with gr.Blocks() as demo_text_to_image:
            with gr.Row():
                with gr.Column():
                    with gr.Group():
                        input_text = gr.Textbox(label="Input text") 
                      
                    btn = gr.Button("Submit")
                with gr.Column():
                    output_text = gr.Image(label="Image", type="pil", )

            gr.on(
                triggers=[input_text.submit, btn.click],
                fn=text_to_image,
                inputs=[input_text],
                outputs=output_text,
                api_name=task,
            )
        def run_text_to_video(prompt):
                import torch
                from diffusers import DiffusionPipeline
                from diffusers.utils import export_to_video

                pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
                pipe.enable_model_cpu_offload()

                # memory optimization
                pipe.enable_vae_slicing()
                video_frames = pipe(prompt, num_frames=64).frames[0]
                video_path = export_to_video(video_frames)
                import cv2
                from PIL import Image
                import torch
                import time
                import uuid

                SUBSAMPLE = 2
                conf_threshold = 0.5    
                cap = cv2.VideoCapture(video_path)

                video_codec = cv2.VideoWriter_fourcc(*"mp4v") # type: ignore
                fps = int(cap.get(cv2.CAP_PROP_FPS))

                desired_fps = fps // SUBSAMPLE
                width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) // 2
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) // 2

                iterating, frame = cap.read()

                n_frames = 0

                name = f"output_{uuid.uuid4()}.mp4"
                segment_file = cv2.VideoWriter(name, video_codec, desired_fps, (width, height)) # type: ignore
                batch = []

                while iterating:
                    frame = cv2.resize( frame, (0,0), fx=0.5, fy=0.5)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    if n_frames % SUBSAMPLE == 0:
                        batch.append(frame)
                    if len(batch) == 2 * desired_fps:
                        end = time.time()
                        start = time.time()
                        # Convert RGB to BGR
                        frame = frame[:, :, ::-1].copy()
                        segment_file.write(frame)

                        batch = []
                        segment_file.release()
                        yield name
                        end = time.time()
                        print("time taken for processing boxes", end - start)
                        name = f"output_{uuid.uuid4()}.mp4"
                        segment_file = cv2.VideoWriter(name, video_codec, desired_fps, (width, height)) # type: ignore

                    iterating, frame = cap.read()
                    n_frames += 1
                # return send_file(os.path.join(video_path, 'video.mp4'), as_attachment=True)
              
        with gr.Blocks() as demo_text_to_video:
            with gr.Row():
                with gr.Column():
                    with gr.Group():
                        input_text = gr.Textbox(label="Input text")
                      
                    btn = gr.Button("Submit")
                with gr.Column():
                    output_text = gr.Video(label="Processed Video", streaming=True, autoplay=True)

            gr.on(
                triggers=[input_text.submit, btn.click],
                fn=run_text_to_video,
                inputs=[input_text],
                outputs=output_text,
                api_name=task,
            )
        def run_image_to_video(image,prompt):
                import torch
                from diffusers import CogVideoXImageToVideoPipeline
                from diffusers.utils import export_to_video, load_image
                from PIL import Image
                if prompt == None or prompt == "":
                    prompt = "A vast, shimmering ocean flows gracefully under a twilight sky, its waves undulating in a mesmerizing dance of blues and greens. The surface glints with the last rays of the setting sun, casting golden highlights that ripple across the water. Seagulls soar above, their cries blending with the gentle roar of the waves. The horizon stretches infinitely, where the ocean meets the sky in a seamless blend of hues. Close-ups reveal the intricate patterns of the waves, capturing the fluidity and dynamic beauty of the sea in motion."
                # image = load_image(image="cogvideox_rocket.png")
                # image_64 = kwargs.get("image")
                # image = image_64.replace('data:image/png;base64,', '')
                pipe = CogVideoXImageToVideoPipeline.from_pretrained(
                    model_id,
                    torch_dtype=torch.bfloat16
                )
                
                pipe.vae.enable_tiling()
                pipe.vae.enable_slicing()

                video = pipe(
                    prompt=prompt,
                    image=Image.open(io.BytesIO(base64.b64decode(image))).convert('RGB'),
                    num_videos_per_prompt=1,
                    num_inference_steps=24,
                    num_frames=24,
                    guidance_scale=6,
                    generator=torch.Generator(device="cuda").manual_seed(42),
                ).frames[0]

                import cv2
                from PIL import Image
                import torch
                import time
                import uuid

                SUBSAMPLE = 2
                conf_threshold = 0.5    
                cap = cv2.VideoCapture(export_to_video(video, "output.mp4", fps=8))

                video_codec = cv2.VideoWriter_fourcc(*"mp4v") # type: ignore
                fps = int(cap.get(cv2.CAP_PROP_FPS))

                desired_fps = fps // SUBSAMPLE
                width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) // 2
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) // 2

                iterating, frame = cap.read()

                n_frames = 0

                name = f"output_{uuid.uuid4()}.mp4"
                segment_file = cv2.VideoWriter(name, video_codec, desired_fps, (width, height)) # type: ignore
                batch = []

                while iterating:
                    frame = cv2.resize( frame, (0,0), fx=0.5, fy=0.5)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    if n_frames % SUBSAMPLE == 0:
                        batch.append(frame)
                    if len(batch) == 2 * desired_fps:
                        end = time.time()
                        start = time.time()
                        # Convert RGB to BGR
                        frame = frame[:, :, ::-1].copy()
                        segment_file.write(frame)

                        batch = []
                        segment_file.release()
                        yield name
                        end = time.time()
                        print("time taken for processing boxes", end - start)
                        name = f"output_{uuid.uuid4()}.mp4"
                        segment_file = cv2.VideoWriter(name, video_codec, desired_fps, (width, height)) # type: ignore

                    iterating, frame = cap.read()
                    n_frames += 1
                # return send_file(os.path.join(export_to_video(video, "output.mp4", fps=8), 'video.mp4'), as_attachment=True)
                   
        with gr.Blocks() as demo_image_to_video:
            with gr.Row():
                with gr.Column():
                    with gr.Group():
                        input_text = gr.Image(label="Upload Image", type="pil" )
                        prompt_text = gr.Textbox(label="Prompt text") 
                      
                    btn = gr.Button("Submit")
                with gr.Column():
                    output_text = gr.Video(label="Processed Video", streaming=True, autoplay=True)


            gr.on(
               triggers=[btn.click],
                fn=run_image_to_video,
                inputs=[input_text,prompt_text],
                outputs=output_text,
                api_name=task,
            )
        DESCRIPTION = """\
        # Huggingface UI
        This is a demo of Huggingface UI.
        """
        with gr.Blocks(css="style.css") as demo:
            gr.Markdown(DESCRIPTION)

            with gr.Tabs():
                if task == "text-generation":
                  with gr.Tab(label=task):
                        demo_text_generation.render()
                elif task == "summarization":
                  with gr.Tab(label=task):
                        demo_summarization.render()
                elif task == "question-answering":
                   with gr.Tab(label=task):
                        demo_question_answering.render()
                elif task == "translation":
                  with gr.Tab(label=task):
                        demo_translation.render()
                elif task == "text-classification":
                    with gr.Tab(label=task):
                            demo_text_classification.render()
                elif task == "sentiment-analysis":
                  with gr.Tab(label=task):
                        demo_sentiment_analysis.render()
                elif task == "ner":
                   with gr.Tab(label=task):
                        demo_ner.render()
                elif task == "fill-mask":
                  with gr.Tab(label=task):
                        demo_fill_mask.render()
                elif task == "text2text-generation":
                   with gr.Tab(label=task):
                        demo_text2text_generation.render()
                elif task == "multiple-choice":
                   with gr.Tab(label=task):
                        demo_multiple_choice.render()
                elif task == "object-detection":
                   with gr.Tab(label=task):
                        demo_object_detection.render()
                elif task == "image-classification":
                    with gr.Tab(label=task):
                        demo_image_classification.render()
                elif task == "image-segmentation":
                   with gr.Tab(label=task):
                        demo_image_segmentation.render()
                elif task == "video-classification":
                    with gr.Tab(label=task):
                        demo_video_classification.render()
                elif task == "automatic-speech-recognition" or task == "speech-to-text":
                   with gr.Tab(label=task):
                        demo_automatic_peech_recognition.render()
                elif task == "text-to-audio" or  task == "text-to-speech":
                    with gr.Tab(label=task):
                            demo_text_to_audio.render()
                elif task == "image-to-text":
                    with gr.Tab(label=task):
                        demo_image_to_text.render()
                elif task == "text-to-image":
                   with gr.Tab(label=task):
                        demo_text_to_image.render()
                elif task == "text-to-video":
                   with gr.Tab(label=task):
                        demo_text_to_video.render()
                elif task == "image-to-video":
                    with gr.Tab(label=task):
                        demo_image_to_video.render()
                else:
                    return {"share_url": "", 'local_url': ""}
        gradio_app, local_url, share_url = demo.launch(share=True, quiet=True, prevent_thread_lock=True, server_name='0.0.0.0',show_error=True)
   
        return {"share_url": share_url, 'local_url': local_url}
    
    @mcp.tool()
    def model_trial(self, **kwargs):
        return {"message": "Done", "result": "Done"}
        from huggingface_hub import login 
        hf_access_token = kwargs.get("hf_access_token", "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")
        login(token = hf_access_token)

        model_id = kwargs.get("model_id", "bigscience/bloomz-1b7")  #"tiiuae/falcon-7b" "bigscience/bloomz-1b7" `zanchat/falcon-1b` `appvoid/llama-3-1b` meta-llama/Llama-3.2-3B` `mistralai/Mistral-7B-v0.1` `bigscience/bloomz-1b7` `Qwen/Qwen2-1.5B`
        dataset_id = kwargs.get("dataset_id","lucasmccabe-lmi/CodeAlpaca-20k")
        num_train_epochs = kwargs.get("num_train_epochs", 3)
        per_device_train_batch_size = kwargs.get("per_device_train_batch_size", 3)
        gradient_accumulation_steps = kwargs.get("gradient_accumulation_steps", 2)
        gradient_checkpointing = kwargs.get("gradient_checkpointing", True)
        optim = kwargs.get("optim", "adamw_torch_fused")
        logging_steps = kwargs.get("logging_steps", "")
        save_strategy = kwargs.get("save_strategy", "epoch")
        learning_rate = kwargs.get("learning_rate", 2e-4)
        bf16 = kwargs.get("bf16", False)
        tf32 = kwargs.get("tf32", False)
        max_grad_norm = kwargs.get("max_grad_norm", 0.3)
        warmup_ratio = kwargs.get("warmup_ratio", 0.03)
        lora_alpha = kwargs.get("lora_alpha", 128)
        lora_dropout = kwargs.get("lora_dropout", 0.05)
        bias = kwargs.get("bias", "none")
        target_modules = kwargs.get("target_modules", "all-linear")
        task_type = kwargs.get("task_type", "CAUSAL_LM")
        use_cpu = kwargs.get("use_cpu", True)
        task = kwargs.get("task", "summarization")
        remove_unused_columns =kwargs.get("remove_unused_columns", False)
        max_seq_length = kwargs.get("max_seq_length", 1024)
        
        import gradio as gr 
        css = """
        .feedback .tab-nav {
            justify-content: center;
        }

        .feedback button.selected{
            background-color:rgb(115,0,254); !important;
            color: #ffff !important;
        }

        .feedback button{
            font-size: 16px !important;
            color: black !important;
            border-radius: 12px !important;
            display: block !important;
            margin-right: 17px !important;
            border: 1px solid var(--border-color-primary);
        }

        .feedback div {
            border: none !important;
            justify-content: center;
            margin-bottom: 5px;
        }

        .feedback .panel{
            background: none !important;
        }


        .feedback .unpadded_box{
            border-style: groove !important;
            width: 500px;
            height: 345px;
            margin: auto;
        }

        .feedback .secondary{
            background: rgb(225,0,170);
            color: #ffff !important;
        }

        .feedback .primary{
            background: rgb(115,0,254);
            color: #ffff !important;
        }

        .upload_image button{
            border: 1px var(--border-color-primary) !important;
        }
        .upload_image {
            align-items: center !important;
            justify-content: center !important;
            border-style: dashed !important;
            width: 500px;
            height: 345px;
            padding: 10px 10px 10px 10px
        }
        .upload_image .wrap{
            align-items: center !important;
            justify-content: center !important;
            border-style: dashed !important;
            width: 500px;
            height: 345px;
            padding: 10px 10px 10px 10px
        }

        .webcam_style .wrap{
            border: none !important;
            align-items: center !important;
            justify-content: center !important;
            height: 345px;
        }

        .webcam_style .feedback button{
            border: none !important;
            height: 345px;
        }

        .webcam_style .unpadded_box {
            all: unset !important;
        }

        .btn-custom {
            background: rgb(0,0,0) !important;
            color: #ffff !important;
            width: 200px;
        }

        .title1 {
            margin-right: 90px !important;
        }

        .title1 block{
            margin-right: 90px !important;
        }

        """

        with gr.Blocks(css=css) as demo:
            with gr.Row():
                with gr.Column(scale=10):
                    gr.Markdown(
                        """
                        # Theme preview: `AIxBlock`
                        """
                    )

           
            # def predict(input_img):
            
                # result = self.action(project, "predict",collection="",data={"img":input_img})
                # print(result)
                # if result['result']:
                #     boxes = result['result']['boxes']
                #     names = result['result']['names']
                #     labels = result['result']['labels']
                    
                #     for box, label in zip(boxes, labels):
                #         box = [int(i) for i in box]
                #         label = int(label)
                #         input_img = cv2.rectangle(input_img, box, color=(255, 0, 0), thickness=2)
                #         # input_img = cv2.(input_img, names[label], (box[0], box[1]), color=(255, 0, 0), size=1)
                #         input_img = cv2.putText(input_img, names[label], (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        
                # return input_img
            
            def download_btn():
                # print(f"Downloading {dataset_choosen}")
                return f'<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css"><a href="#" style="font-size:50px"> <i class="fa fa-download"></i> Download this dataset</a>'
                
            def trial_training(file_upload,dataset_id,model_id):
                read_logs()
                
                from datasets import load_dataset
                if dataset_id == None:
                    dataset_id ="thisisanshgupta/CodeAlpacaSmall"

                import torch
                from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
                from trl import setup_chat_format,SFTTrainer
                
                # Hugging Face model id
                if model_id == None:
                    model_id = "appvoid/llama-3-1b" # or  `appvoid/llama-3-1b` tiiuae/falcon-7b` `mistralai/Mistral-7B-v0.1` `bigscience/bloomz-1b7` `Qwen/Qwen2-1.5B`
                
                # BitsAndBytesConfig int-4 config
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
                )
                
                # Load model and tokenizer
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    device_map="auto",
                    low_cpu_mem_usage=True,
                    use_cache=True
                )
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                tokenizer.padding_side = 'right' # to prevent warnings
                
                # # set chat template to OAI chatML, remove if you start from a fine-tuned model
                model, tokenizer = setup_chat_format(model, tokenizer)
                from peft import LoraConfig
                # LoRA config based on QLoRA paper & Sebastian Raschka experiment
                peft_config = LoraConfig(
                        lora_alpha=32,
                        lora_dropout=0.05,
                        r=16,
                        bias="none",
                        task_type="CAUSAL_LM",
                        
                )
                from transformers import TrainingArguments
                # Load dataset from the hub
                dataset = load_dataset(dataset_id, split="train")
                # Create train and eval splits
                dataset = dataset.shuffle()
                train_dataset = dataset.select(range(0, math.floor(len(dataset)*0.8))) # 80% of dataset for training
                eval_dataset = dataset.select(range(0, math.floor(len(dataset)*0.2))) #  20% of dataset for training
                args = TrainingArguments(
                    output_dir= os.getenv('MODEL_DIR', './data/checkpoint'), # directory to save and repository id
                    num_train_epochs=num_train_epochs,                     # number of training epochs
                    per_device_train_batch_size=3,          # batch size per device during training
                    gradient_accumulation_steps=2,          # number of steps before performing a backward/update pass
                    gradient_checkpointing=True,            # use gradient checkpointing to save memory
                    optim="adamw_torch_fused",              # use fused adamw optimizer
                    logging_steps=10,                       # log every 10 steps
                    save_strategy="epoch",                  # save checkpoint every epoch
                    learning_rate=2e-4,                     # learning rate, based on QLoRA paper
                    bf16=False,                              # use bfloat16 precision
                    tf32=False,                              # use tf32 precision
                    max_grad_norm=0.3,                      # max gradient norm based on QLoRA paper
                    warmup_ratio=0.03,                      # warmup ratio based on QLoRA paper
                    lr_scheduler_type="constant",           # use constant learning rate scheduler
                    push_to_hub=True,                       # push model to hub
                    report_to="tensorboard",                # report metrics to tensorboard
                    use_cpu=use_cpu,
                    remove_unused_columns=remove_unused_columns
                )
               
                # max_seq_length = 1024 # max sequence length for model and packing of the dataset
                from transformers import TrainerCallback

                class TrainOnStartCallback(TrainerCallback):
                    def on_train_begin(self, args, state, control, logs=None, **kwargs):
                        # Log training loss at step 0
                        logs = logs or {}
                        logs["train/loss"] = None  # Replace None with an initial value if available
                        logs["train/global_step"] = 0
                        self.log(logs)

                    def log(self, logs):
                        print(f"Logging at start: {logs}")
                       
                trainer = SFTTrainer(
                    model=model,
                    args=args,
                    train_dataset=train_dataset,
                    peft_config=peft_config,
                    max_seq_length=max_seq_length,
                    tokenizer=tokenizer,
                    packing=True,
                    dataset_kwargs={
                        "add_special_tokens": False,  # We template with special tokens
                        "append_concat_token": False, # No need to add additional separator token
                        'skip_prepare_dataset': True # skip the dataset preparation
                    },
                    callbacks=[TrainOnStartCallback()]
                )
                # start training, the model will be automatically saved to the hub and the output directory
                trainer.train()
                import pathlib
                # save model
                MODEL_DIR = os.getenv('MODEL_DIR', './data/checkpoint')
                FINETUNED_MODEL_NAME = os.getenv('FINETUNED_MODEL_NAME', 'finetuned_model')
                chk_path = str(pathlib.Path(MODEL_DIR) / FINETUNED_MODEL_NAME)
                print(f"Model is trained and saved as {chk_path}")
                trainer.save_model(chk_path)
                # trainer.push_to_hub()
                # free the memory again
                del model
                del trainer
                torch.cuda.empty_cache()
               
                import torch
                from peft import AutoPeftModelForCausalLM
                from transformers import AutoTokenizer, pipeline
                
                # peft_model_id = f"./data/checkpoint/{model_id}"
                # peft_model_id = args.output_dir
                
                # Load Model with PEFT adapter
                model = AutoPeftModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                torch_dtype=torch.float16
                )
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                # load into pipeline
                pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

                from datasets import load_dataset
                from random import randint
                
                
                # Load our test dataset
                eval_dataset = train_dataset #load_dataset("json", data_files="test_dataset.json", split="train")
                rand_idx = randint(0, len(eval_dataset))
                
                # Test on sample
                prompt = pipe.tokenizer.apply_chat_template(eval_dataset[rand_idx]["messages"][:2], tokenize=False, add_generation_prompt=True)
                outputs = pipe(prompt, max_new_tokens=256, do_sample=False, temperature=0.1, top_k=50, top_p=0.1, eos_token_id=pipe.tokenizer.eos_token_id, pad_token_id=pipe.tokenizer.pad_token_id)
                
                print(f"Query:\n{eval_dataset[rand_idx]['messages'][1]['content']}")
                print(f"Original Answer:\n{eval_dataset[rand_idx]['messages'][2]['content']}")
                print(f"Generated Answer:\n{outputs[0]['generated_text'][len(prompt):].strip()}")

                from tqdm import tqdm
                
                
                def evaluate(sample):
                    prompt = pipe.tokenizer.apply_chat_template(sample["messages"][:2], tokenize=False, add_generation_prompt=True)
                    outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95, eos_token_id=pipe.tokenizer.eos_token_id, pad_token_id=pipe.tokenizer.pad_token_id)
                    predicted_answer = outputs[0]['generated_text'][len(prompt):].strip()
                    if predicted_answer == sample["messages"][2]["content"]:
                        return 1
                    else:
                        return 0
                
                success_rate = []
                number_of_eval_samples = 1000
                # iterate over eval dataset and predict
                for s in tqdm(eval_dataset.shuffle().select(range(number_of_eval_samples))):
                    success_rate.append(evaluate(s))
                
                # compute accuracy
                accuracy = sum(success_rate)/len(success_rate)
                
                print(f"Accuracy: {accuracy*100:.2f}%")
            import sys
            from gradio_log import Log
            class Logger:
                def __init__(self, filename):
                    self.terminal = sys.stdout
                    self.log = open(filename, "w")

                def write(self, message):
                    self.terminal.write(message)
                    self.log.write(message)
                    
                def flush(self):
                    self.terminal.flush()
                    self.log.flush()
                    
                def isatty(self):
                    return False    

            sys.stdout = Logger("output.log")
            def read_logs():
                sys.stdout.flush()
                with open("output.log", "r") as f:
                    return f.read()
            # def get_checkpoint_list(project):
            #     print("GETTING CHECKPOINT LIST")
            #     print(f"Proejct: {project}")
            #     import os
            #     checkpoint_list = [i for i in os.listdir("my_ml_backend/models") if i.endswith(".pt")]
            #     checkpoint_list = [f"<a href='./my_ml_backend/checkpoints/{i}' download>{i}</a>" for i in checkpoint_list]
            #     if os.path.exists(f"my_ml_backend/{project}"):
            #         for folder in os.listdir(f"my_ml_backend/{project}"):
            #             if "train" in folder:
            #                 project_checkpoint_list = [i for i in os.listdir(f"my_ml_backend/{project}/{folder}/weights") if i.endswith(".pt")]
            #                 project_checkpoint_list = [f"<a href='./my_ml_backend/{project}/{folder}/weights/{i}' download>{folder}-{i}</a>" for i in project_checkpoint_list]
            #                 checkpoint_list.extend(project_checkpoint_list)
                
            #     return "<br>".join(checkpoint_list)
            with gr.Tabs(elem_classes=["feedback"]) as parent_tabs:
                with gr.TabItem("Demo", id=0):   
                    with gr.Row():
                        gr.Markdown("## Input", elem_classes=["title1"])
                        gr.Markdown("## Output", elem_classes=["title1"])
                    with gr.Row():
                            with gr.Column():
                                gr.Image(elem_classes=["upload_image"], sources="upload", container = False, height = 345,show_label = False)
                            with gr.Column():
                                gr.Image(elem_classes=["upload_image"],container = False, height = 345,show_label = False)
                    # gr.Interface(predict, gr.Image(elem_classes=["upload_image"], sources="upload", container = False, height = 345,show_label = False), 
                    #             gr.Image(elem_classes=["upload_image"],container = False, height = 345,show_label = False), allow_flagging = False ,flagging_mode="auto"          
                    # )

                with gr.TabItem("Train", id=2):
                    gr.Markdown("# seems like you might be referring to a trial training process, possibly in the context of testing a model, running a small-scale experiment, or an initial training phase before scaling up. In both PyTorch and TensorFlow, this could mean training a model on a subset of data or with fewer resources to test the setup, assess performance, or debug before the full training run.")
                    with gr.Column():
                        with gr.Row():
                            with gr.Column():
                                gr.Markdown("## Dataset template to prepare your own and initiate training")
                            with gr.Column(min_width=90):
                                gr.Button("Download this dataset", variant="primary").click(download_btn, gr.HTML())
                            #when the button is clicked, download the dataset from dropdown
                            # download_btn
                        gr.Markdown("## Upload your sample dataset(file or dataset,model from huggingface) to have a trial training")
                        with gr.Row():
                            with gr.Column():
                                # gr.Interface(predict, gr.File(file_types=['tar','zip']),gr.Label(elem_classes=["upload_image"],container = False), allow_flagging = False)  
                                file_upload = gr.File(file_types=['tar','zip'])
                                dataset_id=gr.Textbox(label="Dataset ID(Huggingface)")
                                model_id=gr.Textbox(label="Model ID(Huggingface)")     
                            with gr.Column():
                                console_logs= Log("output.log", dark=True, xterm_font_size=12,height=402)
                                # console_logs= gr.Code(label="", language="shell", elem_classes=["upload_image"], container=False,lines=20)
                        # https://github.com/gradio-app/gradio/issues/2362
                        with gr.Row():
                            gr.Markdown(f"## You can attemp up to {2} FLOps")
                            btn = gr.Button("Submit")
                    gr.on(
                        triggers=[btn.click],
                        fn=trial_training,
                        inputs=[file_upload,dataset_id,model_id],outputs=[console_logs]
                    )
                
        gradio_app, local_url, share_url = demo.launch(share=True, quiet=True, prevent_thread_lock=True, server_name='0.0.0.0',show_error=True)
   
        return {"share_url": share_url, 'local_url': local_url}
#         https://huggingface.co/spaces/Rohit1412/Data_Dynamous_deepfake/blob/main/app.py
# video-classification

# https://huggingface.co/spaces/henryRosero/image-classification/blob/main/app.py
# image-classification

# https://huggingface.co/spaces/kaiku03/NER/blob/main/app.py
# ner

# https://huggingface.co/spaces/DpShirazi/Examiner/blob/main/app.py
# multiple-choice

# https://huggingface.co/spaces/DoomSlayer9743/Text-to-audio/blob/main/app.py
# text-to-audio

# https://huggingface.co/spaces/sairamtelagamsetti/story/blob/main/app.py
# text-to-image


# https://huggingface.co/spaces/szili2011/Prompt2Movie/blob/main/app.py

# text-to-video

# https://huggingface.co/spaces/tdurzynski/object-detection/blob/main/app.py
# object-detection

# https://huggingface.co/spaces/tonyassi/image-segmentation/tree/main
# image-segmentation

# https://huggingface.co/spaces/gradio/question-answering/tree/main
# question-answering


# https://huggingface.co/spaces/ismot/hel5/blob/main/app.py
# text-classification

# https://huggingface.co/spaces/UmeshAdabala/paragraph_summarizer/blob/main/app.py
# text-summarization


# https://huggingface.co/spaces/DoomSlayer9743/Translation/blob/main/app.py
# Translation

        # DESCRIPTION = """\
        # Huggingface UI Demo
        # """

        # with gr.Blocks(css="style.css") as demo:
        #     gr.Markdown(DESCRIPTION)

        #     with gr.Tabs():
        #         if task == "text-generation":
        #           with gr.Tab(label="S2ST"):
        #                 demo_text_generation.render()
        #         elif task == "summarization":
        #           with gr.Tab(label="S2ST"):
        #                 demo_summarization.render()
        #         elif task == "question-answering":
        #            with gr.Tab(label="S2ST"):
        #                 demo_question_answering.render()
        #         elif task == "translation":
        #           with gr.Tab(label="S2ST"):
        #                 demo_translation.render()
        #         elif task == "text-classification":
        #             with gr.Tab(label="S2ST"):
        #                     demo_text_classification.render()
        #         elif task == "sentiment-analysis":
        #           with gr.Tab(label="S2ST"):
        #                 demo_sentiment_analysis.render()
        #         elif task == "ner":
        #            with gr.Tab(label="S2ST"):
        #                 demo_ner.render()
        #         elif task == "fill-mask":
        #           with gr.Tab(label="S2ST"):
        #                 demo_fill_mask.render()
        #         elif task == "text2text-generation":
        #            with gr.Tab(label="S2ST"):
        #                 demo_text2text_generation.render()
        #         elif task == "multiple-choice":
        #            with gr.Tab(label="S2ST"):
        #                 demo_multiple_choice.render()
        #         elif task == "object-detection":
        #            with gr.Tab(label="S2ST"):
        #                 demo_object_detection.render()
        #         elif task == "image-classification":
        #             with gr.Tab(label="S2ST"):
        #                 demo_image_classification.render()
        #         elif task == "image-segmentation":
        #            with gr.Tab(label="S2ST"):
        #                 demo_image_segmentation.render()
        #         elif task == "video-classification":
        #             with gr.Tab(label="S2ST"):
        #                 demo_video_classification.render()
        #         elif task == "automatic-speech-recognition":
        #            with gr.Tab(label="S2ST"):
        #                 demo_automatic_peech_recognition.render()
        #         elif task == "text-to-audio" or  task == "text-to-speech":
        #             with gr.Tab(label="S2ST"):
        #                     demo_text_to_audio.render()
        #         elif task == "image-to-text":
        #             with gr.Tab(label="S2ST"):
        #                 demo_image_to_text.render()
        #         elif task == "text-to-image":
        #            with gr.Tab(label="S2ST"):
        #                 demo_text_to_image.render()
        #         elif task == "text-to-video":
        #            with gr.Tab(label="S2ST"):
        #                 demo_text_to_video.render()
        #         elif task == "image-to-video":
        #             with gr.Tab(label="S2ST"):
        #                 demo_image_to_video.render()
        #         else:
        #             return {"share_url": "", 'local_url': ""}
                
                
                        

        # gradio_app, local_url, share_url = demo.launch(share=True, quiet=True, prevent_thread_lock=True, server_name='0.0.0.0',show_error=True)
   
        # return {"share_url": share_url, 'local_url': local_url}
    
    def download(self, **kwargs):
        return super().download(**kwargs)