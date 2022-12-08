import whisper
import boto3
from urllib.parse import urlparse


def model_fn(model_dir):
    model = whisper.load_model("large-v2")
    return model


def transcribe_from_s3(model, s3_file, language=None):
    s3 = boto3.client('s3')
    o = urlparse(s3_file, allow_fragments=False)
#     print(s3_file)
#     print(o)
    bucket = o.netloc
    key = o.path.lstrip('/')
#     print(bucket)
#     print(key)
    if len(bucket) == 0:
        bucket = 'sagemaker-us-east-1-905847418383'
        key = 'whisper/data/test/he/test-he-000.wav'
    s3.download_file(bucket, key, 'tmp.wav')
    result = model.transcribe('tmp.wav', language=language)
    return result["language"], result["text"]


def predict_fn(data, model):
#     print(data)
    s3_file = data.pop("s3_file")
    language = data.pop("language", None)
#     print(s3_file)
#     print(language)
    detected_language, transcription = transcribe_from_s3(model, s3_file, language)
    
    return {"detected_language": detected_language, "transcription": transcription}
