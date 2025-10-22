Using python 3.10

pip install -r requirements.txt

Run with code:
python grpc_server.py --restore_step 900000 -p config/LJSpeech/preprocess.yaml -m config/LJSpeech/model.yaml -t config/LJSpeech/train.yaml --port 50051 --max_workers 10

Run with docker:
docker run --gpus all -p 50051:50051 fastspeech2-service \
    python grpc_server.py \
    --restore_step 900000 \
    -p config/LJSpeech/preprocess.yaml \
    -m config/LJSpeech/model.yaml \
    -t config/LJSpeech/train.yaml \
    --port 50051 \
    --max_workers 10