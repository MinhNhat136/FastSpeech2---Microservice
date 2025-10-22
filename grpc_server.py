import grpc
from concurrent import futures
import argparse
import yaml
import numpy as np
import torch
import io
import re
from string import punctuation
from scipy.io import wavfile
from g2p_en import G2p
from pypinyin import pinyin, Style

import tts_service_pb2_grpc
import tts_service_pb2

from utils.model import get_model, get_vocoder
from utils.tools import to_device
from text import text_to_sequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"""
RUN:  
python grpc_server.py --restore_step 900000 -p config/LJSpeech/preprocess.yaml -m config/LJSpeech/model.yaml -t config/LJSpeech/train.yaml --port 50051 --max_workers 10
"""

def read_lexicon(lex_path):
    lexicon = {}
    with open(lex_path) as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
    return lexicon

def preprocess_english(text, preprocess_config):
    text = text.rstrip(punctuation)
    lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])

    g2p = G2p()
    phones = []
    words = re.split(r"([,;.\-\?\!\s+])", text)
    for w in words:
        if w.lower() in lexicon:
            phones += lexicon[w.lower()]
        else:
            phones += list(filter(lambda p: p != " ", g2p(w)))
    phones = "{" + "}{".join(phones) + "}"
    phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
    phones = phones.replace("}{", " ")

    sequence = np.array(
        text_to_sequence(
            phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
        )
    )

    return np.array(sequence)

def preprocess_mandarin(text, preprocess_config):
    lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])

    phones = []
    pinyins = [
        p[0]
        for p in pinyin(
            text, style=Style.TONE3, strict=False, neutral_tone_with_five=True
        )
    ]
    for p in pinyins:
        if p in lexicon:
            phones += lexicon[p]
        else:
            phones.append("sp")

    phones = "{" + " ".join(phones) + "}"
    sequence = np.array(
        text_to_sequence(
            phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
        )
    )

    return np.array(sequence)

class TTSServicer(tts_service_pb2_grpc.TTSServiceServicer):
    def __init__(self, model, vocoder, configs):
        self.model = model
        self.vocoder = vocoder
        self.preprocess_config, self.model_config, self.train_config = configs

    def _synthesize_audio(self, text, speaker_id, pitch_control, energy_control, duration_control):
        language = self.preprocess_config["preprocessing"]["text"]["language"]
        if language == "en":
            text_sequence = preprocess_english(text, self.preprocess_config)
        elif language == "zh":
            text_sequence = preprocess_mandarin(text, self.preprocess_config)
        else:
            raise ValueError(f"Unsupported language: {language}")

        ids = [text[:100]]
        raw_texts = [text[:100]]
        speakers = np.array([speaker_id])
        texts = np.array([text_sequence])
        text_lens = np.array([len(text_sequence)])
        batch = (ids, raw_texts, speakers, texts, text_lens, max(text_lens))

        batch = to_device(batch, device)

        with torch.no_grad():
            output = self.model(
                *(batch[2:]),
                p_control=pitch_control,
                e_control=energy_control,
                d_control=duration_control
            )

            mel_predictions = output[1].transpose(1, 2)
            mel_lens = output[9]

            # Convert mel to audio using vocoder
            from utils.model import vocoder_infer
            lengths = mel_lens * self.preprocess_config["preprocessing"]["stft"]["hop_length"]
            wav_predictions = vocoder_infer(
                mel_predictions,
                self.vocoder,
                self.model_config,
                self.preprocess_config,
                lengths=lengths
            )

            wav = wav_predictions[0]

        return wav

    def Synthesize(self, request, context):
        """Synchronous synthesis - returns complete audio"""
        try:
            text = request.text
            speaker_id = request.speaker_id if request.speaker_id else 0
            pitch_control = request.pitch_control if request.pitch_control else 1.0
            energy_control = request.energy_control if request.energy_control else 1.0
            duration_control = request.duration_control if request.duration_control else 1.0
            audio_format = request.audio_format if request.audio_format else "wav"

            if not text:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("Text is required")
                return tts_service_pb2.SynthesizeResponse()

            wav = self._synthesize_audio(text, speaker_id, pitch_control, energy_control, duration_control)

            sampling_rate = self.preprocess_config["preprocessing"]["audio"]["sampling_rate"]

            if audio_format == "wav":
                wav_io = io.BytesIO()
                wavfile.write(wav_io, sampling_rate, wav)
                audio_bytes = wav_io.getvalue()
            else:
                audio_bytes = wav.tobytes()

            duration = len(wav) / sampling_rate

            return tts_service_pb2.SynthesizeResponse(
                audio_data=audio_bytes,
                sample_rate=sampling_rate,
                audio_format=audio_format,
                duration=duration
            )

        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Synthesis failed: {str(e)}")
            return tts_service_pb2.SynthesizeResponse()

    def SynthesizeStream(self, request, context):
        try:
            text = request.text
            speaker_id = request.speaker_id if request.speaker_id else 0
            pitch_control = request.pitch_control if request.pitch_control else 1.0
            energy_control = request.energy_control if request.energy_control else 1.0
            duration_control = request.duration_control if request.duration_control else 1.0

            if not text:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("Text is required")
                return

            wav = self._synthesize_audio(text, speaker_id, pitch_control, energy_control, duration_control)

            chunk_size = 16000  # 1 second of audio at 16kHz
            audio_bytes = wav.tobytes()
            total_chunks = (len(audio_bytes) + chunk_size - 1) // chunk_size

            for i in range(total_chunks):
                start = i * chunk_size
                end = min(start + chunk_size, len(audio_bytes))
                chunk_data = audio_bytes[start:end]

                yield tts_service_pb2.AudioChunk(
                    audio_data=chunk_data,
                    sequence=i,
                    is_final=(i == total_chunks - 1)
                )

        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Synthesis failed: {str(e)}")
            return


def serve(args):
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    print("Loading model...")
    model = get_model(args, configs, device, train=False)

    print("Loading vocoder...")
    vocoder = get_vocoder(model_config, device)

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=args.max_workers),
        options=[
            ('grpc.max_send_message_length', 50 * 1024 * 1024),
            ('grpc.max_receive_message_length', 50 * 1024 * 1024),
        ]
    )

    tts_service_pb2_grpc.add_TTSServiceServicer_to_server(
        TTSServicer(model, vocoder, configs), server
    )

    server.add_insecure_port(f'[::]:{args.port}')

    print(f"Starting gRPC server on port {args.port}...")
    server.start()
    print(f"Server is ready and listening on port {args.port}")

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.stop(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FastSpeech2 gRPC Server")

    parser.add_argument(
        "--restore_step",
        type=int,
        required=True,
        help="Checkpoint step to restore"
    )
    parser.add_argument(
        "-p", "--preprocess_config",
        type=str,
        required=True,
        help="path to preprocess.yaml"
    )
    parser.add_argument(
        "-m", "--model_config",
        type=str,
        required=True,
        help="path to model.yaml"
    )
    parser.add_argument(
        "-t", "--train_config",
        type=str,
        required=True,
        help="path to train.yaml"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=50051,
        help="Port to listen on (default: 50051)"
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=10,
        help="Maximum number of worker threads (default: 10)"
    )

    args = parser.parse_args()
    serve(args)