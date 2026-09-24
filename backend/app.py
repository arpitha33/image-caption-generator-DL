from pathlib import Path
import sys
import tempfile
import uuid

import torch
from PIL import Image
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"

sys.path.insert(0, str(SRC_DIR))

from models import (
    AttentionDecoderLSTM,
    DecoderLSTM,
    EncoderCNN,
    EncoderCNNAttention,
)
from utils import BOS, EOS, Vocabulary
from translator import translate_text
from text_to_speech import text_to_speech

from torchvision import transforms


# ---------------------------------------------------------
# FastAPI setup
# ---------------------------------------------------------

app = FastAPI(
    title="AI Image Captioning API",
    description="Image captioning with translation and text-to-speech.",
    version="1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
# Device
# ---------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------
# Model storage
# ---------------------------------------------------------

models = {}


def load_model(model_type):
    """
    Load either the baseline or attention model.
    The model is loaded only once and then reused.
    """

    if model_type in models:
        return models[model_type]

    if model_type == "attention":
        checkpoint_path = (
            BASE_DIR
            / "outputs_attention"
            / "best_captioner.pt"
        )

        vocab_path = (
            BASE_DIR
            / "outputs_attention"
            / "vocab.json"
        )

    else:
        checkpoint_path = (
            BASE_DIR
            / "outputs_baseline"
            / "best_captioner.pt"
        )

        vocab_path = (
            BASE_DIR
            / "outputs_baseline"
            / "vocab.json"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    vocab = Vocabulary.from_json(vocab_path)

    decoder_type = checkpoint.get(
        "decoder_type",
        "lstm"
    )

    attention = decoder_type == "attention"

    if attention:

        encoder = EncoderCNNAttention(
            pretrained=False,
            train_backbone=checkpoint.get(
                "train_backbone",
                False,
            ),
        ).to(device)

        decoder = AttentionDecoderLSTM(
            vocab_size=checkpoint["vocab_size"],
            embed_dim=checkpoint["embed_dim"],
            hidden_dim=checkpoint["hidden_dim"],
            encoder_dim=(
                checkpoint.get("encoder_dim")
                or encoder.encoder_dim
            ),
            attention_dim=checkpoint.get(
                "attention_dim",
                256,
            ),
            dropout=checkpoint.get(
                "dropout",
                0.1,
            ),
        ).to(device)

    else:

        encoder = EncoderCNN(
            embed_dim=checkpoint["embed_dim"],
            pretrained=False,
            train_backbone=checkpoint.get(
                "train_backbone",
                False,
            ),
        ).to(device)

        decoder = DecoderLSTM(
            vocab_size=checkpoint["vocab_size"],
            embed_dim=checkpoint["embed_dim"],
            hidden_dim=checkpoint["hidden_dim"],
            num_layers=checkpoint.get(
                "num_layers",
                1,
            ),
            dropout=checkpoint.get(
                "dropout",
                0.1,
            ),
        ).to(device)

    encoder.load_state_dict(
        checkpoint["encoder"]
    )

    decoder.load_state_dict(
        checkpoint["decoder"]
    )

    encoder.eval()
    decoder.eval()

    image_size = checkpoint.get(
        "image_size",
        224,
    )

    transform = transforms.Compose(
        [
            transforms.Resize(
                (image_size, image_size)
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[
                    0.485,
                    0.456,
                    0.406,
                ],
                std=[
                    0.229,
                    0.224,
                    0.225,
                ],
            ),
        ]
    )

    models[model_type] = {
        "encoder": encoder,
        "decoder": decoder,
        "vocab": vocab,
        "transform": transform,
    }

    return models[model_type]


# ---------------------------------------------------------
# Generate caption
# ---------------------------------------------------------

def generate_caption(image_path, model_type):
    model = load_model(model_type)

    encoder = model["encoder"]
    decoder = model["decoder"]
    vocab = model["vocab"]
    transform = model["transform"]

    image = (
        transform(
            Image.open(image_path).convert("RGB")
        )
        .unsqueeze(0)
        .to(device)
    )

    with torch.no_grad():

        features = encoder(image)

        ids = decoder.sample(
            features,
            max_len=20,
            bos_id=vocab.word2id[BOS],
            eos_id=vocab.word2id[EOS],
        )

    caption = vocab.decode(
        ids[0].cpu().numpy()
    )

    return caption


# ---------------------------------------------------------
# Generate audio
# ---------------------------------------------------------

def generate_audio(text):
    output_dir = BASE_DIR / "outputs_audio"
    output_dir.mkdir(
        exist_ok=True
    )

    filename = (
        f"caption_{uuid.uuid4().hex}.mp3"
    )

    output_path = (
        output_dir / filename
    )

    text_to_speech(
        text,
        str(output_path),
    )

    return output_path


# ---------------------------------------------------------
# API endpoint
# ---------------------------------------------------------

@app.post("/generate")
async def generate(
    image: UploadFile = File(...),
    model: str = Form("attention"),
    language: str = Form("english"),
):

    model = model.strip().lower()
    language = language.strip().lower()

    if model not in [
        "baseline",
        "attention",
    ]:
        model = "attention"

    if not language:
        language = "english"

    # Save uploaded image temporarily
    suffix = Path(
        image.filename or ".jpg"
    ).suffix

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix,
    ) as temp_file:

        image_bytes = await image.read()

        temp_file.write(
            image_bytes
        )

        temp_image_path = Path(
            temp_file.name
        )

    try:

        # Generate caption
        caption = generate_caption(
            temp_image_path,
            model,
        )

        # Translate
        if language == "english":

            translated_caption = caption

        else:

            translated_caption = translate_text(
                caption,
                language,
            )

            if not translated_caption:
                translated_caption = caption

        # Generate speech
        audio_path = generate_audio(
            translated_caption
        )

        return {
            "success": True,
            "model": model,
            "language": language,
            "caption": caption,
            "translation": translated_caption,
            "audio_url": (
                f"/audio/{audio_path.name}"
            ),
        }

    finally:

        temp_image_path.unlink(
            missing_ok=True
        )


# ---------------------------------------------------------
# Audio endpoint
# ---------------------------------------------------------

@app.get("/audio/{filename}")
def get_audio(filename: str):

    audio_path = (
        BASE_DIR
        / "outputs_audio"
        / filename
    )

    if not audio_path.exists():
        return {
            "error": "Audio file not found"
        }

    return FileResponse(
        audio_path,
        media_type="audio/mpeg",
        filename=filename,
    )


# ---------------------------------------------------------
# Health check
# ---------------------------------------------------------

@app.get("/")
def root():

    return {
        "message": "AI Image Captioning API is running",
        "status": "online",
    }