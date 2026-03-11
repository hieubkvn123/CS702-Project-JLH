import os
import random
from typing import Dict, List, Optional

import torch
import soundfile as sf
from transformers import AutoProcessor, MusicgenForConditionalGeneration


class MusicDiaryGenerator:
    EMOTIONS = [
        "happy",
        "sad",
        "calm",
        "excited",
        "angry",
        "nostalgic",
        "romantic",
        "hopeful",
        "melancholic",
        "other",
    ]

    EMOTION_PROMPT_TEMPLATES = {
        "happy": [
            "happy upbeat pop music with bright piano and energetic drums",
            "joyful pop instrumental with lively rhythm and cheerful melody",
            "uplifting pop track with vibrant synths and playful beat",
        ],
        "sad": [
            "slow emotional piano music expressing sadness",
            "melancholic piano and strings with gentle tempo",
            "soft sad instrumental with expressive piano melody",
        ],
        "calm": [
            "calm relaxing ambient music with soft pads",
            "peaceful ambient soundscape with gentle piano",
            "slow relaxing instrumental with warm atmospheric pads",
        ],
        "excited": [
            "energetic electronic dance music with powerful drums",
            "exciting upbeat EDM track with driving rhythm",
            "high-energy electronic music with bright synth leads",
        ],
        "angry": [
            "aggressive rock music with distorted guitars and heavy drums",
            "intense metal instrumental with fast drums",
            "powerful rock track with gritty guitar riffs",
        ],
        "nostalgic": [
            "nostalgic piano and strings instrumental with warm tone",
            "retro style instrumental with soft synths and emotional melody",
            "melancholic nostalgic music with gentle piano",
        ],
        "romantic": [
            "romantic acoustic guitar melody with warm atmosphere",
            "soft romantic instrumental with piano and strings",
            "gentle love theme with acoustic guitar and light piano",
        ],
        "hopeful": [
            "hopeful uplifting orchestral music with inspiring melody",
            "bright cinematic instrumental expressing hope",
            "uplifting orchestral track with emotional build-up",
        ],
        "melancholic": [
            "melancholic slow piano instrumental with emotional depth",
            "sad cinematic music with piano and soft strings",
            "deep reflective music with minimal piano and ambient texture",
        ],
        "other": [
            "neutral instrumental background music with soft melody",
            "gentle ambient background music with light piano",
            "soft instrumental music suitable for daily reflection",
        ],
    }

    def __init__(
        self,
        model_name: str = "facebook/musicgen-stereo-medium",
        output_dir: str = "generated_music",
        static_url_prefix: str = "/static/generated_music",
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.output_dir = output_dir
        self.static_url_prefix = static_url_prefix.rstrip("/")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        os.makedirs(self.output_dir, exist_ok=True)

        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = MusicgenForConditionalGeneration.from_pretrained(self.model_name).to(self.device)
        self.sampling_rate = self.model.config.audio_encoder.sampling_rate

    def normalize_emotion(self, emotion: str) -> str:
        if not emotion:
            return "other"

        emotion = emotion.lower().strip()

        alias_map = {
            "joy": "happy",
            "joyful": "happy",
            "depressed": "sad",
            "relaxed": "calm",
            "love": "romantic",
            "inspired": "hopeful",
            "anxious": "other",
            "reflective": "other",
            "neutral": "other",
        }

        if emotion in alias_map:
            return alias_map[emotion]

        if emotion in self.EMOTION_PROMPT_TEMPLATES:
            return emotion

        return "other"

    def generate_prompt(self, emotion: str) -> str:
        emotion = self.normalize_emotion(emotion)
        return random.choice(self.EMOTION_PROMPT_TEMPLATES[emotion])

    def _save_audio(self, audio_tensor: torch.Tensor, filename: str) -> str:
        """
        audio_tensor:
          - mono: [samples]
          - stereo: [channels, samples] or [samples, channels]
        """
        audio_np = audio_tensor.detach().cpu().float().numpy()

        # Convert to shape [samples, channels] for soundfile.write
        if audio_np.ndim == 2:
            # MusicGen often returns [channels, samples]
            if audio_np.shape[0] in (1, 2) and audio_np.shape[0] < audio_np.shape[1]:
                audio_np = audio_np.T

        output_path = os.path.join(self.output_dir, filename)
        sf.write(output_path, audio_np, self.sampling_rate)
        return output_path

    def generate_clips(
        self,
        emotion: str,
        num_clips: int = 4,
        max_new_tokens: int = 256,
        base_name: Optional[str] = None,
    ) -> Dict:
        """
        Returns:
        {
            "emotion": "...",
            "prompt": "...",
            "clips": [
                {"id": 1, "label": "...", "file_path": "...", "music_url": "..."},
                ...
            ]
        }
        """
        emotion_norm = self.normalize_emotion(emotion)
        prompt = self.generate_prompt(emotion_norm)

        inputs = self.processor(
            text=[prompt] * num_clips,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            audio_values = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
            )

        # audio_values shape commonly: [B, C, T] or [B, T]
        clips = []
        prefix = base_name or emotion_norm

        for i in range(audio_values.shape[0]):
            filename = f"{prefix}_{i+1}.wav"
            file_path = self._save_audio(audio_values[i], filename)

            clips.append({
                "id": i + 1,
                "label": f"Clip {i+1} — {emotion_norm.capitalize()}",
                "file_path": file_path,
                "music_url": f"{self.static_url_prefix}/{filename}",
                "prompt": prompt,
            })

        return {
            "emotion": emotion_norm,
            "prompt": prompt,
            "clips": clips,
        }

    def generate_one(
        self,
        emotion: str,
        max_new_tokens: int = 256,
        base_name: Optional[str] = None,
    ) -> Dict:
        result = self.generate_clips(
            emotion=emotion,
            num_clips=1,
            max_new_tokens=max_new_tokens,
            base_name=base_name,
        )
        return {
            "emotion": result["emotion"],
            "prompt": result["prompt"],
            "audio": result["clips"][0],
        }

    def notebook_audio(self, file_path: str):
        """
        For Jupyter notebook playback.
        """
        from IPython.display import Audio
        return Audio(filename=file_path)