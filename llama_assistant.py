"""
Llama-3.1-8B-Instruct based assistant for the Music Diary app.

Replaces the three ChatGPT placeholder functions:
  1. guide_conversation  – good-listener chatbot (≤15 turns)
  2. summarize_diary     – first-person diary entry (< 200 words)
  3. extract_emotion     – single dominant emotion keyword
"""

import re
import torch
from typing import List, Dict, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM


# ── Emotion whitelist (matches MusicDiaryGenerator.EMOTIONS) ──────────
VALID_EMOTIONS = [
    "happy", "sad", "calm", "excited", "angry",
    "nostalgic", "romantic", "hopeful", "melancholic", "other",
]


class LlamaAssistant:
    """Thin wrapper around Llama-3.1-8B-Instruct for diary chat tasks."""

    def __init__(
        self,
        model_path: str = "/home/dataset_model/model/Llama-3.1-8B-Instruct",
        device: Optional[str] = None,
        max_model_len: int = 4096,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_model_len = max_model_len

        print(f"[LlamaAssistant] Loading tokenizer from {model_path} …")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"[LlamaAssistant] Loading model to {self.device} …")
        dtype = torch.float16 if "cuda" in self.device else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=self.device,
            trust_remote_code=True,
        )

        self.model.eval()
        print("[LlamaAssistant] Ready.")

    # ──────────────────────────────────────────────────────────────
    #  Low-level generation
    # ──────────────────────────────────────────────────────────────
    def _generate(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """
        Build a Llama-3.1 chat prompt from a list of
        {"role": "system"|"user"|"assistant", "content": "..."} dicts,
        generate, and return only the new assistant text.
        """
        # Use the tokenizer's built-in chat template (Llama-3.1 style)
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_model_len,
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Decode only the newly generated tokens
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return text

    # ──────────────────────────────────────────────────────────────
    #  1. Guide conversation (good listener, ≤ 15 turns)
    # ──────────────────────────────────────────────────────────────
    GUIDE_SYSTEM_PROMPT = (
        "You are a warm, empathetic diary assistant. Your job is to help the "
        "user reflect on their day by asking short, thoughtful follow-up "
        "questions — one at a time. Listen carefully, validate their feelings, "
        "and gently guide them to explore emotions, events, and gratitude.\n\n"
        "Rules:\n"
        "- Ask only ONE question per reply.\n"
        "- Keep each reply under 3 sentences.\n"
        "- Do NOT write the diary entry yourself.\n"
        "- After the user has shared enough (around 7-8 exchanges), let them "
        "know you have enough material and invite them to generate their diary "
        "entry.\n"
        "- The entire conversation must stay within 15 assistant messages."
    )

    def guide_conversation(
        self,
        history: List[Dict[str, str]],
        user_message: str,
    ) -> str:
        """
        Given the conversation history and latest user message, return the
        assistant's next guiding question / reply.

        Parameters
        ----------
        history : list of {"role": "user"|"assistant", "content": str}
            Prior turns (NOT including the current user_message).
        user_message : str
            The latest message from the user (may be "" on first call).
        """
        assistant_turns = sum(1 for m in history if m["role"] == "assistant")

        # Build messages list for the model
        messages = [{"role": "system", "content": self.GUIDE_SYSTEM_PROMPT}]

        # Add conversation history
        for m in history:
            messages.append({"role": m["role"], "content": m["content"]})

        # First call: no user message yet → just generate opening question
        if user_message:
            messages.append({"role": "user", "content": user_message})

        # If we've reached the turn limit, produce a closing message
        if assistant_turns >= 14:
            return (
                "Thank you so much for sharing! I have a wonderful picture of "
                "your day now. Click 'Generate Diary' whenever you're ready."
            )

        reply = self._generate(messages, max_new_tokens=150, temperature=0.7)

        # Safety: if model accidentally produced something very long, truncate
        sentences = re.split(r'(?<=[.!?])\s+', reply)
        if len(sentences) > 4:
            reply = " ".join(sentences[:3])

        return reply

    # ──────────────────────────────────────────────────────────────
    #  2. Summarize diary (first-person, < 200 words)
    # ──────────────────────────────────────────────────────────────
    SUMMARIZE_SYSTEM_PROMPT = (
        "You are a skilled diary writer. Given a conversation between a user "
        "and an assistant about the user's day, write a first-person diary "
        "entry (as if you ARE the user). The entry must:\n"
        "- Be written in first person ('I …').\n"
        "- Be fewer than 200 words.\n"
        "- Capture the key events, feelings, and reflections from the "
        "conversation.\n"
        "- Sound natural, personal, and reflective.\n"
        "- Do NOT include any preamble like 'Here is your diary entry:'. "
        "Just write the diary text directly."
    )

    def summarize_diary(self, conversation: List[Dict[str, str]]) -> str:
        """
        Read the full conversation and return a first-person diary entry
        (< 200 words).
        """
        # Flatten conversation into a readable block for the model
        convo_text = "\n".join(
            f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
            for m in conversation
        )

        messages = [
            {"role": "system", "content": self.SUMMARIZE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Here is the conversation:\n\n"
                    f"{convo_text}\n\n"
                    "Now write the diary entry."
                ),
            },
        ]

        diary = self._generate(messages, max_new_tokens=300, temperature=0.7)

        # Enforce the 200-word limit
        words = diary.split()
        if len(words) > 200:
            diary = " ".join(words[:200]).rstrip(",;:—–-") + "."

        return diary

    # ──────────────────────────────────────────────────────────────
    #  3. Extract dominant emotion
    # ──────────────────────────────────────────────────────────────
    EMOTION_SYSTEM_PROMPT = (
        "You are an emotion classifier. Given a diary conversation or entry, "
        "identify the single most dominant emotion.\n\n"
        "You MUST reply with exactly ONE word from this list:\n"
        "happy, sad, calm, excited, angry, nostalgic, romantic, hopeful, "
        "melancholic, other\n\n"
        "Reply with ONLY that single word. No explanation, no punctuation."
    )

    def extract_emotion(self, conversation: List[Dict[str, str]]) -> str:
        """
        Read the conversation and return a single emotion keyword from the
        valid set.
        """
        convo_text = "\n".join(
            f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
            for m in conversation
        )

        messages = [
            {"role": "system", "content": self.EMOTION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Here is the conversation:\n\n"
                    f"{convo_text}\n\n"
                    "What is the dominant emotion?"
                ),
            },
        ]

        raw = self._generate(messages, max_new_tokens=10, temperature=0.3)

        # Parse: take first word, lowercase, and validate
        emotion = raw.strip().lower().split()[0] if raw.strip() else "other"
        emotion = re.sub(r"[^a-z]", "", emotion)  # remove punctuation

        if emotion not in VALID_EMOTIONS:
            # Fuzzy fallback: check if any valid emotion is a substring
            for valid in VALID_EMOTIONS:
                if valid in emotion or emotion in valid:
                    return valid
            return "other"

        return emotion