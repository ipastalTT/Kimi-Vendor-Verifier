"""Reference prompt-token counts computed from a model's published tokenizer.

The constants in ``testdata/prompt_token_cases`` were collected from the K3
serving stack, whose chat template differs from other Kimi models (e.g. it
adds a thinking-effort block). For other models the expected count is instead
computed from the model's own Hugging Face tokenizer and chat template:

    expected = len(apply_chat_template(messages, tools, add_generation_prompt=True))
               + sum(image_tokens(image) - 1 for each image in a user message)

The chat template emits one ``<|media_pad|>`` per image; serving engines expand
it to the image's token count (as computed by ``media_utils.navit_resize_image``
with the model's ``preprocessor_config.json``), hence the ``- 1``.
"""

from __future__ import annotations

import base64
import importlib.util
import json
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image


# Files needed for the tokenizer, chat template and image token calculation.
_ALLOW_PATTERNS = ["*.json", "*.jinja", "*.py", "tiktoken.model"]
_IGNORE_PATTERNS = ["model.safetensors.index.json", "modeling_*.py"]


class ReferenceCounter:
    def __init__(self, source: str) -> None:
        from transformers import AutoTokenizer

        path = Path(source)
        if not path.is_dir():
            from huggingface_hub import snapshot_download

            path = Path(snapshot_download(
                source, allow_patterns=_ALLOW_PATTERNS, ignore_patterns=_IGNORE_PATTERNS,
            ))
        self.source = source
        self.tokenizer = AutoTokenizer.from_pretrained(str(path), trust_remote_code=True)
        self._media_cfg: dict[str, Any] | None = None
        self._navit_resize_image = None
        preprocessor = path / "preprocessor_config.json"
        media_utils = path / "media_utils.py"
        if preprocessor.is_file() and media_utils.is_file():
            self._media_cfg = json.loads(preprocessor.read_text())["media_proc_cfg"]
            spec = importlib.util.spec_from_file_location("_reference_media_utils", media_utils)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self._navit_resize_image = module.navit_resize_image

    def count(self, request: dict[str, Any]) -> int:
        encoded = self.tokenizer.apply_chat_template(
            request["messages"],
            tools=request.get("tools"),
            add_generation_prompt=True,
            tokenize=True,
        )
        input_ids = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
        return len(input_ids) + sum(self._image_tokens(image) - 1 for image in _user_images(request))

    def _image_tokens(self, image: Image.Image) -> int:
        if self._navit_resize_image is None:
            raise ValueError(f"{self.source} has no image processor config; cannot count image tokens")
        cfg = self._media_cfg
        width, height = image.size
        return self._navit_resize_image(
            width, height, cfg["patch_size"], cfg["merge_kernel_size"], cfg["in_patch_limit"],
            cfg["patch_limit_on_one_side"], cfg["fixed_output_tokens"],
        )["num_tokens"]


def _user_images(request: dict[str, Any]) -> list[Image.Image]:
    """Decode the data-URL images in user messages, as the model's processor does."""
    images = []
    for message in request["messages"]:
        content = message.get("content")
        if message.get("role") != "user" or not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict) or part.get("type") not in ("image_url", "image"):
                continue
            image_url = part.get("image_url")
            url = image_url.get("url") if isinstance(image_url, dict) else image_url
            if not isinstance(url, str) or not url.startswith("data:"):
                raise ValueError(f"reference counting needs data-URL images, got {str(url)[:60]!r}")
            images.append(Image.open(BytesIO(base64.b64decode(url.split(",", 1)[1]))))
    return images
