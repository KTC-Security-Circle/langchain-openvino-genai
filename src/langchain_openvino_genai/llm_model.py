from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import openvino as ov
import openvino_genai
from langchain_core.language_models.llms import LLM
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.outputs import GenerationChunk
from pydantic import BaseModel

from langchain_openvino_genai.genai_helper import ChunkStreamer

if TYPE_CHECKING:
    from collections.abc import Iterator

    from langchain_core.callbacks.manager import CallbackManagerForLLMRun
    from langchain_core.language_models import LanguageModelInput
    from langchain_core.runnables import Runnable

type Device = Literal["CPU", "GPU", "NPU"]


class OpenVINOLLM(LLM):
    """OpenVINO Pipeline API.

    To use, you should have the ``openvino-genai`` python package installed.

    Example using from_model_path:
        .. code-block:: python

            from ov_langchain_helper import OpenVINOLLM
            ov = OpenVINOLLM.from_model_path(
                model_path="./openvino_model_dir",
                device="CPU",
            )
    Example passing pipeline in directly:
        .. code-block:: python

            import openvino_genai

            pipe = openvino_genai.LLMPipeline("./openvino_model_dir", "CPU")
            config = openvino_genai.GenerationConfig()
            ov = OpenVINOLLM.from_model_path(
                ov_pipe=pipe,
                config=config,
            )

    """

    ov_pipe: openvino_genai.LLMPipeline
    tokenizer: openvino_genai.Tokenizer
    config: openvino_genai.GenerationConfig
    streamer: ChunkStreamer
    DEVICE_PRIORITY: ClassVar[set[Device]] = {"NPU", "GPU", "CPU"}

    @classmethod
    def from_model_path(
        cls,
        model_path: str,
        device: str = "AUTO",
        tokenizer: openvino_genai.Tokenizer | None = None,
        **kwargs: object,
    ) -> OpenVINOLLM:
        """Construct the oepnvino object from model_path"""
        try:
            ov_pipe = openvino_genai.LLMPipeline(model_path, device, {}, **kwargs)
        except Exception:
            if device == "AUTO":
                ov_pipe = cls._resolve_load(model_path, **kwargs)
            else:
                raise

        config = ov_pipe.get_generation_config()
        if tokenizer is None:
            tokenizer = ov_pipe.get_tokenizer()
        streamer = ChunkStreamer(tokenizer)

        return cls(
            ov_pipe=ov_pipe,
            tokenizer=tokenizer,
            config=config,
            streamer=streamer,
        )

    @classmethod
    def _resolve_load(cls, model_path: str, **kwargs: object) -> openvino_genai.LLMPipeline:
        available_devices = ov.Core().available_devices
        devices = [d for d in cls.DEVICE_PRIORITY if d in available_devices]

        while True:
            device = f"AUTO:{','.join(devices)}"

            try:
                ov_pipe = openvino_genai.LLMPipeline(
                    model_path,
                    device,
                    {},
                    **kwargs,
                )
            except Exception as e:
                if len(devices) == 1:
                    msg = "No suitable device found for OpenVINO GenAI."
                    raise RuntimeError(msg) from e
                devices = devices[1:]
                continue
            return ov_pipe

    def _call(
        self,
        prompt: str | openvino_genai.TokenizedInputs,
        stop: list[str] | None = None,
        _run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: object,
    ) -> str:
        """Call out to OpenVINO's generate request."""
        if stop is not None:
            self.config.stop_strings = set(stop)
        if not isinstance(self.tokenizer, openvino_genai.Tokenizer):
            tokens = self.tokenizer(prompt, add_special_tokens=False, return_tensors="np")
            input_ids = tokens["input_ids"]
            attention_mask = tokens["attention_mask"]
            prompt = openvino_genai.TokenizedInputs(ov.Tensor(input_ids), ov.Tensor(attention_mask))
        output = self.ov_pipe.generate(prompt, self.config, **kwargs)
        if not isinstance(self.tokenizer, openvino_genai.Tokenizer):
            output = self.tokenizer.batch_decode(output.tokens, skip_special_tokens=True)[0]
        return output

    def _stream(
        self,
        prompt: str | openvino_genai.TokenizedInputs,
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: object,
    ) -> Iterator[GenerationChunk]:
        """Output OpenVINO's generation Stream"""
        from threading import Event, Thread

        if stop is not None:
            self.config.stop_strings = set(stop)
        if not isinstance(self.tokenizer, openvino_genai.Tokenizer):
            tokens = self.tokenizer(prompt, add_special_tokens=False, return_tensors="np")
            input_ids = tokens["input_ids"]
            attention_mask = tokens["attention_mask"]
            prompt = openvino_genai.TokenizedInputs(ov.Tensor(input_ids), ov.Tensor(attention_mask))
        stream_complete = Event()

        def generate_and_signal_complete() -> None:
            """Genration function for single thread"""
            self.streamer.reset()
            self.ov_pipe.generate(prompt, self.config, self.streamer, **kwargs)
            stream_complete.set()
            self.streamer.end()

        t1 = Thread(target=generate_and_signal_complete)
        t1.start()

        for char in self.streamer:
            chunk = GenerationChunk(text=char)
            if run_manager:
                run_manager.on_llm_new_token(chunk.text, chunk=chunk)

            yield chunk

    @property
    def _identifying_params(self) -> dict[str, Any]:
        """Return a dictionary of identifying parameters."""
        return {}

    @property
    def _llm_type(self) -> str:
        return "openvino_pipeline"

    def set_structured_output_config(self, schema: dict | type) -> None:
        """Set the structured output configuration for the model.

        Args:
            schema: A pydantic BaseModel class or a dict representing the schema.
        """
        if schema is None:
            msg = "Schema must be provided for structured output."
            raise ValueError(msg)
        if isinstance(schema, dict):
            self.config.structured_output_config = openvino_genai.StructuredOutputConfig(json_schema=json.dumps(schema))
        elif issubclass(schema, BaseModel):
            self.config.structured_output_config = openvino_genai.StructuredOutputConfig(
                json_schema=json.dumps(schema.model_json_schema())
            )

    def with_structured_output(
        self, schema: dict | type, **_kwargs: object
    ) -> Runnable[LanguageModelInput, dict | BaseModel]:
        """Return a version of this LLM that produces structured output.

        Args:
            schema: A pydantic BaseModel class or a dict representing the schema.
            **kwargs: Additional keyword arguments to pass to the structured output parser.

        Returns:
            A Runnable that produces structured output conforming to the provided schema.
        """
        self.set_structured_output_config(schema)
        output_parser: JsonOutputParser = JsonOutputParser()

        return self | output_parser
        # return self


DEFAULT_SYSTEM_PROMPT = """You are a helpful, respectful, and honest assistant."""
