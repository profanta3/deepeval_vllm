from typing import Optional, Tuple, Union

from deepeval.models.llms.local_model import LocalModel, retry_local
from deepeval.models.llms.utils import trim_and_load_json
from deepeval.utils import (
    check_if_multimodal,
    convert_to_multi_modal_array,
)
from openai.types.chat import ChatCompletion
from pydantic import BaseModel


class VLLMLocalModel(LocalModel):
    """Compatibility wrapper for DeepEval's LocalModel that adds native structured output support."""

    @retry_local
    def generate(
        self, prompt: str, schema: Optional[BaseModel] = None
    ) -> Tuple[Union[str, BaseModel], float]:

        if check_if_multimodal(prompt):
            prompt = convert_to_multi_modal_array(input=prompt)
            content = self.generate_content(prompt)
        else:
            content = prompt

        gen_args = self.generation_kwargs.copy()
        if schema:
            gen_args.update(
                {
                    "extra_body": {
                        "structured_outputs": {"json": schema.model_json_schema()}
                    }
                }
            )

        client = self.load_model(async_mode=False)
        response: ChatCompletion = client.chat.completions.create(
            model=self.name,
            messages=[{"role": "user", "content": content}],
            temperature=self.temperature,
            **gen_args,
        )
        res_content = response.choices[0].message.content

        if schema:
            json_output = trim_and_load_json(res_content)
            return schema.model_validate(json_output), 0.0
        else:
            return res_content, 0.0

    @retry_local
    async def a_generate(
        self, prompt: str, schema: Optional[BaseModel] = None
    ) -> Tuple[Union[str, BaseModel], float]:

        if check_if_multimodal(prompt):
            prompt = convert_to_multi_modal_array(input=prompt)
            content = self.generate_content(prompt)
        else:
            content = prompt

        gen_args = self.generation_kwargs.copy()
        if schema:
            gen_args.update(
                {
                    "extra_body": {
                        "structured_outputs": {"json": schema.model_json_schema()}
                    }
                }
            )

        client = self.load_model(async_mode=True)
        response: ChatCompletion = await client.chat.completions.create(
            model=self.name,
            messages=[{"role": "user", "content": content}],
            temperature=self.temperature,
            **gen_args,
        )
        res_content = response.choices[0].message.content

        if schema:
            json_output = trim_and_load_json(res_content)
            return schema.model_validate(json_output), 0.0
        else:
            return res_content, 0.0
