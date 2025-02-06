#!/usr/bin/env python3
"""
    ezprompt.py
"""

from pydantic import BaseModel
from typing import Optional, Callable

from litellm import completion, acompletion

from . import utils

# --
# Main

class EZPrompt:
    LOG_DIR         = None
    DEFAULT_SYSTEM = "You are a helpful assistant"

    @classmethod
    def set_default_logdir(cls, path: str):
        import os
        cls.LOG_DIR = path
        os.makedirs(path, exist_ok=True)
    
    def __init__(
            self, 
            *, 
            name:str,
            system:Optional[str]=None, 
            template:str, 
            before:Optional[Callable]=None, 
            after:Optional[Callable]=None, 
            response_format:Optional[BaseModel]=None,
            llm_kwargs:dict
        ):
        self.name            = name
        self.counter         = 0
        self.system          = system if system else self.DEFAULT_SYSTEM
        self.template        = template
        self.response_format = response_format
        self.before          = before
        self.after           = after
        self.llm_kwargs      = llm_kwargs
    
    def prompt(self, **inputs):
        # [TODO] not DRY?
        if self.before is not None:
            prompt_kwargs = self.before(**inputs)
            extra_kwargs  = {k:v for k,v in prompt_kwargs.items() if k.startswith('__')}
            prompt_kwargs = {k:v for k,v in prompt_kwargs.items() if k not in extra_kwargs}
        else:
            extra_kwargs  = {}
            prompt_kwargs = inputs
        
        prompt = self.template.format(**prompt_kwargs)
        return prompt, extra_kwargs
    
    def run(self, **inputs):
        prompt, extra_kwargs = self.prompt(**inputs)
        
        response = completion(
            **self.llm_kwargs,
            response_format = self.response_format,
            messages        = [
                {"role": "system", "content": self.system},
                {"role": "user",   "content": prompt}
            ]
        )
        
        output_str = response.choices[0].message.content
        if self.response_format is not None:
            output = self.response_format.model_validate_json(output_str).model_dump()
        else:
            output = {"output_str" : output_str}
        
        if self.after is not None:
            output = self.after(**output, **extra_kwargs)
        
        assert isinstance(output, dict)
        if self.LOG_DIR is not None:
            utils.log(self.LOG_DIR, self.name, self.counter, prompt, output_str, output)
        
        self.counter += 1
        
        return output

    async def arun(self, **inputs):
        # [TODO] very annoying that i have to repeat the whole thing? can i do a synchronous wrapper?
        prompt, extra_kwargs = self.prompt(**inputs)
        
        # vvvvvv ONLY DIFFERENCE FROM SYNC VERSION vvvvvv
        response = await acompletion(
            **self.llm_kwargs,
            response_format = self.response_format,
            messages        = [
                {"role": "system", "content": self.system},
                {"role": "user",   "content": prompt}
            ]
        )
        # ^^^^^^ ONLY DIFFERENCE FROM SYNC VERSION ^^^^^^
        
        output_str = response.choices[0].message.content
        if self.response_format is not None:
            output = self.response_format.model_validate_json(output_str).model_dump()
        else:
            output = {"output_str" : output_str}
        
        if self.after is not None:
            output = self.after(**output, **extra_kwargs)
        
        assert isinstance(output, dict)
        if self.LOG_DIR is not None:
            utils.log(self.LOG_DIR, self.name, self.counter, prompt, output_str, output)
        
        self.counter += 1
        
        return output
    
    def larun(self, **inputs):
        def _fn():
            return self.arun(**inputs)
        
        return _fn

