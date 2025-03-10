#!/usr/bin/env python3
"""
    ezprompt.py
"""

import json
from hashlib import md5
from pathlib import Path
from pydantic import BaseModel
from typing import Optional, Callable
from litellm import completion, acompletion

from . import utils


def _cache_key(*args):
    h = ''
    for x in args:
        if isinstance(x, dict):
            h += md5(json.dumps(x, sort_keys=True).encode()).hexdigest()
        else:
            h += md5(str(x).encode()).hexdigest()
    
    return md5(h.encode()).hexdigest()


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
            llm_kwargs:dict,
            cache_dir:Optional[str]=None,
            no_log:bool=False,
            no_console:bool=False
        ):
        self.name            = name
        self.counter         = 0
        self.system          = system if system else self.DEFAULT_SYSTEM
        self.template        = template
        self.response_format = response_format
        self.before          = before
        self.after           = after
        self.llm_kwargs      = llm_kwargs
        self.do_log          = not no_log
        self.do_console      = not no_console
        
        self.cache_dir = None
        if cache_dir is not None:
            self.cache_dir = Path(cache_dir) / self.name
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
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
    
    def run(self, _cache_idx=None, **inputs):
        prompt, extra_kwargs = self.prompt(**inputs)

        # Check cache
        if self.cache_dir is not None:
            if _cache_idx is None:
                cache_key = _cache_key(self.llm_kwargs, self.system, prompt)
            else:
                cache_key = _cache_key(self.llm_kwargs, self.system, prompt, _cache_idx)
            
            cache_path = Path(self.cache_dir) / f"{cache_key}.json"
            if cache_path.exists():
                return json.loads(cache_path.read_text())
        
        with utils.spinner(f"Running {self.name}"):
            response = completion(
                **self.llm_kwargs,
                num_retries     = 3,
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

        if self.cache_dir is not None:
            cache_path = Path(self.cache_dir) / f"{cache_key}.json"
            cache_path.write_text(json.dumps(output))

        assert isinstance(output, dict)
        if self.do_log and self.LOG_DIR is not None:
            utils.log(self.LOG_DIR, self.name, self.counter, prompt, output_str, output, show_console=self.do_console)
        
        self.counter += 1
        
        return output

    async def arun(self, _cache_idx=None, **inputs):
        # [TODO] very annoying that i have to repeat the whole thing? can i do a synchronous wrapper?
        prompt, extra_kwargs = self.prompt(**inputs)

        if self.cache_dir is not None:
            if _cache_idx is None:
                cache_key = _cache_key(self.llm_kwargs, self.system, prompt)
            else:
                cache_key = _cache_key(self.llm_kwargs, self.system, prompt, _cache_idx)
            
            cache_path = Path(self.cache_dir) / f"{cache_key}.json"
            if cache_path.exists():
                return json.loads(cache_path.read_text())
        
        # vvvvvv ONLY DIFFERENCE FROM SYNC VERSION vvvvvv
        response = await acompletion(
            **self.llm_kwargs,
            num_retries     = 3,
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
        
        if self.cache_dir is not None:
            cache_path = Path(self.cache_dir) / f"{cache_key}.json"
            cache_path.write_text(json.dumps(output))
        
        assert isinstance(output, dict)
        if self.do_log and self.LOG_DIR is not None:
            utils.log(self.LOG_DIR, self.name, self.counter, prompt, output_str, output, show_console=self.do_console)
        
        self.counter += 1
                
        return output
    
    def larun(self, **inputs):
        def _fn():
            return self.arun(**inputs)
        
        return _fn

