#!/usr/bin/env python
"""
    ezprompt/utils.py
"""

import sys
import json
import asyncio
import numpy as np
from pathlib import Path
from pydantic import BaseModel
from tqdm.asyncio import tqdm as atqdm

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.live import Live
from rich.spinner import Spinner
from rich import print as rprint
import time

def spinner(msg, spinner_type="aesthetic"):
    start_time = time.time()
    console = Console()
    
    def get_spinner_text():
        elapsed = int(time.time() - start_time)
        return f"{msg} (elapsed: {elapsed}s)"
    
    spinner_obj = Spinner(spinner_type, text=get_spinner_text())
    
    # Create Live display with the spinner
    live = Live(spinner_obj, console=console, refresh_per_second=16)
    
    # Override the get_renderable method to update the elapsed time
    original_get_renderable = live.get_renderable
    def get_renderable():
        spinner_obj.text = get_spinner_text()
        return original_get_renderable()
    
    live.get_renderable = get_renderable
    
    return live

class PydanticEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, BaseModel):
            return obj.model_dump()
        return super().default(obj)

def log(LOG_DIR, name, counter, prompt, output_str, output, show_console=True):
    
    # JSON
    log_path_json = Path(LOG_DIR) / name / f'{name}-{counter:04d}.json'
    log_path_json.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path_json, 'w') as f:
        json.dump({
            "prompt"     : prompt,
            "output_str" : output_str,
            "output"     : output,
        }, f)
    
    # TXT
    if show_console:
        console = Console(record=True)
        
        # Format prompt
        prompt_panel = Panel(prompt, title=f"{name.upper()} - {counter:04d} - INPUT", border_style="blue")
        
        # Format raw output
        output_str_panel = Panel(output_str, title=f"{name.upper()} - {counter:04d} - OUTPUT - RAW", border_style="yellow")
        
        # Format response
        output_text = Pretty(output)
        response_panel = Panel(output_text, title=f"{name.upper()} - {counter:04d} - OUTPUT - FMT", border_style="green")
        
        console.print(prompt_panel, output_str_panel, response_panel)
        
        # Save to file
        log_path_txt  = Path(LOG_DIR) / name / f'{name}-{counter:04d}.txt'
        log_path_txt.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path_txt, 'w') as f:
            f.write(console.export_text())


# --

class RateLimiter:
    def __init__(self, max_calls, period):
        self.max_calls = max_calls
        self.period = period
        self.submission_times = []
        self.lock = asyncio.Lock()
    
    async def __aenter__(self):
        async with self.lock:
            now = time.monotonic()
            
            # Filter out submission times older than the period
            self.submission_times = [t for t in self.submission_times if now - t < self.period]
            
            if len(self.submission_times) >= self.max_calls:
                print([now - t for t in self.submission_times])
                oldest       = min(self.submission_times)
                expire_time  = oldest + self.period
                sleep_time   = expire_time - now
                print(f"Submission limit reached. Waiting {sleep_time:.2f} seconds before submitting the next request.", file=sys.stderr)
                await asyncio.sleep(sleep_time)
                
                # After sleeping, recalculate the submission times list
                now = time.monotonic()
                self.submission_times = [t for t in self.submission_times if now - t < self.period]
            
            # Add the current time to our submission times
            self.submission_times.append(now)
        
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass




async def arun_batch(futures, max_calls=9999, period=60, delay=0, verbose=True):
    results = {}
    
    rate_limiter = RateLimiter(max_calls=max_calls, period=period)
    
    async def _process_prompt(qid, future):
        if future.is_done:
            return qid, future.value
        
        await asyncio.sleep(np.random.exponential(delay))
        async with rate_limiter:
            if verbose:
                print(f"preparing : {qid}", file=sys.stderr)
            
            try:
                await asyncio.sleep(np.random.exponential(delay))
                
                if verbose:
                    print(f"running   : {qid}", file=sys.stderr)
                
                result = await future()
                
                if verbose:
                    print(f"complete  : {qid}", file=sys.stderr)
                
                return qid, result
            except Exception as e:
                rprint(f"[red]Error processing prompt {qid}[/red]: {e}", file=sys.stderr)
                return qid, None
    
    tasks = [_process_prompt(qid, future) for qid, future in futures.items()]
    
    pbar = atqdm(total=len(futures), desc="arun_batch", disable=not verbose)
    for coro in asyncio.as_completed(tasks):
        qid, result = await coro
        results[qid] = result
        pbar.update(1)
    
    pbar.close()
    return {qid: results[qid] for qid in futures.keys()}


def run_batch(*args, **kwargs):
    # Create a new event loop each time
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(arun_batch(*args, **kwargs))
    finally:
        loop.close()

