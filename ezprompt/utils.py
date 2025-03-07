import json
import asyncio
from pathlib import Path
from tqdm import tqdm
from tqdm.asyncio import tqdm as atqdm

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.pretty import Pretty
from rich.live import Live
from rich.spinner import Spinner
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

def log(LOG_DIR, name, counter, prompt, output_str, output, show_console=True):
    # assert LOG_DIR is not None, "LOG_DIR must be set"
    
    log_path_txt  = Path(LOG_DIR) / f'{name}-{counter:04d}.txt'
    log_path_json = Path(LOG_DIR) / f'{name}-{counter:04d}.json'
    
    with open(log_path_json, 'w') as f:
        json.dump({
            "prompt"     : prompt,
            "output_str" : output_str,
            "output"     : output,
        }, f)
    
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
        with open(log_path_txt, 'w') as f:
            f.write(console.export_text())


# --

class RateLimiter:
    def __init__(self, max_calls, period):
        self.max_calls = max_calls
        self.period = period  # period in seconds (e.g., 60 seconds)
        self.calls = 0
        self.reset_time = time.monotonic() + period
        self.lock = asyncio.Lock()
    
    async def __aenter__(self):
        async with self.lock:
            now = time.monotonic()
            if now >= self.reset_time:
                # Reset the counter and window
                self.calls = 0
                self.reset_time = now + self.period
            if self.calls >= self.max_calls:
                # Delay until the start of the next period
                sleep_time = self.reset_time - now
                print(f"Submission limit reached. Waiting {sleep_time:.2f} seconds before submitting the next request.")
                await asyncio.sleep(sleep_time)
                self.calls = 0
                self.reset_time = time.monotonic() + self.period
            self.calls += 1
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass



async def arun_batch(prompts, max_calls=9999, period=60, show_progress=True):
    results = {}
    
    rate_limiter = RateLimiter(max_calls=max_calls, period=period)
    
    async def _process_prompt(qid, prompt_fn):
        async with rate_limiter:
            print(f"processing: {qid}")
            try:
                result = await prompt_fn()
                return qid, result
            except Exception as e:
                print(f"Error processing prompt {qid}: {e}")
                return qid, None
    
    tasks = [_process_prompt(qid, prompt_fn) for qid, prompt_fn in prompts.items()]
    
    pbar = atqdm(total=len(prompts), desc="arun_batch", disable=not show_progress)
    for coro in asyncio.as_completed(tasks):
        qid, result = await coro
        results[qid] = result
        pbar.update(1)
    
    pbar.close()
    return {qid: results[qid] for qid in prompts.keys()}


def run_batch(*args, **kwargs):
    return asyncio.run(arun_batch(*args, **kwargs))

