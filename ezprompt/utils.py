import json
import asyncio
from pathlib import Path
from tqdm.asyncio import tqdm as atqdm

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.pretty import Pretty

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


async def arun_batch(prompts, max_concurrent=5, show_progress=True):
    results = {}
    sem = asyncio.Semaphore(max_concurrent)
    
    async def _process_prompt(qid, prompt_fn):
        async with sem:
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