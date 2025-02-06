import asyncio
from pathlib import Path
from tqdm.asyncio import tqdm as atqdm

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

def log(LOG_DIR, name, counter, prompt, output_str, output, max_chars=1024):
    assert LOG_DIR is not None, "LOG_DIR must be set"
    
    console  = Console(record=True)
    log_path = Path(LOG_DIR) / f'{name}-{counter:04d}.txt'
    
    # Format prompt
    prompt_panel = Panel(prompt, title=f"{name.upper()} - {counter:04d} - INPUT", border_style="blue")
    
    # Format raw output
    output_str_panel = Panel(output_str[:max_chars], title=f"{name.upper()} - {counter:04d} - OUTPUT - RAW", border_style="yellow")
    
    # Format response
    output_text = Text()
    for k,v in output.items():
        output_text.append(f"## {k}\n", style="bold magenta")
        output_text.append(f"{v}\n\n")
    
    response_panel = Panel(output_text[:max_chars], title=f"{name.upper()} - {counter:04d} - OUTPUT - FMT", border_style="green")
    
    console.print(prompt_panel, output_str_panel, response_panel)
    
    # Save to file
    with open(log_path, 'w') as f:
        f.write(console.export_text())


async def arun_batch(prompts, max_concurrent=5):
    results = {}
    sem = asyncio.Semaphore(max_concurrent)
    
    async def _process_prompt(qid, prompt_fn):
        async with sem:
            try:
                result = await prompt_fn()
                return qid, result
            except Exception as e:
                return qid, None
    
    tasks = [_process_prompt(qid, prompt_fn) for qid, prompt_fn in prompts.items()]
    
    pbar = atqdm(total=len(prompts), desc="arun_batch")
    for coro in asyncio.as_completed(tasks):
        qid, result = await coro
        results[qid] = result
        pbar.update(1)
    
    pbar.close()
    return {qid: results[qid] for qid in prompts.keys()}


def run_batch(*args, **kwargs):
    return asyncio.run(arun_batch(*args, **kwargs))