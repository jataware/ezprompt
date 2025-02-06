#!/usr/bin/env python3
"""
    hello.py
"""

import re
import ezprompt
from ezprompt import EZPrompt
ezprompt.EZPrompt.set_default_logdir('./ezlogs')

llm_kwargs = {
    "temperature" : 0.01,
    "top_p"       : 0.9,
    "model"       : 'gpt-4o-2024-08-06',
    "max_retries" : 3
}

# --
# Prompts

idea_generator = EZPrompt(
    name     = "idea_generator",
    system   = "You are a creative science fiction writer.",
    template = "Give me 5 ideas for a science fiction story involving {TOPIC} and {SETTING}.  Each idea should be in an <idea></idea> tag.",
    before   = lambda topic, setting: {
        "TOPIC"   : topic,
        "SETTING" : setting
    },
    after      = lambda output_str: {"ideas" : re.findall(r"<idea>(.*?)</idea>", output_str, re.DOTALL).strip()},
    llm_kwargs = llm_kwargs
)

title_generator = EZPrompt(
    name     = "title_generator",
    system   = "You are a creative science fiction writer.",
    template = "Give me a title for a science fiction story about this book\n {DESCRIPTION}.",
    before   = lambda description: {
        "DESCRIPTION" : description
    },
    after      = lambda output_str: {"title" : output_str.strip()},
    llm_kwargs = llm_kwargs
)


# --
# Run sync

ideas = idea_generator.run(topic='aliens', setting='space')['ideas']

titles = []
for idea in ideas:
    title = title_generator.run(description=idea)['title']
    titles.append(title)

# --
# Run batched

ideas   = idea_generator.run(topic='aliens', setting='space')['ideas']

tasks   = {i : title_generator.larun(description=idea) for i, idea in enumerate(ideas)}
results = ezprompt.run_batch(tasks, show_progress=False)
titles  = [results[i]['title'] for i in range(len(results))]

