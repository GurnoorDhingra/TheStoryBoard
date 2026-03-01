🎬 AI Storyboard Generator
A multi-agent AI pipeline that transforms a short story into a cinematic storyboard. Built with CrewAI, Fireworks AI, and React.

How It Works
Three specialized agents collaborate in sequence to produce a set of consistent, cinematic image generation prompts:

-Director Agent : Analyzes the story and produces a Visual Bible (color palette, character descriptions, lighting style) and a breakdown of 5 key scenes

-Prompt Engineer Agent : Converts each scene into a detailed image generation prompt using cinematographic language

-Continuity Checker Agent : Reviews all 5 prompts for character, style, and compositional consistency before finalizing

The finalized prompts are passed to the Flux image generation model via Fireworks AI to produce the storyboard images. A React frontend ties everything together into an interactive interface.
