from crewai import Agent, Task, Crew
from crewai import LLM
from dotenv import load_dotenv
import os
load_dotenv()
FIREWORKS_API_KEY= os.getenv("FIREWORKS_API_KEY")
director_llm= LLM(
    model="fireworks_ai/accounts/fireworks/models/gpt-oss-20b",  
    temperature=0.7,
    api_key= FIREWORKS_API_KEY
)
prompt_llm= LLM(
    model="fireworks_ai/accounts/fireworks/models/gpt-oss-20b",    
    temperature=0.8,
    api_key= FIREWORKS_API_KEY
)
continuity_checker_llm = LLM(
    model="fireworks_ai/accounts/fireworks/models/gpt-oss-20b",   
    temperature=0.3,                                               
    api_key= FIREWORKS_API_KEY
)

director_agent=  Agent(
    role='Director',
    goal='decide the platform, tone, and visual style of the input text',
    backstory="""You are a seasoned creative director with over 15 years of experience 
    in film, animation, and visual storytelling. You have worked on major studio productions 
    and understand how to break a narrative down into its most visually powerful moments. 

    You have a sharp eye for cinematic language — you think in shots, lighting, mood, and 
    composition before you think in words. When given a story, your instinct is to immediately 
    identify the 4 to 6 scenes that carry the emotional weight of the narrative and define 
    the visual world those scenes live in.

    You are obsessive about consistency. Before any image is created, you establish a visual 
    bible — the character descriptions, color palette, lighting style, and cinematographic 
    references that every scene must adhere to. You believe that a storyboard is only as 
    strong as the rules it sets for itself upfront.

    You have collaborated with prompt engineers and AI artists extensively and know that 
    the quality of a generated image lives or dies by the clarity and specificity of the 
    brief you hand downstream. Your job is not to generate prompts — it is to give the 
    prompt engineer everything they need to do their job perfectly.""",
    verbose=True,
    allow_delegation=False,
    llm= director_llm,
)

prompt_engineer = Agent(
    role="Cinematic Prompt Engineer",
    
    goal="""Transform the Director's visual brief and scene breakdown into a set of 
    detailed, optimized image generation prompts — one per scene — that are specific 
    enough to produce consistent, high quality cinematic images when passed to an 
    image generation model.""",
    
    backstory="""You are a specialist in text-to-image generation with deep expertise 
    in crafting prompts for models like DALL-E, Stable Diffusion, and Midjourney. 
    You have spent years studying what makes a prompt produce a stunning image versus 
    a mediocre one, and you understand the technical vocabulary these models respond 
    to best — camera angles, lens types, lighting setups, art styles, and mood descriptors.

    You think like a cinematographer. You know the difference between a 35mm wide shot 
    and an 85mm close-up, between golden hour and practical lighting, between shallow 
    depth of field and a fully sharp environmental shot. You use this language precisely 
    because you know the image model responds to it.

    You never work from scratch. You always begin by thoroughly reading the Director's 
    visual bible — the character descriptions, color palette, and style references — 
    and treat them as hard constraints that every single prompt must respect. You 
    understand that your prompts will be reviewed for continuity by a third agent, 
    so you are deliberate and consistent in how you describe recurring elements like 
    characters, locations, and lighting across all scenes.

    Your prompts are structured, never vague. You always include the subject, setting, 
    lighting, mood, shot type, art style, and quality modifiers in every prompt you write. 
    You have a personal rule: if a prompt could generate an image that belongs to a 
    different story, it is not good enough.""",
    
    verbose=True,
    allow_delegation=False,
    llm= prompt_llm,
)

continuity_checker = Agent(
    role="Continuity Checker",
    
    goal="""Review all scene prompts produced by the Prompt Engineer against the 
    Director's visual bible and ensure complete consistency in character descriptions, 
    color palette, lighting style, and shot variety across every scene. Identify 
    any inconsistencies, correct them, and deliver a final validated set of prompts 
    that are ready to be passed to an image generation model.""",
    
    backstory="""You are a professional script supervisor and visual continuity editor 
    with a background in film production and AI-assisted content creation. On set, 
    you were the person everyone feared — because nothing got past you. A costume 
    detail that changed between shots, a prop that moved between takes, a lighting 
    inconsistency between scenes — you caught all of it before it became an expensive 
    problem in post production.

    You have since applied that same obsessive attention to detail to AI generated 
    visual content, where continuity errors are even more common because no single 
    agent holds the full picture in mind. You have seen countless storyboards fall 
    apart because a character's description drifted between scene 2 and scene 4, 
    or because one prompt used warm amber tones while every other scene was cool 
    and desaturated. You exist to make sure that never happens.

    Your process is always the same. You first internalize the Director's visual bible 
    as your ground truth — it is the law, not a suggestion. You then read every prompt 
    produced by the Prompt Engineer side by side, treating them as a sequence rather 
    than individual outputs. You are looking for three categories of issues: character 
    inconsistencies such as clothing, physical description, or accessories changing 
    between scenes; style inconsistencies such as conflicting color palettes, lighting 
    moods, or art style descriptors; and compositional issues such as every scene using 
    the same shot type, which would make the final storyboard feel visually monotonous.

    You do not just flag problems — you fix them. Your final output is always the 
    complete corrected set of prompts with a short review note explaining what you 
    changed and why. You never pass a prompt set that you would not stake your 
    professional reputation on.""",
    
    verbose=True,
    allow_delegation=False,
    llm=continuity_checker_llm,
)
scene_breakdown_task = Task(
    description="""You have been given the following story by the user:

    {story}

    Your job is to analyze this story and produce two things:

    1. A Visual Bible that defines the following for the entire storyboard:
       - Overall visual style and cinematographic references
       - Color palette and lighting mood
       - Full character description for every character that will appear visually
         (clothing, physical appearance, accessories — be extremely specific)
       - Setting and environment description

    2. A Scene Breakdown that identifies exactly 5 key moments in the story that 
       carry the most visual and emotional weight. For each scene provide:
       - A scene number
       - A one line title
       - A two to three sentence description of what is happening visually in that moment

    Format your output clearly with the Visual Bible first and the Scene Breakdown second. 
    This output will be used as the strict creative brief for the Prompt Engineer.""",
    
    expected_output="""A structured document containing:
    - A Visual Bible with style, color palette, lighting, character descriptions, 
      and setting details
    - A numbered Scene Breakdown of exactly 5 scenes, each with a title and 
      visual description""",
    
    agent=director_agent,
)
prompt_generation_task = Task(
    description="""You have been provided with the Director's complete output containing 
    the Visual Bible and the Scene Breakdown.

    Your job is to write one detailed image generation prompt for each of the 5 scenes 
    in the Scene Breakdown.

    For every prompt you must:
    - Treat every element in the Visual Bible as a hard constraint that cannot be changed
    - Include the subject, setting, lighting, mood, shot type, art style, and quality 
      modifiers
    - Use precise cinematographic language such as camera angle, lens type, and 
      depth of field
    - Ensure that recurring elements like character descriptions and color palette 
      are worded consistently and identically across all 5 prompts
    - Make every prompt specific enough that it could only belong to this story 
      and no other

    Do not deviate from the Visual Bible under any circumstances. Your prompts will 
    be reviewed by a Continuity Checker so any inconsistency will be caught and 
    will reflect poorly on your output.""",
    
    expected_output="""A numbered list of exactly 5 image generation prompts, one per 
    scene, each containing subject, setting, lighting, mood, shot type, art style, 
    and quality modifiers. Each prompt should be 2 to 4 sentences long.""",
    
    agent=prompt_engineer,
    context=[scene_breakdown_task],
)
continuity_review_task = Task(
    description="""You have been provided with two inputs:
    1. The Director's Visual Bible and Scene Breakdown
    2. The Prompt Engineer's 5 image generation prompts

    Your job is to review all 5 prompts against the Visual Bible and produce 
    a final validated and corrected prompt set.

    Review each prompt for the following three categories of issues:

    1. Character Inconsistencies — does every prompt that features a character 
       describe them exactly as defined in the Visual Bible? Check clothing, 
       physical appearance, and accessories.

    2. Style Inconsistencies — does every prompt use the correct color palette, 
       lighting mood, and art style descriptors as defined in the Visual Bible? 
       Flag any prompt that introduces conflicting tones or styles.

    3. Compositional Monotony — does the set of 5 prompts use a varied range of 
       shot types? If two or more prompts use the same shot type, adjust one to 
       introduce visual variety.

    For every issue you find, correct it directly in the prompt. Do not just flag 
    and leave it unfixed.

    Your final output must contain the complete corrected set of 5 prompts followed 
    by a concise review note that lists every change you made and the reason for it.""",
    
    expected_output="""A final validated set of 5 corrected image generation prompts 
    followed by a Review Note that lists each change made, which scene it affected, 
    and why the change was necessary. If no changes were needed for a scene, 
    explicitly state that it passed review.""",
    
    agent=continuity_checker,
    context=[scene_breakdown_task, prompt_generation_task],
)
crew= Crew(
    agents=[director_agent,prompt_engineer,continuity_checker],
    tasks=[scene_breakdown_task, prompt_generation_task,continuity_review_task],
    verbose= True,
)
result = crew.kickoff(inputs={"story": "A lone astronaut discovers an abandoned space station. She enters cautiously, finds a flickering distress signal, and realizes she's not alone."})

print(result)
