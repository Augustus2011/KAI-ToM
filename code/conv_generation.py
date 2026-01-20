import pandas as pd
import random
import os
import json
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv("/Users/kunkerdthaisong/GaiToM/.env")

from conv_generation_utils import (
    get_leave_reasons,
    replace_ABCD_with_name,
    populate_template,
    append_data_to_json,
    load_conversation_elements,
    extract_data_fields
)

th_name_list=["Somchai","Somchit","Prasoet","Sombun","Somsak","Narong","Prasit","Somphon","Nittaya","Sombat","Udom","Wichian","Amphon","Thawi","Charoen","Samran","Wichai","Sawat","Chan","Prani"]

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load names dataset
df = pd.read_csv("names_by_birth_year.csv")

# Sort by count and get top 20%
df_sorted = df.sort_values(by="Count", ascending=False)
top_20_count = int(len(df_sorted) * 0.2)
df_top_20 = df_sorted.iloc[:top_20_count]

# Get leave reasons
leave_reasons = get_leave_reasons()

# Sample leave reasons
leave_reason_B, leave_reason_D_1, leave_reason_D_2 = random.sample(leave_reasons, 3)
print(f"Leave reasons: {leave_reason_B}, {leave_reason_D_1}, {leave_reason_D_2}")

# Sample random names
random_names = df_top_20['Name'].sample(n=4).to_list()
A_name, B_name, C_name, D_name = random_names
print(f"Characters: {A_name}, {B_name}, {C_name}, {D_name}")

# Configure parameters
lie_type = "altruistic_white_lies"
emotion = "sad"
set_id = "2-10-5-2"
lie_id = "5-2"
conv_id = 10
truth_id = 1
falsification = False
multiple_liar = True
real_reason_type = False

scenario = "discussing pet care"
relationship = "families (parents, one kid, aunt/uncle)"
situation_topic = "talking about what happened"
situation = "kid is upset"
lie_objective = "comfort feelings"

real_reason_c = "child will be upset"
truth_c = "pet won't return"
lie_c = "pet is safe somewhere"

# Replace placeholders
real_reason_c = replace_ABCD_with_name(real_reason_c, A_name, B_name, C_name, D_name)
lie_c = replace_ABCD_with_name(lie_c, A_name, B_name, C_name, D_name)
truth_c = replace_ABCD_with_name(truth_c, A_name, B_name, C_name, D_name)

# Define conversation templates
template_step_1 = """In this conversation, four characters are participating: {{A: the liar name}}, {{B: the target name}}, {{C: the accomplice name}}, and {{D: the observer name}}.

Start the conversation with all participants {{Topic for the scenario}}.

They are {{relationship descriptor}}, where {{A: the liar name}} and {{C: the accomplice name}} are parents, {{B: the target name}} is kid, and {{D: the observer name}} is aunt/uncle.

Generate a natural conversation between all four characters. Use their names and format the dialogue clearly."""

template_step_2 = """Now, {{B: the target name}} leaves the conversation because of the reason: {{leave reason B}}, while {{D: the observer name}} leaves the conversation because of the reason: {{leave reason D_1}}.

With {{B: the target name}} and {{D: the observer name}} absent, {{A: the liar name}} and {{C: the accomplice name}} continue their conversation, shifting the topic to '{{real_reason_c}}'. Ensure the dialogue flows naturally without directly repeating these exact words.

Remember to indicate that {{B: the target name}} and {{D: the observer name}} are leaving the conversation."""

template_step_3 = """Now, only {{D: the observer name}} returns and rejoins the conversation after previously leaving. Note that {{B: the target name}} has not returned to the conversation yet.

Then the conversation continues and naturally shifts to {{situation_topic}}. Do NOT catch up or recap details from the previous conversation.

And then '{{truth_c}}' is naturally revealed in the conversation.

After engaging briefly, {{D: the observer name}} leaves the conversation again because of the reason: {{leave reason D_2}}.

IMPORTANT:
• Clearly indicate through dialogue only (e.g., "I'm back," or "I have to leave again now.") that {{D: the observer name}} first returns and later departs again.
• Do NOT use third-person narrative or stage directions.
• Avoid mentioning details from previous conversations."""

template_step_4 = """Now {{B: the target name}} returns to the conversation after leaving the conversation.

First have {{B: the target name}} explicitly indicate the return through dialogue. Do NOT catch up or recap details from the previous conversation.

And then {{situation}} naturally unfolds. Make sure the dialogue flows naturally without directly repeating these exact words.

In response, {{A: the liar name}} and {{C: the accomplice name}} tells {{B: the target name}} that '{{the lie}}' to {{lie_objective}}.

IMPORTANT:
• Do NOT use third-person narrative or stage directions.
• Avoid mentioning details from previous conversations."""

# Fill templates
filled_step_1 = populate_template(
    template_step_1, scenario, relationship, situation, lie_objective,
    real_reason_c, lie_c, truth_c, situation_topic,
    leave_reason_B, leave_reason_D_1, leave_reason_D_2,
    A_name, B_name, C_name, D_name
)

filled_step_2 = populate_template(
    template_step_2, scenario, relationship, situation, lie_objective,
    real_reason_c, lie_c, truth_c, situation_topic,
    leave_reason_B, leave_reason_D_1, leave_reason_D_2,
    A_name, B_name, C_name, D_name
)

filled_step_3 = populate_template(
    template_step_3, scenario, relationship, situation, lie_objective,
    real_reason_c, lie_c, truth_c, situation_topic,
    leave_reason_B, leave_reason_D_1, leave_reason_D_2,
    A_name, B_name, C_name, D_name
)

filled_step_4 = populate_template(
    template_step_4, scenario, relationship, situation, lie_objective,
    real_reason_c, lie_c, truth_c, situation_topic,
    leave_reason_B, leave_reason_D_1, leave_reason_D_2,
    A_name, B_name, C_name, D_name
)

# Helper function to generate conversation with OpenAI
def generate_conversation_part(prompt, previous_context=""):
    """
    Generate a conversation part using OpenAI API.
    
    Args:
        prompt: The instruction prompt for this step
        previous_context: Previous conversation parts for context
        
    Returns:
        str: Generated conversation text
    """
    messages = [
        {
            "role": "system",
            "content": "You are a creative writer generating natural, realistic conversations. Write dialogue in a clear format with character names followed by their speech."
        }
    ]
    
    if previous_context:
        messages.append({
            "role": "user",
            "content": f"Previous conversation:\n{previous_context}\n\n{prompt}"
        })
    else:
        messages.append({
            "role": "user",
            "content": prompt
        })
    
    response = client.chat.completions.create(
        model="gpt-4o-2024-08-06",
        messages=messages,
        temperature=0.8,
        max_tokens=1000
    )
    
    tokens=response.usage.completion_tokens

    return response.choices[0].message.content,tokens

# Generate conversation parts
print("\nGenerating Step 1...")
part_1, tokens1 = generate_conversation_part(filled_step_1)

print("Generating Step 2...")
part_2, tokens2 = generate_conversation_part(filled_step_2, part_1)

print("Generating Step 3...")
part_3, tokens3 = generate_conversation_part(filled_step_3, f"{part_1}\n\n{part_2}")

print("Generating Step 4...")
part_4, tokens4 = generate_conversation_part(filled_step_4, f"{part_1}\n\n{part_2}\n\n{part_3}")

# Combine and save
full_context = "\n\n".join([part_1, part_2, part_3, part_4])
full_context_tokens = tokens1 + tokens2 + tokens3 + tokens4
short_context = "\n\n".join([part_2, part_3, part_4])
short_context_tokens = tokens2 + tokens3 + tokens4

# Create data dictionary
data_dict = {
    "set_id": set_id,
    "lie_id": lie_id,
    "conv_id": conv_id,
    "truth_id": truth_id,
    "lie_type": lie_type,
    "emotion": emotion,
    "topic": {
        "scenario": scenario,
        "situation_topic": situation_topic,
        "situation": situation,
        "lie_objective": lie_objective,
        "leave_reason_B": leave_reason_B,
        "leave_reason_D_1": leave_reason_D_1,
        "leave_reason_D_2": leave_reason_D_2
    },
    "relationship": relationship,
    "multiple_liar": multiple_liar,
    "real_reason_type":real_reason_type,
    "characters": {
        "liar": A_name,
        "target": B_name,
        "accomplice": C_name,
        "observer": D_name
    },
    "lie": {
        "real_reason_c": real_reason_c,
        "lie_c": lie_c,
        "truth_c": truth_c,
        "falsification": falsification
    },
    "part_1": part_1,
    "part_2": part_2,
    "part_3": part_3,
    "part_4": part_4,
    "full_context": full_context,
    "short_context": short_context,
    "full_context_tokens": full_context_tokens,
    "Short_context_tokens": short_context_tokens
}

# Save to JSON file
output_file = "generated_conversations.json"
append_data_to_json(data_dict, output_file)

print(f"\n✓ Conversation generated and saved to {output_file}")
print(f"\nGenerated conversation preview:")
print("=" * 80)
print(full_context[:500] + "..." if len(full_context) > 500 else full_context)
print("=" * 80)