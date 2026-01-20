import pandas as pd
import random
import json
import argparse


from openai import OpenAI
from dotenv import load_dotenv
import os



from conv_generation_utils import (
    get_leave_reasons,
    replace_ABCD_with_name,
    append_data_to_json,
    clean_val,
    clean_val2,
)

load_dotenv()
random.seed(42)

# Thai names with Thai characters these famous thai name are from  https://forebears.io/thailand/forenames
TH_NAMES = {
    "Somchai": "สมชัย",
    "Somchit": "สมจิตร",
    "Prasoet": "ประเสริฐ",
    "Sombun": "สมบูรณ์",
    "Somsak": "สมศักดิ์",
    "Narong": "ณรงค์",
    "Prasit": "ประสิทธิ์",
    "Somphon": "สมพล",
    "Nittaya": "นิตยา",
    "Sombat": "สมบัติ",
    "Udom": "อุดม",
    "Wichian": "วิเชียร",
    "Amphon": "อำพล",
    "Thawi": "ทวี",
    "Charoen": "เจริญ",
    "Samran": "สำราญ",
    "Wichai": "วิชัย",
    "Sawat": "สวัสดิ์",
    "Chan": "จันทร์",
    "Prani": "ปราณี"
}



class ConversationGenerator:
    def __init__(self, language="Thai", thai_names=False):


        self.api_key = os.getenv("OPENAI_API_KEY","")
        self.client = OpenAI(api_key=self.api_key)
        self.language = language
        self.thai_names = thai_names  #to gen thai name letter mode but in defualt is 
        
    def generate_conversation_part(self, prompt, previous_context=""):
        """Generate a single conversation part using OpenAI API"""
        if self.language == "Thai":
            system_prompt = "You are a creative writer generating natural, realistic conversations in Thai language. Write dialogue in a clear format with character names followed by their speech. Keep the names in English but the conversation in Thai. Use natural Thai conversation style."
        else:
            system_prompt = "You are a creative writer generating natural, realistic conversations. Write dialogue in a clear format with character names followed by their speech."
        
        messages = [{"role": "system", "content": system_prompt}]
        
        if previous_context:
            messages.append({
                "role": "user",
                "content": f"Previous conversation:\n{previous_context}\n\n{prompt}"
            })
        else:
            messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            messages=messages,
            temperature=0.2,
            max_tokens=1000
        )
        
        tokens = response.usage.completion_tokens
        return response.choices[0].message.content, tokens

    
    def generate_full_conversation(self, config):
        conv_id = config.get('ConversationID', config.get('conv_id', 'unknown'))
        print(f"Generating conversation {conv_id}...")
        
        # Get character names - always use Thai names for Thai language
        if self.language == "Thai":
            available_names = list(TH_NAMES.keys())
            selected_names = random.sample(available_names, 4)
            A_name, B_name, C_name, D_name = selected_names#available_names[:4] #selected_names
            if self.thai_names:
                A_name = TH_NAMES.get(A_name, A_name)
                B_name = TH_NAMES.get(B_name, B_name)
                C_name = TH_NAMES.get(C_name, C_name)
                D_name = TH_NAMES.get(D_name, D_name)

        if self.language == "English":
            # Load names dataset
            script_dir = os.path.dirname(os.path.abspath(__file__))
            names_df = pd.read_csv(os.path.join(script_dir, "names_by_birth_year.csv"))
            names_df_sorted = names_df.sort_values(by="Count", ascending=False)
            top_20_count = int(len(names_df_sorted) * 0.2)
            names_df_top_20 = names_df_sorted.iloc[:top_20_count]
            A_name,B_name,C_name,D_name = names_df_top_20['Name'].sample(n=4).to_list()
        
        # Get leave reasons
        leave_reasons = get_leave_reasons(th=(self.language == "Thai"))
        leave_reason_B, leave_reason_D_1, leave_reason_D_2 = random.sample(leave_reasons, 3)
        
        # Parse relationship to get roles
        relationship_text = config.get('Relationship', '')
        
        # Extract data from config using your column names
        scenario = config.get('Scenario', '')
        situation_topic = config.get('SituationTopic', '')
        situation = config.get('Situation', '')
        lie_objective = config.get('LieObjective', '')
        real_reason = config.get('RealReason', '')
        lie = config.get('Lie', '')
        truth = clean_val2(config.get('Truth', ''))
        
        # Replace placeholders with actual names
        real_reason_c_filled = replace_ABCD_with_name(real_reason, A_name, B_name, C_name, D_name)
        lie_c_filled = replace_ABCD_with_name(lie, A_name, B_name, C_name, D_name)
        truth_c_filled = replace_ABCD_with_name(truth, A_name, B_name, C_name, D_name)
        scenario = replace_ABCD_with_name(scenario, A_name, B_name, C_name, D_name)
        situation_topic = replace_ABCD_with_name(situation_topic, A_name, B_name, C_name, D_name)
        situation = replace_ABCD_with_name(situation, A_name, B_name, C_name, D_name)
        lie_objective = replace_ABCD_with_name(lie_objective, A_name, B_name, C_name, D_name)
        relationship_text = replace_ABCD_with_name(relationship_text, A_name, B_name, C_name, D_name)
        
        
        # Create templates
        template_step_1 = f"""In this conversation, four characters are participating: {A_name} , {B_name} , {C_name} , and {D_name}.

Start the conversation with all participants {scenario}.

Relationship: {relationship_text}.

Generate a natural conversation between all four characters. Use their names and format the dialogue clearly."""

        template_step_2 = f"""Now, {B_name} leaves the conversation because of the reason: {leave_reason_B}, while {D_name} leaves the conversation because of the reason: {leave_reason_D_1}.

With {B_name} and {D_name} absent, {A_name} and {C_name} continue their conversation, shifting the topic to '{real_reason_c_filled}'. Ensure the dialogue flows naturally without directly repeating these exact words.

Remember to indicate that {B_name} and {D_name} are leaving the conversation."""

        template_step_3 = f"""Now, only {D_name} returns and rejoins the conversation after previously leaving. Note that {B_name} has not returned to the conversation yet.

Then the conversation continues and naturally shifts to {situation_topic}. Do NOT catch up or recap details from the previous conversation.

And then '{truth_c_filled}' is naturally revealed in the conversation.

After engaging briefly, {D_name} leaves the conversation again because of the reason: {leave_reason_D_2}.

IMPORTANT:
• Clearly indicate through dialogue only (e.g., "I'm back," or "I have to leave again now.") that {D_name} first returns and later departs again.
• Do NOT use third-person narrative or stage directions.
• Avoid mentioning details from previous conversations."""

        template_step_4 = f"""Now {B_name} returns to the conversation after leaving the conversation.

First have {B_name} explicitly indicate the return through dialogue. Do NOT catch up or recap details from the previous conversation.

And then {situation} naturally unfolds. Make sure the dialogue flows naturally without directly repeating these exact words.

In response, {A_name} and {C_name} tells {B_name} that '{lie_c_filled}' to {lie_objective}.

IMPORTANT:
• Do NOT use third-person narrative or stage directions.
• Avoid mentioning details from previous conversations."""

        # Generate parts
        print("  Generating Part 1...")
        part_1, tokens1 = self.generate_conversation_part(template_step_1)
        
        print("  Generating Part 2...")
        part_2, tokens2 = self.generate_conversation_part(template_step_2, part_1)
        
        print("  Generating Part 3...")
        part_3, tokens3 = self.generate_conversation_part(template_step_3, f"{part_1}\n\n{part_2}")
        
        print("  Generating Part 4...")
        part_4, tokens4 = self.generate_conversation_part(template_step_4, f"{part_1}\n\n{part_2}\n\n{part_3}")
        
        # Combine results
        full_context = "\n\n".join([part_1, part_2, part_3, part_4])
        short_context = "\n\n".join([part_2, part_3, part_4])
        
        # Handle falsification value
        falsification = config.get('Falsification', config.get('falsification'))
        if isinstance(falsification, str):
            falsification_upper = falsification.upper()
            if falsification_upper in ['TRUE', 'YES', '1']:
                falsification = True
            elif falsification_upper in ['FALSE', 'NO', '0']:
                falsification = False
            elif falsification_upper in ['NULL', 'NONE', '', 'NAN']:
                falsification = None
        elif pd.isna(falsification):
            falsification = None
        
        # Handle boolean fields
        multiple_liar = config.get('MultipleLair', config.get('multiple_liar', False))
        if isinstance(multiple_liar, str):
            multiple_liar = multiple_liar.upper() in ['TRUE', 'YES', '1']
        
        real_reason_type = config.get('RealReasontype', config.get('real_reason_type', False))
        if isinstance(real_reason_type, str):
            real_reason_type = real_reason_type.upper() in ['TRUE', 'YES', '1']
        
        # Create data dictionary
        data_dict = {
            "setup_file": config.get('Setupfile', config.get('setup_file', '')),
            "language": config.get('Lang', self.language),
            "set_id": config.get('SetID', config.get('set_id', '')),
            "lie_id": config.get('LieID', config.get('lie_id', '')),
            "conv_id": conv_id,
            "truth_id": config.get('TruthID', config.get('truth_id', '')),
            "lie_type": config.get('LieType', config.get('lie_type', '')),
            "emotion": config.get('Emotion', config.get('emotion', '')),
            "falsification": falsification,
            "multiple_liar": multiple_liar,
            "real_reason_type": real_reason_type,
            "topic": {
                "scenario": scenario,
                "situation_topic": situation_topic,
                "situation": situation,
                "lie_objective": lie_objective,
                "leave_reason_B": leave_reason_B,
                "leave_reason_D_1": leave_reason_D_1,
                "leave_reason_D_2": leave_reason_D_2
            },
            "relationship": relationship_text,
            "characters": {
                "liar": A_name,
                "target": B_name,
                "accomplice": C_name,
                "observer": D_name
            },
            "lie": {
                "real_reason_c": real_reason_c_filled,
                "lie_c": lie_c_filled,
                "truth_c": truth_c_filled
            },
            "entity": {
                "PLACE": clean_val(config.get('PLACE', 0)),
                "PERSON": clean_val(config.get('PERSON', 0)),
                "ORGANISM": clean_val(config.get('ORGANISM', 0)),
                "ARTIFACT/OBJECT": clean_val(config.get('ARTIFACT/OBJECT', 0)),
                "TRADITION/EVENT": clean_val(config.get('TRADITION/EVENT', 0)),
                "ORGANIZATION": clean_val(config.get('ORGANIZATION', 0)),
                "FOOD": clean_val(config.get('FOOD', 0)),
                "TIME": clean_val(config.get('TIME', 0))
            },
            "part_1": part_1,
            "part_2": part_2,
            "part_3": part_3,
            "part_4": part_4,
            "full_context": full_context,
            "short_context": short_context,
            "full_context_tokens": tokens1 + tokens2 + tokens3 + tokens4,
            "short_context_tokens": tokens2 + tokens3 + tokens4,
        }
        
        print(f"  ✓ Completed ({data_dict['full_context_tokens']} tokens)\n")
        return data_dict


def load_config_from_csv(csv_path):
    """Load conversation configurations from CSV file"""
    df = pd.read_csv(csv_path, encoding='utf-8')
    
    print(f"CSV columns found: {df.columns.tolist()}\n")
    
    # Convert DataFrame to list of dictionaries
    configs = df.to_dict('records')
    
    return configs


def main():
    parser = argparse.ArgumentParser(description='Generate conversations from CSV configuration')
    parser.add_argument('--config', '-c', required=True, help='Path to CSV configuration file')
    parser.add_argument('--output', '-o', default='generated_conversations1-th.json', help='Output JSON file path')
    parser.add_argument('--language', '-l', choices=['English', 'Thai'], default='Thai', help='Language for generation')
    parser.add_argument('--thai-names','-tn',action='store_true', help='gen thai name mode (only for Thai language)')
    parser.add_argument('--limit', type=int, help='Limit number of conversations to generate')
    parser.add_argument('--start', type=int, default=0, help='Start index (0-based)')
    
    args = parser.parse_args()
    
    # Load configurations
    print(f"Loading configurations from {args.config}...")
    configs = load_config_from_csv(args.config)
    
    # Apply start and limit
    if args.start > 0:
        configs = configs[args.start:]
        print(f"Starting from index {args.start}")
    
    if args.limit:
        configs = configs[:args.limit]
    
    print(f"Will generate {len(configs)} conversation(s)\n")
    
    # Initialize generator
    generator = ConversationGenerator(language=args.language, thai_names=args.thai_names)
    
    # Generate conversations
    print(f"Generating conversations in {args.language}...\n")
    success_count = 0
    error_count = 0
    
    for i, config in enumerate(configs, 1):
        print(f"[{i}/{len(configs)}]")
        try:
            conversation = generator.generate_full_conversation(config)
            append_data_to_json(conversation, args.output)
            success_count += 1
        except Exception as e:
            print(f"  ✗ Error: {str(e)}\n")
            error_count += 1
            continue
    
    print(f"\n{'='*60}")
    print(f"✓ Successfully generated: {success_count}")
    if error_count > 0:
        print(f"✗ Failed: {error_count}")
    print(f"💾 All conversations saved to {args.output}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()