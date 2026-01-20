import random
import json
import time
from tqdm import tqdm
import concurrent.futures
import pandas as pd
from dotenv import load_dotenv
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


import os


load_dotenv()

# ============================================
# CONFIGURATION - Change these parameters
# ============================================

DATA_PATHS = [
    "/dataset/generated_questions/generated_question_conv1-th.json",
    "/dataset/generated_questions/generated_question_conv3-th.json",]
LLM_NAME = "./Gemma-SEA-LION-v3-9B"  # Change to your model
MAX_WORKERS = 1  # Changed to 1 since we're using GPU inference
USE_COT = True  # Set to True for chain-of-thought reasoning

# ============================================
# Load model and tokenizer
# ============================================
print(f"Loading model: {LLM_NAME}")
tokenizer = AutoTokenizer.from_pretrained(LLM_NAME)
model = AutoModelForCausalLM.from_pretrained(
    LLM_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.eval()  # Set to evaluation mode
print("Model loaded successfully!")

# Set pad token if not already set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ============================================
# Helper Functions
# ============================================
def load_TactfulToM_dataset(data_path):
    with open(data_path) as f:
        dataset = json.load(f)
    df = pd.DataFrame(dataset)
    return df

class LLM:
    def __init__(self, llm_name, max_workers):
        self.llm_name = llm_name
        self.max_workers = max_workers
        self.error_times = 0
        
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.cached_tokens = 0

    def get_token_usage(self):
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cached_tokens": self.cached_tokens,
            "total_tokens": self.total_tokens
        }
    
    def generate_single(self, single_input):
        messages = single_input
        
        # Manually format messages since ThaiLLM doesn't have a chat template
        # Format: System: <system>\nUser: <user>\nAssistant:
        input_text = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                input_text += f"{content}\n\n"
            elif role == "user":
                input_text += f"{content}\n\n"
            elif role == "assistant":
                input_text += f"{content}\n\n"
        
        # Add a prompt for the assistant to respond
        # input_text += "Answer: "
        
        # Tokenize the input
        model_inputs = tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048  # Adjust based on your needs
        ).to(model.device)
        
        # Count prompt tokens
        prompt_token_count = model_inputs.input_ids.shape[1]
        self.prompt_tokens += prompt_token_count

        # Generate response
        with torch.inference_mode(): 
            generate_ids = model.generate( 
                model_inputs.input_ids,
                max_new_tokens=500, 
                repetition_penalty=1.2, 
                num_beams=1, 
                do_sample=True, 
                temperature=0.2, 
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode only the new tokens (excluding the prompt)
        response = tokenizer.batch_decode(
            generate_ids[:, prompt_token_count:],  # Only decode new tokens
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=True
        )[0]
        
        # Update token counts
        completion_token_count = generate_ids.shape[1] - prompt_token_count
        self.completion_tokens += completion_token_count
        self.total_tokens += prompt_token_count + completion_token_count
        
        return response.strip()
    
    def generate_helper(self, args):
        try:
            return self.generate_single(args)
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            self.error_times += 1
            return "ERROR"
    
    def generate_set(self, set_inputs, cot=False):
        # For GPU inference, process sequentially to avoid memory issues
        responses = []
        for single_input in tqdm(set_inputs, desc="Generating responses"):
            response = self.generate_helper(single_input)
            responses.append(response)
        
        if not cot:
            return responses
        else:
            # For COT, append the response and ask for final answer
            new_inputs = []
            for i, single_input in enumerate(set_inputs):
                if responses[i] != "ERROR":
                    # Create a new messages list with the assistant's response
                    new_messages = single_input.copy()
                    new_messages.append({
                        "role": "assistant",
                        "content": responses[i]
                    })
                    new_messages.append({
                        "role": "user",
                        "content": "Therefore, the final answer is:"
                    })
                    new_inputs.append(new_messages)
                else:
                    new_inputs.append(single_input)
            return self.generate_set(new_inputs, cot=False)

def get_user_prompt(question_type, context, question, options=None, information_prompt=None, cot=None):
    context_prompt = f"# Context:\n{context}\n\n"
    
    if not information_prompt:
        information_prompt = ""
    
    question_prompt = f"# Question:\n{question}\n\n"
    
    if question_type == "mcq":
        options_text = "\n".join([f"{i+1}. {option}" for i, option in enumerate(options)])
        option_prompt = f"# Options:\n{options_text}\n\n"
    else:
        option_prompt = ""
    
    if cot:
        cot_prompt = "Let's think step by step:\n"
    else:
        cot_prompt = ""
    
    user_prompt = context_prompt + information_prompt + question_prompt + option_prompt + cot_prompt
    return user_prompt

def get_system_prompt(question_type, cot):
    if not cot:
        if question_type == "binary":
            system_prompt = "You are an expert in social reasoning. Answer the following question with 'Yes' or 'No'. Remember: Your answer should ONLY include 'Yes' or 'No' with nothing else."
        elif question_type == "freeform":
            system_prompt = "You are an expert in social reasoning. Answer the following question with a single sentence."
        elif question_type == "mcq":
            system_prompt = "You are an expert in social reasoning. Answer the following question with the option number of the most appropriate answer. Remember: Your answer should ONLY include the option number with nothing else." 
        elif question_type == "list":
            system_prompt = "You are an expert in social reasoning. List the required items and split them with commas. Remember: Your answer should ONLY include the required items split by commas with nothing else."
    else:
        if question_type == "binary":
            system_prompt = "You are an expert in social reasoning. Think step by step and give a final answer of 'Yes' or 'No' only."
        elif question_type == "freeform":
            system_prompt = "You are an expert in social reasoning. Think step by step and give a final answer of a single sentence."
        elif question_type == "mcq":
            system_prompt = "You are an expert in social reasoning. Think step by step and give a final answer of the option number of the most appropriate answer."
        elif question_type == "list":
            system_prompt = "You are an expert in social reasoning. Think step by step, list the required items and split them with commas."
    return system_prompt

def get_llm_input(question, question_type, context, cot):
    question_text = question["question"]
    system_prompt = get_system_prompt(question_type, cot)
    
    # information prompt
    if "information" in question.keys():
        information_prompt = f"# Information:\n{question['information']}\n\n"
    else:
        target_question_keys = ["fact_question_real_reason", "comprehension_q", "fact_question_truth", "truth_question"]
        information_prompt = ""
        for q_key in target_question_keys:
            if q_key in question.keys():
                information_prompt = f"# Target Question:\n{question[q_key]}\n\n"
                break
    
    # options
    if question_type == "mcq":
        if "wrong_answer" in question.keys():
            wrong_answer_list = question["wrong_answer"]
        elif "wrong_answers" in question.keys():
            wrong_answer_list = question["wrong_answers"]
        else:
            print("No wrong answers in the following question.")
            print(question)
            wrong_answer_list = []
        
        if not type(wrong_answer_list) == list:
            wrong_answer_list = [wrong_answer_list]
        while type(wrong_answer_list[0]) == list:
            wrong_answer_list = wrong_answer_list[0]
        
        options = [question["correct_answer"]] + wrong_answer_list
        mapping = list(range(len(options)))
        random.shuffle(mapping)
        new_options = [options[m] for m in mapping]
        user_prompt = get_user_prompt(question_type, context, question_text, new_options, information_prompt, cot)
    else:
        mapping, options = [], []
        user_prompt = get_user_prompt(question_type, context, question_text, None, information_prompt, cot)
    
    llm_input = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ]
    return llm_input, mapping

def get_results(data_path, llm_name, cot, max_workers):
    # load data
    df = load_TactfulToM_dataset(data_path)
    
    llm = LLM(llm_name, max_workers)
    
    question_categories = [
        "comprehensionQA", "justificationQA", "fact_reasonQA", "fact_truthQA", 
        "beliefQAs", "infoAccessibilityQA_list", "infoAccessibilityQAs_binary", 
        "answerabilityQA_list", "answerabilityQAs_binary", "lieabilityQAs",
        "liedetectabilityQAs_list", "liedetectabilityQAs_binary", "KJability"
    ]
    
    results = []
    for idx, questions_set in df.iterrows():
        if idx % 10 == 0:
            print(f"Progress: {idx} / {df.shape[0]}")
        
        set_results = {cat: [] for cat in question_categories}
        set_inputs = []
        
        # Step 1: Initialize the result list
        for cat in question_categories:
            if cat not in questions_set.keys():
                continue
            
            cat_questions = questions_set[cat]
            if not isinstance(cat_questions, list):
                continue
                
            for question in cat_questions:
                # question_type for different question_category
                if "list" in cat:
                    question_types = ["list"]
                elif "binary" in cat:
                    question_types = ["binary"]
                elif cat == "comprehensionQA":
                    question_types = ["binary"]
                elif cat in ["lieabilityQAs", "KJability"]:
                    question_types = ["mcq"]
                else:
                    question_types = ["mcq"]
                
                results_question = []
                for question_type in question_types:
                    for context_type in ["full_context"]:
                        context = questions_set.get(context_type, "")
                        llm_input, mcq_mapping = get_llm_input(question, question_type, context, cot)
                        set_inputs.append(llm_input)
                        
                        results_question.append({
                            "question": question["question"],
                            "correct_answer": question["correct_answer"],
                            "original_result": "",
                            "clean_result": "",
                            "question_type": question_type,
                            "context_type": context_type,
                            "mcq_mapping": mcq_mapping,
                            "question_id": question.get("q_id", "")
                        })
                set_results[cat].append(results_question)
        
        # Step 2: Use LLM to generate the results
        set_outputs = llm.generate_set(set_inputs, cot)
        print(f"Error times: {llm.error_times}")
        
        # Step 3: Update the result list
        pointer = 0
        for cat in question_categories:
            for i, result_cat in enumerate(set_results[cat]):
                for j, result in enumerate(result_cat):
                    original_output = set_outputs[pointer]
                    set_results[cat][i][j]["original_result"] = original_output
                    pointer += 1
        
        results.append({
            "results": set_results,
            "token_usage": llm.get_token_usage()
        })
        
        # Save intermediate results
        file_name = llm_name.split("/")[-1]
        if cot:
            file_name += "-cot"
        question_type = data_path.split("generated_")[-1].split(".")[0]
        file_name += f"-{question_type}"
        
        with open(f"{file_name}.json", "w", encoding='utf-8') as f:
            json.dump(results, f, indent=3, ensure_ascii=False)
    
    return results

# ============================================
# RUN THE EVALUATION
# ============================================
if __name__ == "__main__":
    for data_path in DATA_PATHS:
        print(f"\n{'='*60}")
        print(f"Processing: {data_path}")
        print(f"{'='*60}\n")
        
        try:
            results = get_results(data_path, LLM_NAME, USE_COT, MAX_WORKERS)
            print(f"✓ Successfully processed {data_path}")
        except Exception as e:
            print(f"✗ Error processing {data_path}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*60)
    print("All processing complete!")
    print("="*60)
