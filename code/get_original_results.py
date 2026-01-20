import argparse
import random
from openai import OpenAI
import json
import time
from tqdm import tqdm
import concurrent.futures
import traceback

import pandas as pd
from dotenv import load_dotenv
import os
load_dotenv()

def load_TactfulToM_dataset(data_path):
    with open(data_path) as f:
        dataset = json.load(f)
    df = pd.DataFrame(dataset)
    return df

class LLM:
    def __init__(self, llm_name, max_workers, inference_mode="sync"):
        self.llm_name = llm_name
        self.max_workers = max_workers
        self.error_times = 0
        self.error_log = []  # Track specific errors

        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.cached_tokens = 0
        self.reasoning_tokens = 0
        self.inference_mode = inference_mode
        
        if "test" in llm_name:
            self.client = None
        elif "gpt" in llm_name:
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        elif "gemini" in llm_name or "qwen3-8b" in llm_name or "qwen-2.5-7b" in llm_name:
            self.client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENROUTER_API_KEY"))
        elif "Gemma-SEA" in llm_name:
            self.client = OpenAI(base_url="https://router.huggingface.co/v1", api_key=os.getenv("HF_TOKEN"))
        elif "typhoon" in llm_name:
            self.client = OpenAI(base_url="https://api.opentyphoon.ai/v1", api_key=os.getenv("TYPHOON_API_KEY1"))
        else: 
            self.client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENROUTER_API_KEY"))

    def build_batch_file(self, inputs, file_path):
        with open(file_path, "w", encoding="utf-8") as f:
            for i, messages in enumerate(inputs):
                if "gpt-5" in self.llm_name:
                    record = {
                        "custom_id": f"req-{i}",
                        "method": "POST",
                        "url": "/v1/responses",
                        "body": {
                            "model": self.llm_name,
                            "input": messages,
                            "max_output_tokens": 2000,
                            "reasoning": {"effort": "low"}
                        }
                    }
                    f.write(json.dumps(record) + "\n")

    def submit_batch(self, batch_file_path):
        file_obj = self.client.files.create(
            file=open(batch_file_path, "rb"),
            purpose="batch"
        )

        # Determine the correct endpoint based on model
        if "gpt-5" in self.llm_name:
            endpoint = "/v1/responses"
        else:
            endpoint = "/v1/chat/completions"

        batch = self.client.batches.create(
            input_file_id=file_obj.id,
            endpoint=endpoint,
            completion_window="24h"
        )

        return batch.id
    
    def wait_for_batch(self, batch_id, poll_interval=30):
        print(f"\nWaiting for batch {batch_id} to complete...")
        print("This may take several hours. You can safely exit and retrieve results later using:")
        print(f"python retrieve_batch_results.py --batch_id {batch_id}")
        
        while True:
            batch = self.client.batches.retrieve(batch_id)
            status = batch.status
            request_counts = batch.request_counts
            
            print(f"\rStatus: {status} | Completed: {request_counts.completed}/{request_counts.total}", end="", flush=True)
            
            if status in ["completed", "failed", "expired"]:
                print()  # New line after status updates
                return batch
            time.sleep(poll_interval)

    def load_batch_results(self, output_file_id, num_requests):
        file_response = self.client.files.content(output_file_id)
        lines = file_response.text.splitlines()

        outputs = ["ERROR"] * num_requests

        for line in lines:
            obj = json.loads(line)
            idx = int(obj["custom_id"].split("-")[1])

            if obj.get("error") is None:
                try:
                    # For GPT-5 responses endpoint
                    response = obj.get("response", {})
                    body = response.get("body", {})
                    output = body.get("output", [])
                    
                    if len(output) > 1 and "content" in output[1]:
                        content = output[1].get("content", [])
                        if len(content) > 0 and "text" in content[0]:
                            outputs[idx] = content[0].get("text")
                        else:
                            print(f"Warning: Unexpected content structure for request {idx}")
                            self.error_times += 1
                    else:
                        print(f"Warning: Unexpected output structure for request {idx}")
                        self.error_times += 1
                except (IndexError, KeyError, AttributeError) as e:
                    print(f"Error parsing response for request {idx}: {str(e)}")
                    self.error_times += 1
                    self.error_log.append({
                        "error": f"Parse error: {str(e)}",
                        "response": obj.get("response", {})
                    })
            else:
                self.error_times += 1
                self.error_log.append(obj["error"])

        return outputs

    def generate_set_batch(self, set_inputs):
        batch_file = "tmp_batch.jsonl"

        self.build_batch_file(set_inputs, batch_file)
        batch_id = self.submit_batch(batch_file)

        print(f"\n{'='*80}")
        print(f"✓ Submitted batch: {batch_id}")
        print(f"✓ Total requests: {len(set_inputs)}")
        print(f"{'='*80}")

        batch = self.wait_for_batch(batch_id)

        if batch.status != "completed":
            raise RuntimeError(f"Batch failed: {batch.status}")

        outputs = self.load_batch_results(
            batch.output_file_id,
            len(set_inputs)
        )

        return outputs
    
    def get_token_usage(self):
        if self.inference_mode == "batch":
            return {
                "prompt_tokens": None,
                "completion_tokens": None,
                "cached_tokens": None,
                "reasoning_tokens": None,
                "total_tokens": None
            }

        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cached_tokens": self.cached_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "total_tokens": self.total_tokens,
        }
    
    def generate_single(self, single_input, llm_name):
        """Generate response with retry logic"""
        max_retries = 3
        retry_delay = 2
        
        if not self.client:
            return ""
        
        for attempt in range(max_retries):
            try:
                if "gpt-5" in llm_name:
                    response = self.client.responses.create(
                        model="gpt-5.1-2025-11-13",
                        input=single_input,
                        max_output_tokens=2000,
                        reasoning={"effort": "low"}
                    )                    

                elif "gpt" in llm_name:
                    response = self.client.chat.completions.create(
                        model=llm_name,
                        messages=single_input,
                        temperature=0.2,
                        max_tokens=8192,
                    )

                elif "gemini" in llm_name:
                    response = self.client.chat.completions.create(
                        model=llm_name,
                        messages=single_input,
                        temperature=0.2,
                        max_tokens=8192,
                        extra_body={"reasoning": {"enabled": False}}
                    )
                elif "Gemma-SEA" in llm_name:
                    response = self.client.chat.completions.create(
                        model="aisingapore/Gemma-SEA-LION-v4-27B-IT",
                        messages=single_input,
                        max_tokens=8192,
                        temperature=0.2,
                    )
                elif "typhoon" in llm_name:
                    response = self.client.chat.completions.create(
                        model="typhoon-v2.1-12b-instruct",
                        messages=single_input,
                        temperature=0.2,
                        max_tokens=8192,
                    )
                else:
                    response = self.client.chat.completions.create(
                        model=llm_name,
                        messages=single_input,
                        temperature=0.2,
                        max_tokens=8192
                    )

                # Track token usage
                self.prompt_tokens += response.usage.input_tokens if "gpt-5" in llm_name else response.usage.prompt_tokens
                self.completion_tokens += response.usage.output_tokens if "gpt-5" in llm_name else response.usage.completion_tokens
                self.total_tokens += (response.usage.input_tokens + response.usage.output_tokens) if "gpt-5" in llm_name else response.usage.total_tokens

                if "gpt-5" in self.llm_name and self.inference_mode != "batch":
                    if hasattr(response.usage, 'input_tokens_details') and response.usage.input_tokens_details:
                        if hasattr(response.usage.input_tokens_details, 'cached_tokens'):
                            self.cached_tokens += response.usage.input_tokens_details.cached_tokens
                    if hasattr(response.usage, 'output_tokens_details') and response.usage.output_tokens_details:
                        if hasattr(response.usage.output_tokens_details, 'reasoning_tokens'):
                            self.reasoning_tokens += response.usage.output_tokens_details.reasoning_tokens

                    return response.output_text
                
                elif "gpt" in self.llm_name and self.inference_mode != "batch":
                    if hasattr(response.usage, 'prompt_tokens_details') and response.usage.prompt_tokens_details:
                        if hasattr(response.usage.prompt_tokens_details, 'cached_tokens'):
                            self.cached_tokens += response.usage.prompt_tokens_details.cached_tokens
                    return response.choices[0].message.content
                
                return response.choices[0].message.content
                
            except Exception as e:
                error_msg = f"Attempt {attempt + 1}/{max_retries} failed: {str(e)}"
                print(f"\n{error_msg}")
                
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    self.error_log.append({
                        "error": str(e),
                        "traceback": traceback.format_exc(),
                        "input": single_input
                    })
                    return "ERROR"
        
        return "ERROR"
    
    def generate_helper(self, args):
        """Wrapper for thread pool execution"""
        try:
            result = self.generate_single(*args)
            if result == "ERROR":
                self.error_times += 1
            return result
        except Exception as e:
            self.error_times += 1
            self.error_log.append({
                "error": str(e),
                "traceback": traceback.format_exc()
            })
            print(f"\nUnexpected error in generate_helper: {str(e)}")
            return "ERROR"
    
    def generate_set(self, set_inputs, cot=False):
        """Generate responses for a set of inputs with parallel processing"""
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            tasks = zip(set_inputs, [self.llm_name] * len(set_inputs))
            responses = list(tqdm(executor.map(self.generate_helper, tasks), total=len(set_inputs)))
        
        if not cot:
            return responses
        else:
            # Chain-of-thought continuation
            new_inputs = []
            for i, single_input in enumerate(set_inputs):
                if responses[i] != "ERROR":
                    new_input = single_input.copy()
                    new_input.append({
                        "role": "assistant",
                        "content": responses[i]
                    })
                    new_input.append({
                        "role": "user",
                        "content": "\n\nTherefore, the final answer is: "
                    })
                    new_inputs.append(new_input)
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
            system_prompt = "You are an expert in social reasoning. List the required items and split them with commas. Remember: Your answer should ONLY include the required items spliited by commas with nothing else."
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

def get_llm_input(question, question_type, context, cot): #inference_mode="sync/batch"
    question_text = question["question"]
    system_prompt = get_system_prompt(question_type, cot)
    
    # Information prompt
    if "information" in question.keys():
        information_prompt = f"# Information:\n{question['information']}\n\n"
    else:
        target_question_keys = ["fact_question_real_reason", "comprehension_q", "fact_question_truth", "truth_question"]
        information_prompt = ""
        for q_key in target_question_keys:
            if q_key in question.keys():
                information_prompt = f"# Target Question:\n{question[q_key]}\n\n"
                break
    
    # Options for MCQ
    if question_type == "mcq":
        if "wrong_answer" in question.keys():
            wrong_answer_list = question["wrong_answer"]
        elif "wrong_answers" in question.keys():
            wrong_answer_list = question["wrong_answers"]
        else:
            print(f"Warning: No wrong answers in question: {question.get('q_id', 'unknown')}")
            wrong_answer_list = []
        
        if not type(wrong_answer_list) == list:
            wrong_answer_list = [wrong_answer_list]
        
        while type(wrong_answer_list[0]) == list:
            wrong_answer_list = wrong_answer_list[0]
        
        options = [question["correct_answer"]] + wrong_answer_list
        mapping = [i for i in range(len(options))]
        # if inference_mode=="sync":
        #     random.shuffle(mapping)
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

def get_results(data_path, llm_name, cot, max_workers, inference_mode="sync"):
    # Load data
    print(f"\nLoading dataset from: {data_path}")
    df = load_TactfulToM_dataset(data_path)
    print(f"Loaded {len(df)} samples")

    llm = LLM(llm_name, max_workers, inference_mode)

    question_categories = [
        "comprehensionQA", "justificationQA", "fact_reasonQA", "fact_truthQA", 
        "beliefQAs", "infoAccessibilityQA_list", "infoAccessibilityQAs_binary", 
        "answerabilityQA_list", "answerabilityQAs_binary", "lieabilityQAs",
        "liedetectabilityQAs_list", "liedetectabilityQAs_binary", "KJability"
    ]

    # BATCH MODE: Collect ALL inputs from ALL rows FIRST
    if inference_mode == "batch":
        print("\n" + "="*80)
        print("BATCH MODE: Collecting all inputs from entire dataset")
        print("="*80)
        
        all_inputs = []
        all_results_structure = []
        
        # STEP 1: Loop through ALL rows and collect ALL inputs
        for idx, questions_set in df.iterrows():
            if idx % 10 == 0:
                print(f"Collecting inputs from sample {idx} / {len(df)}")
            
            set_results = {cat: [] for cat in question_categories}
            
            for cat in question_categories:
                if cat not in questions_set.keys():
                    continue
                
                cat_questions = questions_set[cat]
                if not isinstance(cat_questions, list):
                    print(f"Warning: {cat} is not a list for sample {idx}")
                    continue
                
                for question in cat_questions:
                    # Determine question types
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
                            
                            # Add to all_inputs (will be submitted as ONE batch)
                            all_inputs.append(llm_input)
                            
                            results_question.append({
                                "question": question.get("question", ""),
                                "correct_answer": question.get("correct_answer", ""),
                                "original_result": "",
                                "clean_result": "",
                                "question_type": question_type,
                                "context_type": context_type,
                                "mcq_mapping": mcq_mapping,
                                "question_id": question.get("q_id", "")
                            })
                    
                    set_results[cat].append(results_question)
            
            all_results_structure.append(set_results)

        # write all_results_structure for mcq_mapping in retrieve_batch_results.py after close get_original_results.py execution 
        file_name = llm_name.split("/")[-1]
        if cot:
            file_name += "-cot"
        question_type_suffix = data_path.split(".")[0].split("question_")[-1]
        file_name += f"-{question_type_suffix}"

        output_path = f"results/original/structure_map/{file_name}-all_results_structure.json"
        os.makedirs("results/original/structure_map", exist_ok=True)
        
        with open(output_path, "w", encoding='utf-8') as f:
            json.dump(all_results_structure, f, indent=3, ensure_ascii=False)

        # STEP 2: Submit ONE SINGLE batch with ALL inputs
        print(f"\n{'='*80}")
        print(f"Submitting SINGLE batch with {len(all_inputs)} total requests")
        print(f"(Collected from all {len(df)} samples)")
        print(f"{'='*80}")
        
        all_outputs = llm.generate_set_batch(all_inputs)
        
        print(f"\n{'='*80}")
        print(f"Batch processing complete!")
        print(f"Total outputs received: {len(all_outputs)}")
        print(f"Total errors: {llm.error_times}")
        print(f"{'='*80}")
        
        # STEP 3: Distribute outputs back to results structure
        print("\nDistributing results back to structure...")
        results = []
        pointer = 0
        
        for idx, set_results in enumerate(all_results_structure):
            if idx % 10 == 0:
                print(f"Distributing results for sample {idx} / {len(all_results_structure)}")
            
            for cat in question_categories:
                for i, result_cat in enumerate(set_results[cat]):
                    for j, result in enumerate(result_cat):
                        if pointer < len(all_outputs):
                            set_results[cat][i][j]["original_result"] = all_outputs[pointer]
                            pointer += 1
            
            results.append({
                "results": set_results,
                "token_usage": {
                    "prompt_tokens": None,
                    "completion_tokens": None,
                    "cached_tokens": None,
                    "reasoning_tokens": None,
                    "total_tokens": None
                }
            })
        
        # STEP 4: Save final results
        
        output_path = f"results/original/{file_name}.json"
        os.makedirs("results/original", exist_ok=True)
        
        with open(output_path, "w", encoding='utf-8') as f:
            json.dump(results, f, indent=3, ensure_ascii=False)
        
        print(f"\n✓ Saved all results to: {output_path}")
        print(f"✓ Total samples processed: {len(results)}")
        print(f"✓ Total requests: {pointer}")
        
    # SYNC MODE: Process one row at a time
    else:
        print("\n" + "="*80)
        print("SYNC MODE: Processing samples sequentially")
        print("="*80)
        
        results = []
        
        for idx, questions_set in df.iterrows():
            if idx % 10 == 0:
                print(f"\nProgress: {idx} / {len(df)}")
                print(f"Current error count: {llm.error_times}")
            
            set_results = {cat: [] for cat in question_categories}
            set_inputs = []

            # Step 1: Collect inputs for THIS row only
            for cat in question_categories:
                if cat not in questions_set.keys():
                    continue
                
                cat_questions = questions_set[cat]
                if not isinstance(cat_questions, list):
                    print(f"Warning: {cat} is not a list for sample {idx}")
                    continue
                
                for question in cat_questions:
                    # Determine question types
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
                                "question": question.get("question", ""),
                                "correct_answer": question.get("correct_answer", ""),
                                "original_result": "",
                                "clean_result": "",
                                "question_type": question_type,
                                "context_type": context_type,
                                "mcq_mapping": mcq_mapping,
                                "question_id": question.get("q_id", "")
                            })
                    
                    set_results[cat].append(results_question)

            # Step 2: Process THIS row (parallel within row)
            print(f"Generating {len(set_inputs)} responses for sample {idx}...")
            set_outputs = llm.generate_set(set_inputs, cot)
            print(f"Total errors so far: {llm.error_times}")

            # Step 3: Update results for THIS row
            pointer = 0
            for cat in question_categories:
                for i, result_cat in enumerate(set_results[cat]):
                    for j, result in enumerate(result_cat):
                        if pointer < len(set_outputs):
                            original_output = set_outputs[pointer]
                            set_results[cat][i][j]["original_result"] = original_output
                            pointer += 1

            results.append({
                "results": set_results,
                "token_usage": llm.get_token_usage()
            })

            # Save incremental checkpoint
            file_name = llm_name.split("/")[-1]
            if cot:
                file_name += "-cot"
            question_type_suffix = data_path.split(".")[0].split("question_")[-1]
            file_name += f"-{question_type_suffix}"
            
            output_path = f"results/original/{file_name}.json"
            os.makedirs("results/original", exist_ok=True)
            
            with open(output_path, "w", encoding='utf-8') as f:
                json.dump(results, f, indent=3, ensure_ascii=False)
            
            print(f"Saved checkpoint to {output_path}")
    
    # Save error log if there were errors (both modes)
    if llm.error_log:
        error_log_path = f"results/original/{file_name}_errors.json"
        with open(error_log_path, "w", encoding='utf-8') as f:
            json.dump(llm.error_log, f, indent=3, ensure_ascii=False)
        print(f"\nError log saved to {error_log_path}")
    
    print(f"\nFinal statistics:")
    print(f"Total errors: {llm.error_times}")
    print(f"Token usage: {llm.get_token_usage()}")
    
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", type=str, default="/dataset/generated_questions/generated_question_conv3-en.json", help="Comma-separated paths to dataset files")
    parser.add_argument('--llms', type=str, default="gpt-5.1-2025-11-13", help="Comma-separated list of LLM names")
    parser.add_argument('--max_workers', type=int, default=16, help="Maximum number of parallel workers")
    parser.add_argument('--cot', type=str, default='False', help="Use chain-of-thought (True/False/None for auto)")
    parser.add_argument("--inference_mode", type=str, default="batch", choices=["sync", "batch"], help="Inference mode: sync or batch")

    args = parser.parse_args()

    llm_list = [llm.strip() for llm in args.llms.split(",")]
    path_list = [path.strip() for path in args.paths.split(",")]

    for path in path_list:
        if not os.path.exists(path):
            print(f"Warning: Path does not exist: {path}")
            continue
            
        for llm in llm_list:
            # Determine CoT settings
            if args.cot is not None:
                cot_list = [args.cot.lower() == 'true']
            elif llm in ["Qwen/QwQ-32B", "moonshotai/Kimi-K2-Thinking"]:
                cot_list = [False]
            else:
                cot_list = [True, False]
            
            for cot in cot_list:
                print(f"\n{'='*80}")
                print(f"Processing: {llm} | CoT: {cot} | Path: {path}")
                print(f"{'='*80}")
                
                try:
                    get_results(path, llm, cot, max_workers=args.max_workers, inference_mode=args.inference_mode)
                except Exception as e:
                    print(f"\nFATAL ERROR processing {llm} with CoT={cot}:")
                    print(traceback.format_exc())
                    continue

if __name__ == "__main__":
    main()