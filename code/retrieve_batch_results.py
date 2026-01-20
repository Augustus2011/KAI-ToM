import argparse
import json
import os
from openai import OpenAI
from dotenv import load_dotenv
import pandas as pd

import os

load_dotenv()

def load_TactfulToM_dataset(data_path):
    with open(data_path) as f:
        dataset = json.load(f)
    df = pd.DataFrame(dataset)
    return df

class BatchRetriever:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.error_times = 0
        self.error_log = []
    
    def retrieve_batch_status(self, batch_id):
        """Check the status of a batch"""
        batch = self.client.batches.retrieve(batch_id)
        print(f"\nBatch ID: {batch_id}")
        print(f"Status: {batch.status}")
        print(f"Created at: {batch.created_at}")
        print(f"Request counts: {batch.request_counts}")
        
        if batch.output_file_id:
            print(f"Output file ID: {batch.output_file_id}")
        if batch.error_file_id:
            print(f"Error file ID: {batch.error_file_id}")
        
        return batch
    
    def load_batch_results(self, output_file_id, num_requests):
        """Load results from completed batch"""
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
    
    def list_batches(self, limit=10):
        """List recent batches"""
        batches = self.client.batches.list(limit=limit)
        print(f"\n{'='*80}")
        print("Recent Batches:")
        print(f"{'='*80}")
        for batch in batches.data:
            print(f"\nBatch ID: {batch.id}")
            print(f"  Status: {batch.status}")
            print(f"  Created: {batch.created_at}")
            print(f"  Requests: {batch.request_counts}")
        print(f"{'='*80}\n")
        return batches

def get_question_categories():
    return [
        "comprehensionQA", "justificationQA", "fact_reasonQA", "fact_truthQA", 
        "beliefQAs", "infoAccessibilityQA_list", "infoAccessibilityQAs_binary", 
        "answerabilityQA_list", "answerabilityQAs_binary", "lieabilityQAs",
        "liedetectabilityQAs_list", "liedetectabilityQAs_binary", "KJability"
    ]

def calculate_expected_requests(data_path):
    """Calculate how many requests should be in the batch based on the dataset"""
    df = load_TactfulToM_dataset(data_path)
    question_categories = get_question_categories()
    
    total_requests = 0
    for idx, questions_set in df.iterrows():
        for cat in question_categories:
            if cat not in questions_set.keys():
                continue
            
            cat_questions = questions_set[cat]
            if not isinstance(cat_questions, list):
                continue
            
            # Each question generates one request
            total_requests += len(cat_questions)
    
    return total_requests, len(df)

def build_results_structure(data_path="/results/original/structure_map/gpt-5.1-2025-11-13-conv1-th-all_results_structure.json"):
    """Build the same results structure as in get_results()"""
    df = load_TactfulToM_dataset(data_path)
    question_categories = get_question_categories()

    all_results_structure = []
    # if structure_mapping_path:
    #     with open(structure_mapping_path,"r") as f:
    #         all_results_structure=json.load(f)
    #     return all_results_structure,len(df) #all_results_structure,total_requests

    total_requests = 0

    for idx, questions_set in df.iterrows():
        set_results = {cat: [] for cat in question_categories}
        
        for cat in question_categories:
            if cat not in questions_set.keys():
                continue
            
            cat_questions = questions_set[cat]
            if not isinstance(cat_questions, list):
                continue
            
            for question in cat_questions:
                # Determine question type
                if "list" in cat:
                    question_type = "list"
                elif "binary" in cat:
                    question_type = "binary"
                elif cat == "comprehensionQA":
                    question_type = "binary"
                elif cat in ["lieabilityQAs", "KJability"]:
                    question_type = "mcq"
                else:
                    question_type = "mcq"
                
                # Create result entry
                result_entry = {
                    "question": question[0].get("question", ""),
                    "correct_answer": question[0].get("correct_answer", ""),
                    "original_result": "",
                    "clean_result": "",
                    "question_type": question_type,
                    "context_type": "full_context",
                    "mcq_mapping": question[0].get("mcq_mapping", []),
                    "question_id": question[0].get("question_id", "")
                }
                
                set_results[cat].append(result_entry)
                total_requests += 1
        
        all_results_structure.append(set_results)
    return all_results_structure,total_requests


def retrieve_and_save_results(batch_id, data_path, llm_name, cot):
    """Retrieve batch results and save to the same format as get_results()"""
    
    retriever = BatchRetriever()
    
    # Check batch status
    print("\n" + "="*80)
    print("RETRIEVING BATCH RESULTS")
    print("="*80)
    
    batch = retriever.retrieve_batch_status(batch_id)
    
    if batch.status != "completed":
        print(f"\n⚠️  Batch is not completed yet. Current status: {batch.status}")
        if batch.status == "in_progress":
            print("The batch is still processing. Please wait and try again later.")
        elif batch.status == "failed":
            print("The batch has failed. Check the error file for details.")
        elif batch.status == "expired":
            print("The batch has expired.")
        return None
    
    # Calculate expected requests
    expected_requests, num_samples = calculate_expected_requests(data_path)
    print(f"\nDataset info:")
    print(f"  Total samples: {num_samples}")
    print(f"  Expected requests: {expected_requests}")
    
    # Build results structure
    print("\nBuilding results structure...")
    all_results_structure, total_requests = build_results_structure(data_path)
    
    if total_requests != expected_requests:
        print(f"⚠️  Warning: Mismatch in request count!")
        print(f"  Calculated: {total_requests}")
        print(f"  Expected: {expected_requests}")
    
    # Retrieve outputs
    print(f"\nRetrieving {total_requests} outputs from batch...")
    all_outputs = retriever.load_batch_results(batch.output_file_id, total_requests)
    
    print(f"✓ Retrieved {len(all_outputs)} outputs")
    print(f"✓ Errors encountered: {retriever.error_times}")
    
    # Distribute outputs to results structure
    print("\nDistributing results to structure...")
    results = []
    question_categories = get_question_categories()
    j=0
    for idx, set_results in enumerate(all_results_structure):
        for cat in question_categories:
            for i, result_cat in enumerate(set_results[cat]):
                set_results[cat][i]["original_result"] = all_outputs[j]
                j += 1
        
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
    
    
    # Save results
    file_name = os.path.basename(data_path).split("-all_results_structure")[0]
    if cot:
        file_name += "-cot"
    
    output_path = f"results/original/retrieved/{file_name}.json"
    os.makedirs("results/original/retrieved", exist_ok=True)
    
    with open(output_path, "w", encoding='utf-8') as f:
        json.dump(results, f, indent=3, ensure_ascii=False)
    
    print(f"\n✓ Saved results to: {output_path}")
    
    # Save error log if there were errors
    if retriever.error_log:
        error_log_path = f"results/original/retrieved/{file_name}_errors.json"
        with open(error_log_path, "w", encoding='utf-8') as f:
            json.dump(retriever.error_log, f, indent=3, ensure_ascii=False)
        print(f"✓ Saved error log to: {error_log_path}")
    
    print(f"\nFinal statistics:")
    print(f"  Total requests: {total_requests}")
    print(f"  Successful: {total_requests - retriever.error_times}")
    print(f"  Errors: {retriever.error_times}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Retrieve and save batch inference results")
    parser.add_argument("--batch_id", type=str,default="batch_xxxxxxxxxxx",help="The batch ID to retrieve results from") #batch_69620d66ae6481909428a123d459fbad,batch_69621e7a4fe88190b150b1c943d8b55e,batch_69624040587c8190a38b066e2a9e3ca4,batch_69627bbc4b108190bcdf301167a5be1b,batch_69627d0710388190b33b84c4cabaa4ca,batch_69627d9a1d9081908e4439389e31e5c5
    parser.add_argument("--list", action="store_true", help="List recent batches")
    parser.add_argument("--data_path", type=str, 
                       default="/results/original/structure_map/gpt-5.1-2025-11-13-conv3-th-all_results_structure.json",
                       help="Path to the original dataset JSON file") #0th,0en,1th,1en,3th,3en
    parser.add_argument("--llm_name", type=str, default="gpt-5.1-2025-11-13",
                       help="Name of the LLM model used (for filename)")
    parser.add_argument("--cot", type=str, default="False",
                       help="Whether chain-of-thought was used (True/False)")
    
    args = parser.parse_args()
    
    retriever = BatchRetriever()
    
    if args.list:
        # List recent batches
        retriever.list_batches(limit=20)
        return
    
    if not args.batch_id:
        print("Error: Please provide --batch_id or use --list to see recent batches")
        return
    
    # Convert cot string to boolean
    cot = args.cot.lower() == 'true'
    
    # Retrieve and save results
    retrieve_and_save_results(
        batch_id=args.batch_id,
        data_path=args.data_path,
        llm_name=args.llm_name,
        cot=cot
    )

if __name__ == "__main__":
    main()