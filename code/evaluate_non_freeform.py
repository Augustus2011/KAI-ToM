import json
import re
import argparse
import os
import glob
from pathlib import Path
from prettytable import PrettyTable

question_categories = ["comprehensionQA", "fact_reasonQA", "fact_truthQA", "beliefQAs", "infoAccessibilityQA_list", "infoAccessibilityQAs_binary", "answerabilityQA_list", "answerabilityQAs_binary", "lieabilityQAs","liedetectabilityQAs_list", "liedetectabilityQAs_binary","KJability", "justificationQA"]

# Define subfolders to search for results
RESULT_SUBFOLDERS = ["", "zeroshot", "cot"]

def filter_entry(entry, condition):
    # Updated to handle empty strings as well
    if entry["clean_result"] is None or entry["clean_result"] == "ERROR" or entry["clean_result"] == "":
        return False
    if not condition:
        return True
    if "context" in condition:
        return entry["context_type"] == condition


def _clean(entry, file_name):
    question_type = entry["question_type"]
    is_kjability = True if "KJability" in entry["question_id"] else False
    original_answer = entry["original_result"]
    
    if "Llama" in file_name:
        pattern = r'\\boxed\{(.*?)\}'
        format_output = re.findall(pattern, original_answer, re.DOTALL)
        if format_output:
            original_answer = format_output[0]
    elif "DeepSeek-V3" in file_name:
        for pattern in [r'\\boxed\{(.*?)\}']:
            format_output = re.findall(pattern, original_answer, re.DOTALL)
            if format_output:
                original_answer = format_output[0]
                break

    if question_type == "freeform":
        return original_answer
    
    if question_type == "binary":
        nonpunc_answer = re.sub(r'[^\w\s]', '', original_answer)
        if "yes" in nonpunc_answer.lower().split():
            return "yes"
        elif "no" in nonpunc_answer.lower().split():
            return "no"
        else:
            return "NAN"
    
    if question_type == "mcq":
        # Extract answer from common formats
        if original_answer.lower().startswith("option"):
            original_answer = original_answer[6:]
        else:
            # Try to extract from "Answer: X" or "Answer : X" patterns
            pattern = r'[^}]*answer[^}]*:\s*([^}]+)'
            match = re.search(pattern, original_answer, re.IGNORECASE)
            if match:
                original_answer = match.group(1).strip()
        
        mcq_mapping = entry["mcq_mapping"]
        num_options = len(mcq_mapping)
        
        # Handle empty or invalid answers
        if not original_answer or len(original_answer.strip()) == 0:
            return "NAN"
        
        # Clean the answer: remove brackets, parentheses, dots, etc.
        cleaned = original_answer.strip(" #*:[]().").upper()
        
        # Try to find a number (1, 2, 3, 4)
        number_match = re.search(r'[1-4]', cleaned)
        if number_match:
            num = int(number_match.group())
            if 1 <= num <= num_options:
                return mcq_mapping[num - 1]
        
        # Try to find a letter (A, B, C, D)
        letter_match = re.search(r'[A-D]', cleaned)
        if letter_match:
            letter = letter_match.group()
            # Convert A->1, B->2, C->3, D->4
            num = ord(letter) - ord('A') + 1
            if 1 <= num <= num_options:
                return mcq_mapping[num - 1]
        
        return "NAN"
    
    if question_type == "list":
        clean_result = original_answer.split(",")
        # Remove punctuation and whitespace from each item
        clean_result = [re.sub(r'[^\w\s]', '', r).strip() for r in clean_result]
        # Filter out empty strings
        clean_result = [r for r in clean_result if r]
        return clean_result

def _clean_reasoning(entry, file_name):
    if "QwQ" in file_name or "DeepSeek-R1" in file_name or "Qwen3-30B-A3B" in file_name:
        original_result = entry["original_result"]
        entry["original_result"] = original_result.split("</think>")[-1].strip()
        clean_result = _clean(entry, file_name)
        entry["original_result"] = original_result
        return clean_result

def find_original_file(file_name):
    """Find the original result file in any subfolder"""
    for subfolder in RESULT_SUBFOLDERS:
        if subfolder:
            path = f"./results/original/{subfolder}/{file_name}.json"
        else:
            path = f"./results/original/{file_name}.json"
        if os.path.exists(path):
            return path, subfolder
    return None, None

def clean(file_name, subfolder=""):
    """Clean results from original file and save to clean folder"""
    # Determine the input path
    if subfolder:
        input_path = f"./results/original/{subfolder}/{file_name}.json"
        output_folder = f"./results/clean/{subfolder}"
    else:
        input_path = f"./results/original/{file_name}.json"
        output_folder = "./results/clean"

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    with open(input_path) as f:
        original_result = json.load(f)

    original_result = [item["results"] for item in original_result]

    for i, question_set in enumerate(original_result):
        for category in question_categories:
            if category not in question_set:
                continue
            for j, cat_result in enumerate(question_set[category]):
                for k, entry in enumerate(cat_result):
                    try:
                        if "QwQ" in file_name or "DeepSeek-R1" in file_name or "Qwen3-30B-A3B" in file_name:
                            original_result[i][category][j][k]["clean_result"] = _clean_reasoning(entry, file_name)
                        else:
                            original_result[i][category][j][k]["clean_result"] = _clean(entry, file_name)
                    except Exception as e:
                        print(f"Error cleaning entry {i}-{category}-{j}-{k}: {e}")
                        original_result[i][category][j][k]["clean_result"] = "ERROR"

    output_path = f"{output_folder}/{file_name}.json"
    with open(output_path, "w") as f:
        json.dump(original_result, f, indent=3)

    return output_path

def _main_result(file_name, condition="full_context", subfolder=""):
    """Calculate performance metrics for a cleaned result file"""
    performance_list = {cat:[] for cat in question_categories}
    detailed_results = []

    # Determine the clean file path
    if subfolder:
        clean_path = f"./results/clean/{subfolder}/{file_name}.json"
    else:
        clean_path = f"./results/clean/{file_name}.json"

    if not os.path.exists(clean_path):
        print(f"Warning: Clean file not found: {clean_path}")
        return performance_list, detailed_results

    with open(clean_path) as f:
        original_result = json.load(f)

    for i, question_set in enumerate(original_result):
        for category in question_categories:
            if category not in question_set:
                continue
            for j, cat_result in enumerate(question_set[category]):
                for k, entry in enumerate(cat_result):
                    if not filter_entry(entry, condition) or entry["question_type"]=="freeform":
                        continue
                    
                    try:
                        if entry["question_type"] == "binary":
                            performance = (entry["clean_result"].lower()==entry["correct_answer"].lower())
                        elif entry["question_type"] == "mcq":
                            try:
                                performance = 1 if (int(entry["clean_result"])==0) else 0
                            except:
                                performance = 0
                        elif entry["question_type"] == "list":
                            clean_set, correct = set(entry["clean_result"]), set(entry["correct_answer"])
                            inter = clean_set & correct
                            try:
                                prec, recall = len(inter)/len(correct), len(inter)/len(clean_set)
                                if prec==1 and recall==1:
                                    performance = 1
                                else:
                                    performance = 0
                            except:
                                performance = 0
                        
                        performance_list[category].append(performance)

                        # Collect detailed result
                        q_id = entry.get("question_id", f"{i}-{category}-{j}-{k}")
                        detailed_results.append({
                            "q_id": q_id,
                            "set_index": i,
                            "category": category,
                            "question_type": entry["question_type"],
                            "correct": performance,
                            "clean_result": entry["clean_result"],
                            "correct_answer": entry["correct_answer"]
                        })
                    except Exception as e:
                        print(f"Error processing {file_name} entry {i}-{category}-{j}-{k}: {e}")
                        continue

    for k, v in performance_list.items():
        if not len(v):
            performance_list[k] = 0
        else:
            performance_list[k] = sum(v)/len(v)

    print(f"Performance of {file_name}: {performance_list}")
    return performance_list, detailed_results

def make_prettytable(full_results):
    table = PrettyTable()
    for file_name, result in full_results.items():
        if not table.field_names:
            table.field_names = ["model_type"] + [k for k in result.keys()]
        row = [file_name] + ["{:.2f}".format(result[k]*100) for k in table.field_names[1:]]
        table.add_row(row)
    print(table)
    
    os.makedirs("./cases", exist_ok=True)
    with open("./cases/prettytable.txt", "w") as f:
        f.write(table.get_string())
    return table

def get_all_result_files():
    """Get all result files from all subfolders"""
    files_info = []

    for subfolder in RESULT_SUBFOLDERS:
        if subfolder:
            pattern = f"./results/original/{subfolder}/*.json"
        else:
            pattern = "./results/original/*.json"

        files = glob.glob(pattern)
        for f in files:
            file_name = os.path.basename(f)[:-5]
            files_info.append({
                "file_name": file_name,
                "subfolder": subfolder,
                "full_path": f
            })

    return files_info


def parse_model_info(file_name):
    """Extract model information from file name"""
    # Extract method (zeroshot or cot)
    method = "zeroshot"
    if "-cot-" in file_name.lower():
        method = "cot"

    # Extract conv info
    conv_match = re.search(r'conv(\d+)[-_]?(th|en)?', file_name.lower())
    conv_id = int(conv_match.group(1)) if conv_match else None
    language = conv_match.group(2) if conv_match and conv_match.group(2) else "unknown"

    # Model series classification
    model_name = file_name.lower()
    if "typhoon" in model_name:
        series = "Typhoon"
        thai_capable = True
    elif "sealion" in model_name or "sea-lion" in model_name:
        series = "SEA-LION"
        thai_capable = True
    elif "pathumma" in model_name:
        series = "Pathumma"
        thai_capable = True
    elif "qwen" in model_name:
        series = "Qwen"
        thai_capable = False
    elif "gemma" in model_name:
        series = "Gemma"
        thai_capable = False
    elif "deepseek" in model_name:
        series = "DeepSeek"
        thai_capable = False
    elif "gpt" in model_name:
        series = "GPT"
        thai_capable = False
    elif "gemini" in model_name:
        series = "Gemini"
        thai_capable = False
    elif "kimi" in model_name:
        series = "Kimi"
        thai_capable = False
    else:
        series = "Other"
        thai_capable = False

    # Extract model size
    size_match = re.search(r'(\d+)[bB]', file_name)
    model_size = int(size_match.group(1)) if size_match else None

    return {
        "method": method,
        "conv_id": conv_id,
        "language": language,
        "series": series,
        "thai_capable": thai_capable,
        "model_size": model_size
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--condition', type=str, default="full_context")
    parser.add_argument('--file_name', type=str, default=None)
    parser.add_argument('--subfolder', type=str, default=None, help="Subfolder: zeroshot, cot, or empty for root")
    parser.add_argument('--all', action='store_true', help="Process all files in all subfolders")
    args = parser.parse_args()

    if args.all or not args.file_name:
        files_info = get_all_result_files()
        print(f"Found {len(files_info)} result files")
    else:
        subfolder = args.subfolder or ""
        files_info = [{"file_name": args.file_name, "subfolder": subfolder}]

    full_results = {}
    all_detailed_results = {}

    for file_info in files_info:
        file_name = file_info["file_name"]
        subfolder = file_info["subfolder"]
        display_name = f"{subfolder}/{file_name}" if subfolder else file_name

        try:
            clean(file_name, subfolder)
            result, detailed = _main_result(file_name, args.condition, subfolder)
            full_results[display_name] = result
            all_detailed_results[display_name] = detailed
        except Exception as e:
            print(f"Error processing {display_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Make table
    make_prettytable(full_results)

    # Save detailed results
    os.makedirs("./results", exist_ok=True)
    with open("./results/detailed_results.json", "w") as f:
        json.dump(all_detailed_results, f, indent=2)

    return full_results, all_detailed_results


if __name__ == "__main__":
    main()
