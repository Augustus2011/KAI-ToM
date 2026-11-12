# KAI-ToM: Evaluating Theory of Mind in Large Language Models

## TL;DR

We introduce **KAI-ToM**, a novel benchmark evaluating LLMs' Theory of Mind ability to understand and reason about lies in conversations, uncovering their limited understanding of lies and the motivations behind them under cultural . **KJ2 in paper are KJ3 in our code because KJ2 is the future work where study on self-humble behavior to decline things** 

## Abstract



## Repository Structure

```
tactful-tom/
├── code/
│   ├── evaluate_freeform.py              # Evaluation for short-answer questions
│   ├── evaluate_non_freeform.py          # Evaluation for multiple-choice questions
│   ├── question_generation.ipynb         # Question generation pipeline
│   ├── question_generation_utils.py      # Question generation utilities
│   ├── justification_option_generator.py # Generate justification options
│   ├── conv_gen.py                       # Conversation generation module
│   ├── conv_gen_platform.py              # Streamlit UI for conversation generation
│   ├── conv_generation_utils.py          # Conversation generation utilities
│   └── get_original_results.py           # Results aggregation
|   └── retrieve_batch_results.py         # retrieve batch inferece result from openai apis
├── dataset/
│   ├── elements/                         # Raw conversation elements
│   ├── justification_options/            # Generated options
│   ├── final_set/                        # Complete dataset
│   └── generated_questions/              # Generated question sets
├── environment.yml                       # Conda environment
└── README.md                             # This file
```

## Dataset

### Key Features

- **Real-life Conversations**: Natural, multi-turn dialogues with authentic lies
- **Information Asymmetry**: Carefully designed scenarios where different participants have access to different information
- **Multi-level ToM Reasoning**: Questions spanning  lie understanding, lie reasoning, and belief tracking
- **Multi-language Support**: English and Thai language versions
- **Diverse Categories**: three main types of lies
- **Human-annotated**: Generated through LLM with human validation

### Statistics

- **Total Conversations**: 1072 conversations
- **Total Questions**: 39.2k questions
- **Question Category**:
  - Comprehension (detecting the lie)
  - Justification (understanding motivations)
  - Fact Tracking, Information Accessibility, and Answerability (who knows what)
  - Belief States (first-order and second-order ToM)
  - Lie Detection Ability (who can detect the lie based on each character's belief)
  - Lie Ability (who can lie based on each character's belief)
  - KJability (what types of kreng-jai does liar exhibit)
  - etc ..

### Dataset Structure

```
dataset/
├── final_set/              
│   └── Tactful_conv_set_merged(original-kj0).json
└── generated_questions/    # Question datasets (EN/TH)
|    └── generated_question_conv*.json
└── KaiTom_gendata-kj*+thai+cultural.csv #generative seed
└── KaiTom_gendata-kj*+en.csv #generative seed
```

### Data Format

Each conversation includes:

```json
{
  "set_id": "unique_identifier",
  "characters": {
    "liar": "Character who tells the lie",
    "target": "Character being protected by the lie",
    "accomplice": "Character who helps maintain the lie (if any)",
    "observer": "Neutral observer character"
  },
  "lie": {
    "real_reason_q": "The true prosocial motivation",
    "lie_q": "What was said (the lie)",
    "truth_q": "The actual truth being concealed"
  },
  "full_context": "Complete conversation transcript",
  "comprehensionQA": [...],
  "justificationQA": [...],
  "beliefQAs": [...],
  // ... more question types
}
```

## Installation

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd tactful-tom

# Create a virtual environment
conda env create -f environment.yml
conda activate tactful-tom

# Or use pip
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your API keys (OPENAI_API_KEY, etc.)
```

## Usage

### Running Evaluation

```python
# Evaluate on non-freeform (multiple choice) questions
python code/evaluate_non_freeform.py \
    --model gpt-4 \
    --dataset_path dataset/final_set/Tactful_conv_set_0.json \
    --output_dir results/

```

### Question Generation

If you want to generate questions for new conversations:

```python
from code.question_generation_utils import (
    generate_comprehensionQA,
    generate_justificationQA,
    generate_fact_QA,
    generate_1stbeliefQAs,
    generate_2ndbeliefQAs
)

# Load your conversation data
selected_set = {...}  # Your conversation data

# Generate different question types
comprehension_qas = generate_comprehensionQA(selected_set)
justification_qas = generate_justificationQA(selected_set)
fact_qas = generate_fact_QA(selected_set)
belief_qas_1st = generate_1stbeliefQAs(selected_set)
belief_qas_2nd = generate_2ndbeliefQAs(selected_set)
```

### Conversation Generation Platform

Run the Streamlit-based conversation generation UI:

```bash
streamlit run code/conv_gen_platform.py
```

## Environment Variables

Create a `.env` file in the project root with:

```
OPENAI_API_KEY=your_openai_api_key
OPENROUTER_API_KEY=your_openrouter_api_key  # Optional
HF_TOKEN=your_huggingface_token  # Optional
TYPHOON_API_KEY1=your_typhoon_api_key  # Optional for Thai models
```

## Citation
```bibtex
@inproceedings{kerdthaisong-etal-2026-kaitom,
    title = "{K}ai{T}o{M}: A Benchmark for Culturally Informed Theory-of-Mind in {T}hai Context",
    author = "Kerdthaisong, Kun  and
      Buakhaw, Pasin  and
      Sommuang, Wayupuk  and
      Yongsatianchot, Nutchanon",
    editor = "Zhang, Yue  and
      Xu, Ruifeng  and
      Purwarianti, Ayu  and
      Rong, Tong  and
      Suzuki, Jun",
    booktitle = "Proceedings of the 15th International Joint Conference on Natural Language Processing and the 5th Conference of the {A}sia-Pacific Chapter of the {A}ssociation for {C}omputational {L}inguistics (Volume 1: Long Papers)",
    month = nov,
    year = "2026",
    address = "Hengqin, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.aacl-long.125/",
    doi = "10.18653/2026.aacl-long.125",
    pages = "2732--2755",
    ISBN = "979-8-89176-439-2",
    abstract = "Large Language Models (LLMs) show impressive reasoning capabilities, yet their understanding of culturally specific social dynamics remains limited. We present KaiToM, a Theory of Mind (ToM) benchmark for Thai cultural contexts, focusing on Kreng-Jai, a uniquely Thai concept of social reluctance and consideration. Extending TactfulToM, we construct 882 conversations (39.2K questions) spanning three Kreng-Jai dimensions: White Lies, Avoidance of Confrontation, and Fear-Based Respect. To disentangle language proficiency from cultural knowledge, the dataset includes Thai scenarios at varying cultural depths, general English scenarios, and English translations of Thai content. We evaluate Thai-specialized (Typhoonv2.1, SEA-LION-v3), open-weight (Gemma-3, DeepSeek-V3.2, Kimi-K2, Qwen), and proprietary models (GPT-4o-mini, GPT-5.1, Gemini-3-Flash). Models achieve only moderate performance, struggle with Fear-Based Respect, and cannot reliably identify specific Kreng-Jai types, a failure shared by Thai-specialized models. Human evaluation shows that native speakers significantly outperform models on culturally loaded tasks. Performance also degrades in deep relative to surface-level cultural contexts, and generalist models often perform better on English translations than on the original Thai, suggesting that models capture surface cues but lack deeper cultural alignment. KaiToM supports the development of culturally aware systems for low-resource languages. Code is available at https://github.com/Augustus2011/KAI-ToM."
}
```
---
