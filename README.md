# KAI-ToM: Evaluating Theory of Mind Understanding of White Lies in Large Language Models

## TL;DR

We introduce **KAI-ToM**, a novel benchmark evaluating LLMs' Theory of Mind ability to understand and reason about white lies in real-life conversations, uncovering their limited understanding of white lies and the motivations behind them. **KJ2 in paper are KJ3 in our code**

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

- **Real-life Conversations**: Natural, multi-turn dialogues with authentic white lies
- **Information Asymmetry**: Carefully designed scenarios where different participants have access to different information
- **Multi-level ToM Reasoning**: Questions spanning white lie understanding, white lie reasoning, and belief tracking
- **Multi-language Support**: English and Thai language versions
- **Diverse Categories**: Two main types of white lies:
  - **Altruistic White Lies**: Lies told purely for the benefit of others, where the liar may incur some personal cost or disadvantage
  - **Pareto White Lies**: Lies that create a mutually beneficial outcome, serving both the interests of the person being lied to and the liar themselves
- **Human-in-the-Loop**: Generated through a rigorous multi-stage pipeline with human validation

### Statistics

- **Total Conversations**: 100 unique scenarios
- **Total Questions**: 6,000+ multi-choice and short-answer questions
- **Question Types**:
  - Comprehension (detecting the lie)
  - Justification (understanding motivations)
  - Fact Tracking, Information Accessibility, and Answerability (who knows what)
  - Belief States (first-order and second-order ToM)
  - Lie Detection Ability (who can detect the lie based on each character's belief)
  - Lie Ability (who can lie based on each character's belief)

### Dataset Structure

```
dataset/
├── elements/               # Raw conversation elements by category
│   ├── Tactful_conv_element_0.json  # Pareto white lies
│   ├── Tactful_conv_element_1.json  # Altruistic - childhood imagination
│   ├── Tactful_conv_element_2.json  # Altruistic - emotional soothing
│   ├── Tactful_conv_element_3.json  # Altruistic - avoiding distress
│   └── Tactful_conv_element_4.json  # Altruistic - social harmony
├── justification_options/  # Generated justification options
│   └── justification_option_*.json
├── final_set/              # Completed dataset with all questions
│   └── Tactful_conv_set_*.json
└── generated_questions/    # Question datasets (EN/TH)
    └── generated_question_conv*.json
```

### Data Format

Each conversation includes:

```json
{
  "set_id": "unique_identifier",
  "characters": {
    "liar": "Character who tells the white lie",
    "target": "Character being protected by the lie",
    "accomplice": "Character who helps maintain the lie (if any)",
    "observer": "Neutral observer character"
  },
  "lie": {
    "real_reason_q": "The true prosocial motivation",
    "lie_q": "What was said (the white lie)",
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

# Evaluate on freeform (short answer) questions
python code/evaluate_freeform.py \
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

### Generating Justification Options

For creating justification options using GPT-4:

```python
from code.justification_option_generator import (
    init_openai_client,
    process_single_conversation
)

# Initialize OpenAI client (reads from .env)
init_openai_client()

# Process a conversation
success = process_single_conversation(
    json_path="dataset/elements/Tactful_conv_element_0.json",
    set_id="0-1-0-0",
    output_path="output.json"
)
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

## Important Notes

**Intended Use**: This dataset is for research and evaluation purposes only.

**Disclaimer**: Conversations were generated by GPT-4 and validated by humans. We are not claiming machines have minds or emotions—they need social reasoning capabilities to better understand human communication. While we've ensured content quality, generative models may produce unexpected outputs in freeform contexts.

**Ethical Considerations**: Our evaluation reveals that LLMs underperform compared to humans in white lie understanding. This raises important questions: should LLMs understand white lies to interpret behavior, or also generate them? We encourage responsible use and careful consideration of whether aligning LLMs with human social behaviors, including prosocial deception, is desirable for human-AI interaction.

## License

This project is released for research purposes. Please contact the authors for commercial use inquiries.

---
# KAI-ToM
# KAI-ToM
