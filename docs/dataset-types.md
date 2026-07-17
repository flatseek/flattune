# Dataset Types

Flattune supports multiple dataset types for fine-tuning. Each type produces training examples in a specific format.

## Supported Types

### qa

Question-answer pairs generated from Flatseek records.

**Format:**
```json
{"instruction": "What is the magnitude?", "input": "Record about earthquake", "output": "7.2"}
```

**Best for:** Knowledge retrieval, factual question answering

### instruction

Instruction-input-output triplets for task completion.

**Format:**
```json
{"instruction": "Summarize the following", "input": "Long text...", "output": "Short summary"}
```

**Best for:** Task completion, document processing

### chat

Multi-turn conversations.

**Format:**
```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

**Best for:** Customer support, dialogue systems

### classification

Categorized examples.

**Format:**
```json
{"instruction": "Classify", "input": "Text to classify", "output": "category"}
```

**Best for:** Sentiment analysis, topic classification

### summarization

Document-summary pairs.

**Format:**
```json
{"instruction": "Summarize", "input": "Long document...", "output": "Brief summary"}
```

**Best for:** Document summarization, text condensing

### extraction

Entity extraction from text.

**Format:**
```json
{"instruction": "Extract entities", "input": "Text with entities", "output": "Entity1, Entity2, ..."}
```

**Best for:** Named entity recognition, information extraction

### rag

RAG distillation pairs.

**Format:**
```json
{"instruction": "Based on context", "input": "Context: ... Query: ...", "output": "Answer"}
```

**Best for:** Retrieval-augmented generation

## Built-in Dataset Types (NEW)

The build pipeline supports **17 dataset types** across **8 categories**:

### Knowledge Category

#### facts

Extract factual statements from content.

```json
{"instruction": "Extract factual statements from the following content.", "input": "Text...", "output": "Fact 1\nFact 2..."}
```

#### glossary

Extract and define terms.

```json
{"instruction": "Define the following terms.", "input": "Term: photosynthesis", "output": "The process by which plants convert sunlight into energy."}
```

#### concept

Explain concepts in detail.

```json
{"instruction": "Explain this concept in detail.", "input": "Concept: photosynthesis", "output": "Photosynthesis is the process..."}
```

#### definition

Provide clear definitions.

```json
{"instruction": "Provide a definition for the following.", "input": "Term: algorithm", "output": "A step-by-step procedure for solving a problem."}
```

### QA Category

#### context_qa

Answer questions based on provided context.

```json
{"instruction": "Answer based on the given context.", "input": "Context: ... Question: What caused the earthquake?", "output": "The earthquake was caused by..."}
```

#### direct_qa

Direct question answering without context.

```json
{"instruction": "Answer the question directly.", "input": "Question: What is Python?", "output": "Python is a programming language."}
```

#### yes_no

Yes/No question answering.

```json
{"instruction": "Answer yes or no.", "input": "Question: Is water wet?", "output": "Yes"}
```

### Conversation Category

#### chat

Single/multi-turn chat conversations.

```json
{"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there!"}]}
```

#### dialogue

Roleplay dialogue scenarios.

```json
{"messages": [{"role": "system", "content": "You are a doctor."}, {"role": "user", "content": "I have a headache"}, {"role": "assistant", "content": "Have you taken any medication?"}]}
```

### Instruction Category

#### procedure

Step-by-step procedural instructions.

```json
{"instruction": "Explain the steps to accomplish this task.", "input": "Task: How to make coffee", "output": "1. Boil water\n2. Add coffee grounds\n3. Pour water...\n"}
```

#### howto

How-to instructional content.

```json
{"instruction": "How do I accomplish this task?", "input": "Task: Make coffee", "output": "To make coffee, follow these steps..."}
```

### Classification Category

#### topic_classify

Classify content into topics.

```json
{"instruction": "Classify the topic of this content.", "input": "Text about machine learning...", "output": "technology"}
```

#### sentiment

Determine sentiment of text.

```json
{"instruction": "What is the sentiment of this text?", "input": "I love this product!", "output": "positive"}
```

#### entity_classify

Classify named entities.

```json
{"instruction": "Identify and classify the entities in this text.", "input": "John works at Google in California.", "output": "John: PERSON, Google: ORG, California: LOCATION"}
```

### Summary Category

#### executive_summary

Executive-level summary.

```json
{"instruction": "Provide an executive summary.", "input": "Long report...", "output": "This report summarizes key findings and recommendations."}
```

#### concise_summary

Concise brief summary.

```json
{"instruction": "Summarize this concisely.", "input": "Detailed article...", "output": "Brief overview of main points."}
```

### Extraction Category

#### key_value_extract

Extract key-value pairs.

```json
{"instruction": "Extract key-value pairs from this content.", "input": "Invoice: Total: $100, Date: 2024-01-01...", "output": "Total: $100, Date: 2024-01-01"}
```

### RAG Category

#### citation_qa

QA with citation support.

```json
{"instruction": "Answer with citations from the context.", "input": "Context: [document] Query: What did the report find?", "output": "According to the report (p.5), the findings indicate..."}
```

## Generator Plugins

### qa

Generates question-answer pairs from records.

```yaml
dataset:
  generators:
    - qa
```

### summary

Generates summarization examples.

```yaml
dataset:
  generators:
    - summary
```

### classification

Generates classification examples.

```yaml
dataset:
  generators:
    - classification
```

### comparison

Generates comparison/contrast examples.

```yaml
dataset:
  generators:
    - comparison
```

### conversation

Generates multi-turn conversation examples.

```yaml
dataset:
  generators:
    - conversation
```

### extraction

Generates entity extraction examples.

```yaml
dataset:
  generators:
    - extraction
```

### safety

Generates safety-related examples.

```yaml
dataset:
  generators:
    - safety
```

### custom

Custom generator template for user-defined formats.

```yaml
dataset:
  generators:
    - custom
```

## Combining Generators

You can combine multiple generators:

```yaml
dataset:
  type: qa
  generators:
    - qa
    - summary
    - classification
```

## Custom Generators

Create a custom generator in `src/flattune/dataset/generators/custom.py`:

```python
from flattune.dataset.generators.base import BaseGenerator

class CustomGenerator(BaseGenerator):
    name = "custom"
    supported_types = ["custom"]

    def generate(self, document: dict, config) -> list[dict[str, Any]]:
        return {
            "instruction": "...",
            "input": "...",
            "output": "...",
            "metadata": {}
        }
```

Then add to your config:

```yaml
dataset:
  generators:
    - custom
```

## Build Pipeline Type Selection

With the build pipeline, you can specify dataset types directly:

```bash
flattune build config.yml --types facts --types context_qa --types glossary
```

Available types: `facts`, `glossary`, `concept`, `definition`, `context_qa`, `direct_qa`, `yes_no`, `chat`, `dialogue`, `procedure`, `howto`, `topic_classify`, `sentiment`, `entity_classify`, `executive_summary`, `concise_summary`, `key_value_extract`, `citation_qa`
