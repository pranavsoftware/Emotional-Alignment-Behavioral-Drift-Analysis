# Emotional Alignment & Behavioral Drift Analysis

A comprehensive research framework for analyzing emotion detection consistency and behavioral drift in NLP models using the GoEmotions emotion classification system.

## Overview

This research project investigates how emotion detection models maintain consistency across repeated inputs and detect behavioral drift in emotional categorization. By leveraging the GoEmotions model from Hugging Face, the study processes 28 distinct emotions and evaluates model performance across multiple emotional contexts.

## Key Features

- **Emotion Detection Pipeline**: Utilizes the SamLowe/roberta-base-go_emotions model for multi-label emotion classification
- **Consistency Analysis**: Measures behavioral consistency through cosine similarity analysis of emotion vectors
- **Drift Metrics**: Calculates and tracks behavioral drift across repeated prompts
- **Comprehensive Visualization**: Generates four-panel analysis plots showing emotion distributions and drift patterns
- **Detailed Reporting**: Exports results to CSV for further statistical analysis

## Research Methodology

### Experiment Design

The study tests five emotional categories with multiple prompts per category:

- **Happy**: Positive emotion prompts expressing joy and excitement
- **Sad**: Negative emotion prompts expressing sadness and grief
- **Angry**: Aggressive emotion prompts expressing rage and frustration
- **Neutral**: Factual, emotionally neutral statements
- **Mixed**: Prompts containing multiple conflicting emotions

Each prompt is processed 3 times to detect consistency and drift patterns across identical inputs.

### Metrics Calculated

**Consistency Score**: Average cosine similarity between successive emotion vector representations (0 to 1 scale, where 1 is perfect consistency)

**Behavioral Drift**: Inverse of consistency, representing deviation from expected behavior (calculated as 1 - consistency)

**Standard Deviation**: Measures variability in similarity scores across repeats

**Average Top Emotion Score**: Mean confidence score of the primary detected emotion

## Results Summary

Based on 45 processed samples across 5 emotional categories:

| Category | Samples | Consistency | Drift   | Std Dev | Avg Top Score |
|----------|---------|-------------|---------|---------|---------------|
| Angry    | 9       | 0.9995      | 0.0005  | 0.0012  | 0.8391        |
| Sad      | 9       | 0.9539      | 0.0461  | 0.1210  | 0.7830        |
| Neutral  | 9       | 0.9484      | 0.0516  | 0.0895  | 0.6982        |
| Happy    | 9       | 0.8915      | 0.1085  | 0.2058  | 0.6280        |
| Mixed    | 9       | 0.7620      | 0.2380  | 0.4125  | 0.7625        |

### Key Findings

The model demonstrates exceptional consistency across all emotional categories, with particularly strong performance on anger classification (99.95% consistency). Sadness and neutral categories show robust consistency scores above 94%. Mixed emotions exhibit the highest drift (23.8%), indicating that complex emotional contexts introduce greater variability in detection. The relatively lower consistency in happy emotions (89.15%) suggests that positive emotions have more nuanced expressions with higher variation.

## Sample Results Visualization

![Emotional Alignment & Behavioral Drift Analysis](sample-results.png)

The analysis produces a four-panel visualization:

- **Top-Left Panel**: Dominant emotion detection frequency by category, showing the primary emotional classification for each prompt category
- **Top-Right Panel**: Emotion detection heatmap displaying the distribution of all 28 detected emotions across categories
- **Bottom-Left Panel**: Behavioral drift versus consistency comparison, highlighting the inverse relationship and relative stability of each category
- **Bottom-Right Panel**: Distribution of top emotion scores, showing confidence levels across different emotional contexts

## Installation & Setup

### Requirements

- Python 3.7 or higher
- PyTorch (with CUDA support optional)
- Transformers (Hugging Face)
- Pandas, NumPy, Matplotlib, Seaborn
- scikit-learn, SciPy

### Quick Start

```bash
pip install transformers torch pandas numpy matplotlib seaborn scikit-learn scipy openpyxl
```

Then run the main script:

```bash
python emotion_analysis.py
```

## Usage

### Basic Emotion Detection

```python
from transformers import pipeline

classifier = pipeline(
    task="text-classification",
    model="SamLowe/roberta-base-go_emotions",
    top_k=None,
    device=0  # Use -1 for CPU
)

result = classifier("I'm feeling absolutely wonderful today!")
```

### Running the Complete Study

```python
study = EmotionalAlignmentStudy(
    model_name="SamLowe/roberta-base-go_emotions",
    sample_prompts=3,
    repeats=3
)

results = study.run_experiment()
consistency_metrics = study.calculate_consistency_metrics()
study.create_visualizations()
df_results = study.save_results_to_csv()
```

### Customization

Modify the `emotional_prompts` dictionary to test different text inputs:

```python
emotional_prompts = {
    "your_category": [
        "Your test prompt 1",
        "Your test prompt 2",
        "Your test prompt 3",
    ]
}
```

Adjust experimental parameters:

```python
study = EmotionalAlignmentStudy(
    model_name="SamLowe/roberta-base-go_emotions",
    sample_prompts=5,  # Test 5 prompts per category
    repeats=5          # Repeat each prompt 5 times
)
```

## Detected Emotions

The GoEmotions model detects 28 distinct emotions:

admiration, amusement, anger, annoyance, approval, caring, confusion, curiosity, desire, disappointment, disapproval, disgust, embarrassment, excitement, fear, gratitude, grief, joy, love, nervousness, neutral, optimism, pride, realization, relief, remorse, sadness, surprise

## Output Files

The research generates two output files:

**emotional_alignment_analysis.png**: Four-panel visualization containing emotion detection patterns, heatmaps, consistency metrics, and score distributions

**emotion_research_results.csv**: Detailed results for all processed samples including timestamps, category labels, emotion scores for all 28 emotions, top emotions, and confidence scores

## Model Information

**Model**: SamLowe/roberta-base-go_emotions

**Architecture**: RoBERTa-base fine-tuned on GoEmotions dataset

**Capabilities**: Multi-label emotion classification with 28 emotion categories

**Download**: https://huggingface.co/SamLowe/roberta-base-go_emotions

## Project Structure

```
.
├── emotion_analysis.py              # Main research script
├── README.md                        # This file
├── emotional_alignment_analysis.png # Sample visualization output
└── emotion_research_results.csv     # Sample results data
```

## Computational Performance

- Model Load Time: ~18 seconds (first run)
- Processing Time per Prompt: ~50-100ms
- Total Experiment Time (45 samples): ~2-3 minutes on CPU
- Memory Usage: ~2GB (with GPU acceleration significantly faster)

## Citation

If you use this research in your work, please cite:

```
Emotional Alignment & Behavioral Drift Analysis (2024)
Using GoEmotions Model - https://huggingface.co/SamLowe/roberta-base-go_emotions
```

## License

This project is provided as-is for research and educational purposes.

## Contributing

Contributions and improvements are welcome. Please submit issues or pull requests with:

- Bug reports with reproducible examples
- Suggestions for additional emotion categories
- Improvements to visualization and analysis methods
- Performance optimizations

## Contact & Support

For questions or issues regarding this research:

- Review the inline code documentation
- Check the Hugging Face model card: https://huggingface.co/SamLowe/roberta-base-go_emotions
- Consult the research output CSV for detailed analysis results

## References

- GoEmotions Dataset: https://github.com/google-research/google-research/tree/master/goemotions
- Transformers Library: https://huggingface.co/docs/transformers/
- RoBERTa Model: https://arxiv.org/abs/1907.11692

---

**Last Updated**: October 2024

**Status**: Active Research Project
