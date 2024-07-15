# Semantic Detection System

This project implements a Semantic Detection System using various natural language processing techniques and machine learning models. It allows users to input passages and queries, then generates responses using a language model and analyzes the semantic similarity between the input and output.

## Features

- Process and clean text input
- Generate responses using OpenAI's GPT-3.5 Turbo model
- Calculate semantic similarity using cosine similarity and Euclidean distance
- Visualize semantic understanding through 2D embeddings plot
- Streamlit-based user interface for easy interaction

## Requirements

- Python 3.7+
- Libraries:
  - numpy
  - pandas
  - sentence_transformers
  - scikit-learn
  - langchain
  - langchain_openai
  - streamlit
  - matplotlib

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/semantic-detection-system.git
   cd semantic-detection-system
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Set up your OpenAI API key as an environment variable:
   ```
   export OPENAI_API_KEY='your-api-key-here'
   ```

## Usage

Run the Streamlit app:

```
streamlit run app.py
```

Then, open your web browser and navigate to the provided local URL (usually `http://localhost:8501`).

1. Enter one or more passages in the text area, separating each passage with a new line.
2. Enter a query in the input field.
3. Click the "Run" button to generate a response and analyze semantic similarity.

The app will display:
- A 2D visualization of the semantic similarity between passages and the generated output
- Cosine similarity and Euclidean distance metrics
- A semantic check result (entailment, containment, or neutral)
- The generated LLM output

## How It Works

1. The input passages are processed and cleaned.
2. The query is used to generate a response using the GPT-3.5 Turbo model.
3. Sentence embeddings are created for both the input passages and the generated response.
4. Cosine similarity and Euclidean distance are calculated between the embeddings.
5. The results are visualized and displayed in the Streamlit interface.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
