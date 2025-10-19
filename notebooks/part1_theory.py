"""
Part 1: Theoretical Understanding

This script contains answers to the theoretical questions about AI tools and frameworks.
"""

def print_theory_answers():
    """Print answers to the theoretical questions."""
    print("""
    =============================================
    Part 1: Theoretical Understanding - Answers
    =============================================
    
    Q1: What is the primary purpose of Scikit-learn in the Python ecosystem?
    
    Answer:
    Scikit-learn is a powerful and widely-used open-source machine learning library for Python. 
    Its primary purposes include:
    - Providing simple and efficient tools for data mining and data analysis
    - Offering a consistent interface for various machine learning algorithms
    - Supporting both supervised and unsupervised learning
    - Including tools for model fitting, data preprocessing, model selection, and evaluation
    - Being built on NumPy, SciPy, and matplotlib

    =============================================
    
    Q2: Explain the difference between TensorFlow and PyTorch.
    
    Answer:
    TensorFlow and PyTorch are both popular deep learning frameworks, but they have some key differences:
    
    TensorFlow:
    - Developed by Google Brain
    - Uses static computation graphs by default (though eager execution is now available)
    - Better for production deployment
    - Strong support for distributed computing
    - More comprehensive ecosystem (TFX, TensorBoard, etc.)
    
    PyTorch:
    - Developed by Facebook's AI Research lab
    - Uses dynamic computation graphs by default
    - More pythonic and intuitive for research
    - Better for rapid prototyping
    - Preferred by researchers for its flexibility
    - Easier debugging due to its dynamic nature

    =============================================
    
    Q3: What is spaCy primarily used for in NLP tasks?
    
    Answer:
    spaCy is an open-source software library for advanced natural language processing. 
    Its primary uses include:
    - Tokenization and text processing
    - Part-of-speech tagging
    - Named Entity Recognition (NER)
    - Dependency parsing
    - Word vectors and semantic similarity
    - Text classification
    - Rule-based matching
    - Custom pipeline components
    - Multi-language support
    - Efficient processing of large volumes of text
    
    spaCy is designed specifically for production use and provides a fast and efficient way 
    to process and understand large volumes of text.
    """)

if __name__ == "__main__":
    print_theory_answers()
