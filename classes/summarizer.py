from transformers import pipeline
import torch

class Summarizer:
    """
    A class to perform text summarization using the BART model from Hugging Face's Transformers library.
    Supports both single and batch summarization with optional GPU acceleration.
    """

    def __init__(self):
        """
        Initializes the summarization pipeline using the BART model.
        Automatically selects GPU if available, otherwise defaults to CPU.
        """
        device = 0 if torch.cuda.is_available() else -1
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=device)

    def summarize(self, text, max_length=150, min_length=30):
        """
        Summarizes a single text input using the pre-trained BART model.

        Args:
            text (str): The input text to be summarized.
            max_length (int): The maximum length of the summary (default is 150 tokens).
            min_length (int): The minimum length of the summary (default is 30 tokens).

        Returns:
            str: The summarized text.

        Raises:
            ValueError: If the input text exceeds the model's approximate token limit.
            RuntimeError: If the model fails during summarization.
        """
        if len(text.split()) > 1024:
            raise ValueError("Input text too long for the model. Consider truncating or splitting.")
        try:
            return self.summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']
        except Exception as e:
            raise RuntimeError(f"Summarization failed: {e}")

    def bulk_summarize(self, texts, max_length=150, min_length=30, batch_size=8):
        """
        Summarizes multiple texts in batches to optimize performance.

        Args:
            texts (list of str): A list of texts to summarize.
            max_length (int): The maximum length of each summary (default is 150 tokens).
            min_length (int): The minimum length of each summary (default is 30 tokens).
            batch_size (int): The number of texts to process per batch (default is 8).

        Returns:
            list of str: A list of summarized texts.

        Raises:
            RuntimeError: If any batch fails during summarization.
        """
        summaries = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                batch_summaries = self.summarizer(batch, max_length=max_length, min_length=min_length, do_sample=False)
                summaries.extend([summary['summary_text'] for summary in batch_summaries])
            except Exception as e:
                raise RuntimeError(f"Batch summarization failed at batch {i // batch_size}: {e}")
        return summaries
