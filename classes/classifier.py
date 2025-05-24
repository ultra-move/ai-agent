from transformers import pipeline, set_seed
import torch


class Classifier:
    def __init__(self, classes, model_name="MoritzLaurer/deberta-v3-large-zeroshot-v2.0",
                 hypothesis_template="This text is about {}", random_seed=42, device=None):
        """
        Initializes the zero-shot classifier.

        Args:
            classes (list): Classification labels.
            model_name (str): Model identifier.
            hypothesis_template (str): Template for hypothesis formulation.
            random_seed (int): Random seed for reproducibility.
            device (int or None): Torch device index. Auto-detects if None.
        """
        if not classes:
            raise ValueError("`classes` must be a non-empty list of labels.")

        self.classes = classes
        self.hypothesis_template = hypothesis_template

        # Device autodetection
        if device is None:
            device = 0 if torch.cuda.is_available() else -1

        # Set random seed for reproducibility
        set_seed(random_seed)

        # Initialize pipeline
        try:
            self.model = pipeline(
                "zero-shot-classification",
                model=model_name,
                device=device
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize zero-shot pipeline: {e}")

        print("Zero-shot classifier initialized.")

    def classify(self, text, multi_label=False):
        """
        Classifies a single string.

        Args:
            text (str): Input text.
            multi_label (bool): Whether multiple labels can apply.

        Returns:
            dict: Full classification output.
        """
        text = text.strip()
        return self.model(
            text,
            self.classes,
            hypothesis_template=self.hypothesis_template,
            multi_label=multi_label
        )

    def classify_bulk(self, chunks, multi_label=False, batch_size=None):
        """
        Classifies multiple strings.

        Args:
            chunks (list): List of strings.
            multi_label (bool): Allow multiple labels per chunk.
            batch_size (int or None): Batch size for inference.

        Returns:
            list: List of classification results.
        """
        if not isinstance(chunks, list) or not chunks:
            raise ValueError("`chunks` must be a non-empty list of strings.")

        results = self.model(
            chunks,
            self.classes,
            hypothesis_template=self.hypothesis_template,
            multi_label=multi_label,
            batch_size=batch_size
        )

        # Normalize single vs batch output
        if isinstance(results, dict):
            results = [results]

        return results
