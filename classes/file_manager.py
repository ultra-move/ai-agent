import os
import pdfplumber
from langchain_core.tools import tool

class FileManager:
    """
    A utility class for file operations including listing, reading, and writing files.
    Supports plain text and PDF file reading using pdfplumber.
    """

    @staticmethod
    @tool
    def list_files(directory, extension_filter=None):
        """
        Lists files in a given directory, optionally filtered by extension.

        Args:
            directory (str): The path to the directory to list files from.
            extension_filter (str, optional): Filter files by this extension (e.g., '.txt', '.pdf').

        Returns:
            list: A list of filenames matching the criteria.
        """
        if not os.path.isdir(directory):
            raise ValueError(f"Invalid directory: {directory}")

        return [
            f for f in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, f)) and 
               (extension_filter is None or f.endswith(extension_filter))
        ]
    
    @staticmethod
    @tool
    def read_file(file_path):
        """
        Reads the contents of a file. Uses pdfplumber if the file is a PDF.

        Args:
            file_path (str): The path to the file.

        Returns:
            str: The contents of the file.

        Raises:
            IOError: If the file cannot be read.
        """
        if not os.path.isfile(file_path):
            raise IOError(f"File not found: {file_path}")

        if file_path.lower().endswith('.pdf'):
            try:
                with pdfplumber.open(file_path) as pdf:
                    return '\n'.join(page.extract_text() for page in pdf.pages if page.extract_text()).strip()
            except Exception as e:
                raise IOError(f"Failed to parse PDF: {e}")
        else:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read().strip()
            except Exception as e:
                raise IOError(f"Failed to read file: {e}")
    
    @staticmethod
    @tool
    def write_file(file_path, content):
        """
        Writes content to a file. Overwrites if the file exists.

        Args:
            file_path (str): The path where the file will be written.
            content (str): The content to write to the file.

        Raises:
            IOError: If the file cannot be written.
        """
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
        except Exception as e:
            raise IOError(f"Failed to write file: {e}")
