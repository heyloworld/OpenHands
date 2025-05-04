import os
import re
import time
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from spellchecker import SpellChecker
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TextGenerator:
    """
    A text generation system using a pre-trained GPT-2 model.
    """
    
    def __init__(self, model_name="gpt2", cache_dir="models/saved_models"):
        """
        Initialize the text generator.
        
        Args:
            model_name (str): The name of the pre-trained model to use.
            cache_dir (str): The directory to cache the model.
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.spell = SpellChecker()
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Load the model and tokenizer
        logger.info(f"Loading {model_name} model and tokenizer...")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = GPT2LMHeadModel.from_pretrained(model_name, cache_dir=cache_dir)
        
        # Set device (use GPU if available)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        self.model.to(self.device)
        
        logger.info("Model and tokenizer loaded successfully.")
    
    def correct_typos(self, text):
        """
        Correct typos in the input text.
        
        Args:
            text (str): The input text.
            
        Returns:
            str: The corrected text.
        """
        # Split text into words
        words = text.split()
        
        # Correct misspelled words
        corrected_words = []
        for word in words:
            # Skip punctuation and special characters
            if not re.match(r'^[a-zA-Z]+$', word):
                corrected_words.append(word)
                continue
            
            # Check if the word is misspelled
            if word.lower() in self.spell:
                corrected_words.append(word)
            else:
                # Get the correction
                correction = self.spell.correction(word)
                if correction is not None:
                    logger.info(f"Corrected '{word}' to '{correction}'")
                    corrected_words.append(correction)
                else:
                    # If no correction is found, keep the original word
                    corrected_words.append(word)
        
        return ' '.join(corrected_words)
    
    def clean_generated_text(self, text):
        """
        Clean the generated text.
        
        Args:
            text (str): The generated text.
            
        Returns:
            str: The cleaned text.
        """
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Correct common spelling errors
        text = self.correct_typos(text)
        
        # Ensure proper capitalization
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.capitalize() for s in sentences]
        text = ' '.join(sentences)
        
        return text
    
    def generate_text(self, prompt, max_length=200, temperature=0.7):
        """
        Generate text based on the input prompt.
        
        Args:
            prompt (str): The input prompt.
            max_length (int): The maximum length of the generated text.
            temperature (float): The temperature for sampling.
            
        Returns:
            str: The generated text.
        """
        # Correct typos in the prompt
        corrected_prompt = self.correct_typos(prompt)
        if corrected_prompt != prompt:
            logger.info(f"Corrected prompt: '{corrected_prompt}'")
        
        # Encode the prompt
        input_ids = self.tokenizer.encode(corrected_prompt, return_tensors="pt").to(self.device)
        
        # Generate text
        output = self.model.generate(
            input_ids,
            max_length=min(max_length, 1024),  # GPT-2 has a context window of 1024 tokens
            temperature=temperature,
            top_k=50,
            top_p=0.95,
            repetition_penalty=1.2,
            do_sample=True,
            num_return_sequences=1,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        # Decode the generated text
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Clean the generated text
        cleaned_text = self.clean_generated_text(generated_text)
        
        # Truncate to max_length characters
        if len(cleaned_text) > max_length:
            # Try to truncate at a sentence boundary
            last_period = cleaned_text[:max_length].rfind('.')
            if last_period > 0:
                cleaned_text = cleaned_text[:last_period + 1]
            else:
                # If no sentence boundary found, truncate at max_length
                cleaned_text = cleaned_text[:max_length]
        
        return {
            "original_prompt": prompt,
            "corrected_prompt": corrected_prompt,
            "generated_text": cleaned_text
        }
    
    def generate_from_file(self, input_file, output_file, max_length=200):
        """
        Generate text based on the prompt in the input file and save it to the output file.
        
        Args:
            input_file (str): The path to the input file.
            output_file (str): The path to the output file.
            max_length (int): The maximum length of the generated text.
            
        Returns:
            dict: The generation results.
        """
        # Read the prompt from the input file
        try:
            with open(input_file, 'r') as f:
                prompt = f.read().strip()
        except FileNotFoundError:
            logger.error(f"Input file not found: {input_file}")
            return None
        except Exception as e:
            logger.error(f"Error reading input file: {e}")
            return None
        
        # Generate text
        result = self.generate_text(prompt, max_length=max_length)
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Save the generated text to the output file
        try:
            with open(output_file, 'w') as f:
                f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Original Prompt: {result['original_prompt']}\n")
                if result['original_prompt'] != result['corrected_prompt']:
                    f.write(f"Corrected Prompt: {result['corrected_prompt']}\n")
                f.write(f"\nGenerated Text:\n{result['generated_text']}\n")
        except Exception as e:
            logger.error(f"Error writing to output file: {e}")
            return None
        
        logger.info(f"Generated text saved to {output_file}")
        return result

def main():
    """
    Main function to run the text generator.
    """
    # Initialize the text generator
    generator = TextGenerator()
    
    # Generate text from the prompt file
    input_file = "data/prompt.txt"
    output_file = "results/generated_text.txt"
    
    result = generator.generate_from_file(input_file, output_file)
    
    if result:
        print("\nGeneration completed successfully!")
        print(f"Original Prompt: {result['original_prompt']}")
        if result['original_prompt'] != result['corrected_prompt']:
            print(f"Corrected Prompt: {result['corrected_prompt']}")
        print(f"\nGenerated Text:\n{result['generated_text']}")
        print(f"\nOutput saved to: {output_file}")

if __name__ == "__main__":
    main()