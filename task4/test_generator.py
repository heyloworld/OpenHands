import os
import time
from text_generator import TextGenerator

def test_with_prompts(prompts, output_dir="results/test_outputs"):
    """
    Test the text generator with different prompts.
    
    Args:
        prompts (list): List of prompts to test.
        output_dir (str): Directory to save the outputs.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize the text generator
    generator = TextGenerator()
    
    # Test each prompt
    for i, prompt in enumerate(prompts):
        print(f"\nTesting prompt {i+1}/{len(prompts)}: '{prompt}'")
        
        # Generate text
        result = generator.generate_text(prompt)
        
        # Save the result
        output_file = os.path.join(output_dir, f"prompt_{i+1}.txt")
        with open(output_file, 'w') as f:
            f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Original Prompt: {prompt}\n")
            if prompt != result['corrected_prompt']:
                f.write(f"Corrected Prompt: {result['corrected_prompt']}\n")
            f.write(f"\nGenerated Text:\n{result['generated_text']}\n")
        
        # Print the result
        print(f"Original Prompt: {prompt}")
        if prompt != result['corrected_prompt']:
            print(f"Corrected Prompt: {result['corrected_prompt']}")
        print(f"Generated Text: {result['generated_text'][:50]}...")
        print(f"Output saved to: {output_file}")

def main():
    """
    Main function to test the text generator.
    """
    # Define test prompts, including some with typos
    test_prompts = [
        "who are you?",
        "whoo ar yuo?",  # Typos
        "tell me a short story",
        "tell me a shotr storry",  # Typos
        "what is artificial intelligence?",
        "waht is artifical inteligence?",  # Typos
    ]
    
    # Test the generator with the prompts
    test_with_prompts(test_prompts)

if __name__ == "__main__":
    main()