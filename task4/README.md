# OpenHands Projects

This repository contains two projects:

1. **GPT-2 Text Generation System**: A system for generating text using a pre-trained GPT-2 model.
2. **Q-Learning for Gridworld**: A reinforcement learning system for solving maze-style Gridworld tasks.

## GPT-2 Text Generation System

This project implements a text generation system using a pre-trained GPT-2 model. The system can generate text based on prompts, correct typos in the input, and clean up the generated text.

### Features

- **Pre-trained GPT-2 Model**: Uses a pre-trained GPT-2 model for text generation.
- **Typo Correction**: Automatically corrects typos in the input prompt.
- **Text Cleaning**: Post-processes the generated text to remove extra spaces and correct common spelling errors.
- **Length Control**: Limits the generated text to a specified maximum length.
- **File I/O**: Reads prompts from a file and saves the generated text to a file.
- **Caching**: Caches the model to avoid downloading it every time.

### Project Structure

```
.
├── data/
│   └── prompt.txt                # Input prompt file
├── models/
│   └── saved_models/             # Directory to cache the model
├── results/
│   ├── generated_text.txt        # Output file for generated text
│   └── test_outputs/             # Directory for test outputs
├── text_generator.py             # Main script for text generation
├── test_generator.py             # Script to test the system with different prompts
└── save_model.py                 # Script to save the model
```

### Requirements

- Python 3.6+
- PyTorch
- Transformers
- pyspellchecker

### Usage

#### Basic Usage

To generate text based on the prompt in `data/prompt.txt`:

```bash
python text_generator.py
```

This will read the prompt from `data/prompt.txt`, generate text, and save it to `results/generated_text.txt`.

#### Testing with Different Prompts

To test the system with different prompts, including ones with typos:

```bash
python test_generator.py
```

This will generate text for each test prompt and save the results to `results/test_outputs/`.

### Output Format

The generated text is saved in a file with the following format:

```
Timestamp: YYYY-MM-DD HH:MM:SS
Original Prompt: <original prompt>
Corrected Prompt: <corrected prompt>  # Only included if the prompt was corrected

Generated Text:
<generated text>
```

### Customization

You can customize the system by modifying the following parameters in `text_generator.py`:

- `model_name`: The name of the pre-trained model to use (default: "gpt2").
- `max_length`: The maximum length of the generated text (default: 200).
- `temperature`: The temperature for sampling (default: 0.7).

## Q-Learning for Gridworld

This project implements a Q-learning algorithm to solve maze-style Gridworld tasks. The system uses numpy for efficient calculations and matplotlib for visualizations.

### Features

- **Gridworld Environment**: A customizable grid-based environment with obstacles.
- **Q-Learning Algorithm**: An implementation of the Q-learning algorithm for reinforcement learning.
- **Visualization**: Visualization of the learning curve and path changes during training.
- **Model Saving/Loading**: Functionality to save and load trained models.
- **Real-time Feedback**: Progress updates during training.

### Project Structure

```
.
├── src/
│   ├── env.py                    # Gridworld environment implementation
│   └── train.py                  # Q-learning algorithm implementation
├── models/
│   └── saved_models/             # Directory for saved models
│       └── q_learning_model.npy  # Trained Q-learning model
├── results/
│   ├── figures/                  # Directory for figures
│   │   ├── learning_curve.png    # Learning curve visualization
│   │   └── path_changes.gif      # Path changes visualization
│   └── test_paths/               # Directory for test path visualizations
├── run_training.py               # Script to run the training
├── test_model.py                 # Script to test the trained model
└── visualize.py                  # Script to visualize the results
```

### Requirements

- Python 3.6+
- NumPy
- Matplotlib
- tqdm

### Usage

#### Training

To train the agent, run:

```bash
python run_training.py
```

This will train the agent for 1000 episodes and save the model to `models/saved_models/q_learning_model.npy`. It will also generate visualizations of the learning curve and path changes.

#### Testing

To test the trained model, run:

```bash
python test_model.py
```

This will run 5 tests with the trained model and save the path visualizations to `results/test_paths/`.

### Customization

You can customize the system by modifying the following parameters:

#### Environment Parameters

- `grid_size`: The size of the grid.
- `start_pos`: The starting position (x, y).
- `end_pos`: The ending position (x, y).
- `obstacles`: A list of obstacle positions [(x1, y1), (x2, y2), ...].

#### Q-Learning Parameters

- `learning_rate`: The learning rate.
- `discount_factor`: The discount factor.
- `exploration_rate`: The initial exploration rate.
- `exploration_decay`: The exploration decay rate.
- `num_episodes`: The number of episodes to train for.

### Results

The system generates the following results:

- **Learning Curve**: A plot of the episode returns during training.
- **Path Changes**: An animation of the paths taken by the agent during training.
- **Test Paths**: Visualizations of the paths taken by the agent during testing.

## License

This project is licensed under the MIT License - see the LICENSE file for details.