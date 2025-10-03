# Author's Note:

**Tl;dr** - Check out the `solves_export` folder and the JSON files for consolidated exported data. Mainly the keys of `"thoughts"`, `"training_pairs"`, `"test_pairs"`, `"solve_configuration"` to quickly understand how you might utilize the data in your own ARC-AGI solving efforts. Enjoy!

Below, beginning at the **Cargia** section, is the readme file that Cursor made for me near the end of my efforts on this project, and was created during my preparation to move this project from a private repo to a public repo. 

The **true goal** of this project was to **familiarize myself with Cursor** and/or other AI-first coding tools on a fun, low-risk endeavor so that I could better understand how to efficiently utilize Cursor during a larger, more high-risk project, where primarily-AI-written software might actually be used by someone. I believe I have accomplished this goal, and I'm definitely ready to move on to my next major project. 

But before I do that, I wanted to explain my general motivation for choosing ARC-AGI v2 for this project. 
- I'm a fan of puzzles, games, and rulesets.
- I'm interested in the path by which one arrives at certain conclusions.
- I think I'm good at identifying and explaining paths, and how one might arrive there (i.e., teaching).
- ARC-AGI is a uniquely difficult and interesting benchmark. 

I largely view machine learning as analogous to *human* learning. I believe machine learning should take much more inspiration from human learning, and how we've come to learn how to teach children or adults. It was always strange to me reading about how major ARC-AGI efforts involved simply throwing the text represntation of the input and output grids into an LLM, when the core operations or aspects to the puzzle-solving effort were entirely rooted in the visual nature of the grid and the intuitive "world" in which those puzzles exist. Simply put: humans would never be able to solve a strictly-text-based version of ARC-AGI, so why should we expect LLMs to? 

So my approach to making a dent in the ARC-AGI benchmark was to gather a dataset of me solving the puzzles, using explanations that I might use to teach *a human child* how to solve these types of puzzles. The explanation would not simply be stating the answer or the algorithm, because that doesn't help a human child as much as showing them the steps, the path, along the way to the answer, and the important features that inform *how to arrive at* the answer. Of course the image version of the pairs of grids (and some preamble as to the purpose of the puzzles) would be the only thing needed to explain how to solve each puzzle. 

When Gemma3 was released, I had the idea that I could make my puzzle explanations as a sort of **multi-turn conversation**. Instead of giving all the training pairs of a single puzzle at once and expecting the model to figure it all out, and then solve the test pairs, I would treat it more like how I solve the puzzles: by looking at each pair one at a time and getting a sense of the transformation rule, form a hypothesis, and then seeing if the other pairs support that hypothesis. This construction naturally forms a back-and-forth multimodal-based chatbot-like conversation. There would be some base prompt about ARC-AGI, and then the first training pair (input AND output grid) would be shown. It doesn't matter which one really, any training pair would do. The model would then be responsible for reproducing the reasoning text that I myself would record for each training pair in that puzzle. The next training pair would then be shown to the model, and the model would continue on, seeing if this training pair supports or amends the hypothesis established by the training pairs before it. By the time the test pair is shown, the model would then have the context of all the training pairs in image format (and text format), as well as all the reasoning steps that produced the current hypothesis, and the hypothesis. It would simply need to apply that same logic to the shown input grid, and then, ideally successfully, fill in the output grid for each test pair by supplying the correct text as its final response.

So that's how I designed the data collection GUI: **Cargia**. I would be shown training pairs one at a time, record my thoughts, and produce reasoning traces of me solving puzzles. These "thoughts" would form the basis for training a VLM for a multi-turn conversation on solving a puzzle. 

There were a few unforeseen speed bumps along the way. 
1. **The data explanation process was way more laborious and time-consuming than what I originally thought**. I was hoping that I could do one every 6 minutes, so 10 puzzles per hour, such that I could then do the entire training set in only 100 hours (not too bad). However my average solve time was about 19 min. Furthermore, the actual communication and explanation was **difficult**. I could sometimes solve a new puzzle in my mind in two seconds, based on a single training pair, but then it would take me 5+ minutes to talk about and explain *why* or *how* I came to that conclusion.
2. **Training on long contexts is VRAM-intensive**. The plan as explained above *necessitated* that the model hold an entire puzzle's training pair and text sequence in context so that it would have the very best chance at solving the test pairs at the end. It turns out that training on long context sequences is very VRAM intensive. The average token length of my solves was 7500 tokens (including the images). Although Gemma3 was claimed to have a large context window, I didn't realize just how much VRAM that would take, even with LoRA training. Certainly I was easily mazing out my local 4090, so I then intended to try cloud-based 80GB card, but it seemed like even then training went very slowly (but I also very well may not be doing something right in these steps).
3. **Supervised fine-tuning is not the way.** As someone coming from a more computer-vision based ML background, I have little to no hands-on coding experience with reinforcement learning. Supervised fine-tuning the LLM on such a small dataset of thoughts would likely *not* get me where I wanted to be, which was for the VLM to generalize the very structure of the response flow during the solve. That's because there are many, many ways to say the words and concepts that I wanted to get across, and SFT doesn't correctly account for that. But reinforcment learning does. ARC-AGI seems like the ultimate candidate for VLM-based reinforcement learning because there's a *huge* verifiable reward signal at the end of a sequence of reasoning: namely, did the model get the test output grid *exactly* correct. 
3. **Big labs might just solve it entirely with scaling?** I started working on this in April 2025 and it is now October 2025, and there has been *insane* progress in world model-type of AI models. In my opinioin, having visual priors and understanding of objectness, translation and rotation in space, overlapping dimensions, etc. is paramount to solving ARC-AGI puzzles. Solving these puzzles is not just about understanding images and what's in them; it's about how things in images move or can be transformed and how objects normally interact in the world. This description is much closer to needing a full understanding of how the physical world actually works, and I'm not sure even the most capable text-only LLM can understand that (someday there *likely will be* a model that can solve ACR-AGI with only the text-grids!). 

**So I stopped the project**. I want to move on. Before the competition deadline, I am going to make this repository public and post it in the ARC-AGI Discord, so that anyone can, if they want to, use the thoughts and solves data stored in the `thoughts.db` and `solves.db` database files to help them in their efforts. Additionally, I exported all the data into a `solves_export` folder to consolidate the information into a single puzzle per `JSON` file, along with an `index.json` as a summary of all the solves data. **If you do indeed use this data, I'd love to know about it**. You can cite this project if you want, I don't really care. I care more about *understanding how* the data specifically helped to improve whatever ARC-AGI solver model you are training.


# Cargia: ARC-AGI Data Labeling and Model Training Tool

Cargia is a comprehensive toolkit for annotating ARC-AGI (Abstraction and Reasoning Corpus for Artificial General Intelligence) benchmark tasks and training vision-language models to solve them. The project combines a PyQt6-based GUI for data collection with a sophisticated training pipeline for fine-tuning Gemma3 models.

## 🎯 What is ARC-AGI?

ARC-AGI is a benchmark designed to test AI systems' ability to perform abstract reasoning. Tasks consist of visual grid transformations where the AI must learn the underlying pattern from a few examples and apply it to new test cases. These puzzles test core reasoning abilities that are fundamental to artificial general intelligence.

## 🏗️ Repository Structure

### Core Components

#### **Data Collection GUI** (`cargia/main.py`)
- **PyQt6-based interface** for systematic annotation of ARC-AGI tasks
- **Visual grid rendering** with customizable color mappings
- **Step-by-step reasoning capture** with voice transcription support
- **Metadata annotation** including color maps, character mappings, and spatial invariances
- **SQLite database storage** for structured data management

#### **Training Pipeline** (`cargia/training/`)
- **Gemma3 fine-tuning** with LoRA (Low-Rank Adaptation) for efficient training
- **Multi-modal training** combining vision and language understanding
- **Data augmentation** including color invariance, character invariance, and spatial transforms
- **Custom loss functions** with emphasis on grid output accuracy
- **Cloud deployment support** for RunPod and other GPU platforms

#### **Data Management** (`cargia/data_manager.py`)
- **Structured database schema** for solves, thoughts, and metadata
- **Data validation and cleaning** pipelines
- **Export capabilities** for training data preparation

#### **Text Processing** (`cargia/text_cleaner.py`)
- **LLM-powered text cleaning** using Gemma3 for grammar correction
- **Thought process standardization** for consistent training data

### Key Features

#### **Visual Interface**
- Interactive grid visualization with zoom and pan capabilities
- Real-time color mapping and character substitution
- Side-by-side comparison of input/output pairs
- Progress tracking and session management

#### **Voice Integration**
- **Real-time speech-to-text** using Faster-Whisper
- **Automatic transcription** of reasoning processes
- **Audio device management** and silence detection

#### **Advanced Training**
- **Multi-stage training pipeline** from overfitting tests to full dataset training
- **Custom loss weighting** emphasizing grid accuracy over general text generation
- **Data augmentation** with color, character, and spatial transformations
- **Intermediate supervision** for improved reasoning quality

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended for training)
- PyQt6 dependencies for GUI
- Gemma3 model weights (download from Hugging Face)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd cargia

# Install dependencies
pip install -e .

# For development dependencies
pip install -e ".[dev]"

# Set up environment variables (optional)
export GEMMA3_MODEL_PATH="/path/to/your/gemma3-4b-it-ORIGINAL"
```

### Configuration

The training system uses environment variables for flexible path configuration:

#### **Model Path Configuration**
```bash
# Set the Gemma3 model path
export GEMMA3_MODEL_PATH="/path/to/your/gemma3-4b-it-ORIGINAL"

# Or on Windows
set GEMMA3_MODEL_PATH=C:\path\to\your\gemma3-4b-it-ORIGINAL
```

If `GEMMA3_MODEL_PATH` is not set, the system will default to `gemma3-4b-it-ORIGINAL` (expecting the model to be in the current directory or available via Hugging Face).

#### **Data Paths**
- **GUI data directory**: Set via `settings.json` or GUI settings
- **Training data**: Configured in YAML config files
- **Cloud deployment**: Uses RunPod volume mounts

### Quick Start

#### **Data Collection**
```bash
# Launch the GUI for data annotation
python -m cargia.main
```

#### **Training**
```bash
# Test training pipeline with single sample
python cargia/training/train_cli.py --config cargia/training/configs/step_1_overfit_single.yaml --local

# Full dataset training
python cargia/training/train_cli.py --config cargia/training/configs/step_5_full_dataset.yaml --cloud
```

## 📊 Training Pipeline

The training system follows a systematic approach:

### **Phase 1: Validation Testing**
1. **Single sample overfitting** - Verify core training loop
2. **Eight sample overfitting** - Test memorization capacity
3. **Augmentation testing** - Validate data transformation pipeline

### **Phase 2: Production Training**
4. **Full dataset training** - Complete model fine-tuning
5. **Evaluation and deployment** - Model assessment and serving

### **Configuration Management**
- **YAML-based configs** for reproducible experiments
- **Environment-specific paths** (local vs cloud deployment)
- **Flexible augmentation settings** with validation

## 🔧 Data Augmentation

### **Color Invariance**
- Random color mappings while maintaining visual distinguishability
- Automatic text transformation for color references in reasoning

### **Character Invariance**
- Symbol substitution (digits → letters/symbols) to prevent character bias
- JSON transformation for input/output grids

### **Spatial Transforms**
- Rotation, reflection, and mirroring capabilities
- Metadata-driven transformation application

## 🎨 GUI Features

### **Annotation Workflow**
1. **Load ARC-AGI tasks** from JSON format
2. **Visualize transformation pairs** with interactive grids
3. **Record step-by-step reasoning** via text or voice
4. **Annotate metadata** (colors, characters, spatial properties)
5. **Export structured data** for training

### **Quality Assurance**
- **Visual validation** of color mappings and transformations
- **Consistency checking** across training pairs
- **Progress tracking** and session management

## 🧪 Testing and Validation

### **Test Scripts** (`scripts/`)
- **Color invariance testing** - Validate color transformation accuracy
- **Character invariance testing** - Verify symbol substitution
- **Spatial transform testing** - Check geometric transformations
- **Text cleaning validation** - Ensure LLM cleaning quality
- **Data consolidation testing** - Validate unified data export/import

### **Data Validation**
- **Schema validation** for database consistency
- **Cross-reference checking** between solves and thoughts
- **Export validation** for training data integrity

### **Data Consolidation System**
- **Unified JSON export** - Consolidate fragmented data into single files per solve
- **Easy data access** - Load, filter, and analyze solve data without database queries
- **Portable format** - Share and archive complete solve information
- **Rich metadata** - Include timing, user info, and data quality indicators

See `scripts/README_data_consolidation.md` for detailed documentation.

## 🌐 Cloud Deployment

### **RunPod Integration**
- **Volume mounting** for persistent data storage
- **Docker containerization** for reproducible environments
- **SSH access** for interactive development and monitoring

### **Training Infrastructure**
- **Multi-GPU support** for large-scale training
- **Checkpoint management** and resumable training
- **TensorBoard integration** for monitoring

## 📈 Performance and Monitoring

### **Training Metrics**
- **Exact match accuracy** for grid predictions
- **Token-level accuracy** for reasoning quality
- **Loss decomposition** (text vs grid components)

### **Visualization**
- **Real-time training progress** in GUI
- **Example visualization** of model predictions
- **Error analysis** and debugging tools

## 🔬 Research Applications

Cargia enables research in:
- **Abstract reasoning** and pattern recognition
- **Multi-modal learning** (vision + language)
- **Few-shot learning** and generalization
- **Interpretable AI** through reasoning capture

## 📚 Documentation

- **Training configuration guide** (`cargia/training/configs/README.md`)
- **RunPod deployment guide** (`cargia/training/RUNPOD_VOLUME_GUIDE.md`)
- **API documentation** in code comments and docstrings

## 🤝 Contributing

This project is designed for research and development in AI reasoning. Contributions are welcome for:
- **Data augmentation techniques**
- **Training optimization**
- **GUI improvements**
- **Evaluation metrics**

## 📄 License

[Add your preferred license here]

## 🙏 Acknowledgments

- **ARC-AGI benchmark** creators for the foundational dataset
- **Google's Gemma3** team for the vision-language model
- **Hugging Face** for the transformers library and ecosystem

---

*Cargia represents a comprehensive approach to ARC-AGI research, combining human expertise in data annotation with advanced machine learning techniques for training reasoning-capable AI systems.*