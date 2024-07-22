# GenCo: Generative Retrieval with CoT Prompting

GenCo stands for Generative Retrieval with CoT prompting, a method designed to overcome the limitations of text-to-image retrieval inherent in Vision-Language Models (VLMs) by leveraging Large Language Models (LLMs) for generative retrieval.

## Contributions
This paper makes the following contributions:
1. Experimentally identifies and proves the limitations of existing VLMs in text-to-image retrieval at the part-of-speech level.
2. Proposes the GenCo methodology to address these limitations by utilizing the generative and verification capabilities of LLMs.
3. Evaluates the performance of GenCo from multiple angles and provides research insights.

## Reproducing Experiments
All code required to reproduce the experiments in the paper is provided.

### Step 1: Generating Ground Truth
No existing text-to-image retrieval datasets contain ground truth part-of-speech (POS) annotations for images.
Use `dataset_generation.py` to generate ground truth.
Pre-generated ground truth files are provided as `coco.csv` and `flickr.csv` in the `./data/gt/` directory. 
Download these files and save them in the `data` folder.

### Step 2: Data Preparation
Ensure the `./data` directory contains the following:
- `coco_train_images` and `captions`
- `flickr_images` and `captions`

For images, use the following links:
- COCO: [train2017 dataset](https://cocodataset.org/#download)
- Flickr: [Flickr8k dataset](https://www.kaggle.com/datasets/adityajn105/flickr8k)

### Step 3: Creating FAISS Index
Generate the FAISS index for image retrieval. 
Code and `index.csv` information are provided. 
Save the generated `.index` file in the `faiss` folder.

### Step 4: Running GenCo
Use `main.py` to evaluate the performance of GenCo.
![final_pipeline](https://github.com/user-attachments/assets/608d5c12-7290-4254-b559-5c59641b53d0)

#### Note
The captioning model used in the GenCo pipeline is ExpansionNet_v2, which currently achieves state-of-the-art (SoTA) performance.

Clone the model into the `ExpansionNet_v2` folder:
git clone https://github.com/jchenghu/ExpansionNet_v2.git

### Step 5: Verifying VLM POS Imbalance
Reproduce the experiments verifying the POS imbalance issue in VLMs, as mentioned in Contribution 1, using the code provided in the `motivation` folder.

## Directory Structure
```plaintext
.
├── .git/
├── data/
│   ├── gt/
│   │   ├── coco.csv
│   │   └── flickr.csv
│   ├── coco_train_images/
│   ├── captions/
│   ├── flickr_images/
│   └── captions/
├── ExpansionNet_v2/
├── faiss/
│   ├── index.csv
│   └── *.index
├── motivation/
├── dataset_generation.py
└── main.py
```
## References
For detailed instructions on running the code and experiments, please refer to the respective files and directories.
