## Description
In this competition, participants are challenged to reconstruct a large map from shuffled, overlapping image patches and answer multiple-choice questions based on the reconstructed map.

This task combines:

- Computer Vision (image stitching)
- Spatial reasoning
- Visual understanding

Participants must first correctly assemble the map and then use it to answer questions about locations, structures, and spatial relationships.

⚠️ Note: No dataset will be provided for training. Participants are expected to create or source their own data, simulate similar question formats, or leverage external resources to train their models.

## Evaluation
Submissions will be evaluated using a scoring system with negative marking:

- +1 point for each correct answer
- −0.25 points for each incorrect answer
- 0 points for unanswered questions
- -1 point for hallucinated answer

Participants may use ```5``` to indicate a question is not answered.

The final score will be calculated as:

```
final_score = (number of correct option) - 0.25 x (number of incorrect option) -  (number of hallucinated value)
```

Where:

Predictions with values 1, 2, 3, 4 are treated as attempted answers
Prediction value 5 is treated as unanswered and receives no penalty
Any other output value will be treated as hallucinated value

This evaluation encourages both accuracy and strategic decision-making—participants may choose to skip uncertain questions.


## Submission File
A ```submission.csv``` must be submitted with the following format.

    id,question_num,option
    ques_1,ques_1,1
    ques_2,ques_2,5
    ques_3,ques_3,2
    etc.

The first column ```id``` is same as ```question_num```.
For each question in the test set, you must predict the correct answer option (1, 2, 3, or 4) or 5 for unanswered questions. Any other value will be treated as hallucinated and incur -1 penalty.

## Dataset Description
The competition data is designed to simulate a real-world evaluation setup where the final test set remains hidden.

### Files Provided
* **test.csv** - The test set containing the questions that needs to be answered based on map
    - Columns:
        - `id` or `question_id`: Unique identifier for each question (e.g., ques_1)
        - `question`: Question text that has to be answered
        - `option_1`: Answer choice 1 (option value for submission)
        - `option_2`: Answer choice 2 (option value for submission)
        - `option_3`: Answer choice 3 (option value for submission)
        - `option_4`: Answer choice 4 (option value for submission)
* **patches/** - folder containing PNG patches of map.
    - ⚠️ Note: For anchoring map, **patch_0.png** will always correspond to the top-left corner of the original map
    - Except **patch_0.png**, all patches can be shuffled or rotated
* **sample_submission.csv** - Sample submission file demonstrating the required format for submission.

### Hidden Test Set
* During the final evaluation and leaderboard scoring, the contents of the ```patches/``` folder will be replaced with a hidden test dataset
* The structure will remain the same, but patches will be different and not accessible beforehand
* This ensures a fair evaluation of model generalization
* At test time parent directory to ```test.csv``` will be provided and thereafter the folder structure is same as the one provided in sample dataset
* Folder Structure
  ```
  .
  ├── patches/
  │   ├── patch_0.png
  │   └── patch_1.png
  |   └── ...
  ├── test.csv
  └── sample_submission.csv

  ```

### Submission Requirement
Participants must submit a file named submission.csv:

* The format must strictly follow sample_submission.csv
* Required columns and structure are defined in the sample file and described on this page
* Each row should correspond to a question from test.csv

### Note:

* Do not rely on specific image content in the provided dataset, as it will change during evaluation
* Ensure your pipeline works generically for any valid input image
* All predictions must be generated based on the images referenced in test.csv

## Competition Rules:

- **No late submissions accepted**
- **Submission deadline as per course calendar**
- Notebook/Python file only competition (You'll upload link to the folder containing the jupyter notebook or python file on moodle and any model weights that is required)
- **Proper README to setup the environment without which you will be directly graded 0 (No communication from the TA will be done regarding this)**
- An environment.yml or requirements.txt file that can be used to create environment
- At most 50 questions will be provided 
- Runtime of the notebook shouldn't exceed 1 hr
- Your notebook will be run on 48GB L40s GPU
- Internet will be used only to setup the environment and no internet will be allowed on the final submitted notebook
- Do not cheat
- Okay to consult and discuss idea but final solution should be your own.
- Final grading will be based on final leaderboard standing
- Cite whatever source you'll be using in your notebook

## Environment Setup

### Requirements
- Linux, CUDA 12.6, Python 3.11

### Steps

**1. Run setup (installs dependencies into the current Python environment and downloads weights):**
```bash
bash setup.bash
```

**2. Run inference:**
```bash
python inference.py --test_dir <path_to_test_dir>
```
Where `<path_to_test_dir>` is the directory containing `patches/` and `test.csv`.

If the evaluator creates its own Conda/virtual environment, activate that environment before running setup. No project-specific `conda activate` command is required.

Optional explicit Python binary:
```bash
PYTHON_BIN=python3.11 bash setup.bash
python3.11 inference.py --test_dir <path_to_test_dir>
```

For grading: the test directory will be provided by the evaluator, e.g.:
```bash
python inference.py --test_dir /grading/test_data
```

This produces `submission.csv` in the current directory.

### What setup.bash does
- Verifies Python 3.11
- Installs all dependencies into the current Python environment (see `requirements.txt`)
- Downloads `Qwen2.5-VL-72B-Instruct-AWQ` model weights locally
- Downloads EasyOCR English assets locally
- Inference uses local files only (no internet needed after setup)

### Dependencies
See `requirements.txt`. Key packages:
- **PyTorch**: `torch==2.6.0`, `torchvision==0.21.0`, `torchaudio==2.6.0` (CUDA 12.4 wheels, compatible with CUDA 12.6)
- **Model Loading**: `transformers==4.51.3`, `accelerate==1.6.0`, `huggingface_hub`, `autoawq==0.2.9`
- **VLM**: `Qwen2.5-VL-72B-Instruct-AWQ`, `qwen-vl-utils==0.0.8`
- **Image Processing**: `opencv-python-headless`, `pillow`
- **Text Extraction**: `easyocr`
- **Optimization**: `bitsandbytes` (for 4-bit quantization fallback)
- **Data**: `pandas`, `numpy`

## Sources / Citations
- [Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct-AWQ) - Vision-language model for map understanding
- [Qwen2-VL](https://github.com/QwenLM/Qwen2-VL) — Vision-Language Model for map understanding
- [HuggingFace Transformers](https://github.com/huggingface/transformers)
- OpenCV for image stitching
