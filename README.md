# Modeling-Following-Behavior-of-Human-Driven-Vehicles-Using-Inverse-Soft-Q-Learning


This repository contains code, data processing scripts, and model training routines for the project that investigates human driving behavior in mixed traffic environments using Inverse Soft-Q Learning (IQ-Learn). Our approach is focused on accurately modeling car-following dynamics, capturing latent reward structures from expert demonstrations, and reconstructing realistic driving policies with an Actor-Critic framework.

---

## Project Description

Mixed traffic scenarios feature both autonomous vehicles (AVs) and human-driven vehicles (HDVs), which pose challenges for safety and efficiency. This project aims to analyze and model the following behavior of human-driven vehicles under different leading vehicle scenarios (HDV-following-AV vs. HDV-following-HDV). By leveraging the Waymo Open Dataset for real-world trajectory data, the project applies advanced data preprocessing and coordinate system transformations. The core of the modeling is based on Inverse Soft-Q Learning (IQ-Learn), which avoids explicit reward function recovery and directly learns the optimal soft Q-function. The developed model provides insights into driver behavior variations, such as differences in speed, following distance, and acceleration profiles.

---

## Features

- **Data Collection and Preprocessing**  
  - Extraction of vehicle trajectory data from the Waymo Open Dataset.
  - Smoothing of speed and acceleration using the Savitzky-Golay filter.
  - Transformation from global to ego-vehicle coordinate system.
  - Extraction of straight-driving trajectories and identification of car-following pairs.

- **Model Development and Training**  
  - An Actor-Critic based reinforcement learning framework.
  - Integration of the IQ-Learn algorithm to directly learn optimal soft Q-functions.
  - Use of expert demonstration data to guide the model training and mimic real-world driving policies.

- **Evaluation and Visualization**  
  - Analysis of key metrics such as average speed, vehicle spacing, acceleration profiles, ITTC, and jerk.
  - Visualization tools for reward function distributions, including heatmaps and scatter plots.
  - Comparative analysis of driver behavior under different following scenarios.

- **Extensible Codebase**  
  - Modular design for easy integration of additional features or datasets.
  - Configurable training parameters and model settings through command-line arguments or configuration files.

---

## Getting Started

### Prerequisites

- **Python Version:** Python 3.8 or later
- **Required Libraries:**  
  - PyTorch
  - Numpy
  - OpenCV
  - Matplotlib
  - Scipy
  - Additional packages specified in `requirements.txt`

### Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/your-username/iq-learn-following-behavior.git
   cd iq-learn-following-behavior
   ```

2. **Create a Virtual Environment (Optional but Recommended)**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # For Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare Dataset**

   Download and extract the Waymo Open Dataset (or the specific subset used in this project) to a designated folder. Adjust data paths in the configuration files if necessary.

---

## Usage

### Training the Model

To train the IQ-Learn model on the extracted car-following data, execute:

```bash
python train_iql.py --config config/train_config.yaml
```

This script will:
- Load and preprocess the dataset.
- Initialize the Actor-Critic and IQ-Learn modules.
- Train the model using a mix of expert demonstrations and simulated policy rollouts.
- Save checkpoints and log training metrics.

### Evaluating the Model

After training, evaluate the model performance in a simulated environment:

```bash
python evaluate.py --config config/eval_config.yaml
```

The evaluation script will generate performance metrics (e.g., mean speed, spacing, ITTC) and produce visualizations such as reward function heatmaps.

### Visualization

To visualize the learned reward functions over the state space, run:

```bash
python visualize_reward.py --config config/vis_config.yaml
```

This will open interactive plots that help in assessing the reward landscape across different following scenarios.

---

## Contributing

Contributions are very welcome! If you find issues or have ideas for improvement:
1. Fork the repository.
2. Create a feature or bug fix branch.
3. Submit a Pull Request detailing your changes.

Please ensure your code adheres to the existing style and includes relevant tests and documentation.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Acknowledgments

- **Waymo Open Dataset:** For providing high-quality trajectory data.
- **IQ-Learn & Actor-Critic Frameworks:** The foundational concepts behind our modeling approach.
- All researchers and developers in the field of autonomous driving and reinforcement learning whose work inspired this project.

---

For detailed documentation, please refer to the [project documentation](docs/README.md) or contact the repository maintainer. Enjoy exploring human driving behavior through IQ-Learn!
