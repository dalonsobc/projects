## Adaptive Data Optimization (ADO) for Language Modeling

This project investigates the implementation and effectiveness of the Adaptive Data Optimization (ADO) algorithm within a language modeling context, using the GPT-2 architecture and the Wikitext-2 dataset.

### Overview
The project explores the potential of ADO to improve training efficiency by dynamically adjusting data sampling probabilities during training, based on observed learning performance. Traditional methods use fixed data sampling policies, which may not be optimal. ADO addresses this by continuously updating sampling probabilities across different domains, giving more weight to more challenging domains while maintaining representation of all domains.
The primary goal of the project is to evaluate if ADO's dynamic sampling methodology can lead to improvements in training efficiency and model performance, especially within computationally constrained environments. The project also aims to compare ADO against a baseline model that uses uniform sampling.

### Key Components
Adaptive Data Optimization (ADO) Algorithm
* Dynamic Sampling: ADO dynamically adjusts sampling probabilities for different data domains during training.
* Domain Loss Calculation: The algorithm evaluates losses for each domain, providing insight into domain-specific learning challenges.
* Probability Update Rule: Uses scaling laws to modify domain sampling probabilities, increasing attention to difficult domains while maintaining appropriate representation of easier domains.
* Clipping Function: Enforces probability constraints to prevent extreme adjustments, ensuring no domain is ignored or over-prioritized.

Experimental Setup
* Dataset: A 20% subset of the Wikitext-2 dataset was used for training, preprocessed using GPT-2 tokenizer and standardized to 100 tokens maximum length. The dataset was divided into uniform segments, without thematic consideration, for domain-specific sampling.
* Model: The GPT-2 small model was used as the foundational architecture.
* Training: The models were trained for three epochs.
* Baseline: A baseline model was trained using uniform sampling across all domains, as a control condition.
* Metrics: Training loss, validation loss, perplexity, and domain-specific sampling frequencies were tracked.

Results and Analysis
* Training Efficiency: ADO showed a 24.6% improvement in training time per epoch compared to the baseline, reducing average epoch duration from 1822.11 seconds to 1372.5 seconds.
* Sampling Distribution: The ADO implementation resulted in a uniform distribution of sampling probabilities across all domains.
* Domain-Specific Loss: The study showed varying learning trajectories across the five domains. * Some domains showed consistent improvement. * Other domains showed rapid initial improvement followed by slower progress, or performance plateaus.
* Validation Metrics: Validation loss and perplexity increased over the epochs, indicating potential overfitting issues, particularly for the ADO implementation. The baseline model had better validation metrics, with a final validation loss of 3.2373 and perplexity of 25.4652, while ADO had a validation loss of 3.7221 and a perplexity of 41.3496.

Conclusions
* While the ADO algorithm aims to prioritize more useful data domains, the implementation did not achieve significant differences in sampling probabilities across domains.
* The observed sampling probabilities remained relatively uniform throughout the training, possibly due to over-regularization or clipping of probabilities.
* Overfitting was evident, as validation loss increased across epochs.
* ADO introduces additional computational overhead due to the need to compute domain-specific metrics and update sampling probabilities in real-time.
* The current implementation of ADO did not fully realize its theoretical advantages.

Future Directions
* Recalibrate scaling laws to improve domain-specific predictions, perhaps by experimenting with alternative loss metrics or hyperparameter settings.
* Relax clipping of sampling probabilities to allow for greater variation.
* Experiment with larger, more diverse datasets to better test ADO's capabilities and reduce overfitting tendencies.
* Extend training duration to give ADO more time to adapt sampling probabilities.

Practical Considerations
* ADO's real-time probability updates and domain tracking increase training time and resource usage compared to simpler approaches.
* The added complexity of ADO may discourage adoption, particularly for teams without significant experience in algorithmic optimization.

References
Y. Jiang, A. Zhou, Z. Feng, S. Malladi, and J. Zico Kolter, “ADAPTIVE DATA OPTIMIZATION: DYNAMIC SAMPLE SELECTION WITH SCALING LAWS,” Oct. 2024.  Q. Lhoest et al., “Datasets: A Community Library for Natural Language Processing,” arXiv:2109.02846 [cs], Sep. 2021, Available: https://arxiv.org/abs/2109.02846  T. Wolf et al., “HuggingFace’s Transformers: State-of-the-art Natural Language Processing,” arXiv:1910.03771 [cs], Feb. 2020, Available: https://arxiv.org/abs/1910.03771