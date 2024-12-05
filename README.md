# Comparative Analysis of Video Generation Models: ConvLSTM, PredRNN, and MaxViT

## Table of Contents
- [Abstract](#abstract)
- [Introduction](#introduction)
- [Methodologies](#methodologies)
  - [ConvLSTM](#convlstm)
  - [PredRNN](#predrnn)
  - [MaxViT](#maxvit)
- [Experimental Results](#experimental-results)
- [Discussion](#discussion)
- [Conclusion](#conclusion)
- [References](#references)

---
### Acknowledgements
This project was a collaborative effort with [Huzaifa Khan](https://github.com/huzaifakhan04), who contributed significantly to the development of the PredRNN and ConvLSTM models.

### Data
The dataset used for this project is the [UCF101 dataset](https://www.kaggle.com/datasets/matthewjansen/ucf101-action-recognition/data), which contains videos of various human activities and is widely utilized for action recognition tasks.


## Abstract

Video generation is a complex task that involves modeling both spatial and temporal dependencies within video data. This README.md file presents a comparative analysis of three prominent models used for video generation: Convolutional Long Short-Term Memory (ConvLSTM), PredRNN, and MaxViT. The performance of each model is evaluated based on Mean Squared Error (MSE) and Structural Similarity Index Measure (SSIM) on a standardized test dataset. While ConvLSTM and PredRNN demonstrate superior performance with lower MSE and higher SSIM scores, MaxViT, despite its advanced transformer architecture, underperforms due to its data-hungry nature and lack of inductive bias. This analysis digs into the architectural nuances of each model, their training dynamics, and the implications of their performance metrics.

---

## Introduction

Video generation entails the creation of coherent and realistic video sequences from given inputs. It requires capturing both spatial features within individual frames and temporal dynamics across consecutive frames. Over the years, various architectures have been proposed to tackle this challenge, we'll be looking into 3 different approaches. 

This README focuses on three such architectures:
1. **ConvLSTM**: Integrates convolutional operations within the traditional LSTM framework to better capture spatial dependencies.
2. **PredRNN**: Enhances the LSTM architecture with dual memory states to model complex spatio-temporal relationships.
3. **MaxViT**: Adapts the transformer architecture, known for its prowess in natural language processing, to handle video data through multi-axis attention mechanisms.

The comparative study aims to understand the strengths and limitations of each model in the context of video generation.

---

## Methodologies

### ConvLSTM

**Convolutional Long Short-Term Memory (ConvLSTM)** extends the traditional LSTM by incorporating convolutional structures in its gates. This design allows the model to capture spatial hierarchies within data, making it particularly suitable for tasks involving spatio-temporal sequences like video generation.

**Architectural Overview:**
- **ConvLSTM Cell**: Implements the core LSTM functionality with convolutional operations for input, forget, and output gates.
- **ConvLSTM Network**: Stacks multiple ConvLSTM cells to process sequences of frames, capturing both spatial and temporal dependencies.
- **Seq2Seq Framework**: Utilizes a sequence-to-sequence approach where input frames are processed to generate future frames, maintaining spatial coherence and temporal consistency.

**Key Features:**
- **Spatial-Temporal Modeling**: By integrating convolutional layers, ConvLSTM effectively models both spatial features within frames and temporal dynamics across frames.
- **Inductive Bias**: The convolutional operations introduce an inductive bias that uses the spatial locality of video data, enhancing learning efficiency.

### PredRNN

**PredRNN** introduces a novel architecture that extends the LSTM framework with dual memory states to better capture spatio-temporal dependencies in video data.

**Architectural Overview:**
- **Dual Memory States**: Incorporates both short-term and long-term memory states to separately capture spatial and temporal information.
- **Stacked Layers**: Multiple PredRNN layers are stacked to deepen the model, allowing for more complex temporal dynamics to be learned.
- **Advanced Gates**: Enhances traditional LSTM gates with additional mechanisms to manage the flow of information more effectively.

**Key Features:**
- **Enhanced Memory Mechanism**: The dual memory states allow the model to retain and process information over longer sequences, improving its ability to generate coherent and contextually accurate video frames.
- **Improved Temporal Modeling**: By explicitly modeling temporal dependencies, PredRNN achieves better performance in capturing motion and dynamics within videos.

### MaxViT

**MaxViT** adapts the transformer architecture for video generation by incorporating multi-axis attention mechanisms, which aim to efficiently capture both local and global dependencies in video data.

**Architectural Overview:**
- **Multi-Axis Attention**: Combines blocked local attention with dilated global attention to balance computational efficiency and modeling capacity.
- **Hierarchical Design**: Implements a hierarchical structure that allows the model to process video frames at multiple resolutions, enhancing its ability to capture both fine-grained and global features.
- **Convolutional Integration**: Blends convolutional operations with transformer blocks to introduce spatial inductive biases, mitigating some of the data-hungry nature of pure transformer models.

**Key Features:**
- **Scalable Attention Mechanism**: The multi-axis attention reduces the quadratic complexity of traditional self-attention, enabling the model to handle larger video sequences more efficiently.
- **Global and Local Interactions**: By integrating both global and local attention, MaxViT captures a comprehensive range of dependencies within video data.
- **Flexibility and Efficiency**: Designed to adapt to varying input sizes with linear complexity, making it suitable for diverse video generation tasks.

---

## Experimental Results

The performance of ConvLSTM, PredRNN, and MaxViT was evaluated on a standardized test dataset using two primary metrics: Mean Squared Error (MSE) and Structural Similarity Index Measure (SSIM). Additionally, training time and the number of epochs required to achieve convergence were considered.

### Model Performance Metrics

| Model     | Test Loss (MSE) | Test SSIM   | Training Time         | Epochs |
|-----------|------------------|-------------|-----------------------|--------|
| ConvLSTM  | 0.0012           | 0.9296      |        90 minutes     | 20 |
| PredRNN   | 0.0013           | 0.9215      |       100 minutes     | 20 |
| MaxViT    | 0.0049           | 0.7784      |       385 minutes     | 50 |

All training was done on Colabs's A100 GPU.

### Summary of Results

- **ConvLSTM** achieved the lowest MSE and highest SSIM, indicating superior performance in generating accurate and structurally similar video frames.
- **PredRNN** followed closely behind ConvLSTM with slightly higher MSE and lower SSIM, yet still demonstrating robust performance.
- **MaxViT**, despite its advanced transformer-based architecture, showed higher MSE and significantly lower SSIM compared to the other models. Additionally, it required longer training times and more epochs to converge.

---

## Discussion

The experimental results highlight distinct differences in the performance and training dynamics of ConvLSTM, PredRNN, and MaxViT models for video generation.

### ConvLSTM and PredRNN Superiority

- **Inductive Bias**: Both ConvLSTM and PredRNN incorporate convolutional operations, introducing spatial inductive biases that leverage the inherent spatial locality of video data. This allows them to learn more efficiently from the available data, resulting in lower MSE and higher SSIM scores.
- **Efficient Temporal Modeling**: PredRNN's dual memory states enhance its capability to capture complex temporal dependencies, contributing to its strong performance. ConvLSTM, while simpler, effectively models spatio-temporal relationships, making it highly effective for video generation tasks.
- **Training Efficiency**: These models require fewer epochs and shorter training times to achieve convergence, making them more practical for real-world applications where computational resources and time are constraints.

### MaxViT's Underperformance

- **Data Hunger**: Transformer-based architectures like MaxViT are known for their high capacity and flexibility but are also data-hungry. They typically require large amounts of training data to perform optimally, which may not have been sufficiently available in this study.
- **Lack of Inductive Bias**: Unlike ConvLSTM and PredRNN, MaxViT lacks strong spatial inductive biases, making it harder to learn spatial dependencies efficiently from limited data. This results in higher MSE and lower SSIM, as the model struggles to generate accurate and coherent video frames.
- **Training Complexity**: MaxViT's transformer architecture introduces greater computational complexity, leading to longer training times and the need for more epochs to achieve convergence. This makes it less efficient compared to the recurrent-based models in scenarios with limited computational resources.
- **Mitigation Efforts**: Although MaxViT attempts to mitigate the lack of inductive bias through multi-axis attention and convolutional integrations, these efforts were insufficient to overcome the inherent challenges posed by transformer architectures in this specific application.

### Architectural Nuances

- **ConvLSTM and PredRNN**: Both models leverage recurrent structures enhanced with convolutional operations to maintain spatial and temporal coherence. PredRNN's additional memory states provide a more nuanced understanding of temporal dynamics, albeit with increased complexity.
- **MaxViT**: By employing multi-axis attention, MaxViT aims to balance local and global feature interactions. However, the transformer’s reliance on large datasets and higher computational demands make it less suited for tasks where data is limited or quick training is essential.

---

## Conclusion

This comparative analysis underscores the strengths of recurrent-based models, ConvLSTM and PredRNN, in video generation tasks, particularly in scenarios with limited data and computational resources. Their ability to incorporate spatial inductive biases and efficiently model temporal dynamics results in superior performance metrics compared to transformer-based architectures like MaxViT. While MaxViT offers advanced mechanisms for capturing complex dependencies through multi-axis attention, its data-hungry nature and lack of inductive bias present significant challenges in achieving optimal performance in video generation tasks. Future work may explore hybrid approaches that combine the strengths of both recurrent and transformer-based models to enhance video generation capabilities.

---

## References

1. **ConvLSTM**  
   Shi, X., Chen, Z., Wang, H., Yeung, D.-Y., Wong, W. K., & Woo, W. (2015). *Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting*. [arXiv:1506.04214](https://arxiv.org/abs/1506.04214)

2. **PredRNN**  
   Wang, Y., Wu, H., Zhang, J., Gao, Z., Wang, J., Yu, P. S., & Long, M. (2021). *PredRNN: A Recurrent Neural Network for Spatiotemporal Predictive Learning*. [arXiv:2103.09504](https://arxiv.org/abs/2103.09504)

3. **MaxViT**  
   Tu, Z., Talebi, H., Zhang, H., Yang, F., Milanfar, P., Bovik, A., & Li, Y. (2022). *MaxViT: Multi-Axis Vision Transformer*. [arXiv:2204.01697](https://arxiv.org/abs/2204.01697)

4. **Additional References**  
   - Dosovitskiy, A., et al. (2020). *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*. [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)  
   - Vaswani, A., et al. (2017). *Attention is All You Need*. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)  
   - He, K., et al. (2016). *Deep Residual Learning for Image Recognition*. [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)

