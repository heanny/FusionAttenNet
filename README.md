# FusionAttenNet
This is my Master's graduation project, collaborating with Erasmus Medical Center.

## Abstract for this project

Attention Deficit Hyperactivity Disorder (ADHD) is a complex neurodevelopmental disorder characterized by significant heterogeneity, making accurate prediction and diagnosis particularly challenging. This study introduces FusionAttenNet, a novel deep learning framework designed to advance ADHD research by integrating neuroimaging data with phenotypic information for dual task prediction of both attention problem scores and subject age. Leveraging a ResNet-50 backbone enhanced with spatial, channel, and feature attention mechanisms, FusionAttenNet effectively captures intricate neurobiological patterns often overlooked by conventional models.
To handle the high dimensional nature of structural MRI (sMRI) data, a 3D-to-2D transformation approach was employed, enabling the use of convolutional neural networks while preserving critical spatial information. The model further incorporates key phenotypic variables—such as sex, maternal education level, and aggressive behaviour scores - from two large scale datasets, Generation R and ABCD, to enhance predictive accuracy and interpretability.

Through extensive cross validation and ablation studies, FusionAttenNet demonstrated substantial improvements in R-squared scores compared to baseline models (around 60% for age prediction and around 35% for CBCL 6-18 attention problems score prediction) , particularly excelling in age prediction tasks. While absolute R-squared values remain modest by data science standards, they reflect a significant advancement within ADHD research, where low predictive power is a known challenge. The findings underscore the potential of multi-modal, attention enhanced deep learning models in unravelling complex neurodevelopmental patterns and contribute valuable insights for future clinical applications.
This study not only highlights the effectiveness of integrating cortical and phenotypic data but also sets a precedent for employing advanced machine learning techniques in psychiatric research. The proposed framework opens avenues for more accurate, interpretable, and clinically relevant ADHD prediction models, paving the way for personalized diagnostics and targeted interventions.

## Three step pipeline for 3D vertex-wise sMRI to 512*512 2D four-channelled images
<img width="1100" alt="3d_to_2d" src="https://github.com/user-attachments/assets/7d12c95f-b181-4203-a8a7-3dd710399111" />

## Archtecture of FusionAttenNet
<img width="1100" alt="ml_model" src="https://github.com/user-attachments/assets/aa14de1f-2095-4d91-b2af-0383bacea88d" />




