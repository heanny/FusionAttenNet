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

## Highlights

- Innovative Model Design: FusionAttenNet, a ResNet-50-based architecture enhanced with spatial, channel,
and feature attention mechanisms, optimized for ADHD prediction.

- Dual-Task Prediction: Simultaneous prediction of attention problem scores and age, providing
deeper insights into ADHD-related neurodevelopmental patterns.

- Multi-Modal Data Fusion: Integration of cortical features with key phenotypic data (sex,
maternal educational level, CBCL 6-18 aggressive behaviour scores) for more comprehensive
modelling.

- Enhanced Predictive Accuracy: Achieved higher $R^2$ scores than traditional baselines, demonstrating
substantial improvements within ADHD research. Its robustness was validated through cross-validation, residual and error distribution analyses, and a series of ablation studies.

- Efficient 3D-to-2D Data Transformation: Facilitated the use of CNNs while retaining critical
spatial brain information, streamlining data processing and model training.

## Future work
<ol>
<li>North pole alignment in Mercator projection
 
Instead of fixing one arbitrary orientation, different north-pole settings could be tested. By selecting a small set of predefined orientations and distributing transformed datasets evenly across them, the projection artefacts introduced by 3D→2D flattening may be mitigated. A systematic check for sensitivity to north-pole angle is therefore necessary. From our preliminary sample tests, there is a possibility of moderate performance degradation under certain misalignments, although more extensive evaluation would be required to draw firm conclusions.


<li> Reducing rotation sensitivity (P3CNN-inspired approach)
  
The potential sensitivity to cortical surface rotations observed in this thesis could be alleviated by adopting ideas from the P3CNN paper[^1]. Their pipeline uses:

[^1]: Henschel L, Reuter M. Parameter Space CNN for Cortical Surface Segmentation. Bildverarb Med. 2020 Mar;2020:216-221. doi: 10.1007/978-3-658-29267-6_49. Epub 2020 Feb 12. PMID: 36637373; PMCID: PMC9832244.

- KNN interpolation with inverse distance weighting (Shepard interpolation) to better handle the mismatch between a FreeSurfer triangular mesh and the regular latitude–longitude pixel grid.

- Multi-view alignment: the sphere is rotated such that the north pole aligns with X, Y, and Z axes separately, projected to 2D, and reconstructed back to 3D; results are averaged to reduce orientation bias.

<li>Site-split evaluation
  
Future experiments should evaluate generalization under a site-split setting, where training and testing come from different acquisition sites/scanners. This helps quantify dataset shift and ensures robustness across cohorts.

<li> Graph-based models
  
Projection-free methods such as graph neural networks (GNNs) on the cortical mesh may bypass 3D→2D artefacts altogether, preserving neighborhood structure and potentially improving interpretability.
</ol>
