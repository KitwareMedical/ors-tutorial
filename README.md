# Tutorial for the ORS 2023 Implants Workshop
## Practical considerations for applying and interpreting AI/ML models in orthopedics

Artificial intelligence is not new, but easy access to vast computational resources and more widely available imaging data repositories it is relatively new. Deep learning, or neuronal networks with more than three layers, needed these most recent advances to work well.

This tutorial will introduce you to a few examples of deep learning tasks using open source toolkits.

## Learning objectives

1. Increase familiarity with [MONAI](https://monai.io/)
2. Hands-on example on DL-based image classification
3. Hands-on example on DL-based image segmentation

# Project MONAI

<img src="https://monai.io/assets/img/MONAI-logo_color_full.png"/>

Medical Open Network for Artificial Intelligence (MONAI) is a [PyTorch-based](https://pytorch.org/), [open-source](https://github.com/Project-MONAI/MONAI/blob/dev/LICENSE) framework for deep learning in healthcare imaging, part of the [PyTorch Ecosystem](https://pytorch.org/ecosystem/). Its goal is to accelerate the pace of research and development
by providing a common software foundation and a vibrant community for medical imaging deep learning.

## Why is deep learning succeeding?

1. Performance. People get to do the same tasks faster.

2. Open Science. OS is pervasie in deep learning. From open access publications (arXiv), data (TCIA/IDC, ImageNet, DICOM, FIHR) or algorithms (PyTorch)

## Capabilities

*   Segmentation: [transformers](https://openaccess.thecvf.com/content/WACV2022/papers/Hatamizadeh_UNETR_Transformers_for_3D_Medical_Image_Segmentation_WACV_2022_paper.pdf), contrastive learning, auto3DSeg, nn-Unet
*   Image reconstruction: [SKM-TEA](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/file/03c6b06952c750899bb03d998e631860-Paper-round2.pdf)
*   Longitudinal studies: [Learn2reg](https://research.birmingham.ac.uk/en/publications/learn2reg-comprehensive-multi-task-medical-image-registration-cha)
*   Diagnosis: [CNN based diagnosis](https://www.ijimai.org/journal/bibcite/reference/2944)
*   Prediction: [Survival prediction](https://pubmed.ncbi.nlm.nih.gov/35399868/)

[More...](https://docs.google.com/presentation/d/1n0zEiZ2Iss5MqYWYbSlp_WVJ_LRiLqy9O6ErOjd7Bhc/present?slide=id.p1)

Recently MONAI has been integrated with [Amazon HealthLake](https://catalog.us-east-1.prod.workshops.aws/workshops/ff6964ec-b880-45d4-bc1e-468b0c7fa854/en-US) and [Google Health](https://developer.nvidia.com/blog/monai-drives-medical-ai-on-google-cloud-with-medical-imaging-suite/).


## Resources

-   MONAI Slack: https://forms.gle/QTxJq3hFictp31UM9
-   MONAI Docs:
    -   MONAI Core: https://docs.monai.io/en/stable/
    -   MONAI Label: https://docs.monai.io/projects/label/en/latest/index.html
    -   MONAI Deploy App SDK: https://docs.monai.io/projects/monai-deploy-app-sdk/en/latest/
-   MONAI Github: https://github.com/Project-MONAI
    -   MONAI Core: https://github.com/Project-MONAI/MONAI
    -   MONAI Label: https://github.com/Project-MONAI/MONAILabel
    -   MONAI Deploy: https://github.com/Project-MONAI/monai-deploy
-   MONAI YouTube: https://www.youtube.com/c/Project-MONAI
-   MONAI Twitter: https://twitter.com/ProjectMONAI
-   MONAI Medium: https://monai.medium.com/


# Hands-on tutorial: Let's get started

Classification: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kitwaremedical/ors2023-tutorial/blob/master/mednist_tutorial.ipynb)

Segmentation: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KitwareMedical/ors2023-tutorial/blob/master/spleen_segmentation_3d.ipynb)

Feel free to customize these scripts to use with your data! Just [fork](https://docs.github.com/en/get-started/quickstart/fork-a-repo) this repository.

# About the authors

**Jared Vicory**

<img src="https://www.kitware.com/main/wp-content/uploads/2021/11/Jared_Vicory_768x768.jpg"  width="300" height="300" />

Jared Vicory, Ph.D. is a staff R&D engineer on Kitware’s Medical Computing located in Carrboro, North Carolina. He has many years of experience in medical image processing and analysis, focusing on segmentation, statistical shape analysis, and machine learning. He has applied these techniques in a range of target problems including orthopedic \[1,2,3,4\] and dental/craniofacial applications \[5\].

[More information](https://www.kitware.com/jared-vicory/)


**Beatriz Paniagua**

<img src="https://www.kitware.com/main/wp-content/uploads/2021/11/paniagua-300x300-1.jpeg"  width="300" height="300" />

Beatriz “Bea” Paniagua, Ph.D., is an assistant director of Kitware’s Medical Computing Team located in Carrboro, North Carolina. Her projects largely focus on craniomaxillofacial \[5, 6\], musculoskeletal \[7,8,9\], and morphometric image analysis.

[More information](https://www.kitware.com/beatriz-paniagua/)


**References**

[1. Fracture Fixation Biomechanics Simulator with Adaptive Virtual Coaching (PI. Lewis)](https://reporter.nih.gov/search/XrZbdnSL80qYrii9Xeij_g/project-details/10375473)
[2. Bischoff et al. Verification and Validation of an Open Source–Based Morphology Analysis Platform to Support Implant Design](https://asmedigitalcollection.asme.org/medicaldevices/article-abstract/7/4/040903/376620/Verification-and-Validation-of-an-Open-Source?redirectedFrom=fulltext)
[3. Lewis et al. Virtual Simulation for Interactive Visualization of 3D Fracture Fixation Biomechanics](https://pubmed.ncbi.nlm.nih.gov/34370717/)
[4. Bischoff et al. Incorporating Population-Level Variability in Orthopedic Biomechanical Analysis: A Review](https://asmedigitalcollection.asme.org/biomechanical/article-abstract/136/2/021004/442937/Incorporating-Population-Level-Variability-in?redirectedFrom=fulltext)

[5. Vicory et al. Dental microfracture detection using wavelet features and machine learning](https://pubmed.ncbi.nlm.nih.gov/35505894/)

[6. Vimort et al. Detection of bone loss via subchondral bone analysis](https://pubmed.ncbi.nlm.nih.gov/29769754/)

[7. Paniagua et al. Diagnostic Index: An open-source tool to classify TMJ OA condyles](https://pubmed.ncbi.nlm.nih.gov/28690356/)

[8. Vimort et al. Computing Textural Feature Maps for N-Dimensional images](https://www.insight-journal.org/browse/publication/985)

[9. Vimort et al. Computing Bone Morphometric Feature Maps from 3-Dimensional Images](https://www.insight-journal.org/browse/publication/988)
