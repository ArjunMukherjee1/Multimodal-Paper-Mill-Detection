# Improving machine learning-based paper mill detection in cancer research using full text, image and citation data

**Arjun Mukherjee¹**

¹ Carnegie Mellon University, Pittsburgh, PA, USA

**Corresponding author:** Arjun Mukherjee, Carnegie Mellon University, Pittsburgh, PA, USA — arjunmukherjee2007@gmail.com

## Abstract

**Background:** Paper mills, organizations that create and sell authorship rights to fraudulent research papers, are detrimental to the integrity of academic research, especially cancer literature. A recent study leveraged a BERT-based classifier trained on a paper's title and abstract to identify paper mill products in cancer research, achieving an accuracy of 91% on internal validation and 93% on external validation. This paper investigates whether incorporating a paper's full text, figure images and references alongside its abstract and title can enhance the performance of this detection pipeline.

**Methods:** A dataset was created combining known paper mill publications from Retraction Watch and PubPeer with control papers extracted from high impact journals and a stratified sample of countries, selected to account for linguistic biases. Every paper's title, abstract, figure images, citations and full text (Methods, Results and figure captions) were gathered. Using this dataset, four signals were developed: a BERT-based classifier trained on title and abstracts (mirroring the original study), a BERT-based classifier extending this by integrating full text, a citation signal measuring references to known paper mill articles, and a CLIP-based figure image classifier. These signals were all merged using logistic regression, creating a final fused model.

**Results:** The abstract-only model performed similarly to the original study's model, attaining accuracies of 93.4% and 91.5% on internal and external validation respectively. The full text model showed some improvement upon this, boosting internal validation accuracy to 95.5% and external validation accuracy to 92.1%. Compared to the text-based models, the image classifier was less reliable, yielding an accuracy of 83.6% on internal validation and 76.4% on external validation. The fused model demonstrated the best performance out of all the models, achieving a 98.2% accuracy on internal validation and 96.4% on external validation.

**Conclusions:** Supplementing title and abstract text with image, full-text and citation data can significantly boost the performance of machine learning-based paper mill detection. Paper mills are poisoning the credibility of academic literature and will likely continue to become larger and more evasive. Therefore, developing more comprehensive and reliable ways to detect their work is very important.

**Keywords:** paper mills, machine learning, research integrity, BERT, cancer research, retraction, natural language processing
