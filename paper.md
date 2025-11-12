Lightweight Transformer-Based Cyberbullying
Detection for English, Malayalam, and Manglish
Social Media Texts
Clive Lawrence Xavier
Scholar
Department of Computer Science
Sacred Heart College
Kochi, India
clivelawrence17@gmail.com
C Vishnu Mohan
Assistant Professor
Department of Computer Science
Sacred Heart College
Kochi, India
vishnumohan@shcollege.ac.in
Abstract—Cyberbullying has become one of the principal
threats in this era of the internet, most notably on social networking sites where people—most notably teenagers and young
adults—are frequently exposed to abuse or hate speech. This
project aims to develop and implement a smart system capable
of automatically identifying cyberbullying in social media posts
through the use of Natural Language Processing (NLP) methods.
The system scans user-generated text to identify potentially
abusive, offensive, or threatening words using methods such as
text preprocessing, feature extraction, and classification. Different
machine learning and deep learning models like Support Vector
Machines, Logistic Regression, and transformer-based models
like BERT are explored to enhance the detection mechanism’s
accuracy and context awareness. The project also addresses the
primary issues like handling of slang, sarcasm, ambiguity of
language, and repetitive patterns of undesirable behavior. By
detecting syntactic, semantic, and sentiment-based features, the
system improves explicit and implicit cyberbullying detection.
The developed model will be flexible enough so that it can be
refined step by step based on experimental outcomes and the
dynamics of actual data. It can be applied on content moderation
platforms to recognize toxic messages, enable early intervention,
and ensure safer online communication. The project demonstrates the real-world application of Natural Language Processing
(NLP) and machine learning in limiting the growing problem
of cyberbullying, hence making an effort towards developing a
healthier and more respectful virtual world.
Index Terms—Cyberbullying detection, social media, NLP,
BERT, sentiment analysis, sarcasm detection, content moderation.
I. INTRODUCTION
The rapid proliferation of social networking sites like Facebook, Instagram, Twitter (now X), TikTok, and Reddit has
transformed how individuals connect, communicate, and share
ideas. However, this digital transformation has also given
rise to cyberbullying — the intentional and repeated use of
digital communication to harm, intimidate, or humiliate others.
Cyberbullying can manifest through direct insults, threats,
exclusion, rumor-spreading, and sharing of harmful content.
Research has consistently linked cyberbullying to severe psychological consequences, including depression, anxiety, selfharm tendencies, and in extreme cases, suicide.
Unlike face-to-face bullying, cyberbullying transcends geographical boundaries and time constraints, allowing perpetrators to target victims at any time. The anonymity and
lack of immediate consequences online further embolden
aggressors. The volume of data generated on social media
makes human moderation insufficient. As a result, automated
detection systems powered by NLP have become critical to
timely intervention and prevention.
NLP-based cyberbullying detection leverages linguistic
cues, semantic analysis, and contextual understanding to differentiate between benign conversation and harmful intent. Advances in ML and deep learning have significantly improved
detection capabilities. However, the diversity of languages,
slang, and cultural nuances presents persistent challenges. This
paper reviews literature from 2013 to 2025, highlighting key
developments, methodologies, datasets, and research gaps in
this domain.
II. RELATED WORKS
A. Surveys and Taxonomies
Several surveys provide structured overviews of cyberbullying detection methodologies. Elsafoury et al. [1] categorize
approaches into lexicon-based, classical machine learning,
deep learning, and transfer learning, highlighting that lexiconbased methods are interpretable but limited in scalability,
while deep learning methods achieve higher accuracy but
demand large annotated datasets. They further emphasize
the role of linguistic, syntactic, semantic, and multimodal
features, while noting challenges such as dataset imbalance
and annotation inconsistencies. Mahmud et al. [3] extend these
insights by focusing on low-resource and dialectal languages,
discussing how transliteration, cross-lingual embeddings, and
transfer learning can support detection in Hinglish, Manglish,
or Arabic dialects. Their study stresses that detection systems
trained exclusively on English underperform in multilingual
contexts, leaving many global communities underprotected.
Collectively, these surveys reveal both the strengths and limitations of mainstream approaches while underlining the need
for taxonomies that incorporate multilingual, multimodal, and
culturally adaptive perspectives.
B. Machine Learning and Deep Learning Approaches
Traditional machine learning techniques form the baseline
for cyberbullying detection. Islam et al. [5] demonstrate the
utility of classifiers such as Support Vector Machines (SVM)
and Logistic Regression trained on TF-IDF features, showing
competitive accuracy but vulnerability to noisy and informal
social media text. Desai et al. [14] further explore Decision
Trees and Random Forests, emphasizing that n-gram features
and sentiment scores improve detection, though performance
declines in heavily obfuscated or code-mixed contexts. Deep
learning models provide significant advances in this regard.
Teoh and Varathan [11] show that transfer learning architectures such as BERT outperform traditional approaches by capturing contextual semantics, though with higher computational
requirements. Sayed et al. [17] extend this line of work by
incorporating multi-class classification, differentiating abusive
content based on categories like race, gender, and religion,
thereby enabling more nuanced moderation. Similarly, Nikitha
et al. [10] investigate bilingual settings, demonstrating that
preprocessing strategies such as transliteration handling and
multilingual embeddings are critical for detecting harmful
content in languages like Hinglish. Together, these studies
highlight the shift from handcrafted features toward contextaware deep models, with transformers offering state-of-the-art
performance when resources permit.
C. Ensemble and Hybrid Models
Ensemble and hybrid methods have emerged as effective
strategies for improving robustness. Kumar et al. [12] and
Muneer et al. [13] show that stacking ensembles with transformer models enhance detection accuracy and reduce false
negatives by leveraging complementary strengths of multiple
classifiers. Perera and Fernando [6] propose thematic classification within ensemble frameworks, enabling differentiation
between context-specific abuse such as political, cultural, or
personal attacks, thereby improving interpretability and user
trust. Pradheep et al. [9] advance multimodal detection by
combining textual and image features, a crucial development
for identifying harmful memes where abusive intent is both
visual and textual. These ensemble and hybrid approaches
consistently outperform individual models, though they introduce higher computational costs and reduced interpretability.
Nevertheless, their ability to capture subtle, context-specific,
and multimodal patterns makes them highly relevant for realworld deployment.
D. Feature Engineering and Obfuscation Handling
Effective preprocessing and feature engineering remain indispensable, particularly given the prevalence of obfuscation in
social media text. Zhang et al. [7] demonstrate that combining
user behavior features, such as posting frequency and network
centrality, with linguistic cues enhances early detection in
Chinese social networks. Shekhar and Venkatesan [8] introduce a bag-of-phonetic-codes approach to address deliberate
obfuscation, such as replacing letters with symbols or phonetic
equivalents, thereby improving recall on noisy text. Similarly,
Abdrakhmanov et al. [15] show that lexical and character-level
features, including character n-grams, are effective against
spelling variations and low-resource conditions, serving as preprocessing filters for downstream models. These studies collectively underscore that robust feature engineering—spanning
behavioral signals, phonetic encodings, and character-level
representations—remains a critical step in handling adversarial
and noisy online environments.
E. Real-Time, Cultural, and Ethical Perspectives
Research has increasingly shifted toward addressing practical, cultural, and ethical considerations in cyberbullying detection. Li et al. [2] investigate real-time detection challenges,
showing that asynchronous message flows and random delays
can significantly impair streaming-based moderation, and propose dynamic frameworks for improved responsiveness. Shah
et al. [4] highlight the need for culturally adaptive detection,
arguing that sensitivity thresholds should vary across regions
to account for community norms and prevent over-flagging.
Carter [16], from a social science perspective, emphasizes
the psychological toll of cyberbullying on undergraduate students and stresses that detection systems must be paired
with ethical, user-centered interventions. Collectively, these
studies demonstrate that technological effectiveness alone is
insufficient. Real-time responsiveness, cultural awareness, and
ethical responsibility are equally vital in building trustworthy
and impactful cyberbullying detection systems.
III. METHODOLOGY TRENDS
Across studies, common steps in the cyberbullying detection
pipeline include:
1) Data Collection — Using datasets from Twitter,
Facebook, Instagram, YouTube comments, or custom
crawlers.
2) Preprocessing — Tokenization, stopword removal,
lemmatization, handling slang/obfuscation.
3) Feature Extraction — TF-IDF, word embeddings
(Word2Vec, GloVe, BERT embeddings), sentiment features.
4) Model Training — Traditional ML (SVM, Logistic
Regression), deep learning (CNN, LSTM), transformer
models (BERT, DistilBERT), or ensemble learning.
5) Evaluation — Metrics include Accuracy, Precision,
Recall, F1-score; cross-validation ensures robustness.
IV. CHALLENGES AND RESEARCH GAPS
As summarized in Table I, existing approaches excel in specific domains but often lack generalization across platforms,
languages, and cultural contexts.
Despite progress, several challenges remain:
TABLE I
SUMMARY OF REVIEWED CYBERBULLYING DETECTION STUDIES
Reference Dataset Method Lang. Remarks
[1] Elsafoury et al. 50+ public datasets Survey of ML, DL, TL Multi Taxonomy of features/methods; notes data
imbalance issues
[2] Li et al. Simulated streams Delay-compensation filtering N/A Useful for streaming moderation; not NLPfocused
[3] Mahmud et al. Bengali, Arabic,
Hinglish
Transfer learning Lowresource
Effective for dialect/code-mix; lacks large
datasets
[4] Shah et al. Twitter, Instagram ML + cultural adaptation Multi Adjusts detection by culture; manual calibration needed
[5] Islam et al. Twitter TF-IDF + SVM, LR Eng. Good baseline accuracy; poor sarcasm detection
[6] Perera & Fernando Facebook Theme-specific ML Eng. Detects bullying themes; needs better generalization
[7] Zhang et al. Weibo Behavior + text features Chinese Social patterns boost detection; limited to
China
[8] Shekhar & Venkatesan Twitter Bag-of-phonetic-codes Eng. Handles obfuscated profanity; fails on semantic attacks
[9] Pradheep et al. FB/Twitter
multimodal
Text + image fusion Eng. Captures abusive memes; high annotation
cost
[10] Nikitha et al. Social networks NLP + ML bilingual Eng./Hing. Works on transliterated text; limited scope
[11] Teoh & Varathan Public datasets ML vs BERT, RoBERTa Eng. TL better than ML; high compute cost
[12] Kumar et al. Twitter Ensemble classifiers Eng. Robust detection; complex architecture
[13] Muneer et al. Twitter/Facebook Stacking + BERT Eng. High F1-score; needs large GPUs
[14] Desai et al. Twitter/Facebook NB, DT, RF Eng. Good with feature engineering; weak on
noisy text
[15] Abdrakhmanov et al. Twitter/Facebook Offensive lang. ML Eng. Good pre-filter; no intent classification
[16] Carter Survey Questionnaire study Eng. Focuses on coping strategies; no automation
[17] Sayed et al. Twitter/Reddit NLP + multi-class ML Eng. Labels target group; needs larger dataset
• Sarcasm and Figurative Language — Most models
misclassify sarcastic remarks as non-bullying.
• Low-Resource Languages — Datasets for many regional
languages are scarce.
• Code-Mixed Content — Social media often blends
multiple languages in one post.
• Multimodality — Limited integration of images, videos,
and text for holistic detection.
• Real-Time Detection — High computational cost of
deep models hinders deployment.
• Explainability — Black-box models reduce trust among
users and moderators.
• Ethics and Privacy — Risk of bias and misuse of
detection systems.
V. PROPOSED METHODOLOGY
The proposed system is an adaptive, modular cyberbullying
detection architecture for English, Malayalam, and Manglish
content. It processes social media data (posts, comments,
shares, mentions, etc.) and extracts rich features on content,
context, user behavior, sentiment, and threat levels. The core
models are transformer-based (DistilBERT for English, IndicBERT for Malayalam, and XLM-R for cross-lingual codeswitching) augmented with LoRA adapter modules for efficient domain adaptation. LoRA adapters freeze the base model
weights and inject trainable low-rank matrices, reducing finetuning parameters significantly while preserving performance.
Multiple LoRA “experts” are specialized for phenomena such
as slang, insults, and emojis, and a lightweight gating controller (<200 KB) routes each input to the appropriate experts.
This modular design supports fast prototyping, high accuracy,
privacy, cost-efficiency, and robustness to evolving language
use.
A. Data Collection
Training data is harvested from diverse social media
platforms (e.g., Twitter, Facebook, YouTube) via APIs and
crawlers, ensuring coverage of English, Malayalam script, and
Romanized Malayalam (Manglish). We collect user-generated
text (posts, comments, replies, stories) along with associated
metadata (timestamps, thread context, user profiles). Data is
labeled for cyberbullying/harassment versus benign content.
Language detection and filtering are applied to construct
balanced corpora in each target language or code-mixed form.
Privacy-preserving measures are followed in accordance with
platform policies.
B. Preprocessing and Feature Extraction
Raw text undergoes preprocessing, including lowercasing,
punctuation removal, URL/mention filtering, and Unicode
normalization. Manglish text is transliterated into Malayalam or standardized English tokens. Tokenization is performed using the appropriate transformer tokenizer (WordPiece for DistilBERT/XLM-R, ALBERT tokenizer for IndicBERT). Emojis and emoticons are mapped to embeddings
(e.g., Emoji2Vec).
Feature sets extracted include:
• Content-based features: embeddings, n-grams, profanity/insult lexicon matches.
• Contextual features: conversation thread position, engagement metrics, temporal posting behavior.
• User-behavioral features: frequency of offensive language, historical abuse record, posting patterns.
• Sentiment features: polarity and emotion scores from
pretrained sentiment classifiers.
• Threat score: heuristic indicators based on insults, profanity strength, and direct targeting.
C. Model Architecture and LoRA Adaptation
The base models include DistilBERT for English, IndicBERT for Malayalam, and XLM-R for cross-lingual text.
Each is enhanced with LoRA modules for parameter-efficient
fine-tuning.
Several LoRA experts are defined:
• Slang/Code-switching expert,
• Insult/Harassment expert,
• Emoji and Sarcasm expert.
A lightweight router dynamically selects experts per input.
This gated mechanism is inspired by mixture-of-LoRA approaches, enabling instance-specific adaptation. Final transformer outputs, concatenated with contextual/user features,
feed into a classification head that predicts cyberbullying probabilities. As shown in Fig. 1, the proposed model architecture
follows a structured pipeline starting with data preprocessing,
feature extraction, integration with an LLM, and classification
into cyberbullying or non-cyberbullying content.
D. Federated Distillation for Privacy
To preserve privacy, we simulate federated knowledge distillation. Clients train local LoRA experts on private data and
share only soft predictions on a proxy dataset with the server.
The server aggregates predictions to update global experts
without direct access to raw user data. This maintains privacy
while ensuring global adaptability.
E. Training and Optimization
Models are fine-tuned using cross-entropy loss with frozen
backbone weights. LoRA ranks and learning rates are tuned
for efficiency. For deployment, we apply INT8 quantization
and convert models to ONNX/TFLite formats, reducing model
size and inference latency. Operator fusion further speeds
inference, enabling real-time classification on edge devices.
F. Evaluation
Evaluation is performed on multilingual, code-mixed
datasets, stratified by language and time period. Metrics include accuracy, precision, recall, F1-score, latency, and memory footprint. Robustness is tested under linguistic drift by
evaluating on emerging slang and unseen insults. Comparisons
with fully fine-tuned BERT baselines highlight the proposed
method’s efficiency and adaptability.
Data Collection
(Posts, Comments, Shares, Mentions, Reels, Stories)
Preprocessing
Normalization; LangID (En/Ml/Manglish);
Tokenization/Transliteration; Metadata
Feature Extraction
Content; Context; User Behavior;
Sentiment; Threat Score
Backbone Encoders
DistilBERT (EN), IndicBERT (ML), XLM-R (X-lingual)
Lightweight Router (<200 KB)
Select 1–2 LoRA Experts
LoRA Experts
Slang/Code-switching
Insults/Harassment
Emoji/Sarcasm
Classifier Head
Cyberbullying Probability
Deployment
INT8 Quantization + ONNX/TFLite
Output
Bullying / Non-Bullying
+ Sentiment/Threat
Federated
Distillation
Clients share
soft labels
Server aggregates
& updates experts
Adaptive Modeling
Fig. 1. Pipeline of the proposed multilingual cyberbullying detection system
with adaptive LoRA experts and federated distillation.
G. Research Priorities
Future directions include:
1) Enhancing data augmentation for low-resource languages.
2) Expanding adaptive expert modules for emerging social
media phenomena.
3) Improving privacy-accuracy trade-offs with advanced
federated distillation.
VI. CONCLUSION
This study reviewed the evolution of cyberbullying detection approaches and highlighted their strengths and limitations. While earlier research progressed from keywordbased methods to transformer-driven architectures like BERT
and RoBERTa, challenges remain in handling multilingual,
code-mixed, and multimodal social media content, as well as
ensuring fairness and explainability. To address these gaps,
the proposed methodology integrates comment-level semantics, contextual cues, and user behavior with a lightweight
transformer framework enhanced through techniques such as
Low-Rank Adaptation (LoRA) and Federated Distillation. This
design aims to improve accuracy, adaptability, and privacypreserving deployment across platforms. The approach is
expected to contribute toward building more generalizable and
ethically responsible cyberbullying detection systems that can
operate in real-world environments. Future work will focus
on large-scale experimental validation, multimodal integration,
and enhancing model interpretability to ensure both effectiveness and trust in safeguarding digital communities.
REFERENCES
[1] Elsafoury, F., Pervez, Z., Katsigiannis, S., and Ramzan, N.
(2021). When the timeline meets the pipeline: A survey on automated cyberbullying detection. IEEE Access, 9, 106788–106817.
https://doi.org/10.1109/ACCESS.2021.3098979
[2] Li, Z., Zhang, H., Mu, D., and Guo, L. (2016). Random time delay
effect on out-of-sequence measurements. IEEE Access, 4, 7862–7873.
https://doi.org/10.1109/ACCESS.2016.2610098
[3] Mahmud, T., Ptaszynski, M., Eronen, J., and Masui, F. (2023). Cyberbullying Detection for Low-resource Languages and Dialects: Review
of the State of the Art. Feedback. arXiv preprint.
[4] Shah, V., Sinha, A., Navalkar, N., Gupta, S., Gonsalves, P., and
Malik, A. (2023). ML and natural language processing: Cyberbullying detection system for safer and culturally adaptive digital communities. Journal of Sensor and Internet of Things, 7(2), 115–126.
https://doi.org/10.2478/jsiot-2023-0020
[5] Islam, M. M., Uddin, M. A., Rahman, R., Akhter, A., and Acharjee, U.
K. (2021). Cyberbullying detection on social media platform: Machine
learning based approach. Jagannath University Journal of Science, 10(1),
67–77.
[6] Perera, A., and Fernando, P. (2024). Cyberbullying detection system
on social media using supervised machine learning. Procedia Computer
Science, 234. https://doi.org/10.1016/j.procs.2024.06.200
[7] Zhang, P., Gao, Y., and Chen, S. (n.d.). Detect Chinese cyber bullying
by analyzing user behaviors and language patterns. Shanghai Jiao Tong
University.
[8] Shekhar, A., and Venkatesan, M. (n.d.). A bag-of-phonetic-codes model
for cyber bullying detection in Twitter. National Institute of Technology,
Karnataka.
[9] Pradheep, T., Yogeshwaran, T., Sheeba, J. I., and Devaneyan, S.
P. (2017). Automatic multimodel cyberbullying detection from social
networks. Proceedings of the International Conference on Intelligent
Computing Systems (ICICS 2017). Elsevier SSRN eLibrary – Journal
of Information Systems and eBusiness Network.
[10] Nikitha, G. S., Shenoy, A., Chaturya, K., Latha, J. C., and Shree, J.
M. (2024). Detection of cyberbullying using NLP and machine learning
in social networks for bi-language. International Journal of Scientific
Research and Engineering Trends, 10(1). ISSN 2395-566X.
[11] Teoh, H. T., and Varathan, K. D. (2023). Cyberbullying detection in social networks: A comparison between machine learning
and transfer learning approaches. IEEE Access, 11, 49955–49968.
https://doi.org/10.1109/ACCESS.2023.3275130
[12] Kumar, Y. J. N., Vanapatla, R. R., Pinamoni, V. K., Kandukuri, J., Almusawi, M., Aravinda, K., Kansal, L., and Kalra, R.
(2024). Detecting cyberbullying in social media using text analysis
and ensemble techniques. E3S Web of Conferences, 507, 01069.
https://doi.org/10.1051/e3sconf/202450701069
[13] Muneer, A., Alwadain, A., Ragab, M. G., and Alqushaibi, A.
(2023). Cyberbullying detection on social media using stacking
ensemble learning and enhanced BERT. Information, 14(8), 467.
https://doi.org/10.3390/info14080467
[14] Desai, A., Kalaskar, S., Kumbhar, O., and Dhumal, R. (2021). Cyber
bullying detection on social media using machine learning. ITM Web of
Conferences, 40, 03038. https://doi.org/10.1051/itmconf/20214003038
[15] Abdrakhmanov, R., Kenesbayev, S. M., Berkimbayev, K., Toikenov, G.,
Abdrashova, E., Alchinbayeva, O., and Ydyrys, A. (2024). Offensive
language detection on social media using machine learning. International Journal of Advanced Computer Science and Applications, 15(5).
https://doi.org/10.14569/IJACSA.2024.0150520
[16] Carter, M. A. (2013). Protecting oneself from cyber bullying on social media sites – A study of undergraduate students. Procedia – Social and Behavioral Sciences, 93, 1229–1235.
https://doi.org/10.1016/j.sbspro.2013.10.020
[17] Sayed, F. R., Elnashar, E. H., and Omara, F. A. (2025). Cyberbullying
detection in social media using natural language processing. Scientific
African, 21, e02713. https://doi.org/10.1016/j.sciaf.2025.e02713
