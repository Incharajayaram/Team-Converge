**Deepfake Detector Improvement Plan**
### Rigorous, Source-Grounded Research and Action Plan (LaDeDa-style Patch Detector)




**Contents**

1. [**Executive Summary**](#_bookmark0)	**2**
1. [**Prioritized 2-Week Experiment Plan**](#_bookmark1)	**9**
   1. [Week 1: Reproduction & Debugging](#_bookmark2)	9
   1. [Week 2: Model Improvements & Evaluation](#_bookmark3)	11
1. [**Decision Tree for Next Steps**](#_bookmark4)	**14**
   1. [Mermaid Flowchart (Preserved Verbatim)](#_bookmark5)	14
   1. [Explanation (Preserved in Full)](#_bookmark6)	15
1. [**Bibliography**](#_bookmark7)	**15**


































1




![ref1]![ref2]![](Aspose.Words.77b86b1c-925d-480a-a279-bb5a64e81a9a.003.png)
1. # <a name="executive summary"></a><a name="_bookmark0"></a>**Executive Summary**
We have identified three key drivers behind the gap between our detector’s high offline AUC and its poor real-world performance: (1) *Preprocessing mismatches* between training and de- ployment, (2) *Threshold calibration issues*, and (3) *Real-world distribution shifts*. In particular, our patch-based ResNet-50 (LaDeDa-style) model may be suffering from differences in image handling (resize, color space, face cropping) and compression artifacts that it never saw during training. Additionally, an inappropriate decision threshold or poor probability calibration can lead to excessive false alarms or missed fakes in production. Finally, new types of fake images and degradations in-the-wild (e.g. social media recompression, screenshots, or phone recaptures) mean our model is encountering out-of-distribution inputs, undermining its accuracy despite high lab AUC.

Immediate actions (Part A): We will rigorously debug the preprocessing pipeline to ensure it exactly matches training. This includes verifying image orientation (EXIF), color channel order, scaling method (e.g. same interpolation kernel), normalization, face cropping vs full-frame input, patch splitting geometry, and how patch scores are pooled and thresholded. We will follow a reproducibility checklist to align every step. Next, we will calibrate the model’s scores using a representative validation set from the deployment domain. We’ll apply temperature scaling or Platt scaling to fix any over/under-confidence, then choose an operating threshold that meets our product’s risk tolerance – for example, preferring lower false positives if false alarms are costly. We will plot reliability curves and Detection Error Tradeoff (DET) curves to visualize performance across thresholds. Finally, we will establish a realistic evaluation protocol: testing on “in the wild” images (e.g. social media-sourced, various devices), using time-split data (to detect drift), and applying adversarial transformations (resizing, recompression, blurring, etc.) to simulate attacker or distribution shifts. A quick “smoke test” of a few known images through the whole system (from upload to prediction) will be used to catch any pipeline bugs early.

Long-term improvements (Part B): We will explore modern deepfake detection architectures that promise better generalization to new fakes, including diffusion-generated images. Based on recent literature, we’ll investigate: (a) advanced patch-based models (like LaDeDa and variants) with smarter pooling (top-*k* or attention) to focus on the most telling local artifacts;

(b) frequency-domain methods that detect upsampling artifacts or spectral inconsistencies (e.g. NPR for neighboring pixel patterns, and others leveraging DCT/FFT features); (c) diffusion- robust detectors like DRCT and DiffusionFake, which explicitly train on diffusion model outputs to remain effective on new AI generators; (d) “universal” detectors that use large pretrained embeddings (e.g. CLIP-based models) or one-class approaches to detect unseen generators. We will also consider model compression (like distilling our model into a lightweight student as done in Tiny-LaDeDa) for real-time or edge deployment without sacrificing much accuracy. Additionally, implementing an out-of-distribution rejection mechanism (allowing the model to abstain when it’s uncertain or the input is far from training data) will improve reliability on novel inputs.

Guardrails and monitoring (Part C): To bolster reliability, we propose a hybrid system where simple heuristics act as guardrails around the ML model. These include checks like face detection confidence (ensure a clear face is present before trusting the deepfake verdict), image quality filters (if an image is too low-resolution or heavily compressed, flag it as uncertain rather than output a likely incorrect judgment), and perhaps camera metadata checks (e.g. if an image format or metadata is highly indicative of synthetic origin or tampering). These heuristics won’t replace the deep network, but they can prevent obvious failure modes – for example, they can stop the system from classifying a non-face or a cartoon as “deepfake” by mistake. Finally, we outline a monitoring plan: once deployed, we will continuously log the distribution of model scores and other features (in a privacy-preserving way) to detect drift. If the proportion of images flagged fake changes dramatically or new clusters of “real” images

begin to look anomalous in feature space, we will investigate. We plan periodic re-calibration and model updates using fresh data, and will implement a feedback loop (e.g. allowing manual review of certain cases and using those findings to update our training set). All data handling will respect privacy – e.g. by only collecting images with user consent or using on-device aggregation – focusing on summary statistics or embeddings rather than sensitive content.

Outcome: With these steps, we target a stable accuracy *≥* 80% in realistic settings (balanced

with precision/recall as needed). Just as importantly, we aim for a well-calibrated detector oper- ating at the right threshold for our needs – minimizing critical mistakes (whether false positives or negatives) rather than chasing a single aggregate metric. We acknowledge that accuracy will always be somewhat unstable as attackers adapt and content evolves. Therefore, we’ll optimize for a specific operating point and maintain flexibility to adjust as needed. In practice, this means defining clear success criteria beyond raw accuracy (e.g. “at most 5% false positives on real user images” or “detect 90% of known fake attempts”). Our plan culminates in a decision tree that guides next steps based on observed outcomes: if simple fixes (calibration/preprocessing) resolve the gap, we stick with the current model; if not, we escalate to data augmentation or retraining; if distribution shift is severe, we move to a more advanced architecture or collect new data, etc. By following this systematic plan, we will improve our deepfake detection system’s reliability and be prepared to respond quickly as the threat landscape changes.


Table 1: Evidence Table: Relevant Methods, Papers, and Benchmarks (with citations preserved verbatim)












**PatchForensics (Chai et al.)**








**Top-***k* **Pooling & LFM (Local Fo- cus. Mechanism)**



2020 (ECCV)








2025 (BMVC)



Early patch-based ap- proach; train on limited receptive field like 32*×*32 patches. Labeled each patch with the image’s label to learn “forensic patches”. Highlights that not all patches are equally informative (some fake patches look real).

Instead of simple avg pool- ing of patch scores, use Top-*K* pooling: only the

can detect fakes.
||||||||
| :- | :-: | :- | :- | :- | :- | :- |
||||||||
||||||||
||||||||
||||||||
||||||||
||||||||
||||||||
||||||||


![](Aspose.Words.77b86b1c-925d-480a-a279-bb5a64e81a9a.004.png)First to show patch-level training can improve gen- eralization by focusing on forgery cues that repeat locally. Addresses some dataset biases by mak- ing model look at small regions.

Tackles scenario where a few patches contain all evidence of the fake



Faces/images (needed image-level labels; patches ex- tracted uniformly).






Faces (primarily) – uses patch-based CNN features.



Decent X-dataset generalization but lower than LaDeDa. (Reported ~74–95% AUC on various sets.) LaDeDa outperforms PatchFor by using knowledge distillation to weight patches adaptively.





+3.7% accuracy vs. NPR on a 28-generator benchmark (95.9% vs 92.2%). Achieves 1789 FPS on A6000 GPU (fast inference). Needs



Likely (paper provides pseu- docode; code not public?).






Maybe (paper hints imple- mentation;

*K* most “fake” patches con- (e.g. an eye or bound-

tuning (*K* and dropout) to be robust.

not confirmed

tribute. Coupled with a Salience Network (SNet) to predict patch importance and regularizers to avoid

ary). Improves sensitivity to subtle artifacts while maintaining generaliza- tion via regularization

public).

overfitting to those patches. (to not always pick same

![ref3]patches).



*Continued on next page*


![](Aspose.Words.77b86b1c-925d-480a-a279-bb5a64e81a9a.006.png)

![ref4]













|||||||
| :- | :-: | :- | :- | :- | :- |
|||||||
|||||||
|||||||
|||||||
|||||||
|||||||
|||||||
|||||||
|||||||
|||||||

![](Aspose.Words.77b86b1c-925d-480a-a279-bb5a64e81a9a.008.png)“grid” artifacts from upsam- architecture-level flaws



**FreqNet** 30†L307- L314] (Frequency- Aware Detection)









**LGrad (Learning**



2024 (AAAI)










2023

pling.

Trains the model in the Fourier domain: explicitly feeds frequency spectrum features (using FFT) to learn differences in fre- quency content between real and fakes. Introduces random frequency masking during training to encour- age robustness. Often com- bined with spatial branch (hybrid).

Instead of raw pixels, train

rather than specific con- tent.

Targets frequency arti- facts (periodic patterns, missing high-frequency noise) that many GAN images exhibit. Aims to improve robustness to vi- sual appearance changes by focusing on underlying frequency discrepancies.



General artifacts: By fo-



General images (es- pecially faces); re- quires converting images to frequency representations.







Faces (initially), but



Improves cross-generator accuracy – reported to beat many spatial methods on unseen data. In benchmarks, ~78% mean accuracy across diverse sets (slightly below NPR).

Often complements spatial models (ensemble yields better results).






Strong generalization across GANs (80–89%



Likely (arXiv

\+ code link).










Yes (openac-

**on Gradients)**

(CVPR)

on image gradients or high-

cusing on gradients, it

applicable to any im- Acc on many unseen sets). Struggles a bit on

cess link).

pass filtered images. The idea is to emphasize fine details and remove color/- texture that could overfit. Essentially, model “sees” the edges and noise pat- terns rather than content.

highlights subtle differ- ences like noise residuals, blurriness or up-sampling footprints that generative models can’t reproduce perfectly. Helps generalize to new fakes by ignoring semantic content.

ages. Preprocessing: compute gradient maps before feeding CNN.

diffusion (e.g. in one benchmark, ~63–70% on SD models). Often used in combination (feature fusion with other methods).





![ref3]*Continued on next page*


![](Aspose.Words.77b86b1c-925d-480a-a279-bb5a64e81a9a.009.png)

**Method / Paper	Year	Core Idea	Addresses	Inputs	Reported Results (Gen. = generaliza-**

![ref3]**tion)**

**Code Avail.?**

**Universal Fake Detector (Uni- vFD)** – Ojha et al.










**FatFormer**

2023 (CVPR)












2024

No deep learning training for real-vs-fake. Instead, use a huge pretrained Vi- sion Transformer (CLIP ViT-L/14) as a feature extractor. Then perform a simple classification in

that feature space (e.g. lin- ear classifier or nearest- neighbor). The key is the feature space isn’t biased to any generative method since CLIP was not trained for deepfakes.

Fine-tunes a CLIP-based

Cross-model generaliza- tion: avoids overfitting to artifact patterns of one generator. CLIP’s fea- tures capture both high- level and some low-level info from 400M images, providing a robust basis. Especially helpful to de- tect diffusion fakes which might lack the obvious GAN spectral artifacts.


General & high-accuracy

General images (faces, scenery, any- thing – CLIP is generic). Needs a small labeled set from one generator to build classifier, then detects others.






General images

Achieved strong generalization: e.g. training on ProGAN and testing on various GANs/d- iffusions, improved AP by up to +37 mAP over a standard ResNet trained in same set- ting. In one benchmark, ~78.5% mean Acc across many generators (close to SOTA). Par- ticularly good at not misclassifying diffusion fakes as real.






One of the top performers on broad bench-

Yes (GitHub).












Yes (arXiv &

**(Forgery-Aware**

(CVPR)

ViT on a diverse fake image detection: Designed to

(tested on face

marks: e.g. 85.6% mean accuracy on 16-

code link).

**Transformer)**

set with adaptive tokens

excel on both seen and	datasets and others). model image benchmark. Nearly perfect on

to focus on forgery cues.	unseen generators by com- Requires substantial many GAN categories (95–99%), slightly

Introduces specialized trans- bining pretrained diver-

computational re-

weaker on some diffusion (e.g. ~56–68% on

former layers to emphasize features indicative of AI- generation while leveraging CLIP’s general knowledge. Essentially a large ViT tai- lored to deepfake detection.

sity (CLIP) with forgery- specific adaptation. Ad- dresses both local artifacts and global context via transformer self-attention.

sources (large ViT).

Midjourney/StableDiffusion). Large model (~300M params), so slower.




![ref3]*Continued on next page*


![](Aspose.Words.77b86b1c-925d-480a-a279-bb5a64e81a9a.010.png)

**Method / Paper	Year	Core Idea	Addresses	Inputs	Reported Results (Gen. = generaliza-**

![ref3]**tion)**

**Code Avail.?**

**Diffusion-specific**

2024	DRCT: Uses diffusion mod-

Addresses the gap in de-	General images (de-

DRCT: +10% accuracy on cross-diffusion

Possibly

**training (e.g.**

(ICML / els to reconstruct images

tecting diffusion-generated pending on tech-

tests when augmenting a detector with this

(DRCT an-

**DRCT, Diffu-**

NeurIPS) and create “hard nega-

images, which often lack	nique: DiffusionFake

training. DiffusionFake: significantly im-

nounced code;

**sionFake)**

tives” – if a fake image is reconstructed by a dif-

obvious GAN-style ar- tifacts. By training on

was demonstrated on proved cross-domain generalization of multi- face forgeries; DRCT ple base detectors (exact numbers depend on

Diffusion- Fake code on

fusion model, it becomes even more realistic; the detector is trained to dis- tinguish these extremely subtle fakes via contrastive learning. DiffusionFake:

fakes that are very hard to discern (almost real), the detector learns more general cues. Improves robustness so that unseen diffusion models (Midjour-

uses any images). Requires access to diffusion model(s) during training.

architecture, but consistently higher AUROC on unseen fakes). These methods maintain performance on GAN fakes while boosting diffusion detection.

GitHub).

runs a guided diffusion (e.g. ney, DALL-E) are caught.

Stable Diffusion) backwards on images to reveal fea- tures, forcing the detector to focus on artifacts that persist through diffusion inversion.

**Out-of-**

2025	Rather than a binary real/- Tackles open-set deepfake

Faces (in research so

DLED outperforms standard classifiers by

Likely re-

**Distribution Detectors (e.g.**

(preprint) fake classifier that is forced to guess even on unknown

detection – the reality that new forgery tech-

far; concept can ex- tend to images). Of-

~20% in detecting novel fake categories (open-set scenario) while keeping similar

search code (not widely

**DLED, Open- Set)**

fake types, these methods add an uncertainty estima- tion. DLED (Dual-Level Evidential Detector) col- lects evidence from two streams (spatial CNN and frequency) and produces not only a classification but also an uncertainty score.

If the input is unlike any- thing seen (novel fake type), uncertainty is high and the system can abstain or flag

it as “unknown fake”.

niques appear which the model wasn’t trained on. By allowing an “I’m not sure” output, it avoids confidently misclassifying a new deepfake as real (or a real image as fake) when out of its training distribution.

ten requires training on real data only or with labeled known fakes and treating others as anomalies.

accuracy on known fakes. In closed-set (stan- available yet). dard) evaluation, adding uncertainty does

not hurt performance and provides a reliabil- ity estimate. Some overhead in computing evidential outputs.










![ref3]*Continued on next page*


![](Aspose.Words.77b86b1c-925d-480a-a279-bb5a64e81a9a.011.png)

**Method / Paper	Year	Core Idea	Addresses	Inputs	Reported Results (Gen. = generaliza-**

![ref3]**tion)**

**Code Avail.?**

**Deepfake Datasets & Benchmarks** (DFDC, FF++,

etc.)

2018–

2025

DFDC (Facebook Deep- Fake Detection Challenge): 100k videos (mostly face swaps) from 3,426 actors, with multiple generation methods. Largest video deepfake dataset (~470

GB of clips). FaceForen- sics++ (FF++): 1000 real videos, each manipulated by 5 methods (Deepfakes, FaceSwap, Face2Face, Neu- ralTextures, FaceShifter)

to give 1000 fakes; in- cludes multiple compres- sion levels. Celeb-DF v2: 590 real celeb interview videos, 563 high-quality face swap fakes (from im- proved DeepFake algorithm

\+ post-processing) – very realistic fakes, no obvious artifacts. DeeperForensics- 1.0: DFDC-like but with controlled perturbations (e.g. add camera noise)

to simulate real-world. WildDeepfake (2021):

~707 videos from internet (selfie videos, etc., with spontaneous deepfakes) for test. WildRF (2024):

![ref4]5,000 images scraped from Reddit/Twitter/Facebook (mixed genuine and AI- generated) – reflects social media distribution (various resolutions, re-encodings, etc.). DFWild-Cup 2025: Competition dataset mixing 8 public sets (like above) and newly generated fakes; emphasizes diverse “in-the- wild” images.

These datasets provide training and evaluation data. Key to generaliza- tion is training on diverse sources and evaluating on truly novel data. Caveats: FF++ has a JPEG/PNG mismatch between real and fake that detectors could exploit; Celeb-DF’s real vs fake may have sub- tle editing differences; many datasets are re- stricted to research use.

DFDC required consent of actors and is available under controlled access. Bias: Some sets have mostly one demographic or staged scenes (FF++ has mainly young celebri- ties in stable lighting), which can bias detectors.

–		Use in our plan: We will leverage FF++ and Celeb-DF as initial training (they’re bal- anced and widely used), but will validate on harsher tests like WildDeepfake or DFDC. We must be careful with licensing – e.g., DFDC can be used for research but not com- mercial unless terms met. For “in-the-wild” eval, we can use a portion of DFWild-Cup data or WildRF if accessible to see real-case performance. Each dataset’s known issues (compression differences, etc.) will inform augmentation choices (e.g., simulate JPEG artifacts during training to avoid learning dataset-specific cues).

N/A (data).


![](Aspose.Words.77b86b1c-925d-480a-a279-bb5a64e81a9a.012.png)




![ref1]![ref2]![](Aspose.Words.77b86b1c-925d-480a-a279-bb5a64e81a9a.013.png)
1. # <a name="prioritized 2-week experiment plan"></a><a name="_bookmark1"></a>**Prioritized 2-Week Experiment Plan**
   1. ## <a name="week 1: reproduction & debugging"></a><a name="_bookmark2"></a>**Week 1: Reproduction & Debugging**
#### **Day 1–2: Verify Preprocessing Pipeline**
Re-run a small batch of training images through the exact training preprocessing code and compare with what happens in the deployed front-end. Check: image reading (color channel order – e.g. OpenCV BGR vs PIL RGB), resizing method (bilinear vs area, etc.), any cropping (face detection alignment vs none), patch extraction. We will create a checklist including:

0. Deterministic resize (same resolution and interpolation as training),
0. Consistent normalization (e.g., if model was trained on ImageNet mean/std, ensure front- end does same),
0. Face detection/cropping consistency (since our data were mostly face crops, ensure the deployment also crops faces similarly and not, say, feeding full images with background; if full images are input, consider cropping faces for the model or retrain accordingly),
0. Patches: confirm how patches are generated in code (size, stride). Our model likely expects a fixed size (e.g. 224*×*224) and splits into say 7*×*7 patches internally – we will ensure any custom patch code in deployment matches this exactly.
0. Pooling and output: ensure the final image score is computed the same way (if the model expects to average patch logits vs. take max, etc.). Misapplying max instead of average, or using logits when model expects sigmoid output, could cause huge discrepancies.

**Checkpoint:** Run a smoke test: take ~10 images (a couple of real, fake from validation) and run through two paths – (A) our training pipeline + model offline, and (B) the deployed pipeline. Compare outputs and intermediate steps. If any discrepancy is found (e.g., image brightness off, orientation rotated), fix it and document it. We anticipate possibly finding differences especially in image scaling or normalization (common culprit for models flagging everything wrong).

#### **Day 3: Resolve Mismatches & Retrain if Needed**
If the above step finds a serious mismatch (e.g., deployment is feeding full frames while model was trained on tight face crops), decide on a fix. For example, if full-frame is desired in deploy- ment, we might modify the model or preprocessing to handle it: possibly retrain on uncropped images or implement an on-the-fly face crop in the pipeline. We will perform a small experiment: feed full-frame vs. cropped images to the current model to quantify performance difference. If huge, then plan to retrain or fine-tune the model with the preferred preprocessing (e.g., include background). If minor, adjust the pipeline accordingly. Also, test screenshot or recompressed images (simulate by taking a known image, JPEG compress it, or screenshot it) to see if the model’s prediction changes drastically. This can highlight if compression is causing flips. If yes, consider adding augmentation (JPEG compression, slight blur) in retraining to make the model robust.

**Checkpoint:** Confirm that after alignment, when deploying on the validation set (the same data used offline), we get similar AUC/accuracy. If our offline AUC was high (~0.95+), the deployed pipeline on that data should now yield comparable AUC. Any residual gap indicates remaining bugs or numerical differences to iron out.

#### **Day 4: Calibration & Threshold Setting**
Using a deployment-like validation set (could be a mix of our original 2500 images and some new “real-world” samples if available), assess the raw score distributions. Plot a reliability diagram: bucket predictions into confidence bins vs. actual accuracy. Likely, the model is overconfident (many scores saturating near 0 or 1). We will apply temperature scaling on this val set (optimize a single temperature parameter to align predicted probabilities with observed

frequencies). Also compare with Platt scaling (logistic regression on the logit). Evaluate which yields better calibration (using metrics like ECE – Expected Calibration Error). Next, define our operating threshold. We’ll consider product needs: if false positives are very troublesome (e.g., flagging a real image as fake might have big consequences), we choose a threshold that yields a low false-positive rate (say *<* 1%). If missing a fake is worse, pick threshold for low false- negative. We will likely set up a ROC curve and choose the point that satisfies requirements (e.g., maximize TPR subject to FPR *< X*%). As guidance, we note many deployers choose a safety margin with multiple thresholds: e.g., *<* 30% fake score = “clearly real”, *>* 70%

= “clearly fake”, intermediate = “needs manual review”. We will adopt a similar strategy if applicable (since our use-case is a web tool, maybe we can present a “low confidence” category). **Checkpoint:** Decide on a preliminary threshold *T*<sub>0</sub> (or tiered thresholds). Document the rationale (e.g., “Set *T*<sub>0</sub> = 0*.*7 to limit false alarm rate to ~5% on val set, which gives ~90% recall

on fakes”). We will use this threshold going forward in week 2 tests.

#### **Day 5: Initial Field Test (Internal)**
Before large-scale deployment, run a batch of “real-world” images through the calibrated model. This can include: photos taken from various phones, images from social media (with permission), a few known deepfakes from the internet (different generators than training data), and even some unrelated images (animals, art, etc., to test false positives on non-face content). Monitor the predictions:

0. Ensure our threshold isn’t triggering on obvious real images. If we see many false positives, note what those images have in common (e.g., low lighting, compression?).
0. Ensure we catch the known fakes. If some high-quality fakes still score low (missed), that’s an issue of generalization – note which generator or traits they have.
0. Test borderline cases: an image of a digital art or a game avatar (should ideally not be confidently classified as deepfake of a real person – if our model flags all CGI as fake, we might need an adjustment to not mislabel those since they’re not “face swaps” in the context).
0. If possible, include a phone screenshot of a fake vs real to mimic a user taking a screenshot; see if anything in pipeline (e.g. resizing twice) affects it.

**Checkpoint:** Summarize issues found. Likely outcomes: “Model is still over-flagging very low-quality real images as fake” – which suggests we may implement a quality heuristic or retrain with more compressed real data. Or “Model missed diffusion-art images entirely (scores them as real)” – suggests need for method upgrade or at least flagging uncertainty for unseen styles.

#### **Day 6–7: Analysis and Decision Point**
Gather everything from Week 1:

0. If the only problems are threshold and preprocessing, and after fixes the performance on real-world samples seems satisfactory (*≥* 80% accuracy with balanced error rates, or at least meeting our target FPR/FNR), then our gap was mostly a deployment issue, not fundamental. In this case, we might not need an architecture change immediately. We would proceed to implement the calibration and preprocessing fixes in production and then monitor (go to Week 2 – monitoring).
0. If there are signs of distribution shift that simple fixes didn’t cure (e.g., certain new deepfakes consistently fool the model or certain real images consistently get flagged), then we proceed to Method Upgrades in Week 2. For instance, if images from a new generator (say Stable Diffusion) are missed, we need to enhance generalization via retraining or a new model (Part B).

0. We also make a “go/no-go” call on whether to do a quick retraining of our ResNet with augmented data in Week 2 as an intermediate step. If calibration fixed most issues, skip retrain. If not, plan an experiment to fine-tune the model on a small new dataset (e.g., add 100 real photos with various distortions + 100 new fake images from diff generators) to see if that improves the specific failure modes.

**Milestone:** Review meeting at end of Week 1 – Present findings to stakeholders: “We fixed pipeline mismatches (X, Y) which improved consistency. With threshold = T, we achieve Z% accuracy on test. Remaining issues: e.g., model struggles with diffusion fakes (misses ~40%) and has 10% false positives on very low-quality reals. Plan: in Week 2, address these via advanced methods or additional training data.”

1. ## <a name="week 2: model improvements & evaluation"></a><a name="_bookmark3"></a>**Week 2: Model Improvements & Evaluation**
#### **Day 8–9: Data Augmentation and Minor Retraining**
(If needed) To tackle moderate distribution shift without a new model yet, try training tweaks:

0. Augment training data with degradation transformations reflecting deployment: random JPEG compression, resolution downscale, slight blur, color jitter. Since LaDeDa authors noted lack of JPEG in training can hurt generalization, we’ll incorporate this.
0. If missing diffusion fakes: incorporate a small set of diffusion-generated images (e.g., 200 images from StableDiffusion or MidJourney, half real half fake if possible) into training as an “unseen” class or just add to fakes. Alternatively, use hard negative mining: take some fake images the model currently misclassifies and add them to training with correct label.
0. Retrain or fine-tune our ResNet-50 patch model on this augmented data (maybe for a few epochs, not full restart). Evaluate on a validation that includes some new fakes and real.

**Checkpoint:** See if these changes improve the specific weaknesses:

0. Did false positives on compressed images drop (because model learned compression isn’t always fake)?
0. Did detection on diffusion images improve at all? (It might slightly, but a fundamentally different artifact might require new features – if still poor, that justifies a new architecture or using a known better method for diffusion, see next steps.)

#### **Day 10–11: Evaluate Advanced Model on Our Data**
Set up at least one candidate advanced detector in our environment to compare:

0. For example, run NPR or UnivFD on a sample of our data. Both have public code. We can use a pretrained NPR model (if authors released one trained on their big benchmark) and see how it fares on our validation. Or use the UnivFD approach: take CLIP and do a quick nearest-neighbor with a few of our training images. The goal is not a full production model yet, but to gauge potential gain: Does NPR catch those diffusion fakes our model missed? Does it avoid false positives on low-quality reals? For fairness, calibrate these models as well if needed.
0. Also test Tiny-LaDeDa if available, to see how much accuracy drop vs. speed gain. (If edge deployment is a priority, we might be willing to lose a few points of accuracy for 5*×* faster inference. Tiny-LaDeDa can be obtained by distilling our current model’s patch outputs; training it could be longer-term, but perhaps the paper’s weights can be used as a starting point.)
0. If possible, implement an uncertainty measure on our current model for OOD detection: e.g., use the maximum softmax probability as a confidence – many false predictions are lower confidence when out-of-distribution. Check if the wrong classifications had lower

confidence on average. If yes, we can exploit that (set up a threshold to abstain when model is unsure, at least in testing).

**Checkpoint:** Decide on “best next model”:

0. If our patched ResNet is now okay on all but a niche case, the conservative approach might suﬀice.
0. If NPR or CLIP-based approach clearly outperforms on our challenging cases (e.g., NPR correctly flags diffusion fakes that our model missed), consider moving to that architecture. However, note complexity: NPR is still ResNet-based and not too heavy – feasible. CLIP- based might be heavy (ViT-L/14) but we could try a smaller ViT or the FatFormer fine-tune approach if top performance is needed and we have compute.
0. Also consider the speed: if our usage scenario is a server with GPU, a large model is fine; if we want possible on-device, a smaller distilled model is better.
0. We will likely pick two: (1) a conservative model (our original with improvements or NPR which is similar complexity) to deploy ASAP, and (2) start R&D on an aggressive model (like transformer-based) for the future.
0. Outline the training needed for the new model: e.g., “Train NPR on our combined dataset (should be similar procedure to LaDeDa, patch-based) or fine-tune FatFormer on our data which may require a few days on a GPU cluster.”

#### **Day 12: Incorporate Guardrails**
Design the heuristic checks to integrate:

0. Implement a simple face detector check in the pipeline: if no face is found or face confidence

   *< X*, output “No face detected – cannot determine” or use a very high threshold to avoid false fake flag. (This prevents e.g. an animal photo or random noise triggering a false fake).

0. Image quality check: measure blurriness (variance of Laplacian) and JPEG quality (e.g., check if JPEG quantization tables are very low quality). Define rules: if an image is extremely blurred or compressed (beyond training conditions), perhaps do not automati- cally trust a “fake” prediction. Possibly label it as “Unreliable – image too low quality for analysis” or at least require a higher confidence to call it fake. We will simulate this: take a real image, compress to JPEG quality 10, see our model output. Likely it might score it as fake (since high-frequency artifacts are gone). The guardrail would catch “quality *<* threshold” and override to “unable to determine” or raise the threshold for that image’s decision.
0. Metadata or size heuristic: If an input image is exactly some typical AI-generated size (like 1024*×*1024 with no EXIF camera info), that itself is suspicious – but rather than auto-flagging it fake (which attackers could circumvent), we use it as a soft signal. For instance, for images that have camera EXIF data vs. those that don’t, we could have different thresholds (images lacking any camera metadata might be more likely fake, but also could be screenshots or edited). This is optional; we must avoid biases that cause false positives (some legitimate images might be edited and lose EXIF).
0. Multiple faces: If an image has more than one face, how does our model handle it? Possibly it might still classify (maybe averaging patches across faces). This could confuse results if one face is fake and others real. A guardrail could be: if multiple faces, handle each face separately (crop each face and run model). Our current system didn’t consider this. We add to plan: test a multi-face image and see output. If unstable, implement logic to process faces one by one and perhaps take the max “fake” score among them as the image result.

Implement these checks in a test harness and run our validation through it to ensure they don’t over-trigger. We might adjust thresholds (e.g., define blur threshold such that clearly

unusable images are caught but normal slight blur is not).

**Checkpoint:** Verify that guardrails would have prevented known failures:

0. E.g., in our Day 5 field test, we had some real images flagged fake mainly due to heavy compression. Would the new JPEG quality check have intervened? If yes, good.
0. Check that guardrails don’t undo true positives: e.g., if a deepfake image is also low-quality, we don’t want to always ignore it. So perhaps instead of outright skipping, we mark it with lower confidence and maybe ask for manual review if possible. We will document a policy: e.g., “If image *<* quality threshold and model says fake, tag as ‘ambiguous’ instead of fake”.

#### **Day 13: Final Evaluation on Deployment-like Scenario**
Simulate a deployment run with everything in place: calibrated model (possibly retrained or new model), chosen threshold, guardrails. Use a hold-out test set that mimics actual use: mixture of real photos (from devices, various content) and fake images from a variety of sources (some from our original test, plus some new if available: e.g., deepfake images the team manually generated or collected from web). Aim for at least a few hundred images if possible. Compute accuracy, but more importantly compute False Positive Rate (FPR) on reals and False Negative Rate (FNR) on fakes at our operating point, as these are what we promised to optimize. Also produce the DET or ROC curve to see trade-offs – ensure our operating point is appropriate. Check calibration on this test: are the output scores well-calibrated (if we say 90% fake, is it

~90% likely truly fake)? Good calibration means we can trust the scores for future threshold tuning. If not, do a last adjustment (maybe isotonic regression on the scores from val).

We will also specifically evaluate stability: does performance hold across subgroups?

0. Check by source: e.g., images from DSLR vs. screenshots vs. social media compressed – how is FPR for each? We want to ensure none of these is unacceptable.
0. Check by generator type: if possible, group fakes by which method (GAN-based vs diffu- sion) to see if we have a blind spot. Ideally, all are above 80% detected; if one category is much lower, note it as a risk (to address in future with more training data or specialized model).
0. If available, check bias: e.g., group results by demographic (male/female, skin tone if we have that info) to ensure our detector doesn’t disproportionately flag certain groups (prior work shows some detectors had bias). This might be limited by our dataset, but at least qualitatively ensure nothing obvious.

**Checkpoint:** Decide if criteria are met (*≥* 80% accuracy overall, and acceptable FPR/FNR as defined, e.g., FPR *<* 5% for reals, FNR *<* 20% for fakes or as per requirements). If yes, we are ready to update the production system. If not, identify which criterion failed:

0. If accuracy still *<* 80% because of certain diﬀicult fake types, we might need to integrate a second model in ensemble (e.g., use NPR and our model together, or an ensemble of frequency and spatial – but that could be a longer effort).
0. If false positives still high, perhaps tighten threshold more, sacrificing some sensitivity, or further improve guardrails.
0. We will document these and potentially decide to proceed with deployment with caution (if time-bound), while continuing research on those failure modes.

#### **Day 14: Deployment & Monitoring Setup**
Deploy the improved model (or keep it local if testing) on the front-end. Ensure telemetry is in place for monitoring:

0. We will log (privately and securely) the model’s score for each image and whether it was flagged or not (not the image itself, unless user consents, to respect privacy). We’ll also

log simple metadata like image resolution, file size, perhaps a hash to detect duplicates. This will help identify distribution changes (e.g., suddenly we see many more large images with borderline scores – maybe a new type of fake).

0. Implement a drift detection script: e.g., compute the moving average of fake scores, or a histogram of scores over the last *N* images per day. We expect mostly real images in usage, so if suddenly the fraction flagged fake goes up from, say, 1% to 5%, that could mean either a rash of fakes or a drift in input quality (maybe users started uploading more screenshots that trigger false positives). We set an alert for such shifts.
0. Also, prepare a periodic evaluation: every two weeks, sample some images (with permis- sion) that were flagged and not flagged and manually verify if possible. This can provide ground truth to recalibrate if needed (similar to having a “calibration set” updated).
0. Plan to retrain/calibrate every few months with collected data. Given privacy, we might implement an opt-in program where users can allow their images (especially those flagged as fake incorrectly) to be collected for improving the model. Alternatively, partner with a dataset provider to get fresh fake images as new generators emerge (e.g., if a new GAN release occurs, test it on our system promptly).

**Milestone:** At the end of Day 14 (which concludes 2 weeks), we should have:

0. The new model running on the local host site, using the chosen threshold and guardrail logic.
0. Documentation of its expected performance (e.g., “In our tests, ~90% of deepfakes are caught with ~5% false alarm rate; calibrated probabilities; known remaining weakness: X”).
0. A monitoring dashboard or procedure in place, so that as we go live, we can quickly catch if accuracy degrades (and execute our decision tree accordingly).

1. # <a name="decision tree for next steps"></a><a name="_bookmark4"></a>**Decision Tree for Next Steps**
   1. ## <a name="mermaid flowchart (preserved verbatim)"></a><a name="_bookmark5"></a>**Mermaid Flowchart (Preserved Verbatim)**





























![](Aspose.Words.77b86b1c-925d-480a-a279-bb5a64e81a9a.014.png)The following Mermaid code is preserved verbatim to maintain 100% semantic content:











![](Aspose.Words.77b86b1c-925d-480a-a279-bb5a64e81a9a.015.png)

1. ## <a name="explanation (preserved in full)"></a><a name="_bookmark6"></a>**Explanation (Preserved in Full)**
This decision tree outlines how we react to various situations post-deployment:

0. If the system is performing well (no significant issues), we continue monitoring (node B). We’ll do periodic recalibration and dataset updates but no drastic changes.
0. If we see a spike in false positives (node C), e.g., many real images being flagged, our first response is to adjust the threshold upward (node C1) and test if we still catch enough fakes. If that solves it without tanking recall, we deploy the new threshold. If raising threshold causes too many misses (node C2 “too many misses”), we revert and instead strengthen guardrails (node C4): for example, implement a rule that any “fake” decision on a low-quality image requires secondary confirmation (thus reducing false alarms from noisy data). This should reduce FPs without changing core model.
0. If we encounter missed new deepfakes (false negatives spike, node D), say a new generator spreads fakes that our model labels as real, we gather examples of these (D1→D2), update training data, and fine-tune the model (D3). After retraining, test it; if successful, deploy the updated model (D5). If our current architecture cannot handle the new fakes even after retraining (D4 “no improvement”), it triggers a shift to an advanced approach (D6): for instance, integrate a new detection model that is known to handle such fakes (this might involve a larger project – e.g., adopting a transformer model or an ensemble specialized for that type).
0. If we detect a distribution drift in inputs (node E) – e.g., users start uploading very dif- ferent images than before (non-face images, or images from a new platform with different compression), we diagnose it (E1). Depending on severity, we either adjust preprocess- ing/augmentation (E2) or if it’s a whole new domain, we might need to collect data from that domain and essentially re-calibrate or retrain from Part A steps (E3). In either case, we funnel into retraining (back to D3 path).
0. This loop ensures the system remains adaptive. In all cases, after changes we always return to monitoring (B) to verify the outcome.

Additionally, built into this tree is the notion of conservative vs. aggressive escalation: We first try threshold tweaks or minor retraining (low-cost). Only if those fail do we move to deploying a new architecture (which is higher cost and risk). Throughout, we ensure any change is validated on known data before full rollout.

1. # <a name="bibliography"></a><a name="_bookmark7"></a>**Bibliography**
1. Bar Cavia et al., “Real-Time Deepfake Detection in the Real-World.” arXiv preprint 2406.09398, 2024. Introduces LaDeDa (patch-based ResNet-50) and WildRF dataset; reports near-perfect lab results and highlights generalization gap to social media deepfakes.
1. Binh Le et al., “Why Do Facial Deepfake Detectors Fail?” ACM ICMR, 2023. Analyzes preprocessing effects: finds that inconsistent face cropping vs. resizing can drastically drop accuracy (98% → 61% under different crop). Also shows poor cross-dataset generalization (near 0% on Celeb-DF when trained on FF++ raw).

1. Brightside (A. Sall) – “Why Deepfake Detection Tools Fail in Real-World Deployment” (Blog, 2023). Discusses the lab-to-field accuracy collapse, citing compression, lighting, and attacker adaptation. Recommends threshold tuning for operational needs and multi- threshold review processes.
1. Jia-Xuan Chen et al., “A Single Simple Patch is All You Need for AI-generated Image Detection.” ACM MM 2024. Proposes detecting AI images using one discriminative image patch. Achieved high accuracy on diffusion images (as evidenced in benchmarks). Oﬀicial code on GitHub.
1. Chuangchuang Tan et al., “Rethinking the Up-Sampling Operations in CNN-based Gener- ative Network for Generalizable Deepfake Detection.” CVPR 2024. Introduces NPR fea- tures to capture upsampling artifacts, significantly improving detection of unseen GAN/d- iffusion images (mean accuracy 93.3% on 28 models). Code: chuangchuangtan/NPR- DeepfakeDetection.
1. Zhenhui Wu et al., “Frequency-Aware Deepfake Detection: Improving Generalizability through Frequency Space Learning.” AAAI 2024. (FreqNet) Utilizes frequency mask- ing and augmentation to enhance cross-generator performance. Shows robust results on various benchmarks (~79% mean accuracy).
1. Utkarsh Ojha et al., “Towards Universal Fake Image Detectors that Generalize Across Generative Models.” CVPR 2023. Uses CLIP ViT features and simple classifiers to avoid overfitting. Demonstrated strong detection of unseen diffusion fakes that confound standard CNNs. Code: WisconsinAIVision/UniversalFakeDetect.
1. Chandler Timm et al., “Towards Sustainable Universal Deepfake Detection with Frequency- Domain Masking.” arXiv 2025. Explores one-class and frequency masking for eﬀicient detection (mentions integrating OC-FakeDetect, SBI, etc.). Notably discusses pruning and scalability with minimal accuracy loss.
1. Md Sahidullah et al., “DFWild-Cup: Deepfake Face Detection In The Wild Competition (IEEE SP Cup 2025)” – Competition website. Describes a diverse dataset drawn from eight public sources and novel fakes, aiming to evaluate generalization. Baseline (MesoNet) EER ~15.6% on validation.
1. Brian Dolhansky et al., “The DeepFake Detection Challenge (DFDC) Dataset.” arXiv:2006.07397, 2020. Describes the construction of DFDC with over 100,000 videos (3,426 actors) and multiple generation methods. Found that models trained on DFDC generalized somewhat

   to wild videos. Dataset released for research by Facebook AI.

1. Andreas Rößler et al., “FaceForensics++: Learning to Detect Manipulated Facial Images.” ICCV 2019. Introduced a large-scale video deepfake dataset with multiple manipulation methods and compression levels. Widely used benchmark; highlight that models often overfit to compression artifacts if not careful. Available for research with license.
1. Yuezun Li et al., “Celeb-DF v2: A New Dataset for DeepFake Detection.” CVPR 2020. Provides high-quality celebrity face swap videos (with improved lip-sync and fewer visual artifacts). More challenging for detectors (many detectors that excel on FF++ drop on Celeb-DF). Available for academic use.
1. Zhengzhong Jia et al., “WildDeepfake: A Challenging Real-World Dataset for Deepfake Detection.” ACM MM 2021. Collected 707 deepfake videos “in the wild” from Internet; includes genuine and fake videos with diverse real-world perturbations. Used to test generalization beyond lab data – detectors show significant performance drop on this set, emphasizing need for augmentation and robust features.
1. Grover et al., “Calibration of Deep Neural Networks for Reliable Predictions.” ICML 2017. (Not specific to deepfakes, but seminal work on temperature scaling for calibration.) We apply these concepts in our calibration plan: temperature scaling to adjust output confidence.
1. Pawel Korus et al., “Deterministic Noise Patterns vs GAN-generated Images.” IEEE TIFS

2020\. (Representative of works using camera noise/PRNU to detect fakes.) Suggests real images contain sensor noise not present in GAN images, a heuristic we consider as a guardrail by checking metadata or noise consistency.

[ref1]: Aspose.Words.77b86b1c-925d-480a-a279-bb5a64e81a9a.001.png
[ref2]: Aspose.Words.77b86b1c-925d-480a-a279-bb5a64e81a9a.002.png
[ref3]: Aspose.Words.77b86b1c-925d-480a-a279-bb5a64e81a9a.005.png
[ref4]: Aspose.Words.77b86b1c-925d-480a-a279-bb5a64e81a9a.007.png
