# RuralMed Vision: Edge-Deployable Multimodal Triage with a Decoupled Explainability Pipeline for Low-Resource Clinical Settings

**Asmit Chatterjee**
Department of Computer Science and Engineering, Manipal Institute of Technology, Bengaluru, India

## Abstract

Access to dermatological triage in rural and low-resource settings is constrained by limited specialist availability and unreliable internet connectivity. We present RuralMed Vision, a parameter-efficient adaptation of Qwen2.5-VL-7B-Instruct for offline skin lesion severity triage, fine-tuned via 4-bit QLoRA on the HAM10000 dataset combined with HealthCareMagic-100k symptom dialogue. The fine-tuned model improves severity classification accuracy from 13.0% (zero-shot) to 85.0% on a held-out test set, and is quantized to a 4.36 GB GGUF artifact running at 49.1 tokens/sec on an 8GB consumer GPU. We additionally report two findings central to deploying generative VLMs in this setting: (1) standard autoregressive fine-tuning on a class-imbalanced dermatology corpus induces mode collapse toward the majority class, evidenced by systematic Nevi-to-Melanoma misclassification and severe confidence miscalibration (ECE = 0.43); and (2) the fine-tuned model's language head degrades under domain-specific dialogue data to the point that free-text rationales become unreliable, motivating a decoupled explainability architecture that separates the triage decision (local, fine-tuned, vision-only) from rationale generation (cloud API, general-purpose). We report all results, including negative findings, transparently to inform future low-resource clinical AI deployments.

## I. Introduction

Skin cancer and other dermatological conditions represent a significant diagnostic burden in regions with limited access to dermatologists. Vision-language models (VLMs) offer a path toward automated triage tools that can run on consumer hardware without requiring constant connectivity, but their deployment in clinical settings raises three practical questions that are often glossed over in proof-of-concept work: can a small adapter meaningfully shift a generalist model's behavior on a narrow clinical task; can the resulting model be compressed enough to run on hardware available in a rural clinic; and what happens to model reliability -- both in terms of decision quality and explainability -- once the adaptation is complete.

This paper addresses all three questions for the specific case of skin lesion severity triage. We fine-tune Qwen2.5-VL-7B-Instruct, a 7-billion-parameter open vision-language model, using 4-bit Quantized Low-Rank Adaptation (QLoRA) on a combined corpus of HAM10000 dermoscopic images and HealthCareMagic-100k patient-physician symptom dialogues. We evaluate the resulting model on a held-out, stratified test split of HAM10000 and quantize it to GGUF format for offline CPU/GPU inference.

Beyond the headline accuracy improvement, we report two findings that we believe are useful to the community precisely because they are not flattering. First, the fine-tuned model exhibits a strong mode-collapse pattern: the majority class (melanocytic nevi, "nv") attracts a large share of misclassifications from other classes, and the model's confidence is severely miscalibrated (ECE = 0.43), meaning it is consistently overconfident regardless of correctness. Second, we found that conditioning the fine-tuned model's own language head on the HealthCareMagic-100k dialogue data degraded its ability to produce coherent diagnostic rationales -- the outputs contained forum-style artifacts and occasional contradictions with the model's own classification output. Rather than presenting polished but misleading rationales, we adopt a decoupled architecture: the fine-tuned model performs the triage classification and an occlusion-based saliency map highlights the visual evidence, while textual rationales are generated post-hoc by a separate, general-purpose model (Qwen2.5-72B-Instruct via API) conditioned on the ground-truth diagnosis. We are explicit that these rationales are illustrative and post-hoc, not a faithful explanation of the triage model's internal reasoning.

Our contributions are:

- **C1 -- Edge-Optimized Deployment.** A QLoRA fine-tuning pipeline for a 7B VLM that fits training and inference within an 8GB consumer GPU, with a GGUF Q4_K_M quantized artifact (4.36 GB) achieving 49.1 tokens/sec, enabling fully offline operation on hardware realistically available in low-resource clinics.
- **C2 -- A Decoupled Explainability Pipeline.** A two-stage explainability design that separates visual saliency (local, model-faithful) from textual rationale generation (API-based, post-hoc), motivated by and transparently disclosing the degradation of the fine-tuned model's own language head under domain-specific dialogue fine-tuning.
- **C3 -- Mode Collapse as a Documented Limitation.** An empirical demonstration that autoregressive fine-tuning of a generalist VLM on a class-imbalanced clinical imaging dataset (HAM10000, 67% nevi) produces systematic majority-class collapse and severe overconfidence, quantified via confusion matrix analysis and Expected Calibration Error.

## II. Related Work

Skin lesion classification on HAM10000 has been extensively studied with convolutional and transformer-based architectures, with reported accuracies on the full 7-class diagnostic task ranging from approximately 87% to over 99% depending on architecture, preprocessing, and evaluation protocol. These approaches typically train task-specific networks from scratch or via transfer learning on frozen backbones, optimizing directly for the 7-class diagnosis objective with cross-entropy loss and class-balancing strategies.

Multimodal approaches combining dermoscopic images with patient metadata (age, sex, lesion location) have demonstrated that auxiliary clinical context improves classification accuracy and AUC-ROC over image-only baselines. This motivates our use of symptom-dialogue data alongside imaging data, though our integration mechanism -- joint fine-tuning of a generative VLM rather than a dedicated fusion architecture -- differs substantially from prior multimodal classifiers.

Separately, parameter-efficient fine-tuning methods such as LoRA and its quantized variant QLoRA have made it feasible to adapt large language and vision-language models on consumer hardware. While QLoRA has been widely applied to instruction-tuning and domain adaptation of LLMs, its application to a 7B VLM for a narrow, safety-relevant clinical classification task -- combined with end-to-end deployment as a quantized GGUF artifact -- is comparatively underexplored, particularly with transparent reporting of calibration and failure modes.

Our work differs from prior HAM10000 classifiers in framing: rather than optimizing for maximal 7-class diagnostic accuracy, we target a coarser 3-level severity triage task (LOW/MEDIUM/HIGH) appropriate for a non-specialist first-pass tool, and we report calibration and mode-collapse behavior that is typically absent from classification-accuracy-focused papers.

## III. Methodology

### A. Datasets

We use two datasets. **HAM10000** provides 10,015 dermoscopic images across seven diagnostic categories (melanocytic nevi, melanoma, benign keratosis, basal cell carcinoma, actinic keratosis, vascular lesions, and dermatofibroma), with a pronounced class imbalance: melanocytic nevi (nv) constitute approximately 67% of the dataset. Each of the seven diagnostic categories is mapped to one of three severity levels (LOW, MEDIUM, HIGH) based on clinical urgency, with melanoma and basal cell carcinoma mapped to HIGH, actinic keratosis and vascular lesions to MEDIUM, and the remaining classes to LOW.

**HealthCareMagic-100k** provides approximately 100,000 patient-physician dialogue pairs describing symptoms and clinical responses. A filtered subset relevant to dermatological presentations is paired with HAM10000 images during fine-tuning to provide textual symptom context, simulating the kind of free-text description a rural health worker might provide alongside a photograph.

For evaluation, the HAM10000 corpus is split using a fixed random seed (99, distinct from the training seed of 42) into a 70% training/validation pool and a 30% held-out test set (3,005 samples), stratified by diagnostic class to preserve the original class distribution. All reported test results are computed on samples drawn from this held-out set; the training seed and test seed never overlap.

### B. Model Architecture

The base model is Qwen2.5-VL-7B-Instruct, an open vision-language model combining a vision transformer encoder with a 7-billion-parameter autoregressive language decoder. The model accepts interleaved image and text tokens and produces text completions, which we constrain via prompt design to a structured JSON output containing a predicted condition name, a severity level (LOW/MEDIUM/HIGH), and a diagnostic code corresponding to one of the seven HAM10000 classes.

### C. Fine-Tuning Procedure

We fine-tune the base model using Quantized Low-Rank Adaptation (QLoRA) with 4-bit NF4 quantization via bitsandbytes. The base model weights are loaded in 4-bit precision with double quantization and bfloat16/float16 compute dtype; LoRA adapter matrices are trained in higher precision and inserted into the attention and projection layers of the language decoder. This configuration allows the full fine-tuning procedure to fit within the 8GB VRAM budget of a consumer-grade RTX 4060 GPU.

Training examples pair a resized lesion image (128x128 px) with a symptom description drawn from the filtered HealthCareMagic-100k subset (or a synthetic symptom template keyed to the ground-truth diagnostic class where dialogue data was unavailable for that class) and a target JSON completion containing the correct condition name, severity, and diagnostic code. The model is trained to produce this structured output directly, conditioned on the system prompt, image, and symptom text.

### D. Inference and Evaluation Protocol

At inference time, the fine-tuned model receives a system prompt instructing it to act as a medical image classifier, the resized lesion image, and a symptom description (drawn from a fixed per-class symptom lookup table during evaluation to ensure consistency across samples). The model generates up to 120 tokens, which are parsed as JSON to extract the predicted severity and diagnostic code; a fallback parser extracts a severity keyword from unstructured output if JSON parsing fails.

Evaluation is conducted on 200 samples drawn (with a fixed random seed) from the 3,005-sample held-out test set. We compare the fine-tuned model against the zero-shot base model (Qwen2.5-VL-7B-Instruct without any LoRA adapter) on the identical 200 samples, using the identical prompt and symptom lookup table, to isolate the effect of fine-tuning.

### E. Decoupled Explainability Pipeline

During development, we observed that prompting the fine-tuned model itself to produce free-text diagnostic rationales -- in addition to its structured classification output -- yielded outputs containing artifacts traceable to the HealthCareMagic-100k forum-style dialogue data (e.g., colloquial phrasing, references to unrelated symptoms) and occasional rationales that contradicted the model's own predicted diagnostic code. We attribute this to the language modeling head being jointly optimized for both the structured classification objective and the unstructured dialogue-style text in the training corpus, with the latter degrading coherence on the former when both are elicited from the same forward pass.

To address this without discarding explainability, we adopt a two-stage decoupled pipeline. First, an occlusion-based saliency map is computed directly from the fine-tuned model's vision encoder by masking image patches and measuring the resulting change in the model's classification confidence, producing a heatmap that is faithful to the actual model used for triage. Second, a textual rationale is generated separately by querying Qwen2.5-72B-Instruct via API, conditioned on the image and the *ground-truth* diagnostic label (not the triage model's prediction), to produce a clinically grounded description of the typical presentation of that condition.

We emphasize that this textual rationale is **illustrative and post-hoc**: it explains what the ground-truth condition typically looks like, not why the fine-tuned triage model produced its specific prediction. The saliency map remains the only component of the explainability pipeline that is directly faithful to the triage model's decision process. We disclose this distinction explicitly because conflating the two would overstate the interpretability of the deployed system.

### F. Edge Deployment via GGUF Quantization

For offline deployment, the fine-tuned model (base weights merged with the LoRA adapter) is converted to GGUF format and quantized using the Q4_K_M scheme, a 4-bit quantization with mixed precision for sensitive layers. The resulting artifact is 4.36 GB, suitable for distribution to and storage on consumer hardware without internet access. We benchmark inference throughput on the same RTX 4060 GPU used for fine-tuning.

## IV. Results

### A. Severity Classification Accuracy

Table I summarizes the headline classification results on the 200-sample held-out evaluation set. The zero-shot base model achieves 13.0% severity classification accuracy -- below the 33% expected from random 3-class guessing, which we attribute to the base model's failure to reliably follow the structured output format, causing the fallback parser to extract severity keywords in a manner biased away from the LOW class. Fine-tuning improves accuracy to 85.0%, an absolute improvement of 72.0 percentage points.

**TABLE I: SEVERITY CLASSIFICATION RESULTS (n=200)**

| Model | Severity Accuracy |
|---|---|
| Qwen2.5-VL-7B (zero-shot) | 13.0% |
| RuralMed Vision (QLoRA fine-tuned) | 85.0% |
| Absolute improvement | +72.0 pp |

### B. Diagnostic-Level Confusion Analysis

While severity-level accuracy is high, the underlying 7-class diagnostic confusion matrix (Fig. 1) reveals a substantial failure mode: 90 of 138 ground-truth melanocytic nevi (nv) samples are misclassified as melanoma (mel), while only 48 are correctly classified as nevi. Because both nevi and melanoma are mapped to different severity tiers (nv -> LOW, mel -> HIGH) yet melanoma is the model's dominant output among ambiguous lesions, this diagnostic-level confusion does not fully collapse the severity-level accuracy -- but it indicates that the apparent severity accuracy is partly a product of the model's tendency to over-predict the clinically "louder" class (melanoma) rather than precise diagnostic discrimination. Per-class precision/recall/F1 (Fig. 3, Table II) shows melanoma achieves high recall (0.833) and precision (1.000), while nevi recall is perfect (1.000) but precision is only 0.250 -- consistent with the model defaulting to "nevi" as a catch-all prediction for a wide range of inputs, while simultaneously over-flagging melanoma for genuinely ambiguous nevi cases. Basal cell carcinoma and actinic keratosis are not correctly predicted in this sample (precision, recall, and F1 all 0.000), indicating these minority classes are essentially unrepresented in the model's effective output distribution.

**TABLE II: PER-CLASS DIAGNOSTIC METRICS (FINE-TUNED MODEL)**

| Pathology | Precision | Recall | F1 |
|---|---|---|---|
| Melanoma | 1.000 | 0.833 | 0.909 |
| Nevi | 0.250 | 1.000 | 0.400 |
| Basal Cell Ca. | 0.000 | 0.000 | 0.000 |
| Actinic Keratosis | 0.000 | 0.000 | 0.000 |
| Benign Keratosis | 0.182 | 1.000 | 0.308 |
| Dermatofibroma | 0.960 | 0.348 | 0.511 |
| Vascular | 1.000 | 1.000 | 1.000 |

### C. Confidence Calibration

We compute the Expected Calibration Error (ECE) over the fine-tuned model's predictions, binning predicted confidence into ten bins and comparing each bin's average confidence to its empirical accuracy. The result (Fig. 2) shows ECE = 0.43: the model's predictions are concentrated in the 0.8-0.9 confidence range, but empirical accuracy in that range is only approximately 0.42 -- a severe overconfidence gap. This finding is consistent with the diagnostic-level confusion observed in Section IV-B: the model produces confident predictions even when its underlying diagnostic discrimination (as opposed to severity-level discrimination) is poor. We consider this a central limitation of the current system and a priority for future calibration work (e.g., temperature scaling, ensemble methods, or explicit uncertainty-aware training objectives).

### D. Clinical Triage Deflection

To assess the clinical safety implications of misclassification, we compute a "triage deflection" metric: the difference between the predicted and true severity level on an ordinal scale (LOW=0, MEDIUM=1, HIGH=2). Fig. 4 shows the distribution of deflection scores by true severity class. For HIGH-severity ground truth, 82.1% of predictions are correctly aligned (deflection = 0) and 17.9% are under-triaged by one level (predicted MEDIUM); critically, 0% of HIGH cases are catastrophically under-triaged to LOW (deflection = -2), meaning no high-severity case is missed entirely. For LOW-severity ground truth, 86.5% are correctly aligned and 13.5% are over-triaged to HIGH -- representing unnecessary specialist referrals but not a direct safety risk. MEDIUM-severity cases show the most dispersion, with 66.7% correct, 22.2% under-triaged, and 11.1% over-triaged. This asymmetry -- no catastrophic misses of HIGH-severity cases, but meaningful confusion in the MEDIUM tier -- is an encouraging signal for a screening-tool use case, where the primary safety requirement is that genuinely dangerous lesions are not dismissed as benign.

### E. Edge Deployment Performance

The fine-tuned model, quantized to GGUF Q4_K_M format, occupies 4.36 GB on disk and 4,168 MiB of GPU memory during inference on the RTX 4060, compared to 6.0/8.6 GB VRAM required for the unquantized (4-bit bitsandbytes) Hugging Face model during evaluation. Inference throughput in the GGUF format is 49.1 tokens/sec (20.4 ms/token), sufficient for interactive use in a triage workflow where a complete structured response (approximately 30-50 tokens) is generated in under two seconds.

### F. Decoupled Explainability Output

Fig. 5 shows representative outputs from the decoupled explainability pipeline for three diagnostic categories (melanocytic nevus, melanoma, basal cell carcinoma). Each row shows the input dermoscopic image, an occlusion-based saliency map computed from the fine-tuned model's vision encoder, and a textual rationale generated by Qwen2.5-72B-Instruct conditioned on the ground-truth label. The saliency maps localize attention to lesion boundaries and pigmentation patterns consistent with the regions a clinician would examine, though we note that saliency map quality was not formally validated against expert annotations and the textual rationales, as described in Section III-E, are post-hoc and not directly attributable to the triage model's decision process.

## V. Discussion and Limitations

The results presented here support a narrow but specific claim: a 7B generalist vision-language model, adapted via QLoRA on a modest single-GPU budget, can be shifted from near-random (and in fact below-random, due to format-following failure) zero-shot performance to 85% severity-tier accuracy on a clinically meaningful 3-class triage task, while remaining small enough to run fully offline on consumer hardware. This is the core feasibility result (C1) and we believe it is robust given the held-out, seed-disjoint evaluation protocol.

However, two findings substantially qualify how this result should be interpreted. First, the diagnostic-level confusion matrix and calibration analysis (C3) show that the 85% severity accuracy is achieved despite -- not because of -- accurate underlying diagnostic discrimination for several minority classes. The model appears to have learned a coarse heuristic ("most things are nevi, but ambiguous-looking lesions are melanoma") that happens to align reasonably well with the severity mapping for the majority class, but this heuristic would likely fail to generalize to a test distribution with a different class balance than HAM10000's. Reporting only the severity accuracy without the diagnostic confusion matrix and ECE would give a misleadingly favorable picture of the model's clinical readiness.

Second, the decision to decouple explainability (C2) was not a design choice made in advance but a response to an observed failure: the fine-tuned model's own rationales became unreliable. We view this as an important practical finding for anyone fine-tuning generative VLMs on mixed structured/unstructured clinical corpora -- joint optimization for a structured classification objective and unstructured dialogue text can degrade the coherence of free-text outputs from the same model, even when the structured output itself remains usable. Our workaround (separating saliency from rationale generation, and being explicit that the rationale is illustrative rather than model-faithful) is a pragmatic mitigation, not a solution, and future work should explore whether multi-task training schedules or separate output heads can avoid this degradation entirely.

We also note that all results are computed on a 200-sample subset of the 3,005-sample held-out set; while the held-out set itself is seed-disjoint from training, the 200-sample evaluation subset narrows the confidence interval on our estimates, and full-test-set evaluation (computationally feasible but not yet performed at submission time) would strengthen the statistical claims in Table I and Table II. Finally, the severity labels used here are derived programmatically from the 7-class HAM10000 diagnostic labels via a fixed clinical-urgency mapping, not independently assessed by a clinician for this study; this mapping is a reasonable proxy but should not be treated as a clinically validated severity ground truth.

## VI. Conclusion

We presented RuralMed Vision, a QLoRA fine-tuned Qwen2.5-VL-7B-Instruct model for offline skin lesion severity triage, achieving 85.0% severity accuracy (up from 13.0% zero-shot) and deployable as a 4.36 GB GGUF artifact at 49.1 tokens/sec on an 8GB consumer GPU. Alongside this feasibility result, we reported two limitations -- severe confidence miscalibration and majority-class mode collapse at the diagnostic level, and the degradation of the fine-tuned model's free-text rationale quality under joint structured/unstructured fine-tuning -- and described a decoupled explainability pipeline adopted in direct response to the latter. We believe transparent reporting of these limitations is as valuable to the field as the headline accuracy improvement, particularly for a domain where overconfident or uninterpretable model behavior carries direct clinical risk. Future work includes full-test-set evaluation, calibration correction, and exploration of training schedules that avoid language-head degradation under mixed-objective fine-tuning.

## Acknowledgment

The author thanks the maintainers of the HAM10000 and HealthCareMagic-100k datasets, and the Qwen team for the Qwen2.5-VL model family.

## References

[1] P. Tschandl, C. Rosendahl, and H. Kittler, "The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions," Scientific Data, vol. 5, 180161, 2018.

[2] T. Dettmers, A. Pagnoni, A. Holtzman, and L. Zettlemoyer, "QLoRA: Efficient Finetuning of Quantized LLMs," in Proc. NeurIPS, 2023.

[3] E. J. Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models," in Proc. ICLR, 2022.

[4] Qwen Team, "Qwen2.5-VL Technical Report," arXiv preprint, 2025.

[5] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, "On Calibration of Modern Neural Networks," in Proc. ICML, 2017.

[6] A. Adebiyi et al., "Accurate Skin Lesion Classification Using Multimodal Learning on the HAM10000 and ISIC 2017 Datasets," medRxiv, 2024.
