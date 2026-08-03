{
  "summary": "Audit of quality-filtering/sarvam_quality_classifier.py and quality_filtering.py plus data/sarvam_quality_classifier artifacts found two critical evaluation-validity bugs (hash-collision corruption-kind collapse; 100%-synthetic reject class), one high-severity inflated-metric issue (held-out set dominated by trivially-separable synthetic/template rows), and two medium issues (train/held-out leakage for protected controls; truncation bias from the 900-char annotation cap). A prioritized upgrade plan and recommended Sarvam-vs-deterministic sampling split were delivered to Main.",
  "files": [
    {
      "path": "quality-filtering/sarvam_quality_classifier.py",
      "description": "stable_bucket() at line 129 is reused with modulus 5 both for partition_by_source (train/held-out split) and for the corruption-kind offset in build_training_rows, causing every held-out source doc to always receive the same 2 of 5 BAD_KINDS (residual_markup, link_farm_or_seo). Sarvam never assigns 'reject' to any real source document (verified across all 264 sarvam_source_annotations.jsonl rows), so the reject class is entirely synthetic. protected_controls('train') rows are re-split by the same source-doc hash, leaking some into held-out. capped_text() truncates from the document start at a 900-character cap, likely hiding mid/end-document boilerplate."
    },
    {
      "path": "quality-filtering/quality_filtering.py",
      "description": "Notebook cells sarvam_classifier_results (~line 979) and the Sarvam narrative block (lines 885-894, 1033-1062) report held-out accuracy/reject_recall and per-bad-kind breakdown but do not surface that only 2 of 5 corruption families have meaningful held-out sample sizes, or that zero real documents were ever labeled reject by Sarvam. Deterministic quality-filter heuristics (_hard_flags, REVIEW_REASON_LABELS, evaluate_quality_records) already exist in this file and could be reused to pre-screen real low-quality candidates for Sarvam annotation."
    },
    {
      "path": "quality-filtering/data/sarvam_quality_classifier/classifier_summary.json",
      "description": "held_out_evaluation.bad_kind_breakdown shows residual_markup 41/41 and link_farm_or_seo 41/41 caught, but symbol_debris, repetition_or_keyword_stuffing, and malformed_or_unusable_text each have only 1 held-out document (the single hardcoded adversarial-seed). by_origin.source.accuracy=1.0 with no real reject case present. training.synthetic_bad_documents=528 vs source_documents=264 confirms reject-class synthetic dominance."
    },
    {
      "path": "quality-filtering/data/sarvam_quality_classifier/sarvam_source_annotations.jsonl",
      "description": "All 264 real-document annotations carry label keep or review only; grep for label:\"reject\" returns zero matches, confirming Sarvam never rejected a real document under the current 900-character truncated-from-start annotation cap."
    },
    {
      "path": "quality-filtering/data/sarvam_quality_classifier/held_out_predictions.jsonl",
      "description": "Contains doc_ids protected-train-1 and protected-train-6 despite being tagged for the 'train' split, confirming protected_controls('train') rows leak into held-out via the shared source-doc hash partition."
    }
  ],
  "architecture": "Prioritized fixes: (1) salt the corruption-kind selector hash independently from the train/held-out split hash so all 5 BAD_KINDS get adequate held-out coverage; (2) enlarge and stratify the real source sample to ~1,500-2,000 documents, pre-screening ~15-20% via the notebook's existing deterministic heuristics so genuinely low-quality real documents reach Sarvam and yield real reject labels (target >=50 real held-out rejects); (3) route protected_control rows to their named split directly instead of through the hash partition; (4) add a regression gate requiring minimum per-bad-kind and per-origin held-out sample counts before accepting a run, and report accuracy/recall stratified by origin x type rather than blended; (5) A/B the character cap (900 vs 1500 vs uncapped) to test truncation bias. Sarvam (LLM annotation) should be reserved for real, ambiguous source documents; the 5 synthetic corruption templates and the protected-control set should remain fully deterministic since their labels are correct by construction and gain nothing from LLM judgment."
}