# Validation Flow Diagram - CRF Usage Analysis

## Training Validation Flow (Current Implementation - NO CRF)

```
┌─────────────────────────────────────────────────────────────────┐
│                    Training Loop (train.py)                      │
│                                                                   │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Every N iterations or epochs:                            │  │
│  │  Trigger validation via EvalHook                          │  │
│  └───────────────────────┬───────────────────────────────────┘  │
└────────────────────────────┼────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│            EvalHook._do_evaluate() (eval_hooks.py)              │
│                        Lines 43-55                               │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Calls: single_gpu_test(runner.model, dataloader, ...)    │  │
│  └───────────────────────┬───────────────────────────────────┘  │
└────────────────────────────┼────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│            single_gpu_test() (test.py, lines 38-144)            │
│                                                                   │
│  For each batch in dataloader:                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  with torch.no_grad():                                     │  │
│  │      result = model(return_loss=False, **data)             │  │
│  │                     ▲                                       │  │
│  │                     │ Line 96: Triggers inference mode     │  │
│  └─────────────────────┼───────────────────────────────────────┘  │
└─────────────────────────┼───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│      EncoderDecoder.simple_test() (encoder_decoder.py)          │
│                    Lines 291-331                                 │
│                                                                   │
│  Step 1: Get segmentation logits                                │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  seg_logit = self.inference(img, img_meta, rescale)       │  │
│  │  seg_pred = seg_logit.argmax(dim=1)  # Line 297           │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                   │
│  Step 2: CRF Post-processing? ❌ NO - ALL COMMENTED OUT         │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  # Lines 320-328 (COMMENTED OUT)                          │  │
│  │  # ori_img = np.array(Image.open(...).convert("RGB"))     │  │
│  │  # seg_logit_ori = seg_logit[0].cpu().numpy()             │  │
│  │  # seg_logit = self.DenseCRF(ori_img, seg_logit_ori)      │  │
│  │  # seg_pred = seg_logit.argmax(dim=1)                     │  │
│  │  #                                                          │  │
│  │  # Line 328 also commented:                                │  │
│  │  # seg_pred = self.crf_inference_label(...)               │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                   │
│  Step 3: Return raw predictions                                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  seg_pred = seg_pred.cpu().numpy()  # Line 327            │  │
│  │  seg_pred = list(seg_pred)  # Line 330                    │  │
│  │  return seg_pred  # ⚠️ Raw predictions, NO CRF!           │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│        Collect all results from all batches (test.py)           │
│                      Lines 136-144                               │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  results.extend(result)  # Raw predictions                │  │
│  │  return results                                            │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│         EvalHook.evaluate() → Calculate mIoU                    │
│              (metrics.py, lines 132-168)                         │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  mean_iou(results,  # ⚠️ Raw predictions without CRF      │  │
│  │           gt_seg_maps,                                     │  │
│  │           num_classes, ...)                                │  │
│  │                                                             │  │
│  │  Returns: {'mIoU': 0.518, 'Acc': ..., 'IoU': [...]}       │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## CRF Implementation Available But Not Used

```
┌─────────────────────────────────────────────────────────────────┐
│     DenseCRF Method (encoder_decoder.py, lines 270-288)         │
│                                                                   │
│  ✅ Fully Implemented:                                           │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  def DenseCRF(self, image, probmap, iter_max=10, ...):    │  │
│  │      C, H, W = probmap.shape                               │  │
│  │      U = utils.unary_from_softmax(probmap)                │  │
│  │      dc = dcrf.DenseCRF2D(W, H, C)                        │  │
│  │      dc.setUnaryEnergy(U)                                  │  │
│  │      dc.addPairwiseGaussian(sxy=pos_xy_std, compat=pos_w) │  │
│  │      dc.addPairwiseBilateral(...)                         │  │
│  │      Q = dc.inference(iter_max)                            │  │
│  │      return Q                                              │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                   │
│  ❌ But NOT Called During Validation!                            │
└─────────────────────────────────────────────────────────────────┘
```

## Offline CRF Tool (Separate from Training/Validation)

```
┌─────────────────────────────────────────────────────────────────┐
│        tools/refine_masks.py - Offline Post-processing          │
│                                                                   │
│  Usage: Manually run after training to refine saved masks       │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  python tools/refine_masks.py                             │  │
│  │                                                             │  │
│  │  Process:                                                  │  │
│  │  1. Load saved prediction masks from disk                 │  │
│  │  2. Load original images                                   │  │
│  │  3. Apply CRF: pred = crf_inference_label(img, mask)      │  │
│  │  4. Save refined masks to disk                            │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                   │
│  ⚠️ This is NOT integrated into training validation loop!        │
│  ⚠️ Must be run separately after training                        │
└─────────────────────────────────────────────────────────────────┘
```

## Hypothetical Flow If CRF Were Enabled (Not Current Implementation)

```
┌─────────────────────────────────────────────────────────────────┐
│      EncoderDecoder.simple_test() - WITH CRF ENABLED            │
│                                                                   │
│  Step 1: Get segmentation logits                                │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  seg_logit = self.inference(img, img_meta, rescale)       │  │
│  └───────────────────────────────────────────────────────────┘  │
│                          │                                       │
│                          ▼                                       │
│  Step 2: Apply CRF ✅ (If uncommitted)                          │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  ori_img = np.array(Image.open(...).convert("RGB"))       │  │
│  │  seg_logit_ori = seg_logit[0].cpu().numpy()               │  │
│  │                      │                                      │  │
│  │                      ▼                                      │  │
│  │  ┌────────────────────────────────────────────────────┐   │  │
│  │  │  seg_logit = self.DenseCRF(ori_img, seg_logit_ori) │   │  │
│  │  │                                                      │   │  │
│  │  │  - Unary term from softmax probabilities            │   │  │
│  │  │  - Pairwise Gaussian term (spatial smoothness)      │   │  │
│  │  │  - Pairwise Bilateral term (color similarity)       │   │  │
│  │  │  - Iterate 10 times to refine                        │   │  │
│  │  └────────────────────────────────────────────────────┘   │  │
│  │                      │                                      │  │
│  │                      ▼                                      │  │
│  │  seg_logit = torch.FloatTensor(seg_logit)                 │  │
│  │  seg_pred = seg_logit.argmax(dim=1)                       │  │
│  └───────────────────────────────────────────────────────────┘  │
│                          │                                       │
│                          ▼                                       │
│  Step 3: Return CRF-refined predictions                         │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  seg_pred = seg_pred.cpu().numpy()                        │  │
│  │  return seg_pred  # ✅ CRF-refined predictions!            │  │
│  └───────────────────────────────────────────────────────────┘  │
│                          │                                       │
│                          ▼                                       │
│                    Calculate mIoU                                │
│              (Would likely be higher with CRF)                  │
└─────────────────────────────────────────────────────────────────┘
```

## Summary

| Component | Location | Status | Used in Validation? |
|-----------|----------|--------|---------------------|
| CRF Import | encoder_decoder.py:13-14 | ✅ Present | N/A |
| DenseCRF Method | encoder_decoder.py:270-288 | ✅ Implemented | ❌ No |
| CRF in simple_test | encoder_decoder.py:322-328 | ❌ Commented | ❌ No |
| CRF in aug_test | encoder_decoder.py:333-368 | ❌ Not present | ❌ No |
| Offline CRF Tool | tools/refine_masks.py | ✅ Available | ❌ No (Separate) |
| Validation mIoU | Based on raw predictions | ❌ No CRF | ❌ No |

## Key Takeaways

1. **CRF is implemented but not used** during training validation
2. **Validation mIoU is calculated on raw model predictions**
3. **CRF is only available as an offline post-processing tool**
4. **To enable CRF during validation**: Uncomment lines 320-328 in encoder_decoder.py
5. **Performance impact**: CRF would significantly slow down validation (10x-100x slower)
6. **Likely reason for commenting**: Fair comparison with other methods without post-processing

---

**Analysis Date**: 2025-12-03  
**Repository**: hyzz1/CRAB  
**Branch**: main
