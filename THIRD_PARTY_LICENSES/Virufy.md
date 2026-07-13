## Virufy COVID-19 Open Cough Dataset Attribution

- Dataset: **Virufy COVID-19 Open Cough Dataset (clinical subset)**
- Source repository: https://github.com/virufy/virufy-data
- Maintainer: Virufy.org
- License: **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International
  (CC BY-NC-SA 4.0), with an additional redistribution restriction (see below)**
- License URL: https://creativecommons.org/licenses/by-nc-sa/4.0/

### ⚠️ Important restriction - stricter than the other datasets in this project

Virufy's LICENSE file adds two conditions on top of CC BY-NC-SA 4.0 that do **not**
apply to Coswara, COUGHVID, or Sound-Dr:

1. **No redistribution without written approval.** Quoting the license verbatim:
   > "Redistribution, republishing, or dissemination in any form, source or binary,
   > is not permitted without prior written approval by Virufy."
2. **Non-commercial only**, and **any work using this data must cite Virufy's paper**
   (https://virufy.org/paper).

**Practical consequence for this repository:** the raw Virufy audio/labels
(`external_datasets/virufy-data/`) must not be committed, pushed, or otherwise
redistributed as part of this project without Virufy's prior written permission.
It is excluded via `.gitignore`. Only aggregate, non-redistributive findings
(e.g. "accuracy when evaluated against Virufy's clinical set") should be shared,
not the underlying files.

### Notes

- Only the clinical subset was pulled (121 segmented cough samples from 16
  PCR-confirmed patients) - too small on its own for a statistically meaningful
  cross-dataset evaluation, so it was not used as a primary test set in this project.
- Virufy's larger crowdsourced dataset is not included in the public repository and
  requires contacting Virufy directly (open-data@virufy.org).

### Suggested Citation

Virufy: A Multi-Institutional Study to Build a Screening Tool for COVID-19
Using Crowdsourced Cough Data. https://virufy.org/paper
