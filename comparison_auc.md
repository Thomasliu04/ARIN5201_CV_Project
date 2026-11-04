Metric: **auc**

| Test set | results_resize_b16 | results_crop128_b16 | Δ (last − first) |
|---|---|---|---|
| **Held-out (seen generator)** | | | |
| seen | 100.0 | 99.8 | -0.2 |
| val | 100.0 | 99.7 | -0.3 |
| **Unseen generators** | | | |
| unseen_adm | 93.9 | 92.6 | -1.3 |
| **Bias tests (seen)** | | | |
| all_jpeg444 | 99.9 | 98.8 | -1.1 |
| fake_jpeg420 | 99.8 | 98.5 | -1.3 |
| fake_jpeg444 | 99.9 | 98.7 | -1.2 |
| real_down128 | 93.1 | 78.1 | -15.0 |
| real_jpeg420 | 100.0 | 99.8 | -0.2 |
| size_aligned | 93.1 | 78.1 | -15.0 |
| size_jpeg_aligned | 84.1 | 72.8 | -11.4 |
| **Degradations (seen)** | | | |
| blur0.5 | 99.9 | 99.2 | -0.8 |
| blur1 | 99.9 | 94.6 | -5.3 |
| blur2 | 97.2 | 89.6 | -7.6 |
| down25 | 96.3 | 33.3 | -63.0 |
| down50 | 99.9 | 86.2 | -13.7 |
| down75 | 100.0 | 98.2 | -1.8 |
| jpeg50 | 99.9 | 92.2 | -7.7 |
| jpeg70 | 99.9 | 96.5 | -3.4 |
| jpeg90 | 99.8 | 98.5 | -1.3 |
| **Bias tests (unseen)** | | | |
| unseen_adm_all_jpeg444 | 92.9 | 87.0 | -6.0 |
| unseen_adm_fake_jpeg420 | 89.4 | 85.9 | -3.6 |
| unseen_adm_fake_jpeg444 | 92.8 | 86.6 | -6.2 |
| unseen_adm_real_down128 | 11.2 | 23.2 | +12.0 |
| unseen_adm_real_jpeg420 | 94.1 | 91.6 | -2.5 |
| unseen_adm_size_aligned | 89.1 | 67.5 | -21.7 |
| unseen_adm_size_jpeg_aligned | 90.3 | 66.9 | -23.3 |
| **Degradations (unseen)** | | | |
| unseen_adm_blur0.5 | 94.3 | 85.8 | -8.5 |
| unseen_adm_blur1 | 98.0 | 67.5 | -30.5 |
| unseen_adm_blur2 | 92.8 | 61.9 | -30.9 |
| unseen_adm_down25 | 91.9 | 62.4 | -29.5 |
| unseen_adm_down50 | 99.3 | 92.3 | -6.9 |
| unseen_adm_down75 | 98.2 | 86.2 | -12.0 |
| unseen_adm_jpeg50 | 92.7 | 69.1 | -23.6 |
| unseen_adm_jpeg70 | 95.2 | 80.1 | -15.2 |
| unseen_adm_jpeg90 | 89.5 | 87.1 | -2.4 |
