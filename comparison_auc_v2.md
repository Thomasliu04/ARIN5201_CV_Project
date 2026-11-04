Metric: **auc**

| Test set | results_resize_b16 | results_resize_b16_jpeg | results_crop128_b16 | results_crop128_b16_jpeg | Δ (last − first) |
|---|---|---|---|---|---|
| **Held-out (seen generator)** | | | | | |
| seen | 100.0 | 99.9 | 99.8 | 99.6 | -0.4 |
| val | 100.0 | 99.9 | 99.7 | 99.3 | -0.7 |
| **Unseen generators** | | | | | |
| unseen_adm | 93.9 | 92.4 | 92.6 | 88.4 | -5.5 |
| **Bias tests (seen)** | | | | | |
| all_jpeg444 | 99.9 | 100.0 | 98.8 | 99.2 | -0.8 |
| fake_jpeg420 | 99.8 | 100.0 | 98.5 | 99.2 | -0.6 |
| fake_jpeg444 | 99.9 | 100.0 | 98.7 | 99.1 | -0.8 |
| real_down128 | 93.1 | 91.8 | 78.1 | 78.4 | -14.8 |
| real_jpeg420 | 100.0 | 99.9 | 99.8 | 99.7 | -0.3 |
| size_aligned | 93.1 | 91.8 | 78.1 | 78.4 | -14.8 |
| size_jpeg_aligned | 84.1 | 87.8 | 72.8 | 75.1 | -9.0 |
| **Degradations (seen)** | | | | | |
| blur0.5 | 99.9 | 99.9 | 99.2 | 98.9 | -1.0 |
| blur1 | 99.9 | 99.8 | 94.6 | 94.5 | -5.3 |
| blur2 | 97.2 | 96.6 | 89.6 | 92.1 | -5.1 |
| down25 | 96.3 | 97.1 | 33.3 | 32.6 | -63.7 |
| down50 | 99.9 | 99.9 | 86.2 | 90.8 | -9.2 |
| down75 | 100.0 | 100.0 | 98.2 | 98.7 | -1.3 |
| jpeg50 | 99.9 | 99.9 | 92.2 | 98.0 | -1.8 |
| jpeg70 | 99.9 | 99.9 | 96.5 | 99.0 | -0.8 |
| jpeg90 | 99.8 | 99.9 | 98.5 | 99.4 | -0.4 |
| **Bias tests (unseen)** | | | | | |
| unseen_adm_all_jpeg444 | 92.9 | 95.0 | 87.0 | 86.1 | -6.8 |
| unseen_adm_fake_jpeg420 | 89.4 | 93.1 | 85.9 | 84.4 | -5.0 |
| unseen_adm_fake_jpeg444 | 92.8 | 94.9 | 86.6 | 85.8 | -6.9 |
| unseen_adm_real_down128 | 11.2 | 8.4 | 23.2 | 17.2 | +5.9 |
| unseen_adm_real_jpeg420 | 94.1 | 92.5 | 91.6 | 87.9 | -6.3 |
| unseen_adm_size_aligned | 89.1 | 87.8 | 67.5 | 62.3 | -26.8 |
| unseen_adm_size_jpeg_aligned | 90.3 | 88.4 | 66.9 | 62.1 | -28.2 |
| **Degradations (unseen)** | | | | | |
| unseen_adm_blur0.5 | 94.3 | 93.6 | 85.8 | 80.6 | -13.7 |
| unseen_adm_blur1 | 98.0 | 97.9 | 67.5 | 65.2 | -32.8 |
| unseen_adm_blur2 | 92.8 | 91.7 | 61.9 | 62.5 | -30.3 |
| unseen_adm_down25 | 91.9 | 92.1 | 62.4 | 64.3 | -27.6 |
| unseen_adm_down50 | 99.3 | 99.0 | 92.3 | 93.0 | -6.3 |
| unseen_adm_down75 | 98.2 | 97.6 | 86.2 | 84.3 | -13.9 |
| unseen_adm_jpeg50 | 92.7 | 96.3 | 69.1 | 77.1 | -15.5 |
| unseen_adm_jpeg70 | 95.2 | 96.9 | 80.1 | 85.5 | -9.7 |
| unseen_adm_jpeg90 | 89.5 | 96.3 | 87.1 | 87.8 | -1.7 |
