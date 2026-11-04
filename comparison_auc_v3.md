Metric: **auc**

| Test set | r_b2g_resize_jpeg | r_ba2g_resize_jpeg | r_b2g_crop_jpeg | r_ba2g_crop_jpeg | Δ (last − first) |
|---|---|---|---|---|---|
| **Held-out (seen generator)** | | | | | |
| seen | 100.0 | 99.6 | 99.6 | 98.3 | -1.7 |
| val | 99.9 | 99.5 | 99.0 | 96.8 | -3.1 |
| **Unseen generators** | | | | | |
| unseen_adm | 92.2 | – | 89.0 | – | – |
| unseen_glide | 99.1 | 98.2 | 91.7 | 97.7 | -1.4 |
| **Bias tests (unseen)** | | | | | |
| unseen_glide_all_jpeg444 | 98.5 | 98.3 | 84.8 | 96.2 | -2.3 |
| unseen_glide_fake_jpeg420 | 98.4 | 98.3 | 82.9 | 96.3 | -2.1 |
| unseen_glide_fake_jpeg444 | 98.5 | 98.2 | 84.4 | 96.0 | -2.5 |
| unseen_glide_real_down128 | 36.4 | 59.8 | 23.8 | 50.6 | +14.2 |
| unseen_glide_real_jpeg420 | 99.0 | 98.1 | 91.3 | 96.8 | -2.2 |
| unseen_glide_size_aligned | 98.5 | 97.4 | 69.0 | 85.2 | -13.4 |
| unseen_glide_size_jpeg_aligned | 96.0 | 95.7 | 57.7 | 83.1 | -12.9 |
| **Degradations (unseen)** | | | | | |
| unseen_glide_blur0.5 | 98.9 | 97.7 | 82.5 | 92.0 | -6.9 |
| unseen_glide_blur1 | 98.0 | 95.2 | 56.1 | 59.7 | -38.2 |
| unseen_glide_blur2 | 88.7 | 84.8 | 58.6 | 56.6 | -32.1 |
| unseen_glide_down25 | 89.9 | 85.5 | 48.7 | 54.0 | -35.9 |
| unseen_glide_down50 | 99.1 | 98.1 | 85.6 | 84.9 | -14.2 |
| unseen_glide_down75 | 99.1 | 98.1 | 87.5 | 93.2 | -5.8 |
| unseen_glide_jpeg50 | 99.1 | 99.4 | 75.6 | 88.4 | -10.7 |
| unseen_glide_jpeg70 | 99.4 | 99.6 | 84.1 | 92.3 | -7.1 |
| unseen_glide_jpeg90 | 99.1 | 99.2 | 85.5 | 95.0 | -4.1 |
