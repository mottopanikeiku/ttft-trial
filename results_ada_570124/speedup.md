| label                                     | cache_mode   |   prompt_tokens |   concurrency |   base_p50 |    p50 |   speedup |
|:------------------------------------------|:-------------|----------------:|--------------:|-----------:|-------:|----------:|
| qwen36-27b-ada-channel-r570124            | cold         |            3072 |             1 |      906.1 |  780.4 |      1.16 |
| qwen36-27b-ada-channel-r570124            | cold         |            4096 |             1 |     1254.8 | 1111   |      1.13 |
| qwen36-27b-cold-final-a-r570124           | cold         |            3072 |             1 |      906.1 |  790.2 |      1.15 |
| qwen36-27b-cold-final-a-r570124           | cold         |            4096 |             1 |     1254.8 | 1128.9 |      1.11 |
| qwen36-27b-cold-final-b-r570124           | cold         |            3072 |             1 |      906.1 |  831.6 |      1.09 |
| qwen36-27b-cold-final-b-r570124           | cold         |            4096 |             1 |     1254.8 | 1157.8 |      1.08 |
| qwen36-27b-dense-async-budget4096-r570124 | cold         |            4096 |             1 |     1254.8 | 1094.2 |      1.15 |
| qwen36-27b-dense-async-o3-r570124         | cold         |            4096 |             1 |     1254.8 | 1093.6 |      1.15 |
| qwen36-27b-dense-async-r570124            | cold         |            4096 |             1 |     1254.8 | 1095.8 |      1.15 |
| qwen36-27b-dense-tuned-r570124            | cold         |            3072 |             1 |      906.1 |  795.1 |      1.14 |
| qwen36-27b-dense-tuned-r570124            | cold         |            4096 |             1 |     1254.8 | 1132.6 |      1.11 |
| qwen36-27b-singlechunk-r570124            | cold         |            3072 |             1 |      906.1 |  922.9 |      0.98 |
| qwen36-27b-singlechunk-r570124            | cold         |            4096 |             1 |     1254.8 | 1279.3 |      0.98 |
| qwen36-27b-tuned-singlechunk-r570124      | cold         |            3072 |             1 |      906.1 |  823.1 |      1.1  |
| qwen36-27b-tuned-singlechunk-r570124      | cold         |            4096 |             1 |     1254.8 | 1151.3 |      1.09 |