# Music4All cross-check: complete result tables

This is an **artist/full-mix** codec check, not singer-verified validation. Twelve artist labels, 72 tracks (six per artist), 32 kb/s MP3 mismatch re-encode. Source recordings are already compressed; 71 are reported at 128 kb/s and one at 64 kb/s. CVSM pretraining overlap is present. Macro ranking error compares same-artist different-track positives with other-artist negatives. Bootstrap unit is artist.

Source: [recomputed metrics JSON](/legacy/credible-work-bro/cross%20checking/music4all_rerun/metrics_recomputed.json) (verbatim raw data archived in [raw external and rechecks](raw_external_and_rechecks.md)).

| Model/seed | Artists | Recordings | Native % | Mismatch 32k % | Both 32k % | Mismatch − native pp | Artist-bootstrap 95% CI (pp) |
|---|---:|---:|---:|---:|---:|---:|---|
| adapter_cvsm_a_s0 | 6 | 36 | 15.537 | 10.204 | 19.056 | -5.333 | [-9.944, -0.852] |
| adapter_cvsm_a_s1 | 6 | 36 | 15.426 | 9.981 | 19.463 | -5.444 | [-10.278, -0.796] |
| adapter_cvsm_a_s2 | 6 | 36 | 15.500 | 9.093 | 19.630 | -6.407 | [-11.704, -1.315] |
| adapter_cvsm_art_s0 | 6 | 36 | 13.130 | 17.019 | 16.389 | +3.889 | [0.704, 7.593] |
| adapter_cvsm_art_s1 | 6 | 36 | 13.630 | 5.963 | 16.444 | -7.667 | [-12.426, -3.130] |
| adapter_cvsm_art_s2 | 6 | 36 | 13.444 | 6.778 | 16.333 | -6.667 | [-10.889, -2.574] |
| adapter_ssl_s0 | 6 | 36 | 26.352 | 23.074 | 34.333 | -3.278 | [-6.037, -1.130] |
| adapter_ssl_s1 | 6 | 36 | 26.278 | 22.889 | 33.222 | -3.389 | [-6.370, -0.889] |
| adapter_ssl_s2 | 6 | 36 | 26.685 | 23.704 | 34.389 | -2.981 | [-5.926, -0.667] |
| cvsm_a | 12 | 72 | 12.643 | 22.146 | 17.226 | +9.503 | [7.525, 11.242] |
| cvsm_art | 12 | 72 | 10.274 | 45.530 | 15.139 | +35.257 | [26.229, 43.283] |
| ecapa | 12 | 72 | 19.141 | 27.832 | 21.785 | +8.691 | [5.960, 11.158] |
| heldout_raw_cvsm_a | 6 | 36 | 13.815 | 23.093 | 20.463 | +9.278 | [6.741, 11.722] |
| heldout_raw_cvsm_art | 6 | 36 | 12.889 | 50.167 | 17.593 | +37.278 | [22.000, 51.611] |
| heldout_raw_ssl | 6 | 36 | 25.056 | 38.981 | 31.426 | +13.926 | [5.000, 23.056] |
| mert | 12 | 72 | 20.337 | 44.794 | 20.324 | +24.457 | [18.283, 30.370] |
| ssl | 12 | 72 | 25.686 | 40.522 | 28.683 | +14.836 | [8.569, 21.886] |
| wavlm | 12 | 72 | 26.793 | 28.241 | 26.692 | +1.448 | [0.168, 2.656] |
| yakura | 12 | 72 | 47.294 | 47.639 | 44.937 | +0.345 | [-5.249, 6.006] |

## Per-artist error (%) for every reported model/seed

Each cell is `native / mismatch_32 / both_32`; rows preserve the six-artist test split represented in the JSON.

### adapter_cvsm_a_s0

| Artist | Native | Mismatch 32k | Both 32k |
|---|---:|---:|---:|
| Iron & Wine | 29.111% | 19.556% | 43.667% |
| Men at Work | 33.556% | 20.556% | 34.667% |
| SAAY | 1.222% | 0.333% | 5.889% |
| The Album Leaf | 2.667% | 4.556% | 7.111% |
| Tim McGraw | 1.000% | 1.222% | 1.111% |
| Warpaint | 25.667% | 15.000% | 21.889% |

### adapter_cvsm_a_s1

| Artist | Native | Mismatch 32k | Both 32k |
|---|---:|---:|---:|
| Iron & Wine | 29.111% | 19.667% | 45.778% |
| Men at Work | 33.778% | 20.222% | 34.333% |
| SAAY | 1.111% | 0.556% | 5.889% |
| The Album Leaf | 2.333% | 4.444% | 7.000% |
| Tim McGraw | 1.000% | 1.111% | 1.000% |
| Warpaint | 25.222% | 13.889% | 22.778% |

### adapter_cvsm_a_s2

| Artist | Native | Mismatch 32k | Both 32k |
|---|---:|---:|---:|
| Iron & Wine | 30.000% | 18.222% | 47.444% |
| Men at Work | 34.222% | 19.000% | 35.667% |
| SAAY | 1.111% | 0.222% | 6.000% |
| The Album Leaf | 2.111% | 3.889% | 6.556% |
| Tim McGraw | 1.000% | 0.889% | 1.111% |
| Warpaint | 24.556% | 12.333% | 21.000% |

### adapter_cvsm_art_s0

| Artist | Native | Mismatch 32k | Both 32k |
|---|---:|---:|---:|
| Iron & Wine | 29.778% | 29.556% | 29.667% |
| Men at Work | 15.667% | 23.778% | 25.333% |
| SAAY | 0.333% | 0.444% | 1.333% |
| The Album Leaf | 13.778% | 15.667% | 17.444% |
| Tim McGraw | 1.667% | 3.889% | 2.000% |
| Warpaint | 17.556% | 28.778% | 22.556% |

### adapter_cvsm_art_s1

| Artist | Native | Mismatch 32k | Both 32k |
|---|---:|---:|---:|
| Iron & Wine | 31.000% | 13.333% | 30.333% |
| Men at Work | 15.667% | 5.222% | 23.333% |
| SAAY | 0.333% | 0.111% | 1.444% |
| The Album Leaf | 11.000% | 4.444% | 19.556% |
| Tim McGraw | 1.667% | 1.000% | 2.333% |
| Warpaint | 22.111% | 11.667% | 21.667% |

### adapter_cvsm_art_s2

| Artist | Native | Mismatch 32k | Both 32k |
|---|---:|---:|---:|
| Iron & Wine | 30.222% | 14.444% | 30.778% |
| Men at Work | 16.111% | 6.556% | 24.444% |
| SAAY | 0.222% | 0.111% | 1.444% |
| The Album Leaf | 11.889% | 5.333% | 17.778% |
| Tim McGraw | 1.667% | 1.000% | 2.333% |
| Warpaint | 20.556% | 13.222% | 21.222% |

### adapter_ssl_s0

| Artist | Native | Mismatch 32k | Both 32k |
|---|---:|---:|---:|
| Iron & Wine | 31.333% | 29.556% | 41.556% |
| Men at Work | 43.556% | 40.556% | 55.222% |
| SAAY | 14.889% | 10.333% | 25.889% |
| The Album Leaf | 12.556% | 12.222% | 20.889% |
| Tim McGraw | 3.889% | 3.333% | 9.778% |
| Warpaint | 51.889% | 42.444% | 52.667% |

### adapter_ssl_s1

| Artist | Native | Mismatch 32k | Both 32k |
|---|---:|---:|---:|
| Iron & Wine | 30.333% | 30.111% | 42.000% |
| Men at Work | 42.667% | 38.333% | 52.222% |
| SAAY | 16.667% | 11.333% | 25.000% |
| The Album Leaf | 11.889% | 10.556% | 17.667% |
| Tim McGraw | 3.556% | 3.889% | 9.333% |
| Warpaint | 52.556% | 43.111% | 53.111% |

### adapter_ssl_s2

| Artist | Native | Mismatch 32k | Both 32k |
|---|---:|---:|---:|
| Iron & Wine | 31.333% | 31.778% | 43.444% |
| Men at Work | 43.556% | 40.889% | 54.889% |
| SAAY | 15.778% | 11.444% | 25.222% |
| The Album Leaf | 12.556% | 11.444% | 21.222% |
| Tim McGraw | 4.333% | 3.778% | 9.444% |
| Warpaint | 52.556% | 42.889% | 52.111% |

### cvsm_a

| Artist | Native | Mismatch 32k | Both 32k |
|---|---:|---:|---:|
| Adelitas Way | 2.071% | 14.949% | 5.152% |
| Bedouine | 1.414% | 12.222% | 3.687% |
| Claudia Leitte | 25.253% | 33.889% | 30.000% |
| Elder Island | 21.869% | 33.737% | 27.121% |
| Gridlink | 0.202% | 2.778% | 0.303% |
| Gunship | 20.859% | 33.030% | 29.192% |
| Iron & Wine | 28.333% | 39.646% | 47.626% |
| Men at Work | 25.960% | 38.485% | 30.455% |
| SAAY | 2.172% | 7.727% | 6.061% |
| The Album Leaf | 1.566% | 6.364% | 5.556% |
| Tim McGraw | 0.758% | 9.697% | 1.616% |
| Warpaint | 21.263% | 33.232% | 19.949% |

### cvsm_art

| Artist | Native | Mismatch 32k | Both 32k |
|---|---:|---:|---:|
| Adelitas Way | 0.404% | 36.061% | 0.606% |
| Bedouine | 2.828% | 50.253% | 9.444% |
| Claudia Leitte | 27.929% | 70.152% | 39.293% |
| Elder Island | 19.646% | 52.929% | 24.394% |
| Gridlink | 0.000% | 9.545% | 0.455% |
| Gunship | 4.949% | 56.364% | 10.202% |
| Iron & Wine | 25.051% | 60.455% | 30.960% |
| Men at Work | 14.040% | 62.677% | 25.152% |
| SAAY | 0.253% | 9.394% | 1.515% |
| The Album Leaf | 11.162% | 26.566% | 15.960% |
| Tim McGraw | 1.717% | 42.828% | 1.414% |
| Warpaint | 15.303% | 69.141% | 22.273% |

### ecapa

| Artist | Native | Mismatch 32k | Both 32k |
|---|---:|---:|---:|
| Adelitas Way | 9.343% | 21.162% | 8.485% |
| Bedouine | 28.182% | 32.626% | 27.576% |
| Claudia Leitte | 31.212% | 38.788% | 42.828% |
| Elder Island | 28.182% | 42.071% | 35.354% |
| Gridlink | 2.727% | 1.313% | 1.111% |
| Gunship | 19.495% | 29.394% | 21.667% |
| Iron & Wine | 32.374% | 41.970% | 35.859% |
| Men at Work | 28.889% | 42.172% | 39.545% |
| SAAY | 5.303% | 7.778% | 6.263% |
| The Album Leaf | 6.364% | 14.293% | 6.061% |
| Tim McGraw | 9.343% | 19.899% | 13.333% |
| Warpaint | 28.283% | 42.525% | 23.333% |

### heldout_raw_cvsm_a

| Artist | Native | Mismatch 32k | Both 32k |
|---|---:|---:|---:|
| Iron & Wine | 29.556% | 41.444% | 52.889% |
| Men at Work | 27.444% | 39.556% | 33.889% |
| SAAY | 1.444% | 6.889% | 5.444% |
| The Album Leaf | 2.000% | 6.889% | 6.444% |
| Tim McGraw | 1.000% | 9.444% | 1.556% |
| Warpaint | 21.444% | 34.333% | 22.556% |

### heldout_raw_cvsm_art

| Artist | Native | Mismatch 32k | Both 32k |
|---|---:|---:|---:|
| Iron & Wine | 29.889% | 68.889% | 32.889% |
| Men at Work | 16.000% | 66.778% | 26.222% |
| SAAY | 0.222% | 7.889% | 1.333% |
| The Album Leaf | 12.667% | 31.889% | 19.000% |
| Tim McGraw | 1.556% | 46.444% | 2.000% |
| Warpaint | 17.000% | 79.111% | 24.111% |

### heldout_raw_ssl

| Artist | Native | Mismatch 32k | Both 32k |
|---|---:|---:|---:|
| Iron & Wine | 44.222% | 56.556% | 50.222% |
| Men at Work | 38.889% | 50.556% | 50.222% |
| SAAY | 5.444% | 37.556% | 17.333% |
| The Album Leaf | 3.667% | 3.444% | 8.667% |
| Tim McGraw | 9.889% | 34.667% | 18.667% |
| Warpaint | 48.222% | 51.111% | 43.444% |

### mert

| Artist | Native | Mismatch 32k | Both 32k |
|---|---:|---:|---:|
| Adelitas Way | 3.788% | 37.273% | 3.131% |
| Bedouine | 14.646% | 29.545% | 15.000% |
| Claudia Leitte | 30.404% | 63.384% | 38.737% |
| Elder Island | 22.929% | 39.242% | 20.202% |
| Gridlink | 0.000% | 7.929% | 0.000% |
| Gunship | 23.283% | 63.687% | 26.919% |
| Iron & Wine | 27.727% | 44.747% | 28.687% |
| Men at Work | 19.293% | 47.424% | 17.626% |
| SAAY | 4.646% | 35.354% | 5.859% |
| The Album Leaf | 57.374% | 64.848% | 47.929% |
| Tim McGraw | 14.141% | 43.182% | 13.687% |
| Warpaint | 25.808% | 60.909% | 26.111% |

### ssl

| Artist | Native | Mismatch 32k | Both 32k |
|---|---:|---:|---:|
| Adelitas Way | 16.970% | 32.778% | 5.909% |
| Bedouine | 39.697% | 45.556% | 36.717% |
| Claudia Leitte | 42.980% | 54.293% | 48.434% |
| Elder Island | 37.677% | 42.121% | 40.404% |
| Gridlink | 5.960% | 11.970% | 3.030% |
| Gunship | 20.202% | 58.283% | 39.747% |
| Iron & Wine | 41.515% | 56.465% | 42.222% |
| Men at Work | 39.747% | 54.040% | 47.020% |
| SAAY | 4.343% | 38.485% | 16.111% |
| The Album Leaf | 2.020% | 2.020% | 7.475% |
| Tim McGraw | 10.455% | 37.980% | 16.111% |
| Warpaint | 46.667% | 52.273% | 41.010% |

### wavlm

| Artist | Native | Mismatch 32k | Both 32k |
|---|---:|---:|---:|
| Adelitas Way | 10.808% | 13.586% | 13.333% |
| Bedouine | 39.798% | 41.364% | 39.394% |
| Claudia Leitte | 42.323% | 46.010% | 47.525% |
| Elder Island | 26.616% | 27.374% | 33.939% |
| Gridlink | 0.404% | 1.566% | 0.152% |
| Gunship | 31.869% | 36.818% | 37.273% |
| Iron & Wine | 45.909% | 44.343% | 43.384% |
| Men at Work | 21.515% | 25.556% | 28.586% |
| SAAY | 15.000% | 12.121% | 13.131% |
| The Album Leaf | 36.010% | 37.929% | 31.111% |
| Tim McGraw | 16.768% | 16.263% | 15.606% |
| Warpaint | 34.495% | 35.960% | 16.869% |

### yakura

| Artist | Native | Mismatch 32k | Both 32k |
|---|---:|---:|---:|
| Adelitas Way | 41.010% | 63.081% | 57.273% |
| Bedouine | 45.505% | 52.475% | 34.141% |
| Claudia Leitte | 48.788% | 51.465% | 51.061% |
| Elder Island | 62.778% | 57.172% | 54.091% |
| Gridlink | 44.141% | 26.414% | 24.545% |
| Gunship | 59.545% | 64.192% | 55.556% |
| Iron & Wine | 38.889% | 43.636% | 41.162% |
| Men at Work | 41.263% | 48.737% | 50.960% |
| SAAY | 39.141% | 29.848% | 27.020% |
| The Album Leaf | 52.121% | 54.343% | 56.919% |
| Tim McGraw | 48.384% | 39.545% | 40.657% |
| Warpaint | 45.960% | 40.758% | 45.859% |

