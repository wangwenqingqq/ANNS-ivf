# Data-access amendment, before the first full measurement

The official FTP transfer projected hours at about 11 KB/s and was limited to
300 seconds; the incomplete archive is retained outside Git and never parsed
as a complete dataset. HTTP alternatives returned 403/404, and HTTPS on the FTP
hostname failed certificate verification. No TLS verification was disabled.

Use byte-range downloads of a publicly accessible SIFT mirror instead:
https://huggingface.co/datasets/hhy3/ann-datasets/tree/0bc1ed67fc2626bc6f110c8794a70572681b7395/sift1M
The original source/standard SIFT file interpretation is also documented by
Faiss: https://github.com/facebookresearch/faiss/blob/main/contrib/datasets.py.
Mirror contents are hash-bound, but byte-equivalence to the unavailable complete
official archive is not independently established. Do not call this a verified
reproduction of the standard SIFT1M benchmark.

There is no learn file in this mirror. Replace the planned learn-prefix training
input with the next 8192 base rows, indices [262144,270336), strictly disjoint
from the 262144-row searchable/update pool. Keep all dimensions, live-set sizes,
64x512 successful replacements, query counts, workloads, controls and decision
rules unchanged. This changes the training split before any full result, not
an after-the-fact selection of a better result. Input manifests record offsets,
byte counts and hashes; no vectors or compressed datasets are committed.
The runner accepts `--base` and `--queries`, each either the public full fvecs
file or the required byte prefix. Python 3.9+ and NumPy 1.22+ are sufficient.

The self-test run before this amendment passed all 1275 small fused plans,
20x100 randomized representation events, and the integer-L2/invalid-ID checks.
No full trace or query-quality observation preceded this amendment.
