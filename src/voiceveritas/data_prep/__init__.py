"""Public dataset download & preparation (reproducible benchmarks).

Datasets are prepared into a uniform manifest format with per-dataset
speech-style labels:

============ ============ ============================= =====================
dataset      style/label  source                        license
============ ============ ============================= =====================
librispeech  read (1)     OpenSLR 12 (dev-clean etc.)   CC BY 4.0
ami          spont. (0)   AMI Meeting Corpus (Edinburgh) CC BY 4.0
sbcsae       spont. (0)   OpenSLR 155                   CC BY-ND 3.0 US
coraal       spont. (0)   oraal.uoregon.edu             CC BY-NC-SA 4.0
============ ============ ============================= =====================

Audio is *never* redistributed with the package — only preparation code.
"""

from voiceveritas.data_prep.ami import AMI_SESSIONS, prepare_ami
from voiceveritas.data_prep.download import download_file
from voiceveritas.data_prep.librispeech import prepare_librispeech

__all__ = [
    "AMI_SESSIONS",
    "download_file",
    "prepare_ami",
    "prepare_librispeech",
]
