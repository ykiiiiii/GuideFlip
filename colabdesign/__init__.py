"""ColabDesign, vendored.

Upstream: https://github.com/sokrypton/ColabDesign at e31a56f.

Only the AlphaFold design protocols are kept. The mpnn, tr, rf, seq and esm_msa
submodules are not used here and were not vendored, together with the model weights
that shipped with them; the two imports they provided are removed below. Everything
under af/ and shared/ is upstream's, byte for byte.

See VENDOR_colabdesign.md at the repository root for the full accounting.
"""
import os,jax
# disable triton_gemm for jax versions > 0.3
if int(jax.__version__.split(".")[1]) > 3:
  os.environ["XLA_FLAGS"] = "--xla_gpu_enable_triton_gemm=false"

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from colabdesign.shared.utils import clear_mem
from colabdesign.af.model import mk_af_model

# backward compatability
mk_design_model = mk_afdesign_model = mk_af_model
