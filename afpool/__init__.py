"""afpool: open-set radio-frequency identification of individual drones by acquisition-factor pooling.

Modules:
  signal    burst windows, the encoder views, and the pre-onset noise spectrum
  encoder   identity encoder (f_T and f_A share the architecture) and unit embeddings
  training  training objective of the encoders (reference implementation)
  recr      relative-evidence consensus rejection
  rgd       receiver-grouped discovery with WCCN and its enrolled-only calibration
  metrics   rejection and discovery measures
"""
__version__ = '1.0.0'
