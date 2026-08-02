"""linec_multisensor
===================
Isolated M0/M1/M2 multi-sensor Line-C spectral experiment package.

Read-only reuse of S1-S6 infrastructure (ResNet2D no-BatchNorm encoder,
metadata branch, fusion head, geometry/metric utilities) -- nothing in
src/ml/spectral/{models_spectral,dataset_spectral,train_spectral,eval_spectral}.py
is imported for write access or modified.
"""
