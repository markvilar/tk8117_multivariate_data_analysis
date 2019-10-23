# Data-driven dynamic models and the IDLE model

## PLS-based multivariate metamodeling of dynamic systems

Induction
Deduction

Abduction - Mixture of induction and deduction

Example: Measured absorbance spectra (two ways of analysis)
1) Multivariate calibration (linear PLSR)
2) Semi-causal modelling (non-linear OEMSC)

Example: Hyperspectral images (Hybrid multivariate modelling of causalities)
- Chemical variations
- Physical variations
- Instrument variations
- Measurement conditions
- Outliers
- Measurement noise

1) Weighted least squares (weight so that noise is uniform along wavelengths)

Example: Deshadowing HSI
1) Found difference spectrum between sunny and shady side of a mountain


Example: Oslo HSI
1) Calibration error
2) Multivariate data analysis can discover systematic errors and patterns


## Deep learning

5 layers:
- Input data
- Theory-driven pre-processing
- Data-driven bilinear model development
- Conversion to simpler structure (ICA, MCR, non-negative model)
- Conversion to causal structure

## Example: Thermal camera on high-speed ferry

1) Self-modelling model (subspace model)
2) Anomaly detection
3) Look for pixels with same PCA scores
4) Conversion of thermal video model intro differential equations

## Example: Infrared spectroscopy of an industrial fermentation process at TINE

## Example: Baby RGB video

1) Separate motion and properties

## Preprocesssing

1) MSC
2) Extended MSC
