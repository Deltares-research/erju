# 1/3-Octave Bands — How They Work

## What is an octave?

An **octave** is a doubling of frequency. For example, 10 Hz → 20 Hz is one octave.
A **1/3-octave band** splits each octave into three equal parts on a logarithmic scale.

This means the bands get **wider as frequency increases**, which matches how structures and humans perceive vibration — differences at low frequencies feel larger than differences at high frequencies.

---

## How the centre frequencies are calculated

The formula comes from **ISO 18405:2017** (and ISO 266):

$$f_c^{(n)} = 1000 \cdot 2^{n/3} \quad \text{Hz}$$

where $n$ is an integer index. Setting $n = 0$ gives 1000 Hz (the reference), and stepping down by 1 each time gives the lower bands:

| n | $f_c$ (Hz) | Nominal name |
|---|-----------|--------------|
| -30 | 1.00 | 1 Hz |
| -29 | 1.26 | 1.25 Hz |
| -28 | 1.59 | 1.6 Hz |
| … | … | … |
| -10 | 10.00 | 10 Hz |
| … | … | … |
| 0 | 1000.00 | 1 kHz |

---

## How the band edges are calculated

Each band is centred on $f_c$ with edges placed symmetrically on the log scale:

$$f_\text{lower} = \frac{f_c}{2^{1/6}}, \qquad f_\text{upper} = f_c \cdot 2^{1/6}$$

The factor $2^{1/6} \approx 1.122$ means each edge is half a 1/3-octave step away from the centre.

---

## The 21 bands used in Parquet v2 (1 Hz – 100 Hz)

| # | Nominal $f_c$ | Lower (Hz) | Upper (Hz) | BW (Hz) | Column prefix   |
|---|---------------|------------|------------|---------|-----------------|
| 1 | 1.00          | 0.891      | 1.122      | 0.23    | `fo_oct_1_00hz` |
| 2 | 1.25          | 1.114      | 1.403      | 0.29    | `fo_oct_1_25hz` |
| 3 | 1.60          | 1.425      | 1.796      | 0.37    | `fo_oct_1_60hz` |
| 4 | 2.00          | 1.782      | 2.245      | 0.46    | `fo_oct_2_00hz` |
| 5 | 2.50          | 2.227      | 2.806      | 0.58    | `fo_oct_2_50hz` |
| 6 | 3.15          | 2.806      | 3.536      | 0.73    | `fo_oct_3_15hz` |
| 7 | 4.00          | 3.564      | 4.490      | 0.93    | `fo_oct_4_00hz` |
| 8 | 5.00          | 4.455      | 5.612      | 1.16    | `fo_oct_5_00hz` |
| 9 | 6.30          | 5.613      | 7.072      | 1.46    | `fo_oct_6_30hz` |
| 10 | 8.00         | 7.127      | 8.980      | 1.85    | `fo_oct_8_00hz` |
| 11 | 10.0         | 8.909      | 11.225     | 2.32    | `fo_oct_010hz`  |
| 12 | 12.5         | 11.136     | 14.031     | 2.89    | `fo_oct_012hz`  |
| 13 | 16.0         | 14.254     | 17.959     | 3.71    | `fo_oct_016hz`  |
| 14 | 20.0         | 17.818     | 22.449     | 4.63    | `fo_oct_020hz`  |
| 15 | 25.0         | 22.273     | 28.062     | 5.79    | `fo_oct_025hz`  |
| 16 | 31.5         | 28.063     | 35.358     | 7.29    | `fo_oct_031hz`  |
| 17 | 40.0         | 35.636     | 44.899     | 9.26    | `fo_oct_040hz`  |
| 18 | 50.0         | 44.545     | 56.123     | 11.58   | `fo_oct_050hz`  |
| 19 | 63.0         | 56.127     | 70.715     | 14.59   | `fo_oct_063hz`  |
| 20 | 80.0         | 71.272     | 89.797     | 18.53   | `fo_oct_080hz`  |
| 21 | 100.0        | 89.090     | 100.000    | 10.91   | `fo_oct_100hz`  |

> **Note:** Band 21 is clipped at 100 Hz (the bandpass filter cut-off), so its width is narrower than expected.

---

## How the feature value is computed

For each band, the feature value is the **integral of the Welch PSD** over the band:

$$E_\text{band} = \sum_{f_i \in [f_\text{lower},\ f_\text{upper})} P_{xx}(f_i) \cdot \Delta f \quad [\text{strain}^2]$$

- $P_{xx}$ is the power spectral density from Welch's method (Hamming window, `nperseg=1024`, `nfft=10000`)
- $\Delta f$ is the frequency resolution of the PSD
- The result is **length-independent** — it does not grow with longer signals

Each band value is then reduced across all FO channels to give three columns per band:

| Suffix | Meaning |
|--------|---------|
| `_mean` | Mean across all FO channels |
| `_max`  | Maximum across all FO channels |
| `_std`  | Standard deviation across all FO channels |

So band 11 (10 Hz nominal) produces: `fo_oct_010hz_mean`, `fo_oct_010hz_max`, `fo_oct_010hz_std`.

---

## Why logarithmic bands instead of linear?

| Aspect | Linear (v1) | 1/3-Octave (v2) |
|--------|-------------|-----------------|
| Band spacing | 5 Hz fixed width | Gets wider with frequency |
| Low-freq resolution | 3 bands below 15 Hz | 10 bands below 15 Hz |
| High-freq resolution | Same as low | Wider bands (less needed) |
| Physical basis | Arbitrary | Matches structural / human perception |
| Number of bands | 13 | 21 |
| Total spectral features | 39 | 63 |
