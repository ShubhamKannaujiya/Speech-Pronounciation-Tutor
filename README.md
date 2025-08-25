---

# Speech Recognition Using HMM and LBG Codebook

## Overview

This project implements a **digit-based speech recognition system** using **Hidden Markov Models (HMM)** and **Linde-Buzo-Gray (LBG) codebook generation** for feature extraction. It supports training on recorded audio samples and testing new recordings for digit recognition.

The system includes:

* **Audio Preprocessing:** DC shift removal, normalization, and frame segmentation.
* **Feature Extraction:** LPC coefficients → Cepstral coefficients → Tokhura distance-based quantization.
* **Codebook Generation:** LBG algorithm for creating a compact representation of speech features.
* **HMM Training:** Forward-Backward (Baum-Welch) and Viterbi algorithms for state estimation and model re-estimation.
* **Recognition:** Testing audio against trained models for digit identification.

---

## Table of Contents

1. [Requirements](#requirements)
2. [Installation](#installation)
3. [Usage](#usage)
4. [File Structure](#file-structure)
5. [Features](#features)
6. [Algorithm Overview](#algorithm-overview)
7. [Contributing](#contributing)
8. [License](#license)

---

## Requirements

- Windows OS
- Visual Studio 2010 (or compatible)
- C++ Compiler
- **Libraries:**

  - Windows Multimedia (`winmm.lib`)
  - SAPI (`sapi.lib`) for audio input

---

## Installation

1. Clone the repository:

```bash
git clone <repository_url>
```

2. Open the project in Visual Studio 2010.

3. Ensure all `.cpp` and `.h` files are included.

4. Link required libraries:

   - `winmm.lib`
   - `sapi.lib`

5. Build the solution.

---

## Usage

### Recording Audio

- Launch the application and use **Start Recording** and **Stop Recording** buttons.
- Audio is captured in 16-bit PCM format and stored in memory for processing.

### Training the System

1. Normalize and process the recorded audio to extract features.
2. Generate a **universe of vectors** from training files:

```cpp
generate_universe();
```

3. Generate **LBG codebook** for vector quantization:

```cpp
generate_codebook();
```

4. Initialize HMM models for each digit:

```cpp
set_initial_model();
```

5. Train HMM using Forward-Backward and Viterbi algorithms:

```cpp
train_model(digit, utterance);
calculate_avg_model_param(digit);
store_final_lambda(digit);
```

### Testing/Recognition

1. Process the test audio file to extract features.
2. Generate observation sequences using Tokhura distance:

```cpp
generate_observation_sequence(testFile);
```

3. Use HMM models to compute probability scores and recognize the digit.

---

## File Structure

```
├── _dataset/           # Training audio files (_E_digit_utterance.txt)
├── _lambda/            # HMM trained parameters (A and B matrices)
├── initial_model/      # Initial HMM parameters
├── universe.csv        # Universe vectors of features
├── codebook.csv        # Generated codebook
├── main.cpp            # Main application file
├── speech_processing.cpp # Feature extraction and preprocessing
├── lbg.cpp             # LBG codebook generation
├── hmm.cpp             # HMM training and recognition
├── README.md
```

---

## Features

- Frame-level energy calculation for speech segmentation
- LPC → Cepstral coefficient extraction with **raised sine liftering**
- **Tokhura distance** for similarity measurement
- **LBG algorithm** to generate codebook
- HMM training using **Forward-Backward** and **Viterbi**
- Observation sequence generation for digit recognition

---

## Algorithm Overview

1. **Preprocessing:**

   - Remove DC offset, normalize audio, segment into frames.

2. **Feature Extraction:**

   - Compute LPC coefficients per frame
   - Transform LPC to Cepstral coefficients
   - Apply raised sine window
   - Quantize using LBG codebook

3. **HMM Training:**

   - Initialize matrices `A`, `B`, `π`
   - Compute α (forward), β (backward), γ, ξ matrices
   - Re-estimate model parameters iteratively

4. **Recognition:**

   - Generate observation sequence for test audio
   - Compute probability using trained HMM
   - Predict digit with maximum likelihood

---

## Contributing

Contributions are welcome! If you wish to improve or extend the project:

1. Fork the repository
2. Create a branch: `git checkout -b feature-name`
3. Commit your changes: `git commit -m "Description of change"`
4. Push to branch: `git push origin feature-name`
5. Open a Pull Request

---

## License

This project is **open-source**. You may use and modify it for academic and personal purposes.

---
