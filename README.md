# Fish Nourishment & Mass Predictor (Image + Manual)

This project provides:
- A Streamlit app that supports **manual measurements** input AND **image-based measurement extraction**.
- A simple OpenCV-based measurement extraction (returns pixel measurements). Convert pixels to cm by entering `pixels_per_cm` in the app.
- Hooks to use your existing `clf.joblib` and `reg.joblib` artifacts (place them in `artifacts/`).
- An outline script to train a species classifier (`src/train_species.py`) - you must prepare `data/species_dataset/*` folders.

## How to use

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Place any trained artifacts (optional) into `artifacts/`:
   - `artifacts/clf.joblib` (nourishment classifier trained on tabular data)
   - `artifacts/reg.joblib` (weight regressor trained on tabular data)
   - `artifacts/species_model.h5` (optional CNN that predicts species class id)

3. Run the app:
   ```bash
   streamlit run src/app.py
   ```

4. Image calibration:
   - For converting pixel measurements to centimeters, use the `pixels_per_cm` input in the Image section.
   - To find pixels_per_cm, take a picture containing a ruler and count how many pixels correspond to 1cm (you can estimate by inspecting the image).

## Notes & Next steps
- The extraction method is a controlled-environment method (plain background). It will be less accurate on cluttered backgrounds.
- For higher accuracy, collect many images of fish with a ruler and train a keypoint or regression model that maps images directly to measurements.
- The repository can be extended with: Flask API, React frontend, Dockerfile, and a training pipeline for your tabular models.

