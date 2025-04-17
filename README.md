# Disease Prediction Using Random Forest Classifier

## Project Description
This machine learning project implements a Random Forest Classification model to predict disease occurrence in patients. The model analyzes patient medical data and provides binary predictions (0: No disease, 1: Disease present).

## Project Structure
```
SE22UCSE094/
│
├── main.py                               # Main model implementation
├── student_SE22UCSE094_predictions.csv   # Model predictions output
└── README.md                            # Project documentation
```

## Technical Implementation
### Model Details
- **Algorithm**: Random Forest Classifier
- **Training Split**: 80% training, 20% testing
- **Input**: Patient medical features
- **Output**: Binary classification (0/1)
- **Implementation**: scikit-learn

### Data Processing
1. Data loading from CSV files
2. Null value handling
3. Feature extraction
4. Train-test splitting
5. Model training and evaluation
6. Prediction generation

## Setup and Usage
### Prerequisites
```bash
numpy
pandas
scikit-learn
```

### Running the Model
1. Place your data files:
   - Disease_train.csv (training data)
   - Disease_test.csv (test data)
2. Execute main.py:
   ```bash
   python main.py
   ```
3. Check predictions in output file

## Results
- Model generates patient-specific predictions
- Output format: CSV with patient_id and prediction columns
- Predictions stored in student_SE22UCSE094_predictions.csv

## Future Improvements
- Feature importance analysis
- Model parameter optimization
- Cross-validation implementation
- Additional evaluation metrics

## Contact
For questions or improvements, please raise an issue in the repository.
