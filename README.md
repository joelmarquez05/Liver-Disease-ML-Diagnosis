# Indian Liver Patient Dataset (ILPD) Classification

## 📖 Project Overview

This project presents a comprehensive machine learning approach to predict the presence of liver disease using the Indian Liver Patient Dataset (ILPD). It was developed as a submission for a university-level Kaggle competition.

The primary goal is to build a robust classification model that accurately distinguishes between patients with and without liver disease based on various clinical attributes. The project covers the entire data science workflow, including:

-   **Exploratory Data Analysis (EDA)** to understand data distributions and relationships.
-   **Feature Engineering** to create more meaningful variables.
-   **Data Preprocessing** to handle outliers, skewed data, and scaling.
-   **Model Training and Hyperparameter Tuning** for various classification algorithms.
-   **Ensemble Methods** (Voting and Stacking) to improve predictive performance.
-   **Advanced Techniques** for handling class imbalance, such as SMOTE and optimal threshold tuning.

Everything has been documented on both the Jupyter Notebook and report.pdf.

---

## 📊 Dataset

The dataset used is the **Indian Liver Patient Dataset (ILPD)**. It consists of ten clinical features collected from patients, which are used to predict a binary outcome: whether the patient has liver disease (`0` for Patient) or not (`1` for Healthy).

**Features:**
*   `Age`: Age of the patient.
*   `Female`: Gender of the patient (binary).
*   `TB`: Total Bilirubin.
*   `DB`: Direct Bilirubin.
*   `Alkphos`: Alkaline Phosphatase.
*   `Sgpt`: Alamine Aminotransferase.
*   `Sgot`: Aspartate Aminotransferase.
*   `TP`: Total Proteins.
*   `ALB`: Albumin.
*   `A/R`: Albumin and Globulin Ratio.

A key challenge identified during the EDA was a significant **class imbalance**, with a much larger number of patients diagnosed with liver disease than healthy individuals. This required specialized techniques to avoid building a biased model.

---

## ⚙️ Methodology

The project follows a structured approach to build and evaluate the classification models, with a strong emphasis on robustness and handling data challenges.

#### 1. Exploratory Data Analysis (EDA)
-   **Target Variable Distribution**: Confirmed a significant class imbalance, making F1-score a more suitable evaluation metric than accuracy.
-   **Feature Distribution**: Histograms and boxplots revealed that several key features (`TB`, `DB`, `Alkphos`, `Sgpt`, `Sgot`) were heavily right-skewed and contained numerous outliers.
-   **Correlation Analysis**: A heatmap showed high multicollinearity between certain pairs of features, such as `Sgpt`/`Sgot` (0.91), `TB`/`DB` (0.85), and `TP`/`ALB` (0.80).

#### 2. Feature Engineering
To address multicollinearity and create clinically relevant features, the following steps were taken:
-   **Indirect Bilirubin (IB)**: A new feature `IB = TB - DB` was created. Subsequently, `DB` was dropped to reduce redundancy.
-   **Total Proteins (TP) Removal**: The `TP` feature was dropped due to its high correlation with `ALB`, retaining Albumin as a more specific liver function indicator.

#### 3. Data Preprocessing
-   **Log Transformation**: To normalize the right-skewed distributions of `TB`, `Alkphos`, `Sgpt`, `Sgot`, and the new `IB` feature, a `log1p` transformation was applied.
-   **Robust Scaling**: `RobustScaler` was used instead of `StandardScaler` to handle the large number of outliers effectively. It scales data based on the interquartile range (IQR), making it less sensitive to extreme values.

#### 4. Model Training and Evaluation
A variety of models were trained and evaluated using `StratifiedKFold` cross-validation to preserve class proportions.
-   **Base Models**: Logistic Regression, K-Nearest Neighbors (KNN), and Support Vector Classifier (SVC).
-   **Tree-Based Models**: Random Forest and Extra Trees Classifier.
-   **Ensemble Models**: A `VotingClassifier` (soft voting) and a `StackingClassifier` (with Logistic Regression as the meta-model) were implemented to combine the strengths of the base models.

#### 5. Handling Class Imbalance
-   **SMOTE (Synthetic Minority Over-sampling Technique)**: Integrated into all training pipelines to oversample the minority class (healthy patients) and create a balanced training set for each fold.
-   **Optimal Threshold Tuning**: For each model, the optimal probability threshold was determined using cross-validated predictions to maximize the macro F1-score, ensuring a better balance between precision and recall for both classes.

---

## 📈 Results

The models were systematically evaluated based on their macro F1-score from cross-validation. The **Extra Trees Classifier** emerged as the top-performing model.

| Model                  | CV F1-score (Macro) |
| ---------------------- | ------------------- |
| **ExtraTreesClassifier**   | **0.7062**          |
| StackingClassifier     | 0.6927              |
| VotingClassifier       | 0.6926              |
| RandomForestClassifier | 0.6907              |
| LogisticRegression     | 0.6628              |
| KNeighborsClassifier   | 0.6562              |
| SVC                    | 0.6474              |

The **Extra Trees Classifier** was selected for the final submission due to its superior F1-score and its excellent balance in recall for both classes (0.75 for patients and 0.71 for healthy individuals). This balance makes it a reliable and robust choice for this medical diagnosis problem.

**The final model achieved a public score of 0.6666 in the Kaggle competition.**

---

## 🚀 How to Run

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/joelmarquez05/indian-liver-patient-analysis.git
    cd indian-liver-patient-analysis
    ```

2.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Jupyter Notebook:**
    Navigate to the `notebooks/` directory and open the `ILPD_Analysis_and_Modeling.ipynb` file using Jupyter Notebook or Jupyter Lab to see the full analysis and reproduce the results.

---

## ✍️ Authors

-   **Rebeca Torrecilla**
-   **Joel Márquez**
