from rouge_score import rouge_scorer

openai_output = ["The most common diagnoses in the hospital seem to vary widely among patients. However, some recurring conditions include Acute kidney failure, unspecified, Type 2 diabetes mellitus without complications, and Chronic kidney disease, unspecified. Note that this information is retrieved from a sample of patient records and may not represent the most common diagnoses for all patients in the hospital.",
                 "Based on the data retrieved, there were 10 hospital admissions between the years 2150 and 2160 at Northside Hospital.",
                 "Based on the data I retrieved, the most common types of admission at the hospital are 'EU OBSERVATION,' 'EW EMER.,' 'ELECTIVE,' 'OBSERVATION ADMIT,' and 'DIRECT EMER.'."]

minilm_output=["Based on the data retrieved, it seems that there is no single most common diagnosis in the hospital. The diagnoses are diverse and depend on various factors such as the patient's health condition, age, and other variables. They range from various forms of heart disease, various types of neoplasms, kidney failure and more. Please note that this information might vary as it's based on a limited data sample.",
          "Based on the retrieved data, there were 10 admissions between the years 2150 and 2160. Please note that this data is based on the available information and the actual number might vary.",
          "Based on the data retrieved, the most common admission types at the hospital are 'EW EMER.', 'ELECTIVE', 'DIRECT OBSERVATION', 'OBSERVATION ADMIT', and 'DIRECT EMER.'."]

paraphrase_output=["The most common diagnoses in the hospital appear to include conditions such as acute on chronic systolic heart failure, acute kidney failure, unspecified, type 2 diabetes mellitus without complications, and unspecified essential hypertension. These are derived from a sample of patient records and might not represent the overall most common diagnoses in the hospital. The data also shows that patients have a variety of other diagnoses.",
          "I'm sorry, but I couldn't find the exact number of admissions between the years 2150 and 2160. However, I did find information on several admissions around that time period. For specific numbers, I would need more specific information or access to more detailed records.",
          "Based on the data retrieved, the most common admission types at the hospital are 'EW EMER.', 'OBSERVATION ADMIT', 'URGENT', 'EU OBSERVATION', 'AMBULATORY OBSERVATION', and 'DIRECT OBSERVATION'. However, keep in mind that these are just a few examples from the data available and the results may vary."]

distilroberta_output=["Based on the data retrieved, it's difficult to pinpoint a single most common diagnosis in the hospital as the diagnoses vary widely across patients. However, conditions related to heart diseases, kidney failures, and sepsis appear frequently. Please note that this is a general observation and the most common diagnosis can vary depending on the department or patient population.",
          "Based on the data retrieved, there were several admissions in the hospital between the years 2150 and 2160. However, precise numbers cannot be provided without more specific data.",
          "Based on the retrieved data, the admission types for most patients at the hospital include 'EW EMER.', 'URGENT', 'OBSERVATION ADMIT', 'ELECTIVE', 'AMBULATORY OBSERVATION', and 'EU OBSERVATION'. It seems there is a variety of admission types, thus, it's hard to pinpoint the most common one without further statistical analysis."]

reference=["The most common diagnoses among the patients in this dataset are: Unspecified – 15 occurrences, Acidosis – 5 occurrences, Chronic myeloid leukemia – 4 occurrences, Personal history of irradiation – 4 occurrences, In remission – 4 occurrences, Personal history of antineoplastic chemotherapy – 3 occurrences, BCR/ABL-positive – 3 occurrences, Personal history of other malignant neoplasm of kidney – 3 occurrences, Nausea with vomiting – 3 occurrences, Anemia – 3 occurrences",
           "There were 8 admissions done between the years 2150 and 2160.",
           "The most common admission type at this hospital is 'EW EMER.'"]

scorer = rouge_scorer.RougeScorer(
    ['rouge1', 'rouge2', 'rougeL'],
)

mod_outputs = {"openai": openai_output, "a": minilm_output, "b": paraphrase_output, "c": distilroberta_output}

all_scores = {}

for model in mod_outputs.keys():
    model_output = mod_outputs[model]
    all_scores[model] = []
    for out, ref in zip(model_output, reference):
        all_scores[model].append(scorer.score(ref, out))

average_scores = {}

for model, scores in all_scores.items():
    avg_rougeL_precision = sum(score['rouge1'].precision for score in scores) / len(scores)
    avg_rougeL_recall = sum(score['rouge1'].recall for score in scores) / len(scores)
    avg_rougeL_f1 = sum(score['rouge1'].fmeasure for score in scores) / len(scores)

    average_scores[model] = {
        'rouge1': {
            'precision': avg_rougeL_precision,
            'recall': avg_rougeL_recall,
            'f1': avg_rougeL_f1,
        }
    }

for model, scores in average_scores.items():
    rougeL_scores = scores['rouge1']
    print(f"Model: {model}")
    print(f"  Rouge-1 Precision: {rougeL_scores['precision']:.4f}")
    print(f"  Rouge-1 Recall: {rougeL_scores['recall']:.4f}")
    print(f"  Rouge-1 F1: {rougeL_scores['f1']:.4f}")