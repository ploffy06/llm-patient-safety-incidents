# llm-patient-safety-incidents

## About
Using open-source large language models (LLMs) to summarize patient safety incident reports with two main goals:
- extract key information about what caused the incident and
- categorize the safety issue (e.g. medication errors or patient falls).

## Dataset
The dataset contains 20 incident reports that were generated online with ChatGPT and then incident types were manually labelled afterwards:
- allergy: any incident that induced an allergic reaction
- careless oversight: incidents in which a careless oversight in spite of correct signs in place caused harm to the patient
- clinical handover error: occurred during handover
- communication error: when there has been a misunderstanding
- documentation error
- equipment error: incorrect use of medical equipment
- medication error
- missing patient
- sharps disposal error
- surgical error
- technology error: incident occur due to breakdown in technology


![word count](diagrams/word_counts.png)
![incident types](diagrams/incident_types.png)


## Libraries Used
- pandas
- uuid
- transformers
- matplotlib
- sklearn

## Chosen Model Rationale


## Models and Evaluation
### [bart-large-cnn](https://huggingface.co/facebook/bart-large-cnn)
This zero-shot classifier is bart-large trained on MultiNLI dataset in which the hypothesis is "This text is about {label}". BART is a transformer model with a bidirectional encoder and autoregressive decoder - it is good for text comprehension and generation.

Results are shown in `results/bart.txt` in the following format:
```
{idx}, {id}
actual: {target}, is_correct: {target == predicted}
{prob scores}
{labels}
```

Here are the following evaluation metrics
```
accuracy:  0.6
                         precision    recall  f1-score   support

                allergy       1.00      1.00      1.00         2
     careless oversight       0.00      0.00      0.00         3
  sharps disposal error       0.00      0.00      0.00         1
        equipment error       0.50      1.00      0.67         2
         surgical error       0.50      0.50      0.50         2
    communication error       0.50      1.00      0.67         1
clinical handover error       0.40      0.67      0.50         3
       medication error       1.00      1.00      1.00         1
        missing patient       1.00      1.00      1.00         1
       technology error       1.00      1.00      1.00         1
    documentation error       0.50      0.33      0.40         3

               accuracy                           0.60        20
              macro avg       0.58      0.68      0.61        20
           weighted avg       0.51      0.60      0.53        20
```

and the confusion matrix

![bart cm](diagrams/bart_cm.png)

### [roberta-base-squad2](https://huggingface.co/deepset/roberta-base-squad2)
Roberta-base-squad2 is fine-tuned on RoBERTa base using SQuAD2 dataset for extractive answer questioning. RoBERTa us a transformer model
petrained for Masked Language Modelling (MLM). Standford Question Answering Dataset (SQuAD) comprises of contexts, questions and answers crowdworked on wikipedia articles. The answers are a segment of text from the context or otherwise unanswerable (therefore meaning this model can handle unanswerable questions)

Given we wanted to extract key information (i.e. what caused the incident, what is the incident type), I wanted to text the efficacy of question-answering models.

The results are shown in `results/roberta.txt` in the following format:
```
{idx}: {id}
context: {report description}
summary: {what caused the incident}
incident_type: {incident type}
confidence: {summary score} {incident_type score}
```

By inspection of the results in `results/roberta.txt`, here are my findings:
- As this model is hallucination resistant, it will be be able to extract what is in the context and so it proved to have potential in finding "what caused the incident"
- accuracy of what caused the incident...
- The limitiation to it being hallucination resistant is that it was not able to classify the incident type given the labels appropriately since the label may not be in the context

### [PatientSeek](https://huggingface.co/whyhow-ai/PatientSeek)


## Final Chosen Model


## Evaluation

## Optimization
https://huggingface.co/deepset/tinyroberta-squad2
^ runs at twice the speed as base model

## Next Steps
- larger dataset