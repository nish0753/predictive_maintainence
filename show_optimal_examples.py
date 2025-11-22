import pickle
import pandas as pd

m = pickle.load(open('svm_model.pkl','rb'))
df = pd.read_csv('Manufacturing_dataset.xls')
cls1 = df[df['Optimal Conditions']==1]
if cls1.shape[0]==0:
    print('No class1 rows found')
    raise SystemExit

examples = cls1.sample(n=min(8, len(cls1)), random_state=42)
print('Found', len(cls1), 'class-1 rows; showing examples to use in app:')
for i, row in examples.iterrows():
    inp = {f: float(row[f]) for f in m['features']}
    X = pd.DataFrame([inp])[m['features']]
    Xs = m['scaler'].transform(X)
    pred = m['model'].predict(Xs)[0]
    proba = m['model'].predict_proba(Xs)[0]
    print('\n--- Example index', i, '---')
    for k,v in inp.items():
        print(f"{k}: {v}")
    print(f'Predicted class: {pred}  |  Probability Optimal: {proba[1]:.4f}  |  Probability Not Optimal: {proba[0]:.4f}')
