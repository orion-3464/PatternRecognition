from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# a) Calculation of accuracy for every trained model
print("="*60)
print("a) ACCURACY")
print("="*60)
for model_name, results in trained.items():
	acc = accuracy_score(results['y_true'], results['y_pred'])
	print(f"{model_name}: {acc:.4f}")

# b) Precision, Recall, F1-score for every class
print("\n" + "="*60)
print("b) PER-CLASS METRICS (Precision, Recall, F1-score)")
print("="*60)
for model_name, results in trained.items():
	print(f"\n{model_name}:")
	print(classification_report(results['y_true'], results['y_pred'], 
								target_names=NEW_CLASSES))

# c) Macro-averaged metrics
print("\n" + "="*60)
print("c) MACRO-AVERAGED METRICS")
print("="*60)
for model_name, results in trained.items():
	macro_precision = precision_score(results['y_true'], results['y_pred'], average='macro')
	macro_recall = recall_score(results['y_true'], results['y_pred'], average='macro')
	macro_f1 = f1_score(results['y_true'], results['y_pred'], average='macro')
	print(f"{model_name}:")
	print(f"  Precision: {macro_precision:.4f}, Recall: {macro_recall:.4f}, F1: {macro_f1:.4f}")

# d) Micro-averaged metrics
print("\n" + "="*60)
print("d) MICRO-AVERAGED METRICS")
print("="*60)
for model_name, results in trained.items():
	micro_precision = precision_score(results['y_true'], results['y_pred'], average='micro')
	micro_recall = recall_score(results['y_true'], results['y_pred'], average='micro')
	micro_f1 = f1_score(results['y_true'], results['y_pred'], average='micro')
	print(f"{model_name}:")
	print(f"  Precision: {micro_precision:.4f}, Recall: {micro_recall:.4f}, F1: {micro_f1:.4f}")
