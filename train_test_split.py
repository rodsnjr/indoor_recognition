import os
from shutil import copyfile
from sklearn.cross_validation import train_test_split

arr = os.listdir('./')
x = []
y = []

for label in arr:
	if not os.path.isdir(label) and 'dataset_full':
		continue
	for item in os.listdir('./' + label + '/'):
		x.append(item)
		y.append(label)


# Separei em 0.33, pode mudar para ver outro resultado ou algo assim...
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
x_train_final, x_val, y_train_final, y_val = train_test_split(X_train, y_train, test_size=0.33, random_state=42)

os.mkdir('dataset_full')

try:
	os.mkdir('dataset_full/train')
	os.mkdir('dataset_full/validation')
	os.mkdir('dataset_full/testing')

	for label in arr:
		if not os.path.isdir(label)  and 'dataset_full':
			continue
		os.mkdir(os.path.join('train', label))
		os.mkdir(os.path.join('validation', label))
		os.mkdir(os.path.join('testing', label))
except:
	print("Directory might exist")

for x1, y1 in zip(x_train_final, y_train_final):
	copyfile(os.path.join(y1, x1), os.path.join('dataset_full', 'train', y1, x1))

for x1, y1 in zip(x_val, y_val):
	copyfile(os.path.join(y1, x1), os.path.join('dataset_full', 'validation', y1, x1))

for x1, y1 in zip(X_test, y_test):
	copyfile(os.path.join(y1, x1), os.path.join('dataset_full', 'testing', y1, x1))	
