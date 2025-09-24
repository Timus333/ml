# 1) Install dependencies 
!pip install numpy pandas scikit-learn tensorflow keras matplotlib --quiet

# 2) Imports
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler                                     # 4
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input      # 5

# 3) Load & preprocess
df = pd.read_csv(r"iris.csv")
X = df.iloc[:, :4].astype('float32').to_numpy()           # ensure float32
labels = pd.factorize(df.iloc[:, 4].astype(str))[0]       # integer labels
y = to_categorical(labels, num_classes=3).astype('float32')

# 4) Load & preprocess (ensure all numeric)
df = pd.read_csv(r"titanic.csv",usecols=['Pclass','Gender','Age','Fare','Survived']).dropna()
df['Gender'] = df['Gender'].map({'male':0,'female':1})
X = df[['Pclass','Gender','Age','Fare']].astype('float32').to_numpy()
y = df['Survived'].astype('float32').to_numpy().reshape(-1,1)

# 5. Load MNIST from .csv
df = pd.read_csv(r"mnist_train.csv")
y = df.iloc[:, 0].values                             # First column is label
X = df.iloc[:, 1:].values.reshape(-1, 28, 28, 1)     # Rest is image → reshape to 28x28x1
X = X.astype('float32') / 255.0                      # Normalize to [0,1]
y = to_categorical(y, num_classes=10)                # One-hot encode labels

# 7) Load and preprocess
df = pd.read_csv(r"banknote.csv")  
X = df.iloc[:, :-1].astype('float32').to_numpy()  # all columns except the last
y = df.iloc[:,  -1].astype('float32').to_numpy().reshape(-1, 1)  # last column

# train_test_split
# 3. Split into train/val/test (60/20/20)
X_tmp, X_test, y_tmp, y_test, lbl_tmp, lbl_test = train_test_split(X, y, labels, test_size=0.2, stratify=labels, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_tmp, y_tmp, test_size=0.25, stratify=lbl_tmp, random_state=42)

# 4) Split & scale
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
scaler = StandardScaler().fit(X_tr)
X_tr, X_te = scaler.transform(X_tr).astype('float32'), scaler.transform(X_te).astype('float32')

# 5. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=np.argmax(y, axis=1), random_state=42)

# 7) Train/test split & scaling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test  = scaler.transform(X_test)

# Model created
# 3) Model definition
model = Sequential([
    Input(shape=(4,)),
    Dense(16, activation='relu'),
    Dense(12, activation='relu'),
    Dense(3,  activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 4) Build model with Input()
model = Sequential([
    Input(shape=(4,)),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 5. Define CNN model
model = Sequential([
    Input(shape=(28,28,1)),
    Conv2D(32, 3, activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])
model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])

# 7) Build the model
model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(10, activation='relu'),
    Dense( 1, activation='sigmoid')
])
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#Train the model
# 3) Train
history = model.fit(X_train, y_train,epochs=30, batch_size=16,validation_data=(X_val, y_val),verbose=0)

# 4) Train (with validation split)
h = model.fit(X_tr, y_tr,epochs=30, batch_size=16,validation_split=0.2, verbose=0)

# 5. Train
hist = model.fit(X_train, y_train, epochs=5, batch_size=128, validation_split=0.2, verbose=0)

# 7) Train
history = model.fit(X_train, y_train,epochs=30,batch_size=16,validation_split=0.2,verbose=0)

# Ploting
# 3) Plot metrics
plt.figure(figsize=(8,3))
for i,(key,title) in enumerate([('loss','Loss'),('accuracy','Accuracy')]):
    plt.subplot(1,2,i+1)
    plt.plot(history.history[key],      label='train')
    plt.plot(history.history['val_'+key], label='val')
    plt.title(title); plt.legend()
plt.tight_layout(); plt.show()

# 4) Plot training history
plt.figure(figsize=(8,3))
for i,(key,title) in enumerate([('loss','Loss'),('accuracy','Accuracy')]):
    plt.subplot(1,2,i+1)
    plt.plot(h.history[key],      label='train')
    plt.plot(h.history['val_'+key], label='val')
    plt.title(title); plt.legend()
plt.tight_layout(); plt.show()

# 5. Plot training curves
pd.DataFrame(hist.history)[['loss','val_loss','accuracy','val_accuracy']]\
    .plot(figsize=(10,4), grid=True, title="CNN Training vs Validation Curves")
plt.show()

# 7) Plot training history
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.plot(history.history['loss'],   label='train')
ax1.plot(history.history['val_loss'], label='val')
ax1.set_title('Loss');    ax1.legend()
ax2.plot(history.history['accuracy'],   label='train')
ax2.plot(history.history['val_accuracy'], label='val')
ax2.set_title('Accuracy'); ax2.legend()
plt.tight_layout()
plt.show()

# 3) Final evaluation
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss:    {loss:.4f}")
print(f"Test Accuracy:{acc:.4f}")
