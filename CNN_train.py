import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, Callback, ModelCheckpoint
from tensorflow.keras.regularizers import l2
import tensorflow as tf
import seaborn as sns

# ------------------ 数据加载与准备 -------------------
data_dict = pickle.load(open('./data.pickle', 'rb'))
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels']).astype(int)

x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels)

def augment_data(X, y, noise_factor=0.05):
    X_noisy = X + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X.shape)
    X_scaled = X * np.random.uniform(0.9, 1.1, size=X.shape[0])[:, np.newaxis]
    X_augmented = np.vstack([X, X_noisy, X_scaled])
    y_augmented = np.concatenate([y, y, y])
    return X_augmented, y_augmented

x_train_aug, y_train_aug = augment_data(x_train, y_train)

num_classes = len(np.unique(labels))
y_train_aug = np.eye(num_classes)[y_train_aug]
y_test = np.eye(num_classes)[y_test]


# ------------------ 构建模型 -------------------
model = Sequential([
    Dense(128, input_dim=x_train.shape[1], activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.5),
    Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# ------------------ 对抗训练类 -------------------
class FGM:
    def __init__(self, model):
        self.model = model
        self.perturbations = []

    def attack(self, x_batch, y_batch, epsilon=0.1):
        self.perturbations = []
        with tf.GradientTape() as tape:
            tape.watch(self.model.trainable_variables)
            predictions = self.model(x_batch)
            loss = tf.keras.losses.categorical_crossentropy(y_batch, predictions)
            loss = tf.reduce_mean(loss)
        grads = tape.gradient(loss, self.model.trainable_variables)

        for grad in grads:
            if grad is not None:
                norm = tf.norm(grad)
                if norm != 0:
                    self.perturbations.append(epsilon * grad / norm)
                else:
                    self.perturbations.append(tf.zeros_like(grad))
            else:
                self.perturbations.append(None)

        for i, var in enumerate(self.model.trainable_variables):
            if self.perturbations[i] is not None:
                var.assign_add(self.perturbations[i])

    def restore(self):
        for i, var in enumerate(self.model.trainable_variables):
            if self.perturbations[i] is not None:
                var.assign_sub(self.perturbations[i])


# ------------------ 自定义回调实现对抗训练 -------------------
class AdversarialTrainingCallback(Callback):
    def __init__(self, x, y, batch_size=32, epsilon=0.1):
        super().__init__()
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.fgm = None

    def on_train_begin(self, logs=None):
        self.fgm = FGM(self.model)

    def on_epoch_end(self, epoch, logs=None):
        print(f"FGM adversarial training for epoch {epoch + 1}")
        for i in range(0, len(self.x), self.batch_size):
            x_batch = self.x[i:i+self.batch_size]
            y_batch = self.y[i:i+self.batch_size]
            self.fgm.attack(x_batch, y_batch, self.epsilon)
            self.model.train_on_batch(x_batch, y_batch)
            self.fgm.restore()


# ------------------ 回调函数设置 -------------------
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)
adv_callback = AdversarialTrainingCallback(x_train_aug, y_train_aug, batch_size=32, epsilon=0.1)


# ------------------ 正式训练 -------------------
history = model.fit(
    x_train_aug, y_train_aug,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    callbacks=[early_stopping, checkpoint, adv_callback],
    verbose=1
)

# ------------------ 评估与保存 -------------------
model.load_weights('best_model.h5')
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# 计算各种评估指标
acc = accuracy_score(y_true_classes, y_pred_classes)
precision = precision_score(y_true_classes, y_pred_classes, average='weighted')
recall = recall_score(y_true_classes, y_pred_classes, average='weighted')
f1 = f1_score(y_true_classes, y_pred_classes, average='weighted')

print(f'\nTest Accuracy: {acc * 100:.2f}%')
print(f'Test Precision: {precision * 100:.2f}%')
print(f'Test Recall: {recall * 100:.2f}%')
print(f'Test F1 Score: {f1 * 100:.2f}%')

# 计算每个类别的评估指标
class_precision = precision_score(y_true_classes, y_pred_classes, average=None)
class_recall = recall_score(y_true_classes, y_pred_classes, average=None)
class_f1 = f1_score(y_true_classes, y_pred_classes, average=None)

# ------------------ 可视化 -------------------
# 1. 训练过程中的损失和准确率曲线
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy Curve')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.savefig('training_curves.png')
plt.show()

# 2. 混淆矩阵
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_true_classes, y_pred_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('confusion_matrix.png')
plt.show()

# 3. 整体评估指标的条形图
plt.figure(figsize=(10, 6))
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
values = [acc, precision, recall, f1]
plt.bar(metrics, values, color=['blue', 'green', 'orange', 'red'])
plt.ylim(0, 1.0)
plt.title('Model Performance Metrics')
plt.ylabel('Score')
for i, v in enumerate(values):
    plt.text(i, v + 0.02, f'{v:.4f}', ha='center')
plt.savefig('overall_metrics.png')
plt.show()

# 4. 每个类别的评估指标
plt.figure(figsize=(12, 8))
x = np.arange(len(class_precision))
width = 0.25

plt.bar(x - width, class_precision, width, label='Precision', color='green')
plt.bar(x, class_recall, width, label='Recall', color='orange')
plt.bar(x + width, class_f1, width, label='F1', color='red')

plt.xlabel('Class')
plt.ylabel('Score')
plt.title('Per-Class Performance Metrics')
plt.xticks(x)
plt.ylim(0, 1.0)
plt.legend()
plt.tight_layout()
plt.savefig('per_class_metrics.png')
plt.show()

# 5. 将准确率、精确率、召回率和F1值绘制成雷达图
plt.figure(figsize=(8, 8))
categories = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
values = [acc, precision, recall, f1]

# 闭合雷达图
values = np.concatenate((values, [values[0]]))
categories = np.concatenate((categories, [categories[0]]))

angles = np.linspace(0, 2*np.pi, len(categories)-1, endpoint=False)
angles = np.concatenate((angles, [angles[0]]))

plt.polar(angles, values)
plt.fill(angles, values, alpha=0.1)
plt.xticks(angles[:-1], categories[:-1])
plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ['0.2', '0.4', '0.6', '0.8', '1.0'], color='gray')
plt.ylim(0, 1)

# 在每个轴上添加值标签
for i, (angle, value) in enumerate(zip(angles, values)):
    if i < len(values) - 1:  # 不绘制重复的第一个点
        plt.text(angle, value + 0.05, f'{value:.2f}',
                horizontalalignment='center', verticalalignment='center')

plt.title('Model Performance Radar Chart')
plt.tight_layout()
plt.savefig('radar_chart.png')
plt.show()