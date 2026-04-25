import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import os

from classifier import QuestionClassifier, SimpleTokenizer, CATEGORIES


# =========================
# Training Data (Arabic + English)
# =========================
training_data = [
    # ===== Educational (0) =====
    # English - Programming
    ("what is object oriented programming", 0),
    ("explain inheritance in java", 0),
    ("what is polymorphism", 0),
    ("what are the 4 pillars of OOP", 0),
    ("what are the four pillars of OOP", 0),
    ("explain encapsulation", 0),
    ("what is abstraction in programming", 0),
    ("explain classes and objects", 0),
    ("what is a constructor", 0),
    ("how does web scraping work", 0),
    ("what is beautifulsoup", 0),
    ("explain python classes", 0),
    ("what is a function in programming", 0),
    ("how to use loops in python", 0),
    ("what is an array", 0),
    ("explain linked list", 0),
    ("what is a stack", 0),
    ("what is a queue in programming", 0),
    ("explain sorting algorithms", 0),
    ("what is binary search", 0),
    ("how to use git", 0),
    ("what is version control", 0),
    ("explain design patterns", 0),
    ("what is MVC pattern", 0),
    ("what is an interface in java", 0),
    ("explain abstract class", 0),
    ("what is method overriding", 0),
    ("what is method overloading", 0),
    ("explain exception handling", 0),
    ("what is try catch", 0),
    ("how to read a file in python", 0),
    ("what is a variable", 0),
    ("explain data types", 0),
    ("what is string in programming", 0),
    ("what is an integer", 0),
    ("explain boolean logic", 0),
    ("what is if else statement", 0),
    ("explain while loop", 0),
    ("what is for loop", 0),
    ("explain list comprehension", 0),
    # English - CS / AI
    ("what is machine learning", 0),
    ("explain neural networks", 0),
    ("what is deep learning", 0),
    ("how does backpropagation work", 0),
    ("what is an API", 0),
    ("explain REST API", 0),
    ("what is a database", 0),
    ("how to use SQL", 0),
    ("what is HTML", 0),
    ("explain CSS selectors", 0),
    ("what is recursion", 0),
    ("explain data structures", 0),
    ("what is an algorithm", 0),
    ("explain time complexity", 0),
    ("what is big O notation", 0),
    ("what is a compiler", 0),
    ("explain operating system", 0),
    ("what is a thread", 0),
    ("what is multiprocessing", 0),
    ("explain cloud computing", 0),
    # Arabic - Educational
    ("ما هو البرمجة الكائنية", 0),
    ("اشرح الوراثة في جافا", 0),
    ("ما هو تعدد الأشكال", 0),
    ("ما هي أركان البرمجة الكائنية", 0),
    ("ما هي الأربع ركائز في البرمجة", 0),
    ("اشرح التغليف في البرمجة", 0),
    ("ما هو التجريد", 0),
    ("كيف يعمل تجريف الويب", 0),
    ("ما هي الدوال في البرمجة", 0),
    ("اشرح الحلقات التكرارية", 0),
    ("ما هو التعلم الآلي", 0),
    ("ما هي الشبكات العصبية", 0),
    ("اشرح قواعد البيانات", 0),
    ("ما هو الذكاء الاصطناعي", 0),
    ("ما هي المصفوفات", 0),
    ("اشرح القوائم المتصلة", 0),
    ("ما هو المكدس", 0),
    ("ما هي الخوارزميات", 0),
    ("اشرح هياكل البيانات", 0),
    ("ما هي لغة بايثون", 0),
    ("اشرح لغة جافا", 0),
    ("ما هو الكلاس", 0),
    ("ما هو الكائن في البرمجة", 0),
    ("اشرح الواجهات في جافا", 0),
    ("ما هي قواعد البيانات العلائقية", 0),

    # ===== Legal (1) =====
    # English
    ("what is article 1 of the law", 1),
    ("explain the criminal procedure law", 1),
    ("what are the rights of the accused", 1),
    ("what is the penalty for theft", 1),
    ("explain contract law", 1),
    ("what is civil law", 1),
    ("what are labor rights", 1),
    ("explain the constitution", 1),
    ("what is judicial review", 1),
    ("explain criminal law", 1),
    ("what is habeas corpus", 1),
    ("explain property law", 1),
    ("what is the statute of limitations", 1),
    ("explain family law", 1),
    ("what are human rights", 1),
    ("explain the court system", 1),
    ("what is a legal precedent", 1),
    ("explain intellectual property", 1),
    ("what is due process", 1),
    ("explain tort law", 1),
    ("what is the punishment for murder", 1),
    ("explain legal procedures", 1),
    ("what is a lawsuit", 1),
    ("explain the penal code", 1),
    ("what are constitutional rights", 1),
    ("explain bail and detention", 1),
    ("what is a court order", 1),
    ("explain witness testimony", 1),
    ("what is evidence in court", 1),
    ("explain the appeals process", 1),
    # Arabic
    ("ما هي المادة الأولى من القانون", 1),
    ("اشرح قانون الإجراءات الجنائية", 1),
    ("ما هي حقوق المتهم", 1),
    ("ما هي عقوبة السرقة", 1),
    ("اشرح القانون المدني", 1),
    ("ما هي حقوق العمال", 1),
    ("اشرح الدستور", 1),
    ("ما هو القانون الجنائي", 1),
    ("ما هي حقوق الإنسان", 1),
    ("اشرح قانون الأسرة", 1),
    ("ما هي عقوبة القتل", 1),
    ("اشرح إجراءات المحكمة", 1),
    ("ما هي الدعوى القضائية", 1),
    ("اشرح قانون العقوبات", 1),
    ("ما هي الحقوق الدستورية", 1),
    ("اشرح الكفالة والاحتجاز", 1),
    ("ما هو أمر المحكمة", 1),
    ("اشرح شهادة الشهود", 1),
    ("ما هي الأدلة في المحكمة", 1),
    ("اشرح الاستئناف", 1),

    # ===== General (2) =====
    # English
    ("hello", 2),
    ("hi how are you", 2),
    ("what is the weather today", 2),
    ("tell me a joke", 2),
    ("who are you", 2),
    ("what can you do", 2),
    ("thank you", 2),
    ("goodbye", 2),
    ("help me", 2),
    ("what time is it", 2),
    ("how old are you", 2),
    ("where are you from", 2),
    ("tell me something interesting", 2),
    ("what is your name", 2),
    ("good morning", 2),
    ("good night", 2),
    ("nice to meet you", 2),
    ("how is your day", 2),
    ("what do you think", 2),
    ("can you help me", 2),
    ("how are things going", 2),
    ("what is new", 2),
    ("see you later", 2),
    ("have a nice day", 2),
    ("thanks a lot", 2),
    ("you are welcome", 2),
    ("please help", 2),
    ("I need assistance", 2),
    ("whats up", 2),
    ("hey there", 2),
    # Arabic
    ("مرحبا", 2),
    ("كيف حالك", 2),
    ("ما هو الطقس اليوم", 2),
    ("من أنت", 2),
    ("ماذا تستطيع أن تفعل", 2),
    ("شكرا لك", 2),
    ("مع السلامة", 2),
    ("ساعدني", 2),
    ("صباح الخير", 2),
    ("مساء الخير", 2),
    ("أهلا وسهلا", 2),
    ("كيف الحال", 2),
    ("ما اسمك", 2),
    ("شكرا جزيلا", 2),
    ("تصبح على خير", 2),
    ("يوم سعيد", 2),
    ("كيف يومك", 2),
    ("ما الجديد", 2),
    ("أحتاج مساعدة", 2),
    ("السلام عليكم", 2),
]

# =========================
# Dataset
# =========================
class QuestionDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=50):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, label = self.data[idx]
        encoded = self.tokenizer.encode(text, self.max_len)
        return torch.tensor(encoded, dtype=torch.long), torch.tensor(
            label, dtype=torch.long
        )


# =========================
# Training
# =========================
def train():
    print("=" * 50)
    print("🧠 Training Question Classifier")
    print("=" * 50)

    # Build tokenizer
    tokenizer = SimpleTokenizer(vocab_size=5000)
    texts = [text for text, label in training_data]
    tokenizer.build_vocab(texts)

    print(f"📝 Vocab size: {len(tokenizer.word2idx)}")
    print(f"📊 Training samples: {len(training_data)}")

    # Create dataset
    dataset = QuestionDataset(training_data, tokenizer)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Create model
    model = QuestionClassifier(
        vocab_size=len(tokenizer.word2idx),
        embed_dim=64,
        hidden_dim=128,
        num_classes=3,
        max_len=50,
    )

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train
    num_epochs = 100
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0

        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(output, 1)
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)

        accuracy = correct / total * 100

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch [{epoch+1}/{num_epochs}] Loss: {total_loss:.4f} Accuracy: {accuracy:.1f}%"
            )

    # Save model
    MODEL_DIR = "../models"
    os.makedirs(MODEL_DIR, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "classifier.pth"))
    tokenizer.save(os.path.join(MODEL_DIR, "tokenizer.json"))

    # Save model config
    config = {
        "vocab_size": len(tokenizer.word2idx),
        "embed_dim": 64,
        "hidden_dim": 128,
        "num_classes": 3,
        "max_len": 50,
    }
    with open(os.path.join(MODEL_DIR, "config.json"), "w") as f:
        json.dump(config, f)

    print("=" * 50)
    print(f"✅ Model saved to {MODEL_DIR}/")
    print("=" * 50)

    # Test
    print("\n🧪 Testing:")
    test_questions = [
        "what is inheritance in java",
        "ما هي المادة الأولى من القانون",
        "hello how are you",
        "اشرح البرمجة الكائنية",
        "what is the penalty for murder",
        "مرحبا كيف حالك",
    ]

    model.eval()
    with torch.no_grad():
        for q in test_questions:
            encoded = torch.tensor([tokenizer.encode(q)], dtype=torch.long)
            output = model(encoded)
            probs = torch.softmax(output, dim=1)
            predicted = torch.argmax(output, 1).item()
            confidence = probs[0][predicted].item() * 100

            print(f"  '{q}'")
            print(f"  → {CATEGORIES[predicted]} ({confidence:.1f}%)\n")


if __name__ == "__main__":
    train()
