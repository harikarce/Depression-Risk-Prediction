# 🧠 Depression Risk Prediction

Hi, I'm **Harika**! 👋

This project focuses on **Mental Health Wellness** and **Depression Risk Prediction**. I developed this application using **Python** and **TensorFlow** to analyze various factors and predict the likelihood of depression, helping to raise awareness and provide early insights.

## 🛠️ Tech Stack
* **Python**
* **TensorFlow / Keras** (Deep Learning)
* **Streamlit** (Web Interface)
* **Pandas & NumPy** (Data Processing)

---

## ☁️ Deployment on AWS EC2

I have successfully deployed this application on AWS. If you would like to deploy it yourself, follow these steps:

### 1. Launch an Instance
* Log in to your **AWS Console**.
* Launch a new **EC2 Instance** (Ubuntu is recommended).
* **Important:** In the "Security Group" settings, add a **Custom TCP Rule** to allow traffic on **Port 8501** (0.0.0.0/0).

### 2. Setup & Installation
Connect to your instance via SSH or the EC2 Instance Connect console and run the following commands:

```bash
# 1. Update the system
sudo apt update

# 2. Install Python pip
sudo apt install python3-pip -y

# 3. Clone this repository
git clone <YOUR_GITHUB_URL>

# 4. Navigate into the project folder
cd Depression-Risk-Prediction

# 5. Install dependencies
pip3 install -r requirements.txt
