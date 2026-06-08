import json

notebook_content = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 🍲 Nhận Diện Thức Ăn Bằng Trí Tuệ Nhân Tạo (Zero-Shot AI)\n",
    "\n",
    "Phiên bản này sử dụng mô hình **CLIP của OpenAI**. Mô hình này **đã có sẵn trí thông minh**, hiểu được hình ảnh và văn bản. \n",
    "**ƯU ĐIỂM TUYỆT ĐỐI:** \n",
    "- **KHÔNG CẦN TRAIN (HỌC):** Chạy được luôn ngay lập tức!\n",
    "- **KHÔNG CẦN DATASET:** Bạn thậm chí có thể xóa luôn thư mục `train`.\n",
    "- Nhận diện siêu chuẩn dựa trên kiến thức khổng lồ của AI."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 1. Cài đặt các thư viện AI mạnh nhất hiện nay (Chạy 1 lần)\n",
    "!pip install transformers torch torchvision ipywidgets pillow opencv-python"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import cv2\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import ipywidgets as widgets\n",
    "from IPython.display import display, clear_output\n",
    "from PIL import Image\n",
    "import torch\n",
    "from transformers import pipeline\n",
    "\n",
    "print(\"✅ Đã tải xong thư viện!\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 2. Thông tin các món ăn & Dịch sang tiếng Anh để AI hiểu\n",
    "FOOD_INFO = {\n",
    "    'Cá hú kho': {'english': 'Vietnamese braised fish in clay pot', 'price': '30,000 VNĐ', 'nutrition': '320 Kcal | 20g Protein | 22g Fat'},\n",
    "    'Canh chua có cá': {'english': 'Vietnamese sour soup with fish', 'price': '25,000 VNĐ', 'nutrition': '180 Kcal | 15g Protein | 8g Fat'},\n",
    "    'Canh chua không cá': {'english': 'Vietnamese sour soup with vegetables', 'price': '10,000 VNĐ', 'nutrition': '90 Kcal | 3g Protein | 2g Fat'},\n",
    "    'Canh rau': {'english': 'clear vegetable soup', 'price': '8,000 VNĐ', 'nutrition': '45 Kcal | 2g Protein | 1g Fat'},\n",
    "    'Cơm trắng': {'english': 'a bowl of white rice', 'price': '5,000 VNĐ', 'nutrition': '200 Kcal | 4g Protein | 45g Carbs'},\n",
    "    'Đậu hũ sốt cà': {'english': 'fried tofu in tomato sauce', 'price': '15,000 VNĐ', 'nutrition': '150 Kcal | 10g Protein | 9g Fat'},\n",
    "    'Rau xào': {'english': 'stir-fried green vegetables', 'price': '12,000 VNĐ', 'nutrition': '80 Kcal | 2g Protein | 6g Fat'},\n",
    "    'Sườn nướng': {'english': 'grilled pork chops', 'price': '35,000 VNĐ', 'nutrition': '380 Kcal | 28g Protein | 24g Fat'},\n",
    "    'Thịt kho': {'english': 'Vietnamese braised pork', 'price': '25,000 VNĐ', 'nutrition': '350 Kcal | 22g Protein | 25g Fat'},\n",
    "    'Thịt kho trứng': {'english': 'Vietnamese braised pork with hard-boiled eggs', 'price': '30,000 VNĐ', 'nutrition': '400 Kcal | 25g Protein | 30g Fat'},\n",
    "    'Trứng chiên': {'english': 'fried eggs', 'price': '10,000 VNĐ', 'nutrition': '120 Kcal | 10g Protein | 8g Fat'}\n",
    "}\n",
    "\n",
    "# Danh sách nhãn tiếng Anh để cho vào AI\n",
    "candidate_labels = [info['english'] for info in FOOD_INFO.values()]\n",
    "\n",
    "print(\"✅ Đã cấu hình Menu món ăn!\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 3. Tải Siêu Mô Hình AI (Chỉ tải 1 lần, KHÔNG CẦN TRAIN)\n",
    "print(\"⏳ Đang khởi động AI OpenAI CLIP (Zero-shot)... (Có thể mất 1-2 phút lần đầu để tải)\")\n",
    "classifier = pipeline(\"zero-shot-image-classification\", model=\"openai/clip-vit-large-patch14\")\n",
    "print(\"🚀 SIÊU MÔ HÌNH ĐÃ SẴN SÀNG! Bạn có thể nhận diện ngay lập tức!\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 4. Giao Diện Nhận Diện\n",
    "def predict_food(img_path):\n",
    "    img = Image.open(img_path).convert('RGB')\n",
    "    \n",
    "    # Hiển thị ảnh\n",
    "    plt.figure(figsize=(6,6))\n",
    "    plt.imshow(img)\n",
    "    plt.axis('off')\n",
    "    plt.show()\n",
    "    \n",
    "    print(\"\\n⏳ AI Đang suy nghĩ...\")\n",
    "    \n",
    "    # Phân tích ảnh với AI\n",
    "    result = classifier(img, candidate_labels=candidate_labels)\n",
    "    \n",
    "    best_match_english = result[0]['label']\n",
    "    confidence = result[0]['score'] * 100\n",
    "    \n",
    "    # Tìm lại tên tiếng Việt tương ứng\n",
    "    predicted_vietnamese = \"Unknown\"\n",
    "    for vn_name, info in FOOD_INFO.items():\n",
    "        if info['english'] == best_match_english:\n",
    "            predicted_vietnamese = vn_name\n",
    "            break\n",
    "            \n",
    "    info = FOOD_INFO.get(predicted_vietnamese, {})\n",
    "    \n",
    "    print(\"\\n\" + \"=\"*50)\n",
    "    print(f\"🎯 NHẬN DIỆN MÓN ĂN: {predicted_vietnamese.upper()}\")\n",
    "    print(f\"Độ tự tin của AI: {confidence:.2f}%\")\n",
    "    print(\"-\"*50)\n",
    "    if info:\n",
    "        print(f\"💰 GIÁ TIỀN:    {info['price']}\")\n",
    "        print(f\"⚡ DINH DƯỠNG:  {info['nutrition']}\")\n",
    "    print(\"=\"*50)\n",
    "\n",
    "# Giao diện Upload\n",
    "uploader = widgets.FileUpload(accept='image/*', multiple=False, description='Tải ảnh lên', button_style='success')\n",
    "out = widgets.Output()\n",
    "\n",
    "def on_upload(change):\n",
    "    with out:\n",
    "        clear_output()\n",
    "        if not uploader.value:\n",
    "            return\n",
    "            \n",
    "        if isinstance(uploader.value, dict):\n",
    "            fname = list(uploader.value.keys())[0]\n",
    "            content = uploader.value[fname]['content']\n",
    "        else:\n",
    "            content = uploader.value[0]['content']\n",
    "            \n",
    "        temp_path = 'temp_food_predict.jpg'\n",
    "        with open(temp_path, 'wb') as f:\n",
    "            f.write(content)\n",
    "            \n",
    "        predict_food(temp_path)\n",
    "        uploader.value.clear() if hasattr(uploader.value, 'clear') else None\n",
    "\n",
    "uploader.observe(on_upload, names='value')\n",
    "display(widgets.VBox([\n",
    "    widgets.HTML(\"<h3>📸 BẤM NÚT ĐỂ TẢI ẢNH MÓN ĂN CỦA BẠN LÊN:</h3>\"),\n",
    "    uploader,\n",
    "    out\n",
    "]))"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

with open('Vietnamese Dishes Recognition.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook_content, f, indent=1)

print("Updated Notebook to use Zero-Shot CLIP!")
