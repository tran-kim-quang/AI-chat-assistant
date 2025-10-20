from config.process import text_process, img_process, csv_process

def main():
    print("=" * 50)
    print("TEST 1: Xử lý văn bản")
    print("=" * 50)
    text = text_process("Xin chào, bạn có thể giới thiệu về bản thân không?")
    print(text)
    print()
    
    print("=" * 50)
    print("TEST 2: Xử lý ảnh")
    print("=" * 50)
    img = img_process("test_img/haerin.jpg")
    print(img)
    print()
    
    print("=" * 50)
    print("TEST 3: Xử lý CSV - Thông tin tổng quan")
    print("=" * 50)
    csv_overview = csv_process("test_data.csv")
    print(csv_overview)
    print()
    
    print("=" * 50)
    print("TEST 4: Xử lý CSV - Trả lời câu hỏi")
    print("=" * 50)
    csv_question = csv_process("test_data.csv", "Ai có lương cao nhất và bao nhiêu?")
    print(csv_question)
    print()

if __name__ == "__main__":
    main()