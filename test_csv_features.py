"""
Test các tính năng CSV processing
"""
from config.process import (
    load_csv_from_source, 
    summarize_dataset,
    basic_stats,
    find_missing_values,
    plot_histogram,
    csv_process
)
import pandas as pd
import os

def test_local_file():
    """Test đọc file CSV local"""
    print("=" * 60)
    print("TEST 1: Đọc file CSV local")
    print("=" * 60)
    
    try:
        df = load_csv_from_source("test_data.csv")
        print(f"Đọc thành công: {len(df)} dòng, {len(df.columns)} cột")
        print(f"Các cột: {', '.join(df.columns.tolist())}")
    except Exception as e:
        print(f"❌ Lỗi: {e}")
    print()


def test_url_loading():
    """Test đọc CSV từ URL"""
    print("=" * 60)
    print("TEST 2: Đọc CSV từ GitHub URL")
    print("=" * 60)
    
    # URL mẫu - GitHub raw CSV
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    
    try:
        df = load_csv_from_source(url, max_rows=100)
        print(f"✅ Đọc thành công từ URL: {len(df)} dòng, {len(df.columns)} cột")
        print(f"Các cột: {', '.join(df.columns.tolist())}")
    except Exception as e:
        print(f"❌ Lỗi: {e}")
    print()


def test_summarize():
    """Test tóm tắt dataset"""
    print("=" * 60)
    print("TEST 3: Tóm tắt dataset")
    print("=" * 60)
    
    try:
        df = load_csv_from_source("test_data.csv")
        summary = summarize_dataset(df)
        print(summary)
        print("✅ Tóm tắt thành công")
    except Exception as e:
        print(f"❌ Lỗi: {e}")
    print()


def test_basic_stats():
    """Test thống kê cơ bản"""
    print("=" * 60)
    print("TEST 4: Thống kê cơ bản")
    print("=" * 60)
    
    try:
        df = load_csv_from_source("test_data.csv")
        stats = basic_stats(df)
        print(stats)
        print("✅ Thống kê thành công")
    except Exception as e:
        print(f"❌ Lỗi: {e}")
    print()


def test_missing_values():
    """Test tìm missing values"""
    print("=" * 60)
    print("TEST 5: Tìm missing values")
    print("=" * 60)
    
    try:
        df = load_csv_from_source("test_data.csv")
        missing = find_missing_values(df)
        print(missing)
        print("✅ Phân tích missing values thành công")
    except Exception as e:
        print(f"❌ Lỗi: {e}")
    print()


def test_histogram():
    """Test vẽ histogram"""
    print("=" * 60)
    print("TEST 6: Vẽ histogram")
    print("=" * 60)
    
    try:
        df = load_csv_from_source("test_data.csv")
        # Test với cột Tuổi
        hist = plot_histogram(df, "Tuổi")
        print(hist)
        print("✅ Vẽ histogram thành công")
    except Exception as e:
        print(f"❌ Lỗi: {e}")
    print()


def test_csv_process_questions():
    """Test các loại câu hỏi"""
    print("=" * 60)
    print("TEST 7: Test các loại câu hỏi")
    print("=" * 60)
    
    questions = [
        ("Tóm tắt dataset", "summarize the dataset"),
        ("Thống kê cơ bản", "show basic stats"),
        ("Missing values", "which column has missing values"),
        ("Custom", "Ai có lương cao nhất?")
    ]
    
    for name, question in questions:
        print(f"\n--- {name} ---")
        print(f"Câu hỏi: {question}")
        try:
            result = csv_process("test_data.csv", question)
            print(result[:500])  # In 500 ký tự đầu
            print("✅ Thành công")
        except Exception as e:
            print(f"❌ Lỗi: {e}")
        print()


def test_error_handling():
    """Test xử lý lỗi"""
    print("=" * 60)
    print("TEST 8: Xử lý lỗi")
    print("=" * 60)
    
    # Test file không tồn tại
    print("Test 8.1: File không tồn tại")
    result = csv_process("nonexistent.csv")
    print(result)
    print()
    
    # Test URL không hợp lệ
    print("Test 8.2: URL không hợp lệ")
    result = csv_process("https://invalid-url-12345.com/data.csv")
    print(result)
    print()
    
    # Test cột không tồn tại trong histogram
    print("Test 8.3: Cột không tồn tại")
    df = load_csv_from_source("test_data.csv")
    result = plot_histogram(df, "CộtKhôngTồnTại")
    print(result)
    print()


def main():
    """Chạy tất cả tests"""
    print("\n" + "="*60)
    print("CHẠY TEST CSV PROCESSING")
    print("="*60 + "\n")
    
    # Chỉ chạy tests có thể chạy offline
    test_local_file()
    test_summarize()
    test_basic_stats()
    test_missing_values()
    test_histogram()
    test_error_handling()
    
    # Test online (comment out nếu không có internet)
    print("\n⚠️ Tests yêu cầu internet và API keys:")
    print("- test_url_loading(): Cần internet")
    print("- test_csv_process_questions(): Cần GROQ_API_KEY")
    print("\nBỏ comment trong code để chạy.\n")
    
    # Uncomment để test:
    # test_url_loading()
    # test_csv_process_questions()
    
    print("="*60)
    print("✅ HOÀN THÀNH TẤT CẢ TESTS")
    print("="*60)


if __name__ == "__main__":
    main()

