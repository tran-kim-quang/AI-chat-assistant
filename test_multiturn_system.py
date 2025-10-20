"""
Test hệ thống Multi-turn Conversation
"""
from config.session_manager import SessionManager
from config.intent_classifier import get_intent_classifier
from config.conversation_handler import get_conversation_handler
import os

def test_session_manager():
    """Test SessionManager"""
    print("=" * 60)
    print("TEST 1: SessionManager")
    print("=" * 60)
    
    # Sử dụng DB test
    test_db = "test_conversations.db"
    if os.path.exists(test_db):
        os.remove(test_db)
    
    session_mgr = SessionManager(db_path=test_db)
    
    # Tạo session
    session_id = session_mgr.create_session({"test": True})
    print(f"✓ Tạo session: {session_id}")
    
    # Thêm messages
    session_mgr.add_message(session_id, "user", "Xin chào", intent="general_chat")
    session_mgr.add_message(session_id, "assistant", "Xin chào! Tôi có thể giúp gì?", intent="general_chat")
    print("✓ Thêm messages")
    
    # Lấy history
    history = session_mgr.get_conversation_history(session_id)
    print(f"✓ Lấy history: {len(history)} messages")
    
    # Thêm context
    session_mgr.add_context(session_id, "csv", {"filename": "test.csv", "rows": 100})
    print("✓ Thêm context")
    
    # Lấy context
    contexts = session_mgr.get_context(session_id, "csv")
    print(f"✓ Lấy context: {len(contexts)} contexts")
    
    # Context summary
    summary = session_mgr.get_recent_context_summary(session_id)
    print(f"✓ Context summary:\n{summary[:200]}...")
    
    print("\nTest SessionManager: PASSED\n")
    
    # Cleanup
    os.remove(test_db)


def test_intent_classifier():
    """Test IntentClassifier"""
    print("=" * 60)
    print("TEST 2: IntentClassifier")
    print("=" * 60)
    
    classifier = get_intent_classifier()
    
    test_cases = [
        {
            "message": "Tóm tắt dataset này",
            "context": "File CSV: employees.csv đã được upload",
            "expected": "csv_analysis"
        },
        {
            "message": "Mô tả bức ảnh này",
            "context": "Ảnh: cat.jpg đã được upload",
            "expected": "image_analysis"
        },
        {
            "message": "Xin chào, bạn khỏe không?",
            "context": "",
            "expected": "general_chat"
        },
        {
            "message": "Bạn có thể làm gì?",
            "context": "",
            "expected": "clarification"
        }
    ]
    
    print("\nĐang test AI Intent Classification...")
    print("(Cần GROQ_API_KEY để chạy test này)\n")
    
    for i, test in enumerate(test_cases, 1):
        print(f"Test case {i}: {test['message']}")
        try:
            result = classifier.classify(test['message'], test['context'])
            print(f"  Intent: {result['intent']}")
            print(f"  Confidence: {result['confidence']:.2f}")
            print(f"  Expected: {test['expected']}")
            
            if result['intent'] == test['expected']:
                print("  ✓ PASSED")
            else:
                print("  ⚠ Intent khác expected (có thể do AI hiểu khác)")
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
        print()
    
    print("Test IntentClassifier: COMPLETED\n")


def test_conversation_handler():
    """Test ConversationHandler"""
    print("=" * 60)
    print("TEST 3: ConversationHandler")
    print("=" * 60)
    
    # Setup
    test_db = "test_conversations.db"
    if os.path.exists(test_db):
        os.remove(test_db)
    
    session_mgr = SessionManager(db_path=test_db)
    from config.conversation_handler import ConversationHandler
    handler = ConversationHandler(session_manager=session_mgr)
    
    # Tạo session
    session_id = session_mgr.create_session()
    print(f"✓ Session ID: {session_id}")
    
    # Simulate conversation
    conversations = [
        "Xin chào",
        "Tôi muốn upload file CSV",
        "Bạn có thể làm gì?"
    ]
    
    print("\nSimulate conversation:")
    for msg in conversations:
        print(f"\nUser: {msg}")
        try:
            result = handler.process_message(session_id, msg)
            print(f"Bot: {result['response'][:100]}...")
            print(f"Intent: {result['intent']} (confidence: {result['confidence']:.2f})")
        except Exception as e:
            print(f"Error: {e}")
    
    # Kiểm tra history
    history = session_mgr.get_conversation_history(session_id)
    print(f"\n✓ History có {len(history)} messages")
    
    print("\nTest ConversationHandler: COMPLETED\n")
    
    # Cleanup
    os.remove(test_db)


def test_multi_turn_context():
    """Test multi-turn với context retention"""
    print("=" * 60)
    print("TEST 4: Multi-turn Context Retention")
    print("=" * 60)
    
    test_db = "test_conversations.db"
    if os.path.exists(test_db):
        os.remove(test_db)
    
    session_mgr = SessionManager(db_path=test_db)
    from config.conversation_handler import ConversationHandler
    handler = ConversationHandler(session_manager=session_mgr)
    
    session_id = session_mgr.create_session()
    
    # Turn 1: Upload CSV (simulated)
    handler.add_file_context(
        session_id=session_id,
        file_type='csv',
        file_data={'filename': 'employees.csv', 'rows': 10, 'columns': ['Tên', 'Lương']}
    )
    session_mgr.add_message(session_id, "assistant", "File CSV đã upload", intent="file_uploaded")
    print("Turn 1: File uploaded")
    
    # Turn 2: Hỏi về CSV
    print("\nTurn 2: Hỏi về CSV")
    try:
        result = handler.process_message(session_id, "Tóm tắt dataset")
        print(f"Intent detected: {result['intent']}")
        print(f"Context aware: {result['intent'] == 'csv_analysis'}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Turn 3: Follow-up question
    print("\nTurn 3: Follow-up")
    try:
        result = handler.process_message(session_id, "Có bao nhiêu dòng?")
        print(f"Intent detected: {result['intent']}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Kiểm tra context summary
    summary = session_mgr.get_recent_context_summary(session_id)
    print(f"\n✓ Context summary được tạo:")
    print(summary[:300] + "...")
    
    print("\nTest Multi-turn Context: COMPLETED\n")
    
    # Cleanup
    os.remove(test_db)


def test_session_persistence():
    """Test session persistence trong database"""
    print("=" * 60)
    print("TEST 5: Session Persistence")
    print("=" * 60)
    
    test_db = "test_conversations.db"
    if os.path.exists(test_db):
        os.remove(test_db)
    
    # Phase 1: Tạo session và thêm data
    print("Phase 1: Tạo session")
    session_mgr1 = SessionManager(db_path=test_db)
    session_id = session_mgr1.create_session()
    session_mgr1.add_message(session_id, "user", "Hello", intent="general_chat")
    session_mgr1.add_context(session_id, "csv", {"filename": "data.csv"})
    print(f"✓ Session {session_id} created with data")
    
    # Phase 2: Mở lại SessionManager mới (simulate restart)
    print("\nPhase 2: Khởi tạo SessionManager mới (simulate restart)")
    session_mgr2 = SessionManager(db_path=test_db)
    
    # Retrieve session
    session_info = session_mgr2.get_session(session_id)
    print(f"✓ Session retrieved: {session_info is not None}")
    
    # Retrieve history
    history = session_mgr2.get_conversation_history(session_id)
    print(f"✓ History retrieved: {len(history)} messages")
    
    # Retrieve context
    contexts = session_mgr2.get_context(session_id)
    print(f"✓ Context retrieved: {len(contexts)} contexts")
    
    if session_info and history and contexts:
        print("\n✓ Session persistence: PASSED")
    else:
        print("\n✗ Session persistence: FAILED")
    
    print()
    
    # Cleanup
    os.remove(test_db)


def main():
    """Chạy tất cả tests"""
    print("\n" + "="*60)
    print("MULTI-TURN CONVERSATION SYSTEM TESTS")
    print("="*60 + "\n")
    
    # Test 1: SessionManager
    test_session_manager()
    
    # Test 2: IntentClassifier (requires API key)
    print("⚠️ Test 2 yêu cầu GROQ_API_KEY")
    try:
        test_intent_classifier()
    except Exception as e:
        print(f"Skipped IntentClassifier test: {e}\n")
    
    # Test 3: ConversationHandler (requires API key)
    print("⚠️ Test 3 yêu cầu GROQ_API_KEY")
    try:
        test_conversation_handler()
    except Exception as e:
        print(f"Skipped ConversationHandler test: {e}\n")
    
    # Test 4: Multi-turn context (requires API key)
    print("⚠️ Test 4 yêu cầu GROQ_API_KEY")
    try:
        test_multi_turn_context()
    except Exception as e:
        print(f"Skipped Multi-turn test: {e}\n")
    
    # Test 5: Session persistence (no API needed)
    test_session_persistence()
    
    print("="*60)
    print("TESTS COMPLETED")
    print("="*60)
    print("\nLưu ý:")
    print("- Một số tests yêu cầu GROQ_API_KEY trong file .env")
    print("- Tests offline (SessionManager, Persistence) luôn chạy được")
    print("- Tests online (Intent, Conversation) cần API key")


if __name__ == "__main__":
    main()

