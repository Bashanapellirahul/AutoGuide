from pathlib import Path
from streamlit.testing.v1 import AppTest


def test_widget_greeting_and_history():
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run(timeout=30)
    assert not app.exception
    app.chat_input[0].set_value("Hello").run(timeout=30)
    assert not app.exception
    assert len(app.chat_message) == 2
    assert "Hello!" in app.chat_message[1].markdown[0].value
    next(button for button in app.button if button.label == "Clear conversation").click().run(timeout=30)
    assert not app.exception
    assert len(app.chat_message) == 0


def test_widget_real_recall_answer():
    from rag.retrieval import load_chunks
    import pytest
    if not load_chunks():
        pytest.skip("Local corpus not collected")
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run(timeout=30)
    app.chat_input[0].set_value("2020 Honda Accord airbag recall").run(timeout=30)
    assert not app.exception
    assert not app.error
    assert "HONDA" in app.chat_message[1].markdown[0].value
    assert app.session_state["messages"][-1]["sources"]


def test_widget_tpms_explains_mechanism():
    from rag.retrieval import load_chunks
    import pytest
    if not any(c.get("topic") == "tire pressure monitoring" for c in load_chunks()):
        pytest.skip("TPMS source has not been collected")
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run(timeout=30)
    app.chat_input[0].set_value("What is TPMS").run(timeout=60)
    assert not app.exception
    assert not app.error
    answer = app.session_state["messages"][-1]["content"]
    assert "pressure sensor in each wheel" in answer
    assert "indirect TPMSs" in answer
    assert "Here are relevant passages" not in answer
    assert "not required" not in answer
