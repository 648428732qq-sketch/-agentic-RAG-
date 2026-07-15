custom_css = """
    :root {
        --ink: #111111;
        --muted: #5d584b;
        --paper: #f6edcf;
        --paper-soft: #fffaf0;
        --white: #ffffff;
        --line: #111111;
        --accent: #d8aa28;
        --accent-soft: #f1d98a;
    }

    .progress-text {
        display: none !important;
    }

    body,
    .gradio-container {
        background: var(--paper) !important;
        color: var(--ink) !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei", "PingFang SC", sans-serif !important;
    }

    .gradio-container {
        max-width: 1180px !important;
        width: 100% !important;
        margin: 0 auto !important;
        padding: 22px !important;
    }

    #app-hero {
        background: var(--white);
        border: 2px solid var(--line);
        border-radius: 22px;
        box-shadow: 8px 8px 0 var(--ink);
        padding: 26px 30px;
        margin-bottom: 24px;
    }

    #app-hero .hero-mark {
        display: inline-block;
        background: var(--accent-soft);
        border: 1px solid var(--line);
        border-radius: 999px;
        color: var(--ink);
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 0.04em;
        padding: 5px 12px;
        margin-bottom: 12px;
    }

    #app-hero h1 {
        color: var(--ink) !important;
        font-size: 34px;
        line-height: 1.15;
        margin: 0;
    }

    #app-hero p {
        color: var(--muted);
        margin: 10px 0 0;
        font-size: 15px;
    }

    .paper-card {
        background: var(--paper-soft) !important;
        border: 2px solid var(--line) !important;
        border-radius: 18px !important;
        box-shadow: 6px 6px 0 var(--ink) !important;
        padding: 18px !important;
    }

    .chat-card {
        padding: 18px 18px 14px !important;
    }

    .tabs,
    .tab-nav {
        background: transparent !important;
        border: none !important;
    }

    button[role="tab"] {
        background: var(--white) !important;
        border: 2px solid var(--line) !important;
        border-radius: 999px !important;
        color: var(--ink) !important;
        font-weight: 700 !important;
        margin-right: 8px !important;
        padding: 8px 16px !important;
    }

    button[role="tab"][aria-selected="true"] {
        background: var(--accent-soft) !important;
        color: var(--ink) !important;
    }

    button {
        border: 2px solid var(--line) !important;
        border-radius: 12px !important;
        box-shadow: 3px 3px 0 var(--ink) !important;
        font-weight: 700 !important;
        transition: transform 0.12s ease, box-shadow 0.12s ease !important;
    }

    button:hover {
        transform: translate(1px, 1px) !important;
        box-shadow: 2px 2px 0 var(--ink) !important;
    }

    .primary {
        background: var(--ink) !important;
        color: var(--white) !important;
    }

    .stop {
        background: var(--white) !important;
        color: var(--ink) !important;
    }

    #send-btn,
    #clear-chat-btn {
        min-width: 88px !important;
    }

    input,
    textarea {
        background: var(--white) !important;
        border: 2px solid var(--line) !important;
        border-radius: 14px !important;
        color: var(--ink) !important;
        font-size: 15px !important;
        box-shadow: none !important;
    }

    input:focus,
    textarea:focus {
        border-color: var(--line) !important;
        outline: none !important;
        box-shadow: 0 0 0 3px var(--accent-soft) !important;
    }

    textarea::placeholder,
    input::placeholder {
        color: #7d7665 !important;
    }

    textarea[readonly] {
        background: var(--white) !important;
        color: var(--ink) !important;
    }

    #upload-box,
    [data-testid="file-upload"],
    .file-preview {
        background: var(--white) !important;
        border: 2px dashed var(--line) !important;
        border-radius: 16px !important;
        color: var(--ink) !important;
    }

    #upload-box *,
    [data-testid="file-upload"] *,
    .file-preview * {
        color: var(--ink) !important;
    }

    #file-list-box {
        background: var(--white) !important;
        border: 2px solid var(--line) !important;
        border-radius: 16px !important;
        color: var(--ink) !important;
    }

    #file-list-box textarea {
        background: transparent !important;
        border: none !important;
        color: var(--ink) !important;
        padding: 0 !important;
    }

    #rag-chatbot,
    #rag-chatbot > div,
    #rag-chatbot .chatbot,
    #rag-chatbot .message-wrap,
    #rag-chatbot [class*="chatbot"],
    #rag-chatbot [class*="messages"] {
        background: var(--white) !important;
        border: 2px solid var(--line) !important;
        border-radius: 18px !important;
        color: var(--ink) !important;
    }

    #rag-chatbot > div,
    #rag-chatbot .chatbot,
    #rag-chatbot .message-wrap,
    #rag-chatbot [class*="chatbot"],
    #rag-chatbot [class*="messages"] {
        border: none !important;
        box-shadow: none !important;
    }

    #rag-chatbot .message,
    #rag-chatbot [data-testid="user"],
    #rag-chatbot [data-testid="bot"] {
        border-radius: 14px !important;
        color: var(--ink) !important;
    }

    #rag-chatbot .message.user,
    #rag-chatbot [data-testid="user"] {
        background: #ffe28a !important;
        color: var(--ink) !important;
        border: 2px solid var(--ink) !important;
    }

    #rag-chatbot .message.user *,
    #rag-chatbot [data-testid="user"] * {
        color: var(--ink) !important;
    }

    #rag-chatbot .message.bot,
    #rag-chatbot [data-testid="bot"] {
        background: var(--white) !important;
        color: var(--ink) !important;
        border: 2px solid var(--line) !important;
    }

    #rag-chatbot .message.bot *,
    #rag-chatbot [data-testid="bot"] * {
        color: var(--ink) !important;
    }

    #rag-chatbot pre,
    #rag-chatbot code {
        background: #fff5c7 !important;
        color: var(--ink) !important;
        border: 1px solid var(--line) !important;
        border-radius: 8px !important;
    }

    #rag-chatbot .markdown,
    #rag-chatbot .prose,
    #rag-chatbot p,
    #rag-chatbot li,
    #rag-chatbot span {
        color: var(--ink) !important;
    }

    #chat-input-row {
        align-items: stretch !important;
        gap: 10px !important;
        margin-top: 12px !important;
    }

    #chat-input textarea {
        min-height: 64px !important;
    }

    h1,
    h2,
    h3,
    h4,
    h5,
    h6,
    .prose {
        color: var(--ink) !important;
    }

    .prose a {
        color: var(--ink) !important;
        text-decoration: underline !important;
        text-decoration-color: var(--accent) !important;
        text-decoration-thickness: 2px !important;
    }

    .toast-wrap,
    .toast,
    .notification {
        color: var(--ink) !important;
    }

    footer {
        visibility: hidden;
    }
"""
