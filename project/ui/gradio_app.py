import gradio as gr
from core.chat_interface import ChatInterface
from core.document_manager import DocumentManager
from core.rag_system import RAGSystem
from pathlib import Path
import config
import os

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "..", "assets")

def create_gradio_ui():
    rag_system = RAGSystem()
    rag_system.initialize()
    
    doc_manager = DocumentManager(rag_system)
    chat_interface = ChatInterface(rag_system)
    
    # ===== 自动灌入 markdown_docs 中的预置文件 =====
    reindex_marker = Path(config.MARKDOWN_DIR).parent / ".reindex_required"
    force_reindex = reindex_marker.exists()
    collection_exists = rag_system.vector_db._VectorDbManager__client.collection_exists(rag_system.collection_name)
    preloaded = 0
    
    if collection_exists and not force_reindex:
        try:
            count = rag_system.vector_db._VectorDbManager__client.count(rag_system.collection_name).count
            if count > 1000:
                print(f"Qdrant 已有 {count} 条向量，跳过预置文档灌入")
                preloaded = -1
        except:
            pass

    if force_reindex:
        print(f"检测到重建标记: {reindex_marker.name}，将强制重建知识库索引")
    
    if preloaded >= 0:
        print("正在检查预置文档...")
        rag_system.vector_db.delete_collection(rag_system.collection_name)
        rag_system.vector_db.create_collection(rag_system.collection_name)
        rag_system.parent_store.clear_store()
        
        for md_path in sorted(Path(config.MARKDOWN_DIR).glob("*.md")):
            try:
                parent_chunks, child_chunks = rag_system.chunker.create_chunks_single(md_path)
                if child_chunks:
                    collection = rag_system.vector_db.get_collection(rag_system.collection_name)
                    collection.add_documents(child_chunks)
                    rag_system.parent_store.save_many(parent_chunks)
                    preloaded += 1
                    print(f"  已灌入: {md_path.name} ({len(parent_chunks)}父块/{len(child_chunks)}子块)")
            except Exception as e:
                print(f"  跳过 {md_path.name}: {e}")
        print(f"预置文档灌入完成: {preloaded} 篇")
        if reindex_marker.exists():
            reindex_marker.unlink()
            print("知识库重建完成，已清除重建标记")
    
    def format_file_list():
        files = doc_manager.get_markdown_files()
        if not files:
            return "📭 知识库中暂无文档"
        return "\n".join([f"📄 {f}" for f in files])
    
    def upload_handler(files, progress=gr.Progress()):
        if not files:
            return None, format_file_list()
            
        added, skipped = doc_manager.add_documents(
            files, 
            progress_callback=lambda p, desc: progress(p, desc=desc)
        )
        
        gr.Info(f"✅ 新增: {added} 篇 | 跳过: {skipped} 篇")
        return None, format_file_list()
    
    def clear_handler():
        doc_manager.clear_all()
        gr.Info("🗑️ 已清空全部文档")
        return format_file_list()
    
    def chat_handler(msg, hist):
        for chunk in chat_interface.chat(msg, hist):
            yield chunk
    
    def clear_chat_handler():
        chat_interface.clear_session()
    
    with gr.Blocks(title="中医医院智能知识库") as demo:
        
        # 顶部标题栏
        gr.HTML("""
        <div style="text-align: center; padding: 20px 0 10px 0;">
            <h1 style="font-size: 2em; margin: 0; color: #e67e22;">
                🏥 中医医院智能知识库
            </h1>
            <p style="color: #8b4513; font-size: 1.05em; margin-top: 8px;">
                基于 AI Agent 的中医文献检索与智能问答系统
            </p>
        </div>
        """)
        
        with gr.Tab("📚 知识库管理", elem_id="doc-management-tab"):
            gr.Markdown("## 添加新文档")
            gr.Markdown("上传中医文献 PDF 或 Markdown 文件，重复文件将自动跳过。")
            
            files_input = gr.File(
                label="拖拽或点击上传中医文献",
                file_count="multiple",
                type="filepath",
                height=200,
                show_label=False
            )
            
            add_btn = gr.Button("添加文档", variant="primary", size="md")
            
            gr.Markdown("## 当前知识库文档")
            file_list = gr.Textbox(
                value=format_file_list(),
                interactive=False,
                lines=7,
                max_lines=10,
                elem_id="file-list-box",
                show_label=False
            )
            
            with gr.Row():
                refresh_btn = gr.Button("刷新列表", size="md")
                clear_btn = gr.Button("清空全部", variant="stop", size="md")
            
            add_btn.click(upload_handler, [files_input], [files_input, file_list], show_progress="corner")
            refresh_btn.click(format_file_list, None, file_list)
            clear_btn.click(clear_handler, None, file_list)
        
        with gr.Tab("💬 智能问答"):
            chatbot = gr.Chatbot(
                height=720, 
                placeholder="<strong>请输入您的中医问题</strong><br><em>例如：麻黄汤的组成和功效是什么？ 阴虚火旺有哪些症状？ 黄芪的性味归经和临床应用？</em>",
                show_label=False,
                avatar_images=(None, os.path.join(ASSETS_DIR, "chatbot_avatar.png")),
                layout="bubble"
            )
            
            msg_input = gr.Textbox(
                placeholder="在此输入您的中医问题...",
                show_label=False,
                container=False,
                scale=7
            )
            submit_btn = gr.Button("发送", variant="primary", scale=1)
            
            def respond(message, history):
                history = history or []
                history.append({"role": "user", "content": message})
                full_response = ""
                for chunk in chat_interface.chat(message, history):
                    full_response += chunk
                    # 流式更新最后一条bot消息
                    current = [dict(h) for h in history] + [{"role": "assistant", "content": full_response}]
                    yield "", current
                history.append({"role": "assistant", "content": full_response})
                yield "", history
            
            submit_btn.click(respond, [msg_input, chatbot], [msg_input, chatbot])
            msg_input.submit(respond, [msg_input, chatbot], [msg_input, chatbot])
            chatbot.clear(clear_chat_handler)
    
    return demo
