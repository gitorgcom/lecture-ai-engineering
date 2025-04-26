# --- 課題 ---
import os
import torch
from transformers import pipeline
import streamlit as st
import time
from config import MODEL_NAME

# モデルをキャッシュして再利用
@st.cache_resource
def load_model():
    """LLMモデルをロードする"""
    try:
        # アクセストークンを保存
        hf_token = st.secrets["huggingface"]["token"]
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        st.info(f"Using device: {device}") # 使用デバイスを表示
        
        # text-generation 用に pipeline を修正
        pipe = pipeline(
            "text-generation",  # チャット生成用ではなくテキスト生成用に変更
            model=MODEL_NAME,
            tokenizer=MODEL_NAME,
            device=0 if device == "cuda" else -1  # GPU利用の場合は 0 を指定
        )
        st.success(f"モデル '{MODEL_NAME}' の読み込みに成功しました。")
        return pipe
    except Exception as e:
        st.error(f"モデル '{MODEL_NAME}' の読み込みに失敗しました: {e}")
        st.error("GPUメモリ不足の可能性があります。不要なプロセスを終了するか、より小さいモデルの使用を検討してください。")
        return None

def generate_response(pipe, user_question):
    """LLMを使用して質問に対する回答を生成する"""
    if pipe is None:
        return "モデルがロードされていないため、回答を生成できません。", 0

    try:
        start_time = time.time()

        # GPT-2 用にシンプルなテキスト生成を行う
        outputs = pipe(user_question, max_length=200, num_return_sequences=1)

        # 出力の最初のテキストを取得
        assistant_response = outputs[0]["generated_text"].strip()

        end_time = time.time()
        response_time = end_time - start_time
        print(f"Generated response in {response_time:.2f}s")  # デバッグ用
        return assistant_response, response_time

    except Exception as e:
        st.error(f"回答生成中にエラーが発生しました: {e}")
        # エラーの詳細をログに出力
        import traceback
        traceback.print_exc()
        return f"エラーが発生しました: {str(e)}", 0
