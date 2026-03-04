# ai-agent

Gradio 聊天介面 + LangChain 工具型 agent，支援台股報價、台灣天氣查詢與 PDF 文件索引。

## 安裝

```bash
pip install -r requirements.txt
```

## 設定

### `.env`

參考 `.env.example`，在專案根目錄新增 `.env` 並填入金鑰。

- `CWA_API_KEY`：至 [中央氣象署開放資料平台](https://opendata.cwa.gov.tw/) 註冊後取得。

### `config.yaml`

調整 LLM 設定或 system prompt：

## Langfuse（選用）

Langfuse 用於追蹤 RAG 評估指標（Faithfulness、AnswerRelevancy）。

**1. 啟動本地 Langfuse 伺服器**

```bash
docker compose up -d
```

**2. 設定環境變數**（加入 `.env`）

```
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_HOST=http://localhost:3000
```

**3. 啟用評估**（`config.yaml`）

```yaml
evaluator:
  enabled: true
```

Langfuse 儀表板預設位於 http://localhost:3000。

## 啟動

```bash
python main.py
```
