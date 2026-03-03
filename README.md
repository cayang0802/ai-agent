# ai-agent

Gradio 聊天介面 + LangChain 工具型 agent，支援台股報價、台灣天氣查詢。

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

## 啟動

```bash
python main.py
```
